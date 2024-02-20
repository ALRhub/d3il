
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import hydra
from omegaconf import DictConfig
import einops 
import wandb

from beso.agents.lmp_agents.lmp_modules.utils import State

logger = logging.getLogger(__name__)


class MaskedPlayLMP(nn.Module):
    def __init__(
        self,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        action_decoder: DictConfig,
        state_reconstruction: DictConfig,
        plan_features: int,
        kl_beta: float,
        replan_freq: int = 30,
        device: str = 'cuda',
        alpha: float = 0.0,
        use_goal_in_recognition: bool = False,
        fine_tune_action_decoder: bool=False
    ):
        super(MaskedPlayLMP, self).__init__()
        self.device = device
        self.plan_features = plan_features
        self.plan_proposal = hydra.utils.instantiate(plan_proposal).to(self.device)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition).to(self.device)
        self.action_decoder = hydra.utils.instantiate(action_decoder).to(self.device)
        self.state_reconstruction = hydra.utils.instantiate(state_reconstruction).to(self.device)
        # get prior plan distribution
        # self.plan_prior 
        self.kl_beta = kl_beta
        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None
        # weight for kl loss 
        self.alpha = alpha
        self.use_goal_in_recognition = use_goal_in_recognition
        self.fine_tune_action_decoder = fine_tune_action_decoder

    def compute_loss(self, states, actions, goal):
        '''
        Method to compute the loss for the model during training.
        '''
        mae_loss = None
        if self.use_goal_in_recognition:
            state_sequence = torch.cat([states, goal], dim=1)
        else:
            state_sequence = states
        sampled_plan, mae_loss, state_loss, proposal_loss = self.plan_recognition(state_sequence, actions)

        # state reconstruction
        if self.use_goal_in_recognition:
            goal_state = goal
        else:
            goal_state = einops.rearrange(states[:, -1, :], 'b d -> b 1 d')
            
        # compute state reconstruction loss
        first_state_loss, state_loss_2 = self.state_reconstruction.compute_reconstruction_losses(sampled_plan, states, goal_state)
        
        # action decoder prediction
        action_loss, _ = self.action_decoder.loss_and_act(sampled_plan, states, goal, actions)

        # total loss computation and logging of metrics
        total_loss = 0
        if mae_loss is not None:
            wandb.log({"training/mae_loss": mae_loss})
        if action_loss is not None:
            wandb.log({"training/action_loss": action_loss})
            total_loss += action_loss
        if state_loss is not None:
            wandb.log({"training/state_loss": state_loss})
            total_loss += state_loss
        if proposal_loss is not None:
            total_loss += proposal_loss
            wandb.log({"training/proposal_loss": proposal_loss})
        if first_state_loss is not None:
            wandb.log({"training/first_state_loss": first_state_loss})
            total_loss += first_state_loss
        if state_loss is not None:
            wandb.log({"training/state_loss_2": state_loss_2})
            total_loss += state_loss_2
        wandb.log({"training/plans/plan_mean": sampled_plan.mean()})
        wandb.log({"training/plans/plan_std": sampled_plan.std()})
        wandb.log({"training/plans/plan_max": sampled_plan.max()})
        wandb.log({"training/plans/plan_min": sampled_plan.min()})
        return total_loss
    
    def compute_finetune_loss(self, states, actions, goal):
        '''
        Method to compute the loss for the model during fine tuning.
        This is required to train the plan proposal model if necessary.
        '''
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')
        
        if self.use_goal_in_recognition:
            state_sequence = torch.cat([states, goal], dim=1)
        else:
            state_sequence = states
        
        with torch.no_grad():
            target_plan = self.plan_recognition.predict(state_sequence)
        
        # plan proposal diffusion loss 
        diffusion_loss, sampled_plan = self.plan_proposal.loss_and_pred(first_state, goal, target_plan)
        
        # log the plan statistics
        action_loss = None
        total_loss = diffusion_loss
        wandb.log({"fine_tuning/diffusion_loss": diffusion_loss})
        if action_loss is not None:
            wandb.log({"fine_tuning/action_loss": action_loss})
            total_loss += action_loss
        return total_loss
    
    def freeze_model_weights(self):
        '''
        Freeze the weights of the model during fine tuning.
        '''
        self.plan_recognition.requires_grad_(False)
        if not self.fine_tune_action_decoder:
            self.action_decoder.requires_grad_(False)
    
    @torch.no_grad()
    def compute_val_loss(self, states, actions, goal):
        '''
        Compute the validation loss for the model.
        '''
        if self.use_goal_in_recognition:
            state_sequence = torch.cat([states, goal], dim=1)
        else:
            state_sequence = states
        
        sampled_plan = self.plan_recognition.predict(state_sequence)
        # action decoder prediction
        pred_actions = self.action_decoder.act(sampled_plan, states, goal)
        self.action_decoder.clear_hidden_state()
        # compute the action prediction loss
        mse = F.mse_loss(pred_actions, actions, reduction="none")
        return mse
    
    @torch.no_grad()
    def compute_tuning_val_loss(self, states, actions, goal):
        '''
        Compute the validation loss for the model during fine tuning.
        The focus is on the plan proposal model and a fine tuned action decoder.
        '''
        # plan proposal
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')
        
        if self.use_goal_in_recognition:
            state_sequence = torch.cat([states, goal], dim=1)
        else:
            state_sequence = states
        pred_plan = self.plan_proposal.predict(first_state, goal)
        sampled_plan = self.plan_recognition.predict(state_sequence)
        
        # action decoder prediction
        action_loss = None
        
        action_loss, pred_actions = self.action_decoder.loss_and_act(pred_plan, states, goal, actions)
        self.action_decoder.clear_hidden_state()
        action_loss = F.mse_loss(pred_actions, actions, reduction="none").mean()
    
        plan_loss = F.mse_loss(pred_plan, sampled_plan, reduction="none").mean() #.mean(dim=-1).sum().item()
        wandb.log({"fine_tuning/plan_loss": plan_loss})
        if action_loss is not None:
            total_loss += action_loss 
        return action_loss
    
    def reset(self):
        '''
        Reset method to reset the rollout step counter and the plans and history.
        '''
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0
    
    @torch.no_grad()
    def step(self, state, goal):
        '''
        Method to take a step in the environment.
        '''
        if self.rollout_step_counter % self.replan_freq == 0:
        # replan every replan_freq steps (default 30 i.e every second)
            self.plan = self.get_pp_plan(state, goal)
            self.latent_goal = goal
        # use plan to predict actions with current observations
        # action decoder
        action = self.action_decoder.act(self.plan, state, self.latent_goal)
        self.rollout_step_counter += 1
        return action, self.plan
    
    @torch.no_grad()
    def get_pp_plan(self, state, goal):
        '''
        Method to sample a plan during rollouts
        '''
        sampled_plan = self.plan_recognition.proposal_forward(state, goal)
        return sampled_plan

    def predict_with_plan(
        self,
        obs: Dict[str, Dict],
        latent_goal: torch.Tensor,
        sampled_plan: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            action = self.action_decoder.act(sampled_plan, perceptual_emb, latent_goal)

        return action

    def get_params(self):
        return self.parameters()

