
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig
import einops 
import wandb
from sklearn.neighbors import KernelDensity

from beso.agents.lmp_agents.lmp_modules.utils import State
from beso.agents.lmp_agents.lmp_modules.plan_encoders.diffusion_proposal_net import DiffusionProposalNet

logger = logging.getLogger(__name__)


class PlayLMP(nn.Module):
    def __init__(
        self,
        plan_proposal: DictConfig,
        plan_prior: DictConfig,
        plan_recognition: DictConfig,
        action_decoder: DictConfig,
        action_reconstruction: DictConfig,
        dist: DictConfig,
        plan_features: int,
        kl_beta: float,
        use_zero_mean_guassian_prior: bool = False,
        replan_freq: int = 15,
        train_action_decoder: bool = True,
        use_different_plan_proposal: bool = False,
        fine_tune_action_decoder: bool = False,
        use_action_reconstruction: bool = False,
        use_additional_reconstruction_loss: bool = False,
        device: str = 'cuda',
        alpha: float = 0.0,
        use_goal_in_recognition: bool = False,
    ):
        super(PlayLMP, self).__init__()
        self.device = device
        self.plan_features = plan_features
        self.dist = hydra.utils.instantiate(dist)
        if 'num_sampling_steps' in plan_proposal:
            self.plan_proposal = hydra.utils.instantiate(plan_proposal).to(self.device)
        else:
            self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.dist).to(self.device)
        
        # self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.dist).to(self.device)
        self.use_zero_mean_guassian_prior = use_zero_mean_guassian_prior
        self.plan_prior = hydra.utils.instantiate(plan_prior, dist=self.dist).to(self.device)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.dist).to(self.device)
        self.action_decoder = hydra.utils.instantiate(action_decoder).to(self.device)
        # this model variant has the option to us an action reconstruction model, that is seperate from
        # the final action decoder 
        self.action_reconstruction = hydra.utils.instantiate(action_reconstruction).to(self.device)
        self.use_action_reconstruction = use_action_reconstruction
        self.use_different_plan_proposal = use_different_plan_proposal
        # this model variant has the option to use an additional reconstruction loss
        self.use_additional_reconstruction_loss = use_additional_reconstruction_loss
        self.kl_beta = kl_beta
        # boolean to decide whether to train action decoder in pretraining or finetuning
        self.train_action_decoder = train_action_decoder
        self.fine_tune_action_decoder = fine_tune_action_decoder
        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None
        # weight for kl loss 
        self.alpha = alpha
        self.use_goal_in_recognition = use_goal_in_recognition

    def compute_loss(self, states, actions, goal):
        '''
        Computes the loss during the first training phase
        
        The plan recognition network is trained to reconstruct the plan from the states and goal
        There exist two options for the plan proposal: either a seperate plan prior network 
        is used or the plan prior is directly trained
        Further there exits the option to use an action reconstruction model, 
        that is seperate from the final action decoder.
        
        '''
        if self.use_goal_in_recognition:
            state_sequence = torch.cat([states, goal], dim=1)
        else:
            state_sequence = states
        pr_state, x = self.plan_recognition(state_sequence)
        pr_dist = self.dist.get_dist(pr_state)
        
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')
        
        # plan proposal training or plan prior training
        if self.use_different_plan_proposal:
            if not self.use_zero_mean_guassian_prior:
                pp_state = self.plan_prior(first_state, goal)
        else:
            pp_state = self.plan_proposal(first_state, goal)

        sampled_plan = pr_dist.rsample()  # sample from recognition net
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)

        # action reconstruction, we can either use the action decoder or the additional action reconstruction model
        # to reconstruct the actions to train the latent plans 
        if self.train_action_decoder:
            action_loss, _ = self.action_decoder.loss_and_act(sampled_plan, states, goal, actions)
            self.action_decoder.clear_hidden_state()
        elif self.use_action_reconstruction:
            action_loss, _ = self.action_reconstruction.loss_and_act(sampled_plan, states, goal, actions)
            self.action_reconstruction.clear_hidden_state()
        else:
            action_loss = None
        
        # compute the losses and log them for wandb
        total_loss = 0
        if self.use_zero_mean_guassian_prior:
            kl_loss = self.compute_kl_loss_to_gaussian(pr_state)
        else:
            kl_loss = self.compute_kl_loss(pr_state, pp_state)
        
        if action_loss is not None:
            total_loss += action_loss 
            wandb.log({"training/action_loss": action_loss.item()})
        
        wandb.log({"training/kl_loss": kl_loss.item()})
        wandb.log({"training/plans/plan_mean": sampled_plan.mean()})
        wandb.log({"training/plans/plan_std": sampled_plan.std()})
        wandb.log({"training/plans/plan_max": sampled_plan.max()})
        wandb.log({"training/plans/plan_min": sampled_plan.min()})
        total_loss += kl_loss
        return total_loss
    
    def compute_additional_loss(self, states, actions, goal):
        
        pass
    
    @torch.no_grad()
    def compute_val_loss(self, states, actions, goal):
        '''
        Method to compute the validation loss during the first training phase.
        Depending on the model variant, the validation loss can be computed in different ways.
        '''
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')

        if self.use_different_plan_proposal:
            if not self.use_zero_mean_guassian_prior:
                pp_state = self.plan_prior(first_state, goal)
                pp_dist = self.dist.get_dist(pp_state)
                sampled_plan = pp_dist.rsample()  
            else:
                # if we use a normal gaussian as prior we sample from the recongition
                if self.use_goal_in_recognition:
                    state_sequence = torch.cat([states, goal], dim=1)
                else:
                    state_sequence = states
                pr_state, x = self.plan_recognition(state_sequence)
                pr_dist = self.dist.get_dist(pr_state)
                sampled_plan = pr_dist.rsample()
        else:
            if isinstance(self.plan_proposal, DiffusionProposalNet):
                sampled_plan = self.plan_proposal(first_state, goal)
            else:
                pp_state = self.plan_proposal(first_state, goal)
                pp_dist = self.dist.get_dist(pp_state)
                sampled_plan = pp_dist.rsample() 
            # pp_dist = self.dist.get_dist(pp_state)
        
            sampled_plan = pp_dist.rsample()  # sample from proposal net
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)

        # action decoder to predict the actions from the latent plan
        if self.train_action_decoder:
            pred_actions = self.action_decoder.act(sampled_plan, states, goal)
            self.action_decoder.clear_hidden_state()
        elif self.use_action_reconstruction:
            pred_actions = self.action_reconstruction.act(sampled_plan, states, goal)
            self.action_reconstruction.clear_hidden_state()
        else:
            pred_actions = None
        
        # compute the losses and log them for wandb
        if pred_actions is not None:
            mse = F.mse_loss(pred_actions, actions, reduction="none")
        
            total_loss = mse # action_loss 
        else:
            if self.use_goal_in_recognition:
                state_sequence = torch.cat([states, goal], dim=1)
            else:
                state_sequence = states
            pr_state = self.plan_recognition(state_sequence)
            pr_dist = self.dist.get_dist(pr_state)
            true_plan = pr_dist.rsample()  # sample from recognition net
            plan_loss = F.mse_loss(true_plan, sampled_plan, reduction="none")
            total_loss = plan_loss
        return total_loss

    def compute_finetune_loss(self, states, actions, goal):
        '''
        Method to compute the loss during the second training phase.
        Here the focus is on the plan proposal network and/or the action decoder.
        
        The second phase is used for diffusion models mostly, because they are not able to
        be trained jointly with the latent plans model (based on empirical results).
        '''
        if self.use_goal_in_recognition:
            state_sequence = torch.cat([states, goal], dim=1)
        else:
            state_sequence = states
        
        # with torch.no_grad():
        pr_state, x = self.plan_recognition(state_sequence)
        pr_dist = self.dist.get_dist(pr_state)
        
        # sample from recognition net
        sampled_plan = pr_dist.rsample()  
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)

        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')   

        # plan proposal
        plan_loss, pred_plan = self.plan_proposal.loss_and_pred(first_state, goal, sampled_plan)

        # action reconstruction, we can either use the action decoder or the action reconstruction model
        # to reconstruct the actions
        if self.fine_tune_action_decoder:
            action_loss, _ = self.action_decoder.loss_and_act(sampled_plan, states, goal, actions)
            self.action_decoder.clear_hidden_state()
        else:
            action_loss = None
        
        # compute the losses and log them for wandb
        total_loss = 0
        if action_loss is not None:
            total_loss += action_loss 
            wandb.log({"fine_tuning/action_loss": action_loss.item()})
        
        total_loss += plan_loss
        wandb.log({"fine_tuning/plan_loss": plan_loss})
        return total_loss
    
    def freeze_model_weights(self):
        '''
        Method to freeze the parameters of the model during the second training phase.
        '''
        self.plan_recognition.requires_grad_(False)
        if not self.fine_tune_action_decoder:
            self.action_decoder.requires_grad_(False)
        
    @torch.no_grad()
    def compute_tuning_val_loss(self, states, actions, goal):
        '''
        Method to compute the valdiation loss during the second training phase.
        '''
        if self.use_goal_in_recognition:
            state_sequence = torch.cat([states, goal], dim=1)
        else:
            state_sequence = states
        pr_state, _ = self.plan_recognition(state_sequence)
        pr_dist = self.dist.get_dist(pr_state)
        true_plan = pr_dist.rsample()  # sample from recognition net
        true_plan = einops.rearrange(true_plan, 'b d -> b 1 d')

        # predict a plan from the proposal model
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')
        sampled_plan = self.plan_proposal(first_state, goal)
        
        # compute the plan loss and log them for wandb
        plan_loss = F.mse_loss(true_plan, sampled_plan, reduction="none").mean()
        
        # action decoder prediction of the actions
        pred_actions = self.action_decoder.act(sampled_plan, states, goal)
        self.action_decoder.clear_hidden_state()

        # compute the action loss and log them for wandb
        action_loss = F.mse_loss(pred_actions, actions, reduction="none").mean()
        wandb.log({"fine_tuning/plan_loss": plan_loss})
        total_loss = action_loss # + plan_loss
        return total_loss

    def compute_kl_loss(
        self, 
        pr_state: State, 
        pp_state: State
    ) -> torch.Tensor:
        '''
        Method to compute the KL loss between the recognition and proposal network.
        '''
        pp_dist = self.dist.get_dist(pp_state)  # prior
        pr_dist = self.dist.get_dist(pr_state)  # posterior
        # @fixme: do this more elegantly
        kl_lhs = D.kl_divergence(self.dist.get_dist(self.dist.detach_state(pr_state)), pp_dist).mean()
        kl_rhs = D.kl_divergence(pr_dist, self.dist.get_dist(self.dist.detach_state(pp_state))).mean()

        alpha = self.alpha
        kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled
    
    def compute_kl_loss_to_gaussian(self, pr_state):
        '''
        Method to compute the KL loss between the recognition and a simple Gauissna prior.
        '''
        pr_dist = self.dist.get_dist(pr_state)  # posterior
         # Create the prior distribution
        prior_dist = Normal(torch.zeros_like(pr_dist.mean), torch.ones_like(pr_dist.variance))
        var_dist = Normal(pr_dist.mean, torch.exp(0.5 * pr_dist.variance))
        
        # pr_mean = pr_dist.mean
        # pr_var = torch.exp(0.5 * pr_dist.variance)
        
        # kl_loss = 0.5 * torch.sum(pr_var / 1 + pr_var.pow(2) + (pr_mean - 0).pow(2) - 1).sum(dim=1)
        
         # Compute the KL divergence between the variational and prior distributions
        kl_loss = torch.distributions.kl_divergence(var_dist, prior_dist).sum(dim=1).mean()
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled
    
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
        action = self.action_decoder.act(self.plan, state, self.latent_goal)
        self.rollout_step_counter += 1
        return action, self.plan
    
    @torch.no_grad()
    def get_pp_plan(self, state, goal):
        '''
        Method to get a plan from the proposal network.
        '''
        with torch.no_grad():
            # ------------Plan Proposal------------ #
            if isinstance(self.plan_proposal, DiffusionProposalNet):
                sampled_plan = self.plan_proposal(state, goal)
            else:
                pp_state = self.plan_proposal(state, goal)
                pp_dist = self.dist.get_dist(pp_state)
                sampled_plan = pp_dist.rsample()  
                if self.dist.dist == "discrete":
                    sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)
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

