
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


class PlayLMP(nn.Module):
    def __init__(
        self,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        action_decoder: DictConfig,
        dist: DictConfig,
        plan_features: int,
        kl_beta: float,
        replan_freq: int = 30,
        device: str = 'cuda',
        alpha: float = 0.0,
        use_goal_in_recognition: bool = False,
    ):
        super(PlayLMP, self).__init__()
        self.device = device
        self.plan_features = plan_features
        self.dist = hydra.utils.instantiate(dist)
        self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.dist).to(self.device)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.dist).to(self.device)
        self.action_decoder = hydra.utils.instantiate(action_decoder).to(self.device)
        self.kl_beta = kl_beta
        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None
        # weight for kl loss 
        self.alpha = alpha
        self.use_goal_in_recognition = use_goal_in_recognition

    def compute_loss(self, states, actions, goal):

        # plan recognition
        # state_sequence = torch.cat([states, goal], dim=1)
        # goal_state = einops.rearrange(states[:, -1, :], 'b d -> b 1 d')
        if self.use_goal_in_recognition:
            state_sequence = torch.cat([states, goal], dim=1)
        else:
            state_sequence = states
        pr_state, x = self.plan_recognition(state_sequence)
        pr_dist = self.dist.get_dist(pr_state)
        
        # plan proposal
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')
        pp_state = self.plan_proposal(first_state, goal)
        pp_dist = self.dist.get_dist(pp_state)
        
        sampled_plan = pr_dist.rsample()  # sample from recognition net
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)

        # action decoder
        action_loss, _ = self.action_decoder.loss_and_act(sampled_plan, states, goal, actions)

        kl_loss = self.compute_kl_loss(pp_state, pr_state)
        wandb.log({"training/kl_loss": kl_loss.item(),
                   "training/action_loss": action_loss.item(),})
        total_loss = action_loss + kl_loss
        return total_loss
    
    
    def forward(self, states, goal):
        goal = einops.rearrange(states[:, -1, :], 'b d -> b 1 d')
        # plan proposal
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')
        pp_dist = self.plan_proposal(first_state, goal)

        # sample from recognition net
        sampled_plan = pp_dist.rsample() 

        # action decoder
        action = self.action_decoder(sampled_plan, states, goal)

        return action
    
    @torch.no_grad()
    def compute_val_loss(self, states, actions, goal):

        # plan proposal
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')
        
        pp_state = self.plan_proposal(first_state, goal)
        pp_dist = self.dist.get_dist(pp_state)
        
        sampled_plan = pp_dist.rsample()  # sample from proposal net
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)

        # action decoder
        self.action_decoder.clear_hidden_state()
        pred_actions = self.action_decoder.act(sampled_plan, states, goal)

        mse = F.mse_loss(pred_actions, actions, reduction="none")
        
        total_loss = mse # action_loss 
        return total_loss

    def compute_kl_loss(
        self, 
        pr_state: State, 
        pp_state: State
    ) -> torch.Tensor:

        pp_dist = self.dist.get_dist(pp_state)  # prior
        pr_dist = self.dist.get_dist(pr_state)  # posterior
        # @fixme: do this more elegantly
        kl_lhs = D.kl_divergence(self.dist.get_dist(self.dist.detach_state(pr_state)), pp_dist).mean()
        kl_rhs = D.kl_divergence(pr_dist, self.dist.get_dist(self.dist.detach_state(pp_state))).mean()

        alpha = self.alpha
        kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled
    
    def reset(self):
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0
    
    @torch.no_grad()
    def step(self, state, goal):
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
        
        with torch.no_grad():
            # ------------Plan Proposal------------ #
            pp_state = self.plan_proposal(state, goal)
            pp_dist = self.dist.get_dist(pp_state)
        
            sampled_plan = pp_dist.rsample()  # sample from proposal net
            if self.dist.dist == "discrete":
                sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)
        self.action_decoder.clear_hidden_state()
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

