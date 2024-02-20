#!/usr/bin/env python3

from typing import Tuple

import torch
from torch.distributions import Independent, Normal
import torch.nn as nn
import torch.nn.functional as F

from beso.agents.lmp_agents.lmp_modules.utils import * 


class MLPPriorNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        perceptual_features: int,
        latent_goal_features: int,
        plan_features: int,
        activation_function: str,
        use_goal_in_prior: bool,
        use_state_in_prior: bool,
        dist: Distribution,
    ):
        super(MLPPriorNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.dist = dist
        self.perceptual_features = perceptual_features
        self.latent_goal_features = latent_goal_features
        self.plan_features = plan_features
        self.use_goal_in_prior = use_goal_in_prior
        self.use_state_in_prior = use_state_in_prior
        # the model can be f(s, g) prior or state only prior f(s) or goal only prior f(g)
        if self.use_goal_in_prior and self.use_state_in_prior:
            self.in_features = self.perceptual_features + self.latent_goal_features
        elif self.use_goal_in_prior:
            self.in_features = self.latent_goal_features
        elif self.use_state_in_prior:
            self.in_features = self.perceptual_features
        
        self.act_fn = getattr(nn, activation_function)()
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.hidden_size),  # shape: [N, 136]
            self.act_fn,
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.act_fn,
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.act_fn,
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.act_fn,
        )
        self.fc_state = self.dist.build_state(self.hidden_size, self.plan_features)
        
    def forward(self, initial_percep_emb: torch.Tensor, latent_goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.use_goal_in_prior and self.use_state_in_prior:
            x = torch.cat([initial_percep_emb, latent_goal], dim=-1)
        elif self.use_goal_in_prior:
            x = latent_goal
        elif self.use_state_in_prior:
            x = initial_percep_emb
        x = self.fc_model(x)
        my_state = self.fc_state(x)
        state = self.dist.forward_dist(my_state)
        return state

