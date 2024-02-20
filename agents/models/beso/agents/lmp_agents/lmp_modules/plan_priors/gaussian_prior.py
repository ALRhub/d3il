#!/usr/bin/env python3

from typing import Tuple

import torch
from torch.distributions import Independent, Normal
import torch.nn as nn
import torch.nn.functional as F

from beso.agents.lmp_agents.lmp_modules.utils import * 


class GaussianNetwork(nn.Module):
    def __init__(
        self,
        plan_features: int,
        dist: Distribution,
    ):
        super(GaussianNetwork, self).__init__()
        self.dist = dist
        self.plan_features = plan_features
        # the model can be f(s, g) prior or state only prior f(s) or goal only prior f(g)
        if self.use_goal_in_prior and self.use_state_in_prior:
            self.in_features = self.perceptual_features + self.latent_goal_features
        elif self.use_goal_in_prior:
            self.in_features = self.latent_goal_features
        elif self.use_state_in_prior:
            self.in_features = self.perceptual_features
        
        self.fc_state = self.dist.build_state(self.plan_features, self.plan_features)
        
    def forward(self, initial_percep_emb: torch.Tensor, latent_goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        state = self.dist.forward_dist(my_state)
        return state