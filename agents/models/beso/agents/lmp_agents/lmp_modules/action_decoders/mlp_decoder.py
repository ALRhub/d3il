

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from functools import partial

import numpy as np
import hydra
from omegaconf import ListConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from beso.networks.scaler.scaler_class import Scaler

logger = logging.getLogger(__name__)


class ActionDecoder(nn.Module):
    def act(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_and_act(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_goal = einops.repeat(latent_goal, 'b 1 d -> b (1 t) d', t=perceptual_emb.shape[1])
        if len(latent_plan.shape) == 2:
            latent_plan = einops.rearrange(latent_plan, 'b d -> b 1 d')
        latent_plan = einops.repeat(latent_plan, 'b 1 d -> b (1 t) d', t=perceptual_emb.shape[1])
        
        # encode the input for the diffusion policy
        x = torch.cat([perceptual_emb, latent_plan], dim=-1)

        # get the loss and the action


    def clear_hidden_state(self) -> None:
        raise NotImplementedError

    def _sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise 


class SimpleMLPPolicyNetwork(ActionDecoder):
    
    def __init__(
        self,
        perceptual_features: int,
        latent_goal_features: int,
        model: OmegaConf,
        plan_features: int,
        hidden_size: int,
        out_features: int
    ) -> None:
        super(SimpleMLPPolicyNetwork, self).__init__()
        in_features = perceptual_features + latent_goal_features + plan_features
        model['input_dim'] = in_features
        self.out_features = out_features
        self.model = hydra.utils.instantiate(model)
    
    def set_bounds(self, scaler: Scaler, window_size, device):
        min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(device)
        max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(device)
        
        # reshape for usage
        self.action_min_bound = min_action
        self.action_max_bound = max_action
    
    def loss_and_act(
        self, 
        latent_plan: torch.Tensor, 
        perceptual_emb: torch.Tensor, 
        latent_goal: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_goal = einops.repeat(latent_goal, 'b 1 d -> b (1 t) d', t=perceptual_emb.shape[1])
        if len(latent_plan.shape) == 2:
            latent_plan = einops.rearrange(latent_plan, 'b d -> b 1 d')
        latent_plan = einops.repeat(latent_plan, 'b 1 d -> b (1 t) d', t=perceptual_emb.shape[1])
        x = torch.concat([latent_plan, perceptual_emb, latent_goal], dim=-1)
        pred_actions = self.model(x)
        # loss
        loss = self._loss(pred_actions, actions)
        return loss, pred_actions

    def clear_hidden_state(self) -> None:
        self.hidden_state = None

    def act(
        self, 
        latent_plan: torch.Tensor, 
        perceptual_emb: torch.Tensor, 
        latent_goal: torch.Tensor
    ) -> torch.Tensor:
        x = torch.concat([latent_plan, perceptual_emb, latent_goal], dim=-1)
        pred_actions = self.model(x)
        return pred_actions
    
    def loss(
        self, 
        latent_plan: torch.Tensor, 
        perceptual_emb: torch.Tensor, 
        latent_goal: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        x = torch.concat([latent_plan, perceptual_emb, latent_goal], dim=-1)
        pred_actions = self.model(x)
        return self._loss(pred_actions, actions)

    def _loss(
        self,
        pred_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        mse = nn.functional.mse_loss(pred_actions, actions, reduction="none").mean().item()
        return mse
    
    def forward(  # type: ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
         
        batch_size, seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        if len(latent_plan.shape) == 2:
            latent_plan = einops.repeat(latent_plan, 'b d -> b t d', t=seq_len)
        else:
            latent_plan = einops.repeat(latent_plan, 'b 1 d -> b (t 1) d', t=seq_len) if latent_plan.shape[1] == 1 else latent_plan 
        if len(latent_plan.shape) == 2:
            latent_goal = einops.repeat(latent_goal, 'b d -> b t d', t=seq_len)  if latent_goal.shape[1] == 1 else latent_goal 
        else:
            latent_goal = einops.repeat(latent_goal, 'b 1 d -> b (t 1) d', t=seq_len) 

        x = torch.cat([latent_plan, perceptual_emb, latent_goal], dim=-1)
        pred_actions = self.model(x)
        return pred_actions
    
        