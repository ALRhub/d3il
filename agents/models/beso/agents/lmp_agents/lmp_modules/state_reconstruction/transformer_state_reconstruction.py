import logging
import math 
from typing import Optional

from omegaconf import DictConfig
import torch 
import torch.nn as nn
import numpy as np
import einops 

from beso.networks.transformers.mingpt_policy import Block

logger = logging.getLogger(__name__)


class TransformerStateReconstruction(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        window_size: int,
        device: str,
        input_pdrop: float,
        dropout_p: float,
        ) -> None:
        super().__init__()
        
        self.device = device
        # self.plan_embedding = nn.Linear(input_dim, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.seq_len = window_size - 1
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size+2, hidden_dim))
        self.dropout = nn.Dropout(input_pdrop)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, 
            num_heads, 
            dim_feedforward=256, 
            dropout=dropout_p,
            batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=self.output_norm)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerStateReconstruction):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        
        torch.nn.init.normal_(self.mask_token, std=.02)
    
    def compute_reconstruction_losses(self, latent_plan, states, goal_state=None):
        if latent_plan.ndim == 2:
            latent_plan = einops.rearrange(latent_plan, 'b d -> b 1 d')
        first_state = einops.rearrange(states[:, 0, :], 'b d -> b 1 d')
        
        if goal_state is None:
            goal_state = einops.rearrange(states[:, -1, :], 'b d -> b 1 d')
            missing_states = states[:, :-1, :]
        else:
            missing_states = states[:, 1:, :]
        
        
       #  pred_goal = self.predict_goal(first_state, latent_plan)
        pred_first_state = self.predict_first_state(goal_state, latent_plan)
        # pred_missing_states = self.predict_missing_states(first_state, latent_plan, goal_state)
        pred_goal = self.predict_goal(first_state, latent_plan)
        # goal_loss = torch.mean(torch.abs(pred_goal - goal_state))
        first_state_loss = torch.mean(torch.abs(pred_first_state - first_state))
        # state_loss = torch.mean(torch.abs(pred_missing_states - missing_states))
        goal_loss = torch.mean(torch.abs(pred_goal - goal_state))
        return first_state_loss, goal_loss
    
    def predict_goal(self, first_state, latent_plan):
        
        plan_embedding = latent_plan # self.plan_embedding(latent_plan)
        first_state_embedding = self.state_embedding(first_state)
        missing_token = self.mask_token.repeat(first_state_embedding.shape[0], 1, 1) # torch.zeros_like(first_state_embedding).to(self.device)
        
        x = torch.cat((plan_embedding, first_state_embedding, missing_token), dim=1)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.output_norm(x)
        pred_goal = self.state_head(x[:, -1, :])
        
        return pred_goal

    def predict_first_state(self, goal_state, latent_plan):
        
        plan_embedding = latent_plan # self.plan_embedding(latent_plan)
        goal_embedding = self.state_embedding(goal_state)
        missing_token = torch.zeros_like(goal_embedding).to(self.device)
        
        x = torch.cat((plan_embedding, missing_token, goal_embedding), dim=1)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.output_norm(x)
        pred_first_state = self.state_head(x[:, 0, :])
        
        return pred_first_state

    def predict_missing_states(self, first_state, latent_plan, goal_state):
        
        plan_embedding = self.plan_embedding(latent_plan)
        goal_embedding = self.state_embedding(goal_state)
        first_state_embedding = self.state_embedding(first_state)
        missing_token = torch.zeros((first_state.shape[0], self.seq_len, first_state_embedding.shape[-1])).to(self.device)
        
        x = torch.cat((plan_embedding, first_state_embedding, missing_token, goal_embedding), dim=1)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_norm(x)
        pred_missing_states = self.state_head(x[:, 2:-1, :])
        
        return pred_missing_states