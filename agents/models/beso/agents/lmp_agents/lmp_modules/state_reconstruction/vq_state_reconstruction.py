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
        plan_dim: int,
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
        self.plan_embedding = nn.Linear(plan_dim, hidden_dim)
        self.enc_states_embedding = nn.Linear(plan_dim, hidden_dim)
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
        self.state_pred_head = nn.Linear(hidden_dim, state_dim)
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
    
    def compute_reconstruction_losses(self, latent_states, states, goal_state=None):

        x = self.enc_states_embedding(latent_states)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.output_norm(x)
        pred_states  = self.state_pred_head(x)
        if pred_states.shape[1] != states.shape[1]:
            pred_states = pred_states[:, :-1, :]
        state_loss = torch.nn.functional.mse_loss(pred_states, states)
        
        return state_loss, None