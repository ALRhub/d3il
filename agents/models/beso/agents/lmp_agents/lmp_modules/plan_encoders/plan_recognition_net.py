#!/usr/bin/env python3

from typing import Tuple

import torch
from torch.distributions import Independent, Normal
import torch.nn as nn
import torch.nn.functional as F
import hydra
import einops 

from beso.agents.lmp_agents.lmp_modules.utils import * 


class PlanRecognitionNetwork(nn.Module):
    '''
    Simple RNN based plan recognition network.
    '''
    def __init__(
        self,
        hidden_size: int,
        in_features: int,
        plan_features: int,
        action_space: int,
        birnn_dropout_p: float,
        min_std: float,
        dist: Distribution,
    ):
        super(PlanRecognitionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.plan_features = plan_features
        self.dist = dist
        self.action_space = action_space
        self.min_std = min_std
        self.in_features = in_features
        self.birnn_model = nn.RNN(
            input_size=self.in_features,
            hidden_size=self.hidden_size,
            nonlinearity="relu",
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=birnn_dropout_p,
        )  # shape: [N, seq_len, 64+8]
        # self.mean_fc = nn.Linear(in_features=2 * self.hidden_size, out_features=self.plan_features)  # shape: [N, seq_len, 4096]
        # self.variance_fc = nn.Linear(in_features=2 * self.hidden_size, out_features=self.plan_features)  # shape: [N, seq_len, 4096]
        self.fc_state = self.dist.build_state(2 * self.hidden_size, self.plan_features)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        x, hn = self.birnn_model(state)
        x = x[:, -1]  # we just need only last unit output
        my_state = self.fc_state(x)
        state = self.dist.forward_dist(my_state)
        return state, x

    def __call__(self, *args, **kwargs):
        mean, std = super().__call__(*args, **kwargs)
        pr_dist = Independent(Normal(mean, std), 1)
        return pr_dist



class PlanRecognitionTransformersNetwork(nn.Module):
    '''
    Transformers based plan recognition network.
    '''
    def __init__(
        self,
        num_heads: int,
        obs_dim: int,
        num_layers: int,
        encoder_hidden_size: int,
        fc_hidden_size: int,
        plan_features: int,
        in_features: int,
        action_space: int,
        encoder_normalize: bool,
        positional_normalize: bool,
        position_embedding: bool,
        max_position_embeddings: int,
        dropout_p: bool,
        dist: Distribution,
        use_mean_embedding: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.plan_features = plan_features
        self.action_space = action_space
        self.padding = False
        self.dist = dist
        self.hidden_size = fc_hidden_size
        self.position_embedding = position_embedding
        self.encoder_normalize = encoder_normalize
        self.positional_normalize = positional_normalize
        
        self.use_mean_embedding = use_mean_embedding
        mod = self.in_features % num_heads
        if mod != 0:
            print(f"Padding for Num of Heads : {num_heads}")
            self.padding = True
            self.pad = num_heads - mod
            self.in_features += self.pad
        
        self.position_embeddings = nn.Embedding(max_position_embeddings, self.in_features)
        self.state_encoder = nn.Linear(obs_dim, self.in_features)  
        encoder_layer = nn.TransformerEncoderLayer(
            self.in_features, 
            num_heads, 
            dim_feedforward=encoder_hidden_size, 
            dropout=dropout_p,
            batch_first=True
        )
        # define the norm layer
        encoder_norm = nn.LayerNorm(self.in_features) if encoder_normalize else None
        self.layernorm = nn.LayerNorm(self.in_features)
        self.dropout = nn.Dropout(p=dropout_p)
        # define the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        # define the output layer and the distribution output
        self.fc = nn.Linear(in_features=self.in_features, out_features=fc_hidden_size)
        self.fc_state = self.dist.build_state(fc_hidden_size, self.plan_features)

    def forward(self, state: torch.Tensor) -> Tuple[State, torch.Tensor]:
        '''
        Encoder the state and return the mean and variance of the distribution
        '''
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = (
            torch.cat([state, torch.zeros((batch_size, seq_len, self.pad)).to(state.device)], dim=-1)
            if self.padding
            else state
        )
        if self.position_embedding:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=state.device).unsqueeze(0)
            position_embeddings = self.position_embeddings(position_ids)
            # reshape to all batches
            position_embeddings = einops.repeat(position_embeddings, '1 t d -> (1 b) t d', b=batch_size)
            state_emb = self.state_encoder(state)
            x = state_emb + position_embeddings
        
        if self.positional_normalize:
            x = self.layernorm(x)
        # forward pass
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        if self.use_mean_embedding:
            x = torch.mean(x, dim=1)  # gather all the sequence info
        else:
            x = x[:, -1]  # we just need only last unit output
        my_state = self.fc_state(x)
        state = self.dist.forward_dist(my_state)
        return state, x


class NoDistPlanRecognitionTransformersNetwork(nn.Module):
    '''
    Transformers based plan recognition network without distribution output
    '''
    def __init__(
        self,
        num_heads: int,
        obs_dim: int,
        num_layers: int,
        encoder_hidden_size: int,
        fc_hidden_size: int,
        plan_features: int,
        in_features: int,
        action_space: int,
        encoder_normalize: bool,
        positional_normalize: bool,
        position_embedding: bool,
        max_position_embeddings: int,
        dropout_p: bool,
        use_mean_embedding: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.plan_features = plan_features
        self.action_space = action_space
        self.padding = False
        self.hidden_size = fc_hidden_size
        self.position_embedding = position_embedding
        self.encoder_normalize = encoder_normalize
        self.positional_normalize = positional_normalize
        
        self.use_mean_embedding = use_mean_embedding
        mod = self.in_features % num_heads
        if mod != 0:
            print(f"Padding for Num of Heads : {num_heads}")
            self.padding = True
            self.pad = num_heads - mod
            self.in_features += self.pad
        
        self.position_embeddings = nn.Embedding(max_position_embeddings, self.in_features)
        self.state_encoder = nn.Linear(obs_dim, self.in_features)  
        encoder_layer = nn.TransformerEncoderLayer(
            self.in_features, 
            num_heads, 
            dim_feedforward=encoder_hidden_size, 
            dropout=dropout_p,
            batch_first=True
        )
        
        encoder_norm = nn.LayerNorm(self.in_features) if encoder_normalize else None
        self.layernorm = nn.LayerNorm(self.in_features)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.batchnorm = nn.BatchNorm1d(self.plan_features)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        
        self.fc = nn.Linear(in_features=self.in_features, out_features=self.plan_features)

    def forward(self, state: torch.Tensor) -> Tuple[State, torch.Tensor]:
        batch_size, seq_len = state.shape[0], state.shape[1]
        '''state = (
            torch.cat([state, torch.zeros((batch_size, seq_len, self.pad)).to(state.device)], dim=-1)
            if self.padding
            else state
        )'''
        if self.position_embedding:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=state.device).unsqueeze(0)
            position_embeddings = self.position_embeddings(position_ids)
            # reshape to all batches
            position_embeddings = einops.repeat(position_embeddings, '1 t d -> (1 b) t d', b=batch_size)
            state_emb = self.state_encoder(state)
            x = state_emb + position_embeddings
            # x = x.permute(1, 0, 2)
        # else:
            # padd the perceptual embeddig
            # x = self.positional_encoder(state.permute(1, 0, 2))  # [s, b, emb]
        if self.positional_normalize:
            x = self.layernorm(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        
        x = self.fc(x)
        if self.use_mean_embedding:
            x = torch.mean(x, dim=1)  # gather all the sequence info
        else:
            x = x[:, -1]  # we just need only last unit output
        x = self.batchnorm(x)
        return x



class StateEncoderTransformersNetwork(nn.Module):
    '''
    Transformers based plan recognition network without distribution output
    '''
    def __init__(
        self,
        num_heads: int,
        obs_dim: int,
        num_layers: int,
        encoder_hidden_size: int,
        fc_hidden_size: int,
        plan_features: int,
        in_features: int,
        action_space: int,
        encoder_normalize: bool,
        positional_normalize: bool,
        position_embedding: bool,
        max_position_embeddings: int,
        dropout_p: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.plan_features = plan_features
        self.action_space = action_space
        self.padding = False
        self.hidden_size = fc_hidden_size
        self.position_embedding = position_embedding
        self.encoder_normalize = encoder_normalize
        self.positional_normalize = positional_normalize
        
        mod = self.in_features % num_heads
        if mod != 0:
            print(f"Padding for Num of Heads : {num_heads}")
            self.padding = True
            self.pad = num_heads - mod
            self.in_features += self.pad
        
        self.position_embeddings = nn.Embedding(max_position_embeddings, self.in_features)
        self.state_encoder = nn.Linear(obs_dim, self.in_features)  
        encoder_layer = nn.TransformerEncoderLayer(
            self.in_features, 
            num_heads, 
            dim_feedforward=encoder_hidden_size, 
            dropout=dropout_p,
            batch_first=True
        )
        
        encoder_norm = nn.LayerNorm(self.in_features) if encoder_normalize else None
        self.layernorm = nn.LayerNorm(self.in_features)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.batchnorm = nn.BatchNorm1d(self.plan_features)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        
        self.fc = nn.Linear(in_features=self.in_features, out_features=self.plan_features)

    def forward(self, state: torch.Tensor) -> Tuple[State, torch.Tensor]:
        """
        Encode a sequence of states into an embedding for the VQVAE.
        """
        batch_size, seq_len = state.shape[0], state.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=state.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        # reshape to all batches
        position_embeddings = einops.repeat(position_embeddings, '1 t d -> (1 b) t d', b=batch_size)
        state_emb = self.state_encoder(state)
        x = state_emb + position_embeddings

        if self.positional_normalize:
            x = self.layernorm(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.layernorm(x)
        x = self.fc(x)
        return x