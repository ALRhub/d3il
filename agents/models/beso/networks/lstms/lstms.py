from typing import Tuple
import numbers 

import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
import einops
import logging
from omegaconf import OmegaConf
import numpy as np
import torch.jit as jit

from beso.networks.utils import return_activiation_fcn


class LayerNorm(jit.ScriptModule):
    def __init__(self, normalized_shape: int):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    @jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False)
        return mu, sigma

    @jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.weight + self.bias
    

class LSTMDecoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_lstm_layers: int,
        dropout: float,
        linear_dropout: float,
        output_dim: int,
        batch_first: bool = True,
        linear_dim: int = 0,
        num_linear_layers: int = 0,
        linear_activation: str = "relu",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_linear_layers = num_linear_layers
        self.output_dim = output_dim
        self.linear_dim = linear_dim
         # stack the desired number of hidden layers
        self.lin_layers = nn.ModuleList([nn.Linear(self.input_dim, self.linear_dim)])
        self.lin_layers.extend(
            [
                nn.Linear(self.linear_dim, self.linear_dim)
                for i in range(1, self.num_linear_layers)
            ]
        )
        self.lstm = nn.LSTM(
            input_size=self.linear_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=batch_first,
            dropout=dropout,
        ).to(self.device)
        self.linear_dropout = nn.Dropout(linear_dropout)
        self.act = return_activiation_fcn(linear_activation)
        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        
        self.lin_norm_layer = LayerNorm(self.linear_dim).to(self.device)
        self.lstm_norm_layer = LayerNorm(self.hidden_dim).to(self.device)

    def forward(
        self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if hidden is None:
            hidden = self.initHidden(input.shape[0])
        for idx, layer in enumerate(self.lin_layers):
            if idx == 0:
                x = self.linear_dropout(self.act(layer(input)))
            else:
                x = self.linear_dropout(self.act(layer(x)))
            # x = self.lin_norm_layer(x)
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out = self.lstm_norm_layer(lstm_out)
        output = self.out_layer(lstm_out)
        return output, hidden

    def initHidden(self, batch_size):
        return (
            torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(
                self.device
            ),
            torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(
                self.device
            ),
        )
    
