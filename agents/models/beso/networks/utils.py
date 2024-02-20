import math

import torch
from inspect import isfunction
import torch.nn as nn
import torch.jit as jit
import numbers
from typing import Tuple
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F


from beso.networks.vision_modules.vision_modules import GlobalAvgPool2d, GlobalMaxPool2d, SpatialSoftArgmax


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



def return_activiation_fcn(activation_type: str):
    # build the activation layer
    if activation_type == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activation_type == "tanh":
        act = torch.nn.Sigmoid()
    elif activation_type == "ReLU":
        act = torch.nn.ReLU()
    elif activation_type == "PReLU":
        act = torch.nn.PReLU()
    elif activation_type == "softmax":
        act = torch.nn.Softmax(dim=-1)
    elif activation_type == "Mish":
        act = torch.nn.Mish()
    elif activation_type == nn.GELU():
        act = nn.GELU()
    else:
        act = torch.nn.PReLU()
    return act


def load_spatial_module(module: str):
    if module == " GlobalAvgPool2d":
        model = GlobalAvgPool2d()
    elif module == "GlobalMaxPool2d":
        model = GlobalMaxPool2d()
    elif module == "SpatialSoftArgmax":
        model = SpatialSoftArgmax()
    else:
        ValueError("Module is not implemented! Please check spelling.")
    return model


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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


class Residual(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


