
from collections import namedtuple
import math
from typing import Union

import torch 
from torch.distributions import Independent, Normal, OneHotCategoricalStraightThrough  # type: ignore
import torch.nn as nn
import torch.nn.functional as F


DiscState = namedtuple("DiscState", ["logit"])
ContState = namedtuple("ContState", ["mean", "std"])

State = Union[DiscState, ContState]


class PositionalEncoding(nn.Module):
    """Implementation from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :]
        return x


class Distribution:
    def __init__(self, **kwargs):
        self.dist = kwargs.get("dist")
        assert self.dist == "discrete" or self.dist == "continuous"
        if self.dist == "discrete":
            self.category_size = kwargs.get("category_size")
            self.class_size = kwargs.get("class_size")

    def get_dist(self, state):
        if self.dist == "discrete":
            shape = state.logit.shape
            logits = torch.reshape(state.logit, shape=(*shape[:-1], self.category_size, self.class_size))
            return Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        elif self.dist == "continuous":
            return Independent(Normal(state.mean, state.std), 1)

    def detach_state(self, state):
        if self.dist == "discrete":
            return DiscState(state.logit.detach())
        elif self.dist == "continuous":
            return ContState(state.mean.detach(), state.std.detach())

    def sample_latent_plan(self, distribution):
        sampled_plan = distribution.sample()
        if self.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)
        return sampled_plan

    def build_state(self, hidden_size, plan_features):
        fc_state = []
        if self.dist == "discrete":
            fc_state += [nn.Linear(hidden_size, plan_features)]
        elif self.dist == "continuous":
            fc_state += [nn.Linear(hidden_size, 2 * plan_features)]
        return nn.Sequential(*fc_state)

    def forward_dist(self, x):
        if self.dist == "discrete":
            prior_logit = x
            state = DiscState(prior_logit)  # type: State
        elif self.dist == "continuous":
            mean, var = torch.chunk(x, 2, dim=-1)
            min_std = 0.0001
            std = F.softplus(var) + min_std
            state = ContState(mean, std)
        return state