import numpy as np
import torch
from torch import nn
import normflows as nf


class NFlow(torch.nn.Module):
    def __init__(self, action_dim, obs_dim, num_flows, hidden_units=128, latent_size=2, hidden_layers=2) -> None:
        super().__init__()

        flows = []
        for _ in range(num_flows):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        q0 = nf.distributions.ClassCondDiagGaussian(shape=action_dim, num_classes=obs_dim)

        self.normflow = nf.ClassCondFlow(q0=q0, flows=flows)

    def get_params(self):
        return self.parameters()

    def loss(self, x, y):
        return self.normflow.forward_kld(x, y)

    def forward(self, contexts):
        return self.normflow.sample(num_samples=1, y=contexts)
