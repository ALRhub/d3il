import torch
import torch.nn as nn

from beso.networks.utils import return_activiation_fcn


class TwoLayerPreActivationResNetLinear(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 100,
            activation: str = 'relu',
            dropout_rate: float = 0.25,
            spectral_norm: bool = False,
            use_norm: bool = False,
            norm_style: int = 'BatchNorm'
    ) -> None:
        super().__init__()
        if spectral_norm:
            self.l1 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
            self.l2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        else:
            self.l1 = nn.Linear(hidden_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_norm = use_norm
        self.act = return_activiation_fcn(activation)

        if use_norm:
            if norm_style == 'BatchNorm':
                self.normalizer = nn.BatchNorm1d(hidden_dim)
            elif norm_style == 'LayerNorm':
                self.normalizer = torch.nn.LayerNorm(hidden_dim, eps=1e-06)
            else:
                raise ValueError('not a defined norm type')

    def forward(self, x):
        x_input = x
        if self.use_norm:
            x = self.normalizer(x)
        x = self.l1(self.dropout(self.act(x)))
        if self.use_norm:
            x = self.normalizer(x)
        x = self.l2(self.dropout(self.act(x)))
        return x + x_input