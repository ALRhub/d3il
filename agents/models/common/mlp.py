from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .utils import return_activiation_fcn


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


class MLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network which can be generated with different
    activation functions with and without spectral normalization of the weights
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            use_spectral_norm: bool = False,
            device: str = 'cuda'
    ):
        super(MLPNetwork, self).__init__()
        self.network_type = "mlp"
        # define number of variables in an input sequence
        self.input_dim = input_dim
        # the dimension of neurons in the hidden layer
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        # number of samples per batch
        self.output_dim = output_dim
        self.dropout = dropout
        self.spectral_norm = use_spectral_norm
        # set up the network
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        self.layers.extend(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim)
                for i in range(1, self.num_hidden_layers)
            ]
        )
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # build the activation layer
        self.act = return_activiation_fcn(activation)
        self._device = device
        self.layers.to(self._device)

    def forward(self, x):

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                out = layer(x)
            else:
                if idx < len(self.layers) - 2:
                    out = layer(out)  # + out
                else:
                    out = layer(out)
            if idx < len(self.layers) - 1:
                out = self.act(out)
        return out

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()


class ResidualMLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "Mish",
            use_spectral_norm: bool = False,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            device: str = 'cuda'
    ):
        super(ResidualMLPNetwork, self).__init__()
        self.network_type = "mlp"
        self._device = device
        # set up the network

        assert num_hidden_layers % 2 == 0
        if use_spectral_norm:
            self.layers = nn.ModuleList([spectral_norm(nn.Linear(input_dim, hidden_dim))])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout_rate=dropout,
                    spectral_norm=use_spectral_norm,
                    use_norm=use_norm,
                    norm_style=norm_style
                )
                for i in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.to(self._device)

        # self.robot_enc_layers = nn.Sequential(nn.Linear(4, 256),
        #                                       torch.nn.Mish(),
        #                                       nn.Linear(256, 128))
        #
        # self.box_enc_layers = nn.Sequential(nn.Linear(18, 256),
        #                                     torch.nn.Mish(),
        #                                     nn.Linear(256, 128))

    def forward(self, x):

        # robot_encoding = x[:, :4]
        # box_encoding = x[:, 4:]
        #
        # robot_encoding = self.robot_enc_layers(robot_encoding)
        # box_encoding = self.box_enc_layers(box_encoding)
        # box_encoding, _ = torch.max(box_encoding, dim=1)
        #
        # x = torch.concatenate((robot_encoding, box_encoding.unsqueeze(1)), dim=-1)

        for idx, layer in enumerate(self.layers):
            x = layer(x.to(torch.float32))
        return x

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()


class CriticMLP(nn.Module):

    def __init__(
            self,
            q1: DictConfig,
            q2: DictConfig,
            device: str
    ):
        super(CriticMLP, self).__init__()
        self.network_type = "critic"
        self._device = device
        # set up the network
        self.q1 = q1.to(self._device)
        self.q2 = q2.to(self._device)

    def forward(self, state, y):
        state = state.to(self._device)
        y = y.to(self._device)
        if len(y.shape) == 3:
            fused = torch.cat([state.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
            B, N, D = fused.size()
            fused = fused.reshape(B * N, D)
        # x = torch.cat([state, action], dim=-1)#
        return self.q1(fused), self.q2(fused)

    def get_device(self, device: torch.device):
        self._device = device

    def get_params(self):
        return self.parameters()