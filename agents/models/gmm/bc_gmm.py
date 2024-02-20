import torch
from torch import distributions as D
from torch import nn
from torch.nn import functional as F

from agents.models.common.mlp import MLPNetwork, ResidualMLPNetwork
from agents.models.common.utils import return_activiation_fcn
from agents.models.gmm.tanh_wrapped_distribution import TanhWrappedDistribution

STD_ACTIVATIONS = {"softplus": F.softplus, "exp": torch.exp}


class BC_GMM(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_hidden_layers: int,
            mlp_output_dim: int,
            dropout: int,
            activation: str,
            use_spectral_norm: bool,
            output_dim: int,
            n_gaussians: int,
            min_std: float,
            std_activation: str,
            use_tanh_wrapped_distribution: bool,
            low_noise_eval: bool,
            device: str,
    ):
        super(BC_GMM, self).__init__()
        if std_activation not in STD_ACTIVATIONS:
            raise ValueError(f"std_activation must be one of {list(STD_ACTIVATIONS.keys())}")

        self.output_dim = output_dim
        self.n_gaussians = n_gaussians

        self.min_std = min_std
        self.std_activation = std_activation

        self.mlp = ResidualMLPNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=mlp_output_dim,
            dropout=dropout,
            activation=activation,
            use_spectral_norm=use_spectral_norm,
            device=device,
        )

        self.mlp_output_act = return_activiation_fcn(activation)

        self.mean_mlp = nn.Linear(in_features=mlp_output_dim, out_features=self.output_dim * self.n_gaussians,
                                  device=device)
        self.std_mlp = nn.Linear(in_features=mlp_output_dim, out_features=self.output_dim * self.n_gaussians,
                                 device=device)
        self.logits_mlp = nn.Linear(in_features=mlp_output_dim, out_features=self.n_gaussians, device=device)

        self.use_tanh_wrapped_distribution = use_tanh_wrapped_distribution
        self.low_noise_eval = low_noise_eval

    def forward(self, x: torch.Tensor) -> D.Distribution:
        out = self.mlp(x)
        out = self.mlp_output_act(out)

        gmm_means = self.mean_mlp(out)
        gmm_stds = self.std_mlp(out)
        gmm_weight_logits = self.logits_mlp(out)

        if not self.use_tanh_wrapped_distribution:
            gmm_means = 0.01 * torch.tanh(gmm_means)

        if self.low_noise_eval and (not self.training):
            gmm_stds = torch.ones_like(gmm_stds) * 1e-4
        else:
            gmm_stds = STD_ACTIVATIONS[self.std_activation](gmm_stds) + self.min_std

        gmm_means = gmm_means.reshape(gmm_means.shape[0], gmm_means.shape[1], self.n_gaussians, self.output_dim)
        gmm_stds = gmm_stds.reshape(gmm_stds.shape[0], gmm_stds.shape[1], self.n_gaussians, self.output_dim)
        # gmm_weight_logits = gmm_weight_logits.reshape(gmm_weight_logits.shape[0], gmm_weight_logits.shape[1], self.n_gaussians)

        mix = D.Categorical(logits=gmm_weight_logits)
        comp = D.Independent(D.Normal(loc=gmm_means, scale=gmm_stds), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        if self.use_tanh_wrapped_distribution:
            return TanhWrappedDistribution(gmm, scale=1.9)  # TODO this is hardcoded because maximum absolute value of action is <2.1 in block push
        else:
            return gmm

    def get_params(self):
        return self.parameters()