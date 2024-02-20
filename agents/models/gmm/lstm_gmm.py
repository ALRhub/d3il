from typing import Optional, Tuple
import torch
from torch import distributions as D
from torch import nn
from torch.nn import functional as F

from agents.models.common.mlp import ResidualMLPNetwork

STD_ACTIVATIONS = {"softplus": F.softplus, "exp": torch.exp}


class LSTM_GMM(nn.Module):
    def __init__(
            self,
            input_dim: int,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            lstm_dropout: float,
            bidirectional: bool,
            output_dim: int,
            n_gaussians: int,
            min_std: float,
            std_activation: str,
            use_tanh_wrapped_distribution: bool,
            low_noise_eval: bool,
            device: str,
    ):
        super(LSTM_GMM, self).__init__()
        if std_activation not in STD_ACTIVATIONS:
            raise ValueError(f"std_activation must be one of {list(STD_ACTIVATIONS.keys())}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gaussians = n_gaussians

        self.min_std = min_std
        self.std_activation = std_activation

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.mlp = nn.Linear(in_features=lstm_hidden_size, out_features=lstm_hidden_size, device=device)

        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)

        self.mean_mlp = nn.Linear(in_features=lstm_output_dim, out_features=self.output_dim * self.n_gaussians,
                                  device=device)
        self.std_mlp = nn.Linear(in_features=lstm_output_dim, out_features=self.output_dim * self.n_gaussians,
                                 device=device)
        self.logits_mlp = nn.Linear(in_features=lstm_output_dim, out_features=self.n_gaussians, device=device)

        self.use_tanh_wrapped_distribution = use_tanh_wrapped_distribution
        self.low_noise_eval = low_noise_eval

    def forward(
            self, x: torch.Tensor, only_last: bool = False,
            init_hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[D.Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        assert x.shape[-1] == self.input_dim

        lstm_out, final_hidden_states = self.lstm(x, init_hidden_states)

        if only_last:
            lstm_out = lstm_out[:, -1]

        mlp_out = self.mlp(lstm_out)
        mlp_out = F.relu(mlp_out)

        gmm_means = self.mean_mlp(lstm_out)
        gmm_stds = self.std_mlp(lstm_out)
        gmm_weight_logits = self.logits_mlp(lstm_out)

        assert not self.use_tanh_wrapped_distribution, "Not implemented yet"

        if not self.use_tanh_wrapped_distribution:
            gmm_means = 2.1 * torch.tanh(gmm_means)

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

        return gmm, final_hidden_states

    def get_params(self):
        return self.parameters()