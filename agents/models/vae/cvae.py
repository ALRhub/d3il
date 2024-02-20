from turtle import forward
from omegaconf import DictConfig
import torch
import torch.nn as nn
import hydra


class VariationalEncoder(nn.Module):

    def __init__(
        self,
        model_config: DictConfig,
        latent_dim: int,
        device: str
        ):
        super(VariationalEncoder, self).__init__()
        self.device = device
        self.enc = hydra.utils.instantiate(model_config, output_dim=model_config.hidden_dim).to(device)
        self.mean = nn.Linear(model_config.hidden_dim, latent_dim).to(device)
        self.log_std = nn.Linear(model_config.hidden_dim, latent_dim).to(device)
    
    def forward(self, x):

        x = self.enc(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std


class VariationalAE(nn.Module):

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        device: str
    ) -> None:
        super(VariationalAE, self).__init__()
        self.device = device

        self.encoder = hydra.utils.instantiate(encoder)
        self.decoder = hydra.utils.instantiate(decoder, input_dim=encoder.latent_dim+decoder.input_dim,
                                               hidden_dim=encoder.model_config.hidden_dim,
                                               num_hidden_layers=encoder.model_config.num_hidden_layers)

        self.latent_dim = encoder.latent_dim

    def forward(self, state, action):
        mean, std = self.encoder(torch.cat([state, action], axis=-1))
        # reparametrization trick for backprop
        z = mean + std * torch.randn_like(std)
        u = self.decoder(torch.cat([state, z], dim=-1))
        return u, mean, std

    @torch.no_grad()
    def predict(self, state):

        z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        if len(state.shape) == 3:
            z = z.unsqueeze(0)
        out = self.decoder(torch.cat([state, z], dim=-1))
        return out
    
    def get_params(self):
        return self.parameters()