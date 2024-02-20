import torch
import torch.nn as nn
from omegaconf import DictConfig
import hydra
from typing import Optional
from einops import einops

from agents.models.common.vision_modules import CoordConv


class EBMMLP(nn.Module):
    def __init__(
        self, 
        mlp: DictConfig, 
        device: str
    ) -> None:
        super().__init__()
        self.mlp = hydra.utils.instantiate(mlp)
        self.device = device

    def forward(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        state = state.to(self.device)
        action = action.to(self.device)
        
        if len(action.shape) == 3 and len(state.shape) == 2:
            # bs = batch size   ws = window size    asp = action space  ts = train samples
            state = einops.repeat(state, 'bs asp -> bs ts asp', ts=action.size(1))
            if goal is not None:
                goal = einops.repeat(goal, 'bs asp -> bs ts asp', ts=action.size(1))
        elif len(state.shape) == 3 and len(action.shape) == 2:
            action = einops.repeat(action, 'bs asp -> bs ts asp', ts=state.size(1))
        
        # if EBM doesnt work try to fix it here
        if len(state.shape) == 3 and len(action.shape) == 3:
            if goal is not None:
                state_goal = torch.cat([state, goal], dim=-1)
                fused = torch.cat([state_goal, action], dim=-1)
            else:
                fused = torch.cat([state, action], dim=-1)
            B, N, D = fused.size()
            fused = fused.reshape(B * N, D)
            out = self.mlp(fused)
            return out.view(B, N)
        else:
            if goal is not None:
                state_goal = torch.cat([state, goal], dim=-1)
                fused = torch.cat([state_goal, action], dim=-1)
            else:
                fused = torch.cat([state, action], dim=-1)
            out = self.mlp(fused)
            return out

    def get_device(self, device: torch.device):
        self.device = device
        self.mlp.get_device(device)
        self.mlp.to(device)
    
    def get_params(self):
        return self.mlp.parameters()


class JointEBMMLP(nn.Module):
    def __init__(self, mlp: DictConfig) -> None:
        super().__init__()
        self.mlp = hydra.utils.instantiate(mlp)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:

        if action is None:
            out = self.mlp(state)
            return out
        else:
            state = state.to(self.device)
            action = action.to(self.device)
            if len(action.shape) == 3:
                fused = torch.cat([state.unsqueeze(1).expand(-1, action.size(1), -1), action], dim=-1)
                B, N, D = fused.size()
                fused = fused.reshape(B * N, D)
                out = self.mlp(fused)
                return out.view(B, N)
            elif len(state.shape) == 3 and len(action.shape) == 2:
                fused = torch.cat([state, action.unsqueeze(1).expand(-1, state.size(1), -1)], dim=-1)
                B, N, D = fused.size()
                fused = fused.reshape(B * N, D)
                out = self.mlp(fused)
                return out.view(B, N)
            else:
                fused = torch.cat([state, action], dim=1)
                # fused.requires_grad = True
                out = self.mlp(fused)
                return out

    def get_device(self, device: torch.device):
        self.device = device
        self.mlp.get_device(device)
        self.mlp.to(device)
    
    def get_params(self):
        return self.mlp.parameters()


class LatentMLPNetwork(nn.Module):

    def __init__(
        self,
        state_encoder: DictConfig,
        target_encoder: DictConfig,
        latent_ebm: DictConfig
    ):
        super(LatentMLPNetwork, self).__init__()
        self.network_tactionpe = "LatentMLPEBM"
        # define number of variables in an state sequence
        self.state_encoder = hydra.utils.instantiate(state_encoder)
        self.target_encoder = hydra.utils.instantiate(target_encoder)
        self.latent_ebm = hydra.utils.instantiate(latent_ebm)


    def forward(self, state: torch.Tensor, target: torch.Tensor):
        state = state.to(self.device)
        target = target.to(self.device)   

        if len(target.shape) == 3:
            latent_state = self.state_encoder(state)
            latent_target = self.target_encoder(target)
            latent_state = latent_state.unsqueeze(1).expand(-1,latent_target.size(1), -1)
            fused = torch.cat([latent_state, latent_target], dim=-1)
            B, N, D = fused.size()
            fused = fused.reshape(B * N, D)
            out = self.latent_ebm(fused)
            return out.view(B, N)
        elif len(state.shape) == 3 and len(target.shape) == 2:
            latent_state = self.state_encoder(state)
            latent_target = self.target_encoder(target)
            latent_target = latent_target.unsqueeze(1).expand(-1, latent_state.size(1), -1)
            fused = torch.cat([latent_state, latent_target], dim=-1)
            B, N, D = fused.size()
            fused = fused.reshape(B * N, D)
            out = self.latent_ebm(fused)
            return out.view(B, N)
        else:
            latent_state = self.state_encoder(state)
            latent_target = self.target_encoder(target)
            fused = torch.cat([latent_state, latent_target], dim=-1)
            # fused.requires_grad = True
            out = self.latent_ebm(fused)
            return out

    def get_device(self, device: torch.device):
        self.device = device
        self.state_encoder.get_device(device)
        self.target_encoder.get_device(device)
        self.latent_ebm.get_device(device)

    def get_params(self):
        return self.parameters()


class TwoWaactionsEBMMLP(nn.Module):
    def __init__(self, mlp) -> None:
        super().__init__()
        self.mlp =  hydra.utils.instantiate(mlp)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        action = action.to(self.device)
        if len(action.shape) == 3:
            fused = torch.cat([state.unsqueeze(1).expand(-1, action.size(1), -1), action], dim=-1)
            B, N, D = fused.size()
            fused = fused.reshape(B * N, D)
            out = self.mlp(fused)
            return out.view(B, N)
        elif len(state.shape) == 3 and len(action.shape) == 2:
            fused = torch.cat([state, action.unsqueeze(1).expand(-1, state.size(1), -1)], dim=-1)
            B, N, D = fused.size()
            fused = fused.reshape(B * N, D)
            out = self.mlp(fused)
            return out.view(B, N)
        else:
            fused = torch.cat([state, action], dim=1)
            # fused.requires_grad = True
            out = self.mlp(fused)
            return out

    def get_device(self, device: torch.device):
        self.device = device
        self.mlp.get_device(device)
        self.mlp.to(device)

    def get_params(self):
        return self.mlp.parameters()

class EBMConvMLP(nn.Module):
    def __init__(
        self, 
        small_cnn: DictConfig, 
        mlp: DictConfig, 
        coord_conv: bool = False,
        device: str = 'cuda'
    ) -> None:
        super().__init__()
        self.device = device
        self.coord_conv = coord_conv
        if coord_conv:
            small_cnn.in_channels = small_cnn.in_channels + 2
        self.cnn = hydra.utils.instantiate(small_cnn)
        self.mlp = hydra.utils.instantiate(mlp)

        print(
            "CNN parameters: {}".format(sum(p.numel() for p in self.cnn.parameters()))
        )
        print(
            "mlp parameters: {}".format(sum(p.numel() for p in self.mlp.parameters()))
        )

    def get_device(self, device: torch.device):
        self.device = device
        self.cnn.get_device(device)
        self.mlp.get_device(device)
        self.cnn.to(device)
        self.mlp.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)  # [B, D, V1, V2]
        action = action.to(self.device)  # [B, N, dim_samples]
        if self.coord_conv:
            state = CoordConv()(state)
        out = self.cnn(state)  # [B, 32]
        if len(action.shape) == 3:
            fused = torch.cat(
                [out.unsqueeze(1).expand(-1, action.size(1), -1), action], dim=-1
            )  #
            B, N, D = fused.size()
            fused = fused.reshape(B * N, D)
            out = self.mlp(fused)
            return out.view(B, N)
        else:
            fused = torch.cat([out, action], dim=1)
            # fused.requires_grad = True
            out = self.mlp(fused)
            return out

    def get_params(self):
        return self.parameters() 