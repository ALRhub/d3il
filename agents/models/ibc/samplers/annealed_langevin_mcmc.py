from ast import arg
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch

from agents.scaler.scaler_class import Scaler



class AnnealedLangevinMCMCSampler:
    """
    Sampler class for the Langevin MCMC sampling strategy, which is used to sample from a given ebm
    for training and inference. The class is initialized by hydra with a given DictConfig.
    """

    def __init__(
        self,
        step_lr: float,
        denoise: bool,
        n_steps_per_noise: int,
        device,
        delta_action_clip: int = 0.1
    ):
        super(AnnealedLangevinMCMCSampler, self).__init__()
        self._delta_action_clip = delta_action_clip
        self._step_lr = step_lr
        self._denoise = denoise
        self._n_steps_per_noise = n_steps_per_noise
        self.bounds = None
        self.device = None
        self.type = "AnnealedLangevinMCMCSampler"
        self.device = device

    def get_bounds(self, scaler: Scaler):
        self.bounds = scaler.y_bounds

    @torch.no_grad()
    def _conditional_annealed_langevin_update_step(
        self, score_model: nn.Module, input: torch.Tensor, action: torch.Tensor, step_size: float, sigma: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO
        """
        bounds = torch.from_numpy(self.bounds).to(
            self.device
        )  # [2 ,X] with X action dim
        min_bound = bounds[0, :] - bounds[0, :]*0.1
        max_bound = bounds[1, :] + bounds[1, :]*0.1
        noise = torch.randn_like(action, device=self.device) 
        # compute the estimated score of the sample
        grad = score_model(input, action, sigma)
        # compute the gradient norm
        # grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
        # noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
        # get the annealed dynamics
        dynamics = step_size * grad + noise * torch.sqrt(step_size * 2)
        # grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
        action = action + dynamics 
        # compute additional monitoring stuff 
        # snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
        # keep the action in the allowed normalized range
        action = action.clamp(min=min_bound, max=max_bound)
        return action

    @torch.no_grad()
    def _annealed_langevin_update_step(
        self, score_model: nn.Module, x: torch.Tensor, step_size, sigma
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO
        """
        bounds = torch.as_tensor(self.bounds).to(
            self.device
        )  # [2 ,X] with X action dim

        noise = torch.randn_like(x) * np.sqrt(step_size * 2)
        # compute the estimated score of the sample
        grad = score_model(x, noise)
        # compute the gradient norm
        # grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
        # noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
        # get the annealed dynamics
        dynamics = step_size * grad + noise * np.sqrt(step_size * 2)
        # grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
        x = x + dynamics

        # compute additional monitoring stuff
        # snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
        # keep the action in the allowed normalized range
        x = x.clamp(min=bounds[0, :], max=bounds[1, :])

        """wandb.log(
            {
                "grad_norm": grad_norm.item(),
                "nosie_norm": noise_norm.item(),
                "grad_average": grad_mean_norm.item(),
                "snr": snr,
            }
        )"""
        return x

    def annealed_langevin_dynamics(self, input: torch.Tensor, ebm: nn.Module, noise_ratios: list) -> torch.Tensor:
        """
        Infer the best input for the conditional EBM given a current observation. In contrast to the
        gen_trainining_samples method we run 2 iterations of the MCMC chain to find the best possible samples.
        The second loop uses a very fixed step_size to finetune possible solutions.

        :param  input:      torch.Tensor of the input used to predict the best action
        :param  ebm:        nn.Module of the energy model used for inference

        :return:            torch.Tensor of the best action sampled from the model

        """
        ebm.eval()
        batch_size = input.size(0)
        # only get one sample to optimize for every input observation
        size = (batch_size, self.bounds.shape[1])
        x = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        x = torch.from_numpy(x).to(self.device).to(torch.float32)
        # iterate self._iters times to generate samples with low energy
        for idx, sigma in enumerate(noise_ratios):
            step_size = self._step_lr * (sigma / noise_ratios[-1]) ** 2
            for k in range(self._n_steps_per_noise):
                # let's do the Langevin update step
                x = self._conditional_annealed_langevin_update_step(ebm, input, x, step_size, sigma)

        # finisht by the denoising step
        x = self._denoising_update_step(ebm, input, x, noise_ratios[-1])
        return x

    @torch.no_grad()
    def _denoising_update_step(self, score_model, input, action, sigma):
        """
        Final denoising step to remove the noise in the predicted data
        """
        grad = score_model(input, action, sigma)
        x = action + sigma ** 2 * grad
        return x
