from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampler_base import ConditionalEBMSamplerBase
from agents.utils.scaler import Scaler

# based on the implementation of Kevin Zakka's repo: https://github.com/kevinzakka/ibc
class DerivativeFreeOptimizer(ConditionalEBMSamplerBase):
    def __init__(
        self,
        noise_scale: float,
        noise_shrink: float,
        iters: int,
        train_samples: int,
        inference_samples: int,
        device: str
    ):
        super(DerivativeFreeOptimizer, self).__init__()
        self._noise_scale = noise_scale
        self._noise_shrink = noise_shrink
        self._train_samples = train_samples
        self._iters = iters
        self._inference_samples = inference_samples
        self.bounds = None
        self.device = device
        self.type = "DerivativeFreeOptimizer"

    def get_bounds(self, scaler: Scaler):
        self.bounds = scaler.y_bounds

    def _sample(self, num_samples: int, input:torch.Tensor) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution.
        The samples are generated in the dimensions [N, X] and returned as tensors.

        :param num_samples: int with number of generated samples
        :return:            torch.tensor samples[N, X]
        """
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(
            self.bounds[0, :], self.bounds[1, :], size=size
        )  # [N, X] with X: action dim
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def gen_train_samples(self, batch_size: int, ebm: nn.Module, input: torch.Tensor) -> torch.Tensor:
        """
        For every training sample in batch B with dimensions X we generate T samples.
        The number of generated samples T is dependent on the hyperparameter self._train_samples
        The generated samples are returned in the shape [B, T, X]

        :param batch_size: int with number of samples in the batch
        :param ebm:        nn.Module class with the energy-based model

        :return:           torch.Tensor samples in shape [B, T, X]
        """
        # del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        noise_scale = self._noise_scale
        bounds = torch.as_tensor(self.bounds).to(
            self.device
        )  # [2 ,X] with X action dim
        samples = self._sample(
            batch_size * self._train_samples
        )  # [B*T, X] with X: action dim

        samples = samples.reshape(
            input.size(0), self._inference_samples, -1
        )  # [B, num_samples, X]

        
        for _ in range(self._iters):
            # Compute energies.
            energies = ebm(input, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self._inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self._noise_shrink


        return samples # [B, T, X]

    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Optimize for the best action given an EBM.

        :param x:        observation for the conditional EBM
        :param ebm:      nn.Module class with the energy-based model

        :return:         torch.Tensor samples with the best action given the observation x
        """
        noise_scale = self._noise_scale
        bounds = torch.as_tensor(self.bounds).to(
            self.device
        )  # [2 ,X] with X action dim

        samples = self._sample(
            x.size(0) * self._inference_samples
        )  # [B * num_samples, X]

        samples = samples.reshape(
            x.size(0), self._inference_samples, -1
        )  # [B, num_samples, X]

        for _ in range(self._iters):
            # Compute energies.
            energies = ebm(x, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self._inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self._noise_shrink

        # Return target with highest probability.
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]


class AutoregressiveDerivativeFreeOptimizer(DerivativeFreeOptimizer):

    def __init__(
        self, 
        noise_scale: float, 
        noise_shrink: float, 
        train_iters: int, 
        inference_iters: int,
        train_samples: int, 
        device: str,
        inference_samples: int):
        super().__init__(noise_scale, noise_shrink, inference_iters, train_samples, inference_samples, device)
        self._inference_iters = inference_iters
        self._train_iters = train_iters

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution.
        The samples are generated in the dimensions [N, X] and returned as tensors.

        :param num_samples: int with number of generated samples
        :return:            torch.tensor samples[N, X]
        """
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(
            self.bounds[0, :], self.bounds[1, :], size=size
        )  # [N, X] with X: action dim
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)
    
    def _single_dim_sample(self, num_samples: int, dim_index)->torch.Tensor:
        size = (num_samples, 1)
        samples = np.random.uniform(
            self.bounds[0, dim_index] - self.bounds[0, dim_index]*0.1, 
            self.bounds[1, dim_index] + self.bounds[1, dim_index]*0.1, 
            size=size
        )  # [N, X] with X: action dim
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)
    
    def gen_single_dim_train_samples(self, batch_size: int, ebm: nn.Module, input: torch.Tensor, action_dim_index: int):
        
        samples = self._single_dim_sample(
            batch_size * self._train_samples, action_dim_index
        )  # [B*T, X] with X: action dim
        # noise_scale = self._noise_scale

        bounds = torch.as_tensor(self.bounds[:, action_dim_index]).to(
            self.device
        )  # [2] with X action dim
        # min_bound = bounds[0] - bounds[0]*0.1
        # max_bound = bounds[1] + bounds[1]*0.1

        samples = samples.reshape(
            batch_size, self._train_samples, 1
        )  # [B, num_samples, 1]

        """for _ in range(self._train_iters):
            # Compute energies.
            energies = ebm.single_dimension_forward(input, samples, action_dim_index)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self._train_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=min_bound, max=max_bound)
            noise_scale *= self._noise_shrink"""

        return samples  # [B, T, X]

    
    def gen_train_samples(self, batch_size: int, ebm: nn.Module, input: torch.Tensor) -> torch.Tensor:
        """
        For every training sample in batch B with dimensions X we generate T samples.
        The number of generated samples T is dependent on the hyperparameter self._train_samples
        The generated samples are returned in the shape [B, T, X]

        :param batch_size: int with number of samples in the batch
        :param ebm:        nn.Module class with the energy-based model

        :return:           torch.Tensor samples in shape [B, T, X]
        """
        samples = self._sample(
            batch_size * self._train_samples
        )  # [B*T, X] with X: action dim
        noise_scale = self._noise_scale

        bounds = torch.as_tensor(self.bounds).to(
            self.device
        )  # [2 ,X] with X action dim

        samples = samples.reshape(
            batch_size, self._train_samples, bounds.shape[1]
        )  # [B, num_samples, X]

        for _ in range(self._train_iters):
            # Compute energies.
            energies = ebm(input, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self._train_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self._noise_shrink

        return samples  # [B, T, X]
    
    @torch.no_grad()
    def infer(self, input: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Optimize for the best action given an EBM.

        :param x:        observation for the conditional EBM
        :param ebm:      nn.Module class with the energy-based model

        :return:         torch.Tensor samples with the best action given the observation x
        """ 
        bounds = torch.as_tensor(self.bounds).to(
                self.device
            )  # [2 ,X] with X action dim

        bounds[0, :] -= bounds[0, :]*0.1
        bounds[1, :] += bounds[1, :]*0.1
        current_input = input
        best_actions = torch.zeros((input.size(0), bounds.size(-1)), device=self.device)

        for idx in range(bounds.size(-1)):

            noise_scale = self._noise_scale

            samples = self._single_dim_sample(
                input.size(0) * self._inference_samples,
                dim_index=idx
            )  # [B * num_samples, 1]

            samples = samples.reshape(
                input.size(0), self._inference_samples, -1
            ).to(self.device)  # [B, num_samples, 1]

            if idx > 0:
                current_input = torch.cat([current_input, best_actions[:, idx-1].reshape(input.size(0), 1)], dim=-1)

            for _ in range(self._inference_iters):
                # Compute energies.
                energies = ebm.single_dimension_forward(current_input, samples, idx)
                probs = F.softmax(-1.0 * energies, dim=-1)

                # Resample with replacement.
                idxs = torch.multinomial(probs, self._inference_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                samples = samples + torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=bounds[0, idx], max=bounds[1, idx])

                noise_scale *= self._noise_shrink

            probs = F.softmax(-1.0 * energies, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            best_actions[:, idx] = samples[torch.arange(samples.size(0)), best_idxs, 0][:]

        return best_actions
        # return samples[torch.arange(samples.size(0)), best_idxs, :]