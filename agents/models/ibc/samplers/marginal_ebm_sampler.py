


import hydra
from omegaconf import OmegaConf
import torch
import numpy as np

from torch_implicit_BC.agents.stochastic_optimization.sampler_base import MarginalEBMSamplerBase


class MarginalEBMSampler(MarginalEBMSamplerBase):

    def __init__(
        self,
        ebm_config: OmegaConf,
        langevin_mcmc_sampler: OmegaConf,
        train_samples: int

        ) -> None:
        self._ebm = hydra.utils.instantiate(ebm_config)
        self._langevin_sampler = hydra.utils.instantiate(langevin_mcmc_sampler)
        self._train_samples = train_samples

    
    def get_bounds(self, bounds: np.ndarray)->None:
        """_summary_

        Args:
            bounds (np.ndarray): _description_
        """
        self.bounds = bounds
        self._langevin_sampler.get_bounds(bounds)
    
    def get_device(self, device: str):
        self.device = device

    def infer(self, input):
        self._langevin_sampler(input, self._ebm)
    
    def gen_train_samples(
        self,
        batch_size: int,
        random_start_points: bool = True,
        chosen_start_points=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_
        """
        samples, norm, _ = self._langevin_sampler.gen_train_samples(
            batch_size, self._ebm, random_start_points, chosen_start_points
        )
        # reshape into 3d form and return the generated samples
        return samples, norm


    def _sample_from_ebm(self, batch_size: int, num_samples: int):
        """
        TODO add text
        """
        size = (batch_size * num_samples, self.bounds.shape[1])
        # random sampling to generate the starting points 
        x = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)

    