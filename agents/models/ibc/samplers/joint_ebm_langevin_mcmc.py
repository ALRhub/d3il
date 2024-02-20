from ast import arg
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from agents.samplers import ExponentialSchedule, PolynomialSchedule
from .sampler_base import MarginalEBMSamplerBase


# general Langevin MCMC sampler for joint distribution EBMs
class JointLangevinMCMCSampler(MarginalEBMSamplerBase):
    """
    Sampler class for the Langevin MCMC sampling strategy, which is used to sample from a given ebm
    for training and inference. The class is initalized by hydra with a given DictConfig.
    """

    def __init__(
        self,
        noise_scale: float,
        noise_shrink: float,
        train_samples: int,
        train_iterations: int = 25,
        inference_iterations: int = 100,
        sampler_stepsize_init: float = 5e-1,
        sampler_2nd_inference_stepsize_init: float = 1e-5,
        sampler_stepsize_decay: float = 0.8,  # if using exponential langevin rate.
        sampler_stepsize_final: float = 1e-5,
        sampler_stepsize_power: float = 2.0,
        use_polynomial_rate: bool = True,  # default is exponential
        delta_action_clip: int = 0.1,
    ):
        super(JointLangevinMCMCSampler, self).__init__()

        self._noise_scale = noise_scale
        self._noise_shrink = noise_shrink
        self._train_samples = train_samples
        self._delta_action_clip = delta_action_clip
        self._sampler_stepsize_init = sampler_stepsize_init
        self._sampler_2nd_inference_stepsize_init = sampler_2nd_inference_stepsize_init
        self._sampler_stepsize_decay = sampler_stepsize_decay
        self._train_iterations = train_iterations
        self._inference_iterations = inference_iterations
        self._gradient_scale = 0.5  # from the IBC paper
        self._use_polynomial_rate = (use_polynomial_rate,)  # default is exponential
        self._sampler_stepsize_power = sampler_stepsize_power
        self._sampler_stepsize_final = sampler_stepsize_final
        self.bounds = None
        self.device = None
        self.type = "LangevinMCMCSampler"

        # the Langevin MCMC uses a learning rate decay scheduler to adapt the step size of the sampling
        # based on the paper implementations
        if self._use_polynomial_rate:
            self._schedule = PolynomialSchedule(
                self._sampler_stepsize_init,
                self._sampler_stepsize_final,
                self._sampler_stepsize_power,
                self._inference_iterations,
            )
        else:
            self._schedule = ExponentialSchedule(
                self._sampler_stepsize_init, self._sampler_stepsize_decay
            )

    def get_device(self, device: torch.device):
        self.device = device

    def get_bounds(self, bounds: np.ndarray):
        self.bounds = bounds

    def _langevin_update_step(
        self, ebm: nn.Module, input: torch.Tensor, step_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: this is the Langevin MCMC equation used in the IBC paper and not the correct one.
        Single update step fo the Langevin MCMC sampler method. In this method we set the gradient of
        the ebm to false for the time of the method and set it back to true after the new sample
        is calculated. In addition we have to set the action input gradient to True to calculate the
        derivate of the energy w.r.t to the action input.
        In addition the gradient norm is calcualted and returned to regularize the update steps for a
        more stable training. The gradient norm implementation is based on the Wasserstein Gradient norm

        :param  ebm:        nn.Module of the ebm
        :param  input:      torch.Tensor of the current observation or input
        :step_size:         int of the current size of the langevin update step

        :return:            [
                                torch.Tensor of the new input sample,
                                torch.Tensor of energy of the input state with the previous sample
                                torch.Tensor of gradient norm of the update step
        ]
        """
        bounds = torch.as_tensor(self.bounds).to(
            self.device
        )  # [2 ,X] with X action dim
        delta_action_clip = self._delta_action_clip * 0.5*(bounds[1, :] - bounds[0, :])
        ebm.requires_grad = False
        # input.requires_grad = True
        input = input.detach()
        input.requires_grad = True
        # compute the energy of the current sample
        e = ebm(input)
        sum_e = e.mean()
        # calculate the derivative w.r.t the input action of the conditional EBM

        grad = autograd.grad(outputs=sum_e, inputs=input, create_graph=True)[0]
        # activate gradient computation again
        ebm.requires_grad = True

        # compute gradient norm:
        # from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py WGAN training
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        grad_abs = torch.abs(grad)
        grad_norm, _ = torch.max(grad_abs, 1)
        # clip delta norm values below 0
        # same clipping used in the original ibc implementation
        # clip the delta to max(0, grad_delta)
        grad_delta = grad_norm - 1
        grad_delta = grad_delta.clamp(min=0, max=1e10)
        # get the mean of the delta norm
        wgan_grad_norm = torch.mean(grad_delta.pow(2))

        # get the noise
        noise = torch.randn_like(input) * self._noise_scale

        # this is in the Langevin dynamics equation.
        dynamics = step_size * self._gradient_scale * grad - step_size * noise
        # this is in the Langevin dynamics equation.
        dynamics = step_size * self._gradient_scale * grad - step_size * noise
        input = input - dynamics
        # keep the input in the allowed normalized range
        input = input.clamp(min=bounds[0, :], max=bounds[1, :])

        return input, e, wgan_grad_norm

    def _langevin_mcmc(
        self,
        ebm: nn.Module,
        start_points: torch.Tensor,
        return_intermediate_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Langevin Dynamics MCMC sampling method to draw samples from a distribution using the Langevin Dynamics

        :param  ebm:            nn.Module of the energy model to draw samples form
        :param  input:          torch.Tensor of the ebm state input
        :param  start_points:   torch.Tensor of a chosen start point if wanted else None is recommended
        :param  return_intermediate_step:   Boolean to decide if intermediate steps of the sampling chain are returned

        :return:                [
                                    torch.Tensor of sampled actions
                                    torch.Tensor of the action norms
                                    torch.Tensor of energy values of the samples
        ]
        # TODO how to define output types if 2 output types are possible
        """
        l_samples = []
        l_energies = []
        l_norms = []
        x = start_points
        l_samples.append(start_points)
        step_size = self._sampler_stepsize_init
        # iterate self._num_iterations times to generate samples with low energy
        for idx in range(self._train_iterations):
            # do the Langevin update step
            x, e, grad_norm = self._langevin_update_step(ebm, x, step_size)
            l_samples.append(x.detach())
            l_norms.append(grad_norm.detach())
            l_energies.append(e.detach())
            # adapt the step size
            step_size = self._schedule.get_rate(idx + 1)

        if return_intermediate_steps:
            return l_samples, l_norms, l_energies
        else:
            return l_samples[-1], l_norms[-1], l_energies[-1]

    def gen_train_samples(
        self,
        batch_size: int,
        ebm: nn.Module,
        random_start_points: bool = True,
        chosen_start_points=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to generate training samples with the Langevin MCMC sampling method for training an ebm.
        It is different to the inference method in the number of update steps of the Langevin MCMC chain
        and the step size used for the chain. Moreover, instead of starting from random sapmples, the
        starting points of the sampling chain can be chosen freely.
        For every true action sample in the batch self._train_samples negative samples will be generated
        to optimize the model.

        :param  batch_size:     int with the number of true actions in a training batch
        :param  ebm:            nn.Module of the ebm model
        :param  random_start_points:    boolean if the sampling chain should be started from a random point
        :param  chosen_start_points:    None or torch.Tensor of the start points to sample from if chosen

        :return:                [
                                    torch.Tensor of all sampled actions for the batch
                                    torch.Tensor of all norms of the sampled actions
        ]

        TODO check when intermediate steps are required
        boolean of random_start_point used for possible future extensions with buffer
        # TODO implement start point buffer
        """
        # generate random start points sampled from a uniform distribution if chosen
        if random_start_points:
            size = (batch_size, self._train_samples, self.bounds.shape[1])  # [B, T, X]
            start_points = np.random.uniform(
                self.bounds[0, :], self.bounds[1, :], size=size
            )
            start_points = torch.as_tensor(
                start_points, dtype=torch.float32, device=self.device
            )
        else:
            # here maybe a defined buffer will be added in the future to store good samples and use them
            # in the next iteration to reduce the number of update steps and increase training performance
            start_points = chosen_start_points

        # generate the samples with input: [ B, Z] Z: dim of obs
        samples, norm, _ = self._langevin_mcmc(
            ebm, start_points, return_intermediate_steps=False
        )
        # reshape into 3d form and return the generated samples
        return samples.reshape(batch_size, self._train_samples, -1), norm

    def infer(self, input, ebm: nn.Module) -> torch.Tensor:
        """
        Infer the best input for the conditional EBM given a current observation. In contrast to the
        gen_trainining_samples method we run 2 iterations of the MCMC chain to find the best possible samples.
        The second loop uses a very fixed step_size to finetune possible solutions.

        :param  input:      torch.Tensor of the input used to predict the best action
        :param  ebm:        nn.Module of the energy model used for inference

        :return:            torch.Tensor of the best action sampled from the model

        TODO:    add funtionalty to return multiple best samples for applications where multimodality
        plays a cruical role and we want to get different solutions for the same context
        """
        argmax = True
        batch_size = input.size(0)
        step_size = self._sampler_stepsize_init

        if argmax:
            # get more samples for every input and use argmax to determine the best one
            size = (batch_size * self._train_samples, self.bounds.shape[1])
            x = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
            x = x.reshape(batch_size, self._train_samples, self.bounds.shape[1])
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            # only get one sample to optimize for every input observation
            size = (batch_size, self.bounds.shape[1])
            x = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        # iterate self._iters times to generate samples with low energy
        for idx in range(self._inference_iterations):
            # let's do the Langevin update step
            x, _, _ = self._langevin_update_step(ebm, x, step_size)
            # adapt the step size
            step_size = self._schedule.get_rate(idx + 1)

        # 2nd loop with fixed low step size:
        step_size = self._sampler_2nd_inference_stepsize_init
        for idx in range(self._inference_iterations):
            # let's do the Langevin update step
            x, _, _ = self._langevin_update_step(ebm, x, step_size)

        if argmax:
            # return the sample with the highest energy for every observation in the batch
            energies = ebm(x)
            probs = F.softmax(-1.0 * energies, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            return x[torch.arange(x.size(0)), best_idxs, :]

        else:
            # since there is only one sample for every  batch dimension we can directly return it
            return x