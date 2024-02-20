from ast import Tuple
from datetime import datetime
import os
import einops
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from agents.models.ibc.samplers.sampler_base import ConditionalEBMSamplerBase
from agents.models.ibc.samplers.schedulers import PolynomialSchedule
import torch.nn.functional as F


class NoiseSampler(ConditionalEBMSamplerBase):
    """
    Sampler class for the proposal network strategy, where the negative samples are produced
    using another network that is not conditioned on the goal (or observation)
    IMPORTANT: Does only support Diffusion networks atm
    """
    def __init__(
        self,
        sampler_stepsize_init: float,
        sampler_stepsize_init_infer: float,
        delta_action_clip: float,
        inference_iterations: int,
        train_iterations: int,
        sampler_stepsize_final: float,
        second_inference_stepsize_init: float,
        sampler_stepsize_power: float,
        train_samples: int,
        inference_samples: int,
        noise_scale: float,
        noise_scale_infer: float,
        second_infer: bool,
        adjustable_noise: float,
        device: str,
    ):
        super(NoiseSampler, self).__init__()

        self.sampler_stepsize_init = sampler_stepsize_init
        self.sampler_stepsize_init_infer = sampler_stepsize_init_infer
        self.delta_action_clip = delta_action_clip
        self.inference_iterations = inference_iterations
        self.train_iterations = train_iterations
        self.sampler_stepsize_final = sampler_stepsize_final
        self.second_inference_stepsize_init = second_inference_stepsize_init
        self.sampler_stepsize_power = sampler_stepsize_power
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.noise_scale = noise_scale
        self.noise_scale_infer = noise_scale_infer
        self.device = device
        self.second_infer = second_infer
        self.type = "noiseSampler"
        self.adjustable_noise = adjustable_noise
        
        self.infer_schedule = PolynomialSchedule(
            self.sampler_stepsize_init,
            self.sampler_stepsize_final,
            self.sampler_stepsize_power,
            self.inference_iterations,
        )

        self.train_schedule = PolynomialSchedule(
            self.sampler_stepsize_init,
            self.sampler_stepsize_final,
            self.sampler_stepsize_power,
            self.train_iterations,
        )

    def get_device(self, device: torch.device):
        self.device = device

    def get_bounds(self, scaler: np.ndarray):
        self.bounds = scaler.y_bounds
        self.scaler = scaler

    def gen_train_samples(self, batch_size: int, ebm: torch.nn.Module, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor, count: int):
        
        state = einops.rearrange(state, 'bs ad asp -> bs (ad asp)')
        action = einops.rearrange(action, 'bs ad asp -> bs (ad asp)')
        if goal is not None:
            goal = einops.rearrange(goal, 'bs ad asp -> bs (ad asp)')

        samples = einops.repeat(action, 'bs asp -> bs neg asp', neg=self.train_samples)
        noise = torch.randn_like(samples) * self.adjustable_noise
        samples = samples + noise

        samples, _, _ = self._langevin_mcmc(ebm, state, samples, goal)

        return samples

    
    def infer(self, state: torch.Tensor, ebm: torch.nn.Module, goal) -> torch.Tensor:
        """
        Infer the best input for the conditional EBM given a current observation. In contrast to the
        gen_trainining_samples method we run 2 iterations of the MCMC chain to find the best possible samples.
        The second loop uses a very fixed step_size to finetune possible solutions.

        :param  input:      torch.Tensor of the input used to predict the best action
        :param  ebm:        nn.Module of the energy model used for inference

        :return:            torch.Tensor of the best action sampled from the model
        plays a crucial role and we want to get different solutions for the same context
        """

        batch_size = state.size(0)

        if len(state.shape) == 3:
            state = einops.rearrange(state, 'bs ad asp -> bs (ad asp)')
            if goal is not None:
                goal = einops.rearrange(goal, 'bs ad asp -> bs (ad asp)')            

        size = (batch_size * self.inference_samples, self.bounds.shape[1])
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        samples = torch.as_tensor(samples, dtype=torch.float32, device=self.device)

        step_size = self.sampler_stepsize_init_infer

        x = torch.reshape(samples, (batch_size, self.inference_samples, self.bounds.shape[1]))
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        l_samples = []
        l_samples.append(x)

        for idx in range(self.inference_iterations):
            x, _, _, _ = self._langevin_update_step(ebm, state, x, step_size, goal, self.noise_scale_infer)
            x = x.detach()
            l_samples.append(x)
            step_size = self.infer_schedule.get_rate(idx + 1)

        if self.second_infer:
            step_size = self.second_inference_stepsize_init
            for idx in range(self.inference_iterations):
                x, _, _, _ = self._langevin_update_step(ebm, state, x, step_size, goal, self.noise_scale_infer)
                x = x.detach()
        
        #visualizeMultipathMCMCSampling(int(round(datetime.now().timestamp())), 10.0, 10.0, 20.0, 10.0, l_samples, self.scaler, ebm)

        energies = ebm(state, x, goal)
        probs = F.softmax(-1.0 * energies, dim=-1)
        m = torch.distributions.Categorical(probs)
        best_idxs = m.sample()
        return x[torch.arange(x.size(0)), best_idxs, :]

    def _langevin_mcmc(
        self,
        ebm: torch.nn.Module,
        input: torch.Tensor,
        start_points: torch.Tensor,
        goal: torch.Tensor
    ):
        x = start_points
        step_size = self.sampler_stepsize_init
        grad_norm = 0
        e = 0
        for idx in range(self.train_iterations):
            x, e, _, grad_norm = self._langevin_update_step(ebm, input, x, step_size, goal, self.noise_scale)
            step_size = self.train_schedule.get_rate(idx + 1)

        return x, grad_norm, e

    def compute_gradient(
        self,
        ebm,
        input,
        action,
        goal,
        graph = False
    ):
        ebm.requires_grad = False
        # input.requires_grad = True
        action = action.detach()
        action.requires_grad = True
        # compute the energy of the current sample
        e = ebm(input, action, goal)
        # calculate the derivative w.r.t the input action of the conditional EBM
        grad = torch.autograd.grad(
            outputs=e,
            inputs=action,
            grad_outputs=torch.ones(e.size(), device=self.device),
            create_graph=graph,
            retain_graph=graph
        )[0]

        #grad = grad * -1.0

        # activate gradient computation again
        ebm.requires_grad = True
        # compute gradient norm:
        # from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py WGAN training
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        # grad_norm = torch.sqrt(torch.sum(grad**2, dim=1) + 1e-12)
        
        grad_norm = torch.linalg.norm(grad, ord=float('inf'), dim=1)

        return grad, grad_norm, e
    
    def _langevin_update_step(
        self, 
        ebm: torch.nn.Module, 
        input: torch.Tensor, 
        action: torch.Tensor, 
        step_size: int,
        goal: torch.Tensor,
        noise_scale: float
    ):
        # This effectively scales the gradient as if the actions were
        # in a min-max range of -1 to 1.
        bounds = torch.as_tensor(self.bounds).to(
            self.device
        )  # [2 ,X] with X action dim
        delta_action_clip = self.delta_action_clip * 0.5*(bounds[1, :] - bounds[0, :])
        
        grad, grad_norm, e = self.compute_gradient(ebm, input, action, goal)
        
        grad_delta = grad_norm - 1
        grad_delta = grad_delta.clamp(min=0, max=1e10)
        # get the mean of the delta norm
        wgan_grad_norm = torch.mean(grad_delta.pow(2))

        # get the noise
        noise = torch.randn_like(action) * noise_scale

        # this is in the Langevin dynamics equation.
        dynamics = step_size * 2 * grad + step_size * noise

        dynamics = dynamics.clamp(min=-delta_action_clip, max=delta_action_clip)
        action = action - dynamics
        # keep the action in the allowed normalized range
        action = action.clamp(min=bounds[0, :], max=bounds[1, :])

        return action, e, wgan_grad_norm, grad_norm
