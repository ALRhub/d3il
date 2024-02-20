
from typing import Tuple, Optional

import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from agents.models.ibc.samplers.schedulers import ExponentialSchedule, PolynomialSchedule
from agents.models.ibc.samplers.sampler_base import ConditionalEBMSamplerBase

from einops import einops

# based on the implementation of google/ibc
class LangevinMCMCSampler(ConditionalEBMSamplerBase):
    """
    Sampler class for the Langevin MCMC sampling strategy, which is used to sample from a given ebm
    for training and inference. The class is initialized by hydra with a given DictConfig.
    """

    def __init__(
        self,
        noise_scale: float,
        noise_scale_infer: float,
        noise_shrink: float,
        train_samples: int,
        inference_samples: int,
        device: str,
        train_iterations: int = 25,
        inference_iterations: int = 100,
        sampler_stepsize_init: float = 5e-1,
        sampler_stepsize_init_infer: float = 5e-1,
        second_inference_stepsize_init: float = 1e-5,
        sampler_stepsize_decay: float = 0.8,  # if using exponential langevin rate.
        sampler_stepsize_final: float = 1e-5,
        sampler_stepsize_power: float = 2.0,
        use_polynomial_rate: bool = True,  # default is exponential
        delta_action_clip: int = 0.1,
        second_inference_iteration: bool = True,
    ):
        super(LangevinMCMCSampler, self).__init__()

        self.noise_scale = noise_scale
        self.noise_scale_infer = noise_scale_infer
        self._noise_shrink = noise_shrink
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.delta_action_clip = delta_action_clip
        self.sampler_stepsize_init = sampler_stepsize_init
        self.sampler_stepsize_init_infer = sampler_stepsize_init_infer
        self.second_inference_stepsize_init = second_inference_stepsize_init
        self._sampler_stepsize_decay = sampler_stepsize_decay
        self.train_iterations = train_iterations
        self.inference_iterations = inference_iterations
        self._gradient_scale = 0.5  # from the IBC paper
        self._use_polynomial_rate = use_polynomial_rate # default is exponential
        self.sampler_stepsize_power = sampler_stepsize_power
        self.sampler_stepsize_final = sampler_stepsize_final
        self.second_infer = second_inference_iteration
        self.bounds = None
        self.device = device
        self.type = "LangevinMCMCSampler"

        # the Langevin MCMC uses a learning rate decay scheduler to adapt the step size of the sampling
        # based on the paper implementations
        if self._use_polynomial_rate:
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
        else:
            self._schedule = ExponentialSchedule(
                self._sampler_stepsize_init, self._sampler_stepsize_decay
            )

    def get_device(self, device: torch.device):
        self.device = device

    def get_bounds(self, scaler: np.ndarray):
        self.bounds = scaler.y_bounds

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
        dynamics = step_size * self._gradient_scale * grad + step_size * noise

        dynamics = dynamics.clamp(min=-delta_action_clip, max=delta_action_clip)
        action = action - dynamics
        # keep the action in the allowed normalized range
        action = action.clamp(min=bounds[0, :], max=bounds[1, :])

        return action, e, wgan_grad_norm, grad_norm

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

    def gen_train_samples(
        self,
        batch_size: int,
        ebm: nn.Module,
        input: torch.Tensor,
        goal: torch.Tensor,
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
        :param  input:          torch.Tensor of all inputs for the true actions of the ebm in the batch
        :param  random_start_points:    boolean if the sampling chain should be started from a random point
        :param  chosen_start_points:    None or torch.Tensor of the start points to sample from if chosen

        :return:                [
                                    torch.Tensor of all sampled actions for the batch
                                    torch.Tensor of all norms of the sampled actions
        ]
        boolean of random_start_point used for possible future extensions with buffer
        """
        # generate random start points sampled from a uniform distribution if chosen
        if random_start_points:
            size = (batch_size, self.train_samples, self.bounds.shape[1])  # [B, T, X]
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
            
        input = torch.squeeze(input)
        if goal is not None:
            goal = torch.squeeze(goal)

        # generate the samples with input: [ B, Z] Z: dim of obs
        samples, norm, _ = self._langevin_mcmc(
            ebm, input, start_points, goal
        )
        # reshape into 3d form and return the generated samples
        return samples.reshape(batch_size, self.train_samples, -1)

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


# class which uses the true equation for the langevin mcmc update steps 
class CorrectLangevinMCMCSampler(LangevinMCMCSampler):

    def __init__(
        self, 
        noise_scale: float, 
        noise_shrink: float, 
        train_samples: int, 
        train_iterations: int = 25, 
        inference_iterations: int = 100, 
        sampler_stepsize_init: float = 0.5, 
        sampler_2nd_inference_stepsize_init: float = 0.00001, 
        sampler_stepsize_decay: float = 0.8, 
        sampler_stepsize_final: float = 0.00001, 
        sampler_stepsize_power: float = 2, 
        use_polynomial_rate: bool = True, 
        delta_action_clip: int = 0.1
        ):
        super().__init__(
            noise_scale, 
            noise_shrink, 
            train_samples, 
            train_iterations, 
            inference_iterations, 
            sampler_stepsize_init, 
            sampler_2nd_inference_stepsize_init, 
            sampler_stepsize_decay, 
            sampler_stepsize_final, 
            sampler_stepsize_power, 
            use_polynomial_rate, 
            delta_action_clip
            )

    def _langevin_update_step(
        self, ebm: nn.Module, input: torch.Tensor, action: torch.Tensor, step_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: this is the correct Langevin MCMC equation not used in the IBC paper.
        Single update step fo the Langevin MCMC sampler method. In this method we set the gradient of
        the ebm to false for the time of the method and set it back to true after the new sample
        is calculated. In addition we have to set the action input gradient to True to calculate the
        derivate of the energy w.r.t to the action input.
        In addition the gradient norm is calcualted and returned to regularize the update steps for a
        more stable training. The gradient norm implementation is based on the Wasserstein Gradient norm

        :param  ebm:        nn.Module of the ebm
        :param  input:      torch.Tensor of the current observation or input
        :param  action:     torch.Tensor of the previous action sample, which we optimize
        :step_size:         int of the current size of the langevin update step

        :return:            [
                                torch.Tensor of the new action sample,
                                torch.Tensor of energy of the input state with the previous sample
                                torch.Tensor of gradient norm of the update step
        ]
        """
        bounds = torch.as_tensor(self.bounds).to(
            self.device
        )  # [2 ,X] with X action dim
        ebm.requires_grad = False
        # input.requires_grad = True
        action = action.detach()
        action.requires_grad = True
        # compute the energy of the current sample
        e = ebm(input, action)
        sum_e = e.sum()
        # calculate the derivative w.r.t the input action of the conditional EBM

        grad = autograd.grad(outputs=sum_e, inputs=action, create_graph=True)[0]
        # activate gradient computation again
        ebm.requires_grad = True

        # compute gradient norm:
        # from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py WGAN training
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        grad_norm = torch.sqrt(torch.sum(grad**2, dim=1) + 1e-12)
        # clip delta norm values below 0
        # same clipping used in the original ibc implementation
        # clip the delta to max(0, grad_delta)
        grad_delta = grad_norm - 1
        grad_delta = grad_delta.clamp(min=0)
        # get the mean of the delta norm
        wgan_grad_norm = torch.mean(grad_delta**2)

        # get the noise
        noise = torch.randn_like(action) * 1

        # this is in the Langevin dynamics equation.
        dynamics = step_size * self._gradient_scale * grad - torch.sqrt(step_size) * noise
        action = action - dynamics
        # keep the action in the allowed normalized range
        action = action.clamp(min=bounds[0, :], max=bounds[1, :])

        return action, e, wgan_grad_norm


class InputandOutputLangevinMCMCSampler(LangevinMCMCSampler):

    def __init__(
        self, 
        noise_scale: float, 
        noise_shrink: float, 
        train_samples: int, 
        train_iterations: int = 25, 
        inference_iterations: int = 100, 
        sampler_stepsize_init: float = 0.5, 
        sampler_2nd_inference_stepsize_init: float = 0.00001, 
        sampler_stepsize_decay: float = 0.8, 
        sampler_stepsize_final: float = 0.00001, 
        sampler_stepsize_power: float = 2, 
        use_polynomial_rate: bool = True, 
        delta_action_clip: int = 0.1,
        input_train_iterations: int = 10):
        super().__init__(
            noise_scale, 
            noise_shrink, 
            train_samples, 
            train_iterations, 
            inference_iterations, 
            sampler_stepsize_init, 
            sampler_2nd_inference_stepsize_init, 
            sampler_stepsize_decay, sampler_stepsize_final, 
            sampler_stepsize_power, use_polynomial_rate, 
            delta_action_clip)
        
        self.input_bounds = None
        self._input_train_iterations = input_train_iterations
        # the Langevin MCMC uses a learning rate decay scheduler to adapt the step size of the sampling
        # based on the paper implementations
        if self._use_polynomial_rate:
            self._input_schedule = PolynomialSchedule(
                self._sampler_stepsize_init,
                self._sampler_stepsize_final,
                self._sampler_stepsize_power,
                self._inference_iterations,
            )
        else:
            self._input_schedule = ExponentialSchedule(
                self._sampler_stepsize_init, self._sampler_stepsize_decay
            )
    
    def set_input_bounds(self, bounds: np.ndarray):
        self.input_bounds = bounds
    
    def gen_input_train_samples(
        self,
        batch_size: int,
        ebm: nn.Module,
        targets: torch.Tensor,
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
        :param  targets:          torch.Tensor of all targets of the true actions of the ebm in the batch
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
            size = (batch_size, self._train_samples, self.input_bounds.shape[1])  # [B, T, X]
            start_points = np.random.uniform(
                self.input_bounds[0, :], self.input_bounds[1, :], size=size
            )
            start_points = torch.as_tensor(
                start_points, dtype=torch.float32, device=self.device
            )
        else:
            # here maybe a defined buffer will be added in the future to store good samples and use them
            # in the next iteration to reduce the number of update steps and increase training performance
            start_points = chosen_start_points

        # generate the samples with input: [ B, Z] Z: dim of obs
        samples, norm, _ = self._input_langevin_mcmc(
            ebm, start_points, targets, return_intermediate_steps=False
        )
        # reshape into 3d form and return the generated samples
        return samples.reshape(batch_size, self._train_samples, -1), norm
    
    def _input_langevin_mcmc(
        self,
        ebm: nn.Module,
        start_points: torch.Tensor,
        targets: torch.Tensor,
        return_intermediate_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Langevin Dynamics MCMC sampling method to draw input samples from a distribution using the Langevin Dynamics

        :param  ebm:            nn.Module of the energy model to draw samples form
        :param  targets:          torch.Tensor of the ebm action targets
        :param  start_points:   torch.Tensor of a chosen start point if wanted else None is recommended
        :param  return_intermediate_step:   Boolean to decide if intermediate steps of the sampling chain are returned

        :return:                [
                                    torch.Tensor of sampled actions
                                    torch.Tensor of the action norms
                                    torch.Tensor of energy values of the samples
        ]
        """
        l_samples = []
        l_energies = []
        l_norms = []
        x = start_points
        l_samples.append(start_points)
        step_size = self._sampler_stepsize_init
        # iterate self._num_iterations times to generate samples with low energy
        for idx in range(self._input_train_iterations):
            # do the Langevin update step
            x, e, grad_norm = self._input_langevin_update_step(ebm, x, targets, step_size)
            l_samples.append(x.detach())
            l_norms.append(grad_norm.detach())
            l_energies.append(e.detach())
            # adapt the step size
            step_size = self._input_schedule.get_rate(idx + 1)

        if return_intermediate_steps:
            return l_samples, l_norms, l_energies
        else:
            return l_samples[-1], l_norms[-1], l_energies[-1]
    
    def _input_langevin_update_step(
        self, ebm: nn.Module, input: torch.Tensor, action: torch.Tensor, step_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        # Same as normal langevin MCMC but for inputs of the EBM and not targets!

        :param  ebm:        nn.Module of the ebm
        :param  input:      torch.Tensor of the current observation or input
        :param  action:     torch.Tensor of the previous action sample, which we optimize
        :step_size:         int of the current size of the langevin update step

        :return:            [
                                torch.Tensor of the new action sample,
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
        action = action.detach()
        action.requires_grad = True
        # compute the energy of the current sample
        e = ebm(input, action)
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
        dynamics = dynamics.clamp(min=-delta_action_clip, max=delta_action_clip)
        input = input - dynamics
        # keep the action in the allowed normalized range
        input = input.clamp(min=bounds[0, :], max=bounds[1, :])

        return input, e, wgan_grad_norm