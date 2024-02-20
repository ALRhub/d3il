from functools import partial

import torch 
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import DictConfig
import einops 

from beso.agents.diffusion_agents.k_diffusion.gc_sampling import *

    
class DiffusionProposalNet(nn.Module):
    '''
    Diffusion Proposal Network to generate latent plans based on the current observation and desired goal
    '''
    def __init__(
        self,
        model: DictConfig,
        rho: float,
        plan_dim: int,
        sampler_type: str,
        num_sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        sigma_sample_density_type: str,
        sigma_sample_density_mean: float,
        sigma_sample_density_std: float,
        noise_scheduler: str,
        device: str,
    ) -> None:
        super().__init__()
        self.model = hydra.utils.instantiate(model)
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_sample_density_type = sigma_sample_density_type
        self.sigma_sample_density_mean = sigma_sample_density_mean
        self.sigma_sample_density_std = sigma_sample_density_std
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.plan_dim = plan_dim
    
    def forward(self, state: torch.Tensor, latent_goal: torch.Tensor, extra_args={}):
        '''
        Main forward method to generate latent plans given the current goal and state 
        '''
        # get the extra args if any
        n_sampling_steps = extra_args['n_sampling_steps'] if 'n_sampling_steps' in extra_args else self.num_sampling_steps
        noise_scheduler = extra_args['noise_scheduler'] if 'noise_scheduler' in extra_args else self.noise_scheduler
        sampler_type = extra_args['sampler_type'] if 'sampler_type' in extra_args else self.sampler_type
        # get the noise schedule
        sigmas = self.get_noise_schedule(n_sampling_steps, noise_scheduler)
        noisy_plan = torch.randn((len(state), 1, self.plan_dim), device=state.device) * self.sigma_max
        # sample the plan in n steps
        denoised_plan = self.sample_loop(
            sigmas, 
            noisy_plan, 
            state, 
            latent_goal, 
            sampler_type,
            extra_args=extra_args,
        )
        return denoised_plan
        
    def loss_and_pred(self, state: torch.Tensor, latent_goal: torch.Tensor, latent_plan: torch.Tensor, extra_args={}):
        '''
        Main training method to compute the loss and predict the latent plan for the action decoder 
        '''
        if len(state.shape) == 2:
            state = einops.rearrange(state, 'b d -> b 1 d')
        if state.shape[1] > 1:
            first_state = state[:, 0, :]
            first_state = einops.rearrange(first_state, 'b d -> b 1 d')
        else:
            first_state = state
        if len(latent_plan.shape) == 2:
            latent_plan = einops.rearrange(latent_plan, 'b d -> b 1 d')
            # latent_plan = einops.repeat(latent_plan, 'b 1 d -> b (1 t) d', t=state.shape[1])
    
        loss = self.compute_loss(first_state, latent_goal, latent_plan)
        pred_plan = self.forward(first_state, latent_goal, extra_args=extra_args)
        return loss, pred_plan
        
    def compute_loss(self, state: torch.Tensor, latent_goal: torch.Tensor, latent_plan: torch.Tensor):
        '''
        Compute the denoising loss during training 
        '''
        noise = torch.randn_like(latent_plan) # * self.sigma_max
        sigma = self.make_sample_density()(shape=(len(latent_plan),), device=self.device)
        loss = self.model.loss(state, latent_plan, latent_goal, noise, sigma)
        return loss 
    
    @torch.no_grad()
    def predict(self, state: torch.Tensor, latent_goal: torch.Tensor, extra_args={}):
        '''
        Predict the plan given the state and latent goal
        '''
        # get the extra args if any
        n_sampling_steps = extra_args['n_sampling_steps'] if 'n_sampling_steps' in extra_args else self.num_sampling_steps
        noise_scheduler = extra_args['noise_scheduler'] if 'noise_scheduler' in extra_args else self.noise_scheduler
        sampler_type = extra_args['sampler_type'] if 'sampler_type' in extra_args else self.sampler_type
        sigmas = self.get_noise_schedule(n_sampling_steps, noise_scheduler)
        noisy_plan = torch.randn((len(state), 1, self.plan_dim), device=state.device) * self.sigma_max
        # sample the plan in n steps
        denoised_plan = self.sample_loop(
            sigmas, 
            noisy_plan, 
            state, 
            latent_goal, 
            sampler_type,
            extra_args=extra_args,
        )
        return denoised_plan
        
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        '''
        Return the noise schedule for the denoising process
        '''
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def make_sample_density(self):
        '''
        Return the sample density function for the training process of the noise distribution 
        '''
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)
        raise ValueError('Unknown sample density type')
    
    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        state: torch.Tensor, 
        goal: torch.Tensor, 
        sampler_type: str,
        extra_args={}, 
        ):
        """
        Main method to generate samples depending on the chosen sampler type
        """
        # get the s_churn 
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        # extra_args.pop('s_churn', None)
        # extra_args.pop('use_scaler', None)
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x:extra_args[x] for x in keys}
        else:
            reduced_args = {}
        
        if use_scaler:
            scaler = self.scaler
        else:
            scaler=None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min, disable=True)
        # ODE deterministic 
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True) 
        # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas), disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0    