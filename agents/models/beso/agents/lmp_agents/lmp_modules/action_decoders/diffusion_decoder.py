
from functools import partial
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from collections import deque

import numpy as np
import hydra
from omegaconf import ListConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import hydra

from beso.networks.scaler.scaler_class import Scaler
from beso.agents.lmp_agents.lmp_modules.k_diffusion.plan_gc_sampling import *

logger = logging.getLogger(__name__)

from .action_decoder import ActionDecoder
from beso.agents.diffusion_agents.k_diffusion.score_gpts import Block


class DiffusionActionDecoder(nn.Module):

    def __init__(
        self,
        model:OmegaConf,
        window_size: int,
        goal_window_size: int,
        rho: float,
        sampler_type: str,
        num_sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        sigma_sample_density_type: str,
        sigma_sample_density_mean: float,
        sigma_sample_density_std: float,
        noise_scheduler: str,
        action_dim: int,
        latent_plan_dim: int,
        perceptual_emb_dim: int,
        latent_goal_dim: int,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.latent_plan_dim = latent_plan_dim
        self.perceptual_emb_dim = perceptual_emb_dim
        self.latent_goal_dim = latent_goal_dim
        self.model = hydra.utils.instantiate(model).to(self.device)
        self.sampler_type = sampler_type
        self.noise_scheduler = noise_scheduler
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_sample_density_type = sigma_sample_density_type
        self.sigma_sample_density_mean = sigma_sample_density_mean
        self.sigma_sample_density_std = sigma_sample_density_std

        self.window_size = window_size
        self.goal_window_size = goal_window_size
        self.obs_context = deque(maxlen=self.window_size)
        self.goal_context = deque(maxlen=self.goal_window_size)
        # if we use DiffusionGPT we need an action context and use deques to store the actions
        self.action_context = deque(maxlen=self.window_size-1)
        self.que_actions = True

    @torch.no_grad()
    def act(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor,  extra_args={}) -> torch.Tensor:
        
        
        n_sampling_steps = extra_args['n_sampling_steps'] if 'n_sampling_steps' in extra_args else self.num_sampling_steps
        noise_scheduler = extra_args['noise_scheduler'] if 'noise_scheduler' in extra_args else self.noise_scheduler
        sampler_type = extra_args['sampler_type'] if 'sampler_type' in extra_args else self.sampler_type
        
        # get the noise schedule
        sigmas = self.get_noise_schedule(n_sampling_steps, noise_scheduler)        
        # for the rollout we store the obersevations in a deque
        # this is the case for a batch size of 1
        if perceptual_emb.shape[0] == 1:
            self.obs_context.append(perceptual_emb) # this automatically manages the number of allowed observations
            input_state = torch.concat(tuple(self.obs_context), dim=1)

            # get the initial action
            # if we have a previous action, we need to concatenate it to the input state
            if len(self.action_context) > 0:  
                previous_a = torch.cat(tuple(self.action_context), dim=1)
                x = torch.randn((len(input_state), 1, self.action_dim), device=input_state.device) * self.sigma_max
                a_T = torch.cat([previous_a, x], dim=1)
            else:
                a_T = torch.randn((len(input_state), perceptual_emb.shape[1], self.action_dim), device=input_state.device) * self.sigma_max
        else:
            input_state = perceptual_emb
            a_T = torch.randn((len(input_state), perceptual_emb.shape[1], self.action_dim), device=input_state.device) * self.sigma_max
        
        # sample the plan in n steps
        a_0 = self.sample_loop(
            sigmas, 
            a_T, 
            input_state,
            latent_goal,
            latent_plan,  
            sampler_type,
            extra_args=extra_args,
        )
        # if a_0.size()[1] > 1 and len(a_0.size()) ==3:
        #     a_0 = a_0[:, -1, :]
        last_action = a_0[:, -1, :]
        last_action = einops.rearrange(last_action, 'b d -> b 1 d')
        self.action_context.append(last_action)
        
        if a_0.size()[1] > 1 and perceptual_emb.shape[0] == 1:
            a_0 = a_0[:, -1, :]
            
        return a_0

    def loss(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn_like(actions)
        sigma = self.make_sample_density()(shape=(len(actions),), device=self.device)
        loss = self.model.loss(perceptual_emb, actions, latent_goal, latent_plan, noise, sigma)
        return loss 

    def loss_and_act(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = self.loss(latent_plan, perceptual_emb, latent_goal, actions)
        action = self.act(latent_plan, perceptual_emb, latent_goal)
        return loss, action
    
    def set_bounds(self, scaler: Scaler, window_size, device):
        self.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(device)
        self.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(device)

    def clear_hidden_state(self) -> None:
        self.obs_context.clear()
        self.goal_context.clear()
        self.action_context.clear()

    def forward(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.act(latent_plan, perceptual_emb, latent_goal)
    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
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

    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        state: torch.Tensor, 
        goal: torch.Tensor, 
        plan: torch.Tensor,
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
            x_0 = sample_lms(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min, disable=True)
        # ODE deterministic 
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True) 
        # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, plan, sigmas, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, plan, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, plan,  sigmas[-2].item(), sigmas[0].item(), len(sigmas), disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, plan, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0    
    
    def make_sample_density(self):

        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)
        raise ValueError('Unknown sample density type')



class LatentPlansDiffusionGPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        state_dim: int,
        latent_plan_dim: int,
        device: str,
        goal_conditioned: bool,
        action_dim: int,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        sigma_vocab_size: int,
        goal_drop: float = 0.1,
        plans_drop: float = 0.1,
        linear_output = False,
    ):
        super().__init__()
        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + 2 * obs_seq_len + 2
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len + 2
        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.plan_emb = nn.Linear(latent_plan_dim, embed_dim)
        self.drop = nn.Dropout(embed_pdrob)
        
        # needed for classifier guidance learning
        self.cond_mask_prob = goal_drop
        self.cond_mask_prob_plans = plans_drop
        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim
        # transformer
        self.blocks = nn.Sequential(
            *[Block(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size
            ) for _ in range(n_layers)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        # we need another embedding for the sigma
        self.sigma_emb = nn.Linear(1, embed_dim) 
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        # action pred module 
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), 
                nn.GELU(),  
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim)
        
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, LatentPlansDiffusionGPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(
        self, 
        states,
        actions, 
        goals,
        plan,
        sigma,
        uncond: Optional[bool] =False,
        keep_last_actions: Optional[bool] = False
    ):  
        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the sigma embedding
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas.to(torch.float32))
        if len(states.shape) == 3:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        
        if len(plan.shape) == 2:
            plan = einops.rearrange(plan, 'b d -> b 1 d')
        # define the total length of the input sequence
        seq_length = self.goal_seq_len + 1 + t*2
        # get the beginning of the state action pairs
        
        if self.goal_conditioned:
            second_half_idx = self.goal_seq_len + 2 
        else:
            second_half_idx = 2
        # during training randomly mask out the goal
        # to train the conditional model with classifier-free guidance wen need 
        # to 0 out some of the conditional during training with a desired probability
        # it is usually in the range of 0,1 to 0.2 
        if self.training:
            goals = self.mask_cond(goals, self.cond_mask_prob)
            plan = self.mask_cond(plan, self.cond_mask_prob_plans)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)  
            plan = torch.zeros_like(plan).to(self.device)
        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        goal_embed = self.tok_emb(goals)
        action_embed = self.action_emb(actions)
        plan_embed = self.plan_emb(plan)
        
        # if not uncond:
        if self.goal_conditioned:
            position_embeddings = self.pos_emb[
            :, :(t + self.goal_seq_len), :
            ]  # each position maps to a (learnable) vector
        else: # without goal conditioning we only have the obs sequence 
            position_embeddings = self.pos_emb[
            :, :t, :
            ]
        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the masking would not make sense
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        # the action get the same position embedding as the related states 
        action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])
        plan_x = self.drop(plan_embed)
        # now for the complicated part
        # we need to stack the input in the following order:
        # [sigma_emb, s_g1, .., sg_n, s_1, a_1, s_2, a_2, ..., s_n, a_n]
        # first stack actions and states in the way: [s_1, a_1, s_2, a_2, ..,]
        sa_seq = torch.stack([state_x, action_x], dim=1
                            ).permute(0, 2, 1, 3).reshape(b, 2*t, self.embed_dim)
        
        # next we stack everything together 
        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, plan_x, goal_x, sa_seq], dim=1)
        else:
            input_seq = torch.cat([emb_t,  plan_x, sa_seq], dim=1)
        '''else:
            position_embeddings = self.pos_emb[
            :, :(t), :
            ]  # each position maps to a (learnable) vector
            state_x = self.drop(state_embed + position_embeddings)
            # the action get the same position embedding as the related states 
            action_x = self.drop(action_embed + position_embeddings)
            sa_seq = torch.stack([state_x, action_x], dim=1
                                ).permute(0, 2, 1, 3).reshape(b, 2*t, self.embed_dim)
            
            # next we stack everything together 
            input_seq = torch.cat([emb_t, sa_seq], dim=1)'''
        
        # Note we need to also adept the action masks 
        x = self.blocks(input_seq)
        x = self.ln_f(x)
        
        # now we want the last half of the output
        x = x[:, second_half_idx:, :]
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # we need to check this for inference and adapt the max seq len accord
        if x.size()[1] < 2*self.obs_seq_len:
            x_len = int(x.size()[1]/2)
            x = x.reshape(b, x_len, 2, self.embed_dim).permute(0, 2, 1, 3)
        else:
            x = x.reshape(b, self.obs_seq_len, 2, self.embed_dim).permute(0, 2, 1, 3)
        # get the outputs related to the actions
        action_outputs = x[:, 1]
        pred_actions = self.action_pred(action_outputs)
        if keep_last_actions:
            pred_actions = torch.cat([actions[:, :-1, :], pred_actions[:, -1, :].reshape(1, 1, -1)], dim=1)

        return pred_actions
    
    def mask_cond(self, cond, cond_mask_prob, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and cond_mask_prob > 0.:
            # TODO Check which one is correct
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * cond_mask_prob) # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            # mask = torch.bernoulli(torch.ones((bs, t, 1), device=cond.device) * self.cond_mask_prob)
            # mask = einops.repeat(mask, 'b t 1 -> b t (1 d)', d=d)
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()