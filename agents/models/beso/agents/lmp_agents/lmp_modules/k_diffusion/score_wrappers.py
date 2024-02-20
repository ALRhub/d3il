from multiprocessing.sharedctypes import Value

import hydra
from torch import DictType, nn
from beso.agents.diffusion_agents.k_diffusion.utils import append_dims
import torch 
'''
Wrappers for the score-based models based on Karras et al. 2022
They are used to get improved scaling of different noise levels, which
improves training stability and model performance 

Code is adapted from:

https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
'''


class GCDenoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1.):
        super().__init__()
        self.inner_model = hydra.utils.instantiate(inner_model)
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, state, action, goal, plan, noise, sigma, **kwargs):
        pred_last = False
        if 'pred_last_action_only' in kwargs.keys():
            if kwargs['pred_last_action_only']:
                pred_last = True
                noise[:, :-1, :] = 0
                noised_input = action + noise * append_dims(sigma, action.ndim)
            else:

                noised_input = action + noise * append_dims(sigma, action.ndim)
            kwargs.pop('pred_last_action_only')
        else:
            noised_input = action + noise * append_dims(sigma, action.ndim)
            
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        # noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output = self.inner_model(state, noised_input * c_in, goal, plan, sigma, **kwargs)
        target = (action - c_skip * noised_input) / c_out
        if pred_last:
            return (model_output[:, -1, :] - target[:, -1, :]).pow(2).mean()
        else:
            return (model_output - target).pow(2).flatten(1).mean()

    def forward(self, state, action, goal, plan, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(state, action * c_in, goal, plan, sigma, **kwargs) * c_out + action * c_skip

    def get_params(self):
        return self.inner_model.parameters()


class GCDenoiserWithVariance(GCDenoiser):
    def loss(self, state, action, goal, plan, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output, logvar = self.inner_model(state, noised_input * c_in, goal, plan, sigma, return_variance=True, **kwargs)
        logvar = append_dims(logvar, model_output.ndim)
        target = (action - c_skip * noised_input) / c_out
        losses = ((model_output - target) ** 2 / logvar.exp() + logvar) / 2
        return losses.flatten(1).mean()