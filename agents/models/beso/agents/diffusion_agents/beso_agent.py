from faulthandler import disable
from functools import partial
import os
import logging
from collections import deque


import einops
from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb

from beso.agents.base_agent import BaseAgent
from beso.networks.ema_helper.ema import ExponentialMovingAverage
from beso.networks.scaler.scaler_class import Scaler
from beso.agents.diffusion_agents.k_diffusion.gc_sampling import *
import beso.agents.diffusion_agents.k_diffusion.utils as utils
from beso.agents.diffusion_agents.k_diffusion.score_gpts import DiffusionGPT

# A logger for this file
log = logging.getLogger(__name__)


class BesoAgent(BaseAgent):

    def __init__(
            self,
            model: DictConfig,
            input_encoder: DictConfig,
            optimization: DictConfig,
            device: str,
            obs_modalities: list,
            goal_modalities: list,
            target_modality: str,
            max_train_steps: int,
            max_epochs: int,
            train_method: str,
            eval_every_n_steps: int,
            use_ema: bool,
            goal_conditioned: bool,
            pred_last_action_only: bool,
            rho: float,
            num_sampling_steps: int,
            lr_scheduler: DictConfig,
            sampler_type: str,
            sigma_data: float,
            sigma_min: float,
            sigma_max: float,
            sigma_sample_density_type: str,
            sigma_sample_density_mean: float,
            sigma_sample_density_std: float,
            decay: float,
            update_ema_every_n_steps: int,
            window_size: int,
            goal_window_size: int,
            use_kde: bool=False,
            patience: int=10,
    ):
        super().__init__(model, input_encoder, optimization, obs_modalities, goal_modalities, target_modality, device, max_train_steps, eval_every_n_steps, max_epochs)

        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), decay, self.device)
        self.use_ema = use_ema
        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler,
            optimizer=self.optimizer
        )
        # define the goal conditioned flag for the model 
        self.gc = goal_conditioned
        # define the training method
        self.train_method = train_method
        self.epochs = max_epochs
        # all diffusion stuff for inference
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        # training sample density
        self.sigma_sample_density_type = sigma_sample_density_type
        self.sigma_sample_density_mean = sigma_sample_density_mean
        self.sigma_sample_density_std = sigma_sample_density_std
        # define the usage of exponential moving average
        self.decay = decay
        self.update_ema_every_n_steps = update_ema_every_n_steps
        self.patience = patience
        # get the window size for prediction
        self.window_size = window_size
        self.goal_window_size = goal_window_size
        # bool if the model should only output the last action or all actions in a sequence
        self.pred_last_action_only = pred_last_action_only
        # set up the rolling window contexts
        self.obs_context = deque(maxlen=self.window_size)
        self.goal_context = deque(maxlen=self.goal_window_size)
        # if we use DiffusionGPT we need an action context and use deques to store the actions
        self.action_context = deque(maxlen=self.window_size-1)
        self.que_actions = True
        # use kernel density estimator if true
        self.use_kde = use_kde
        self.noise_scheduler = 'exponential'

    def get_scaler(self, scaler: Scaler):
        """
        Define the scaler from the Workspace class used to scale state and output data if necessary
        """
        self.scaler = scaler

    def set_bounds(self, scaler: Scaler):
        """
        Define the bounds for the sampler class
        """
        self.model.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(self.device)
        self.model.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(self.device)
    
    def train_agent(self, train_loader, test_loader):
        """
        Train the agent on a given number of epochs or steps
        """
        if self.train_method == 'epochs':
            self.train_agent_on_epochs(train_loader, test_loader, self.epochs)
        elif self.train_method == 'steps':
            self.train_agent_on_steps(train_loader, test_loader)
        else:
            raise ValueError('Either epochs or n_steps must be specified!')
    
    def train_agent_on_epochs(self, train_loader, test_loader, epochs):
        """
        Train the agent on a given number of epochs 
        """
        self.step = 0
        interrupt_training = False
        best_test_mse = 1e10
        mean_mse = 1e10
        generator = iter(train_loader)
        for epoch in tqdm(range(epochs)):
            for batch in test_loader:
                test_mse = []
                mean_mse = self.evaluate(batch)
                test_mse.append(mean_mse)
            avrg_test_mse = sum(test_mse) / len(test_mse)
            interrupt_training, best_test_mse = self.early_stopping(best_test_mse, mean_mse, self.patience, epochs)    
            
            if interrupt_training:
                log.info('Early stopping!')
                break
            
            batch_losses = []
            for batch in train_loader:
                batch_loss = self.train_step(batch)
                batch_losses.append(batch_loss)
                self.steps += 1
                if self.steps % self.eval_every_n_steps == 0:
                    self.lr_scheduler.step()
                wandb.log(
                {
                    "training/loss": batch_loss,
                    "training/test_loss": avrg_test_mse
                }
            )
            wandb.log(
                {
                    'training/epoch_loss': np.mean(batch_losses),
                    'training/epoch_test_loss': avrg_test_mse,
                    'training/epoch': epoch,
                }
            )
            log.info("Epoch {}: Mean test mse is {}".format(epoch, avrg_test_mse))
            # log.info('New best test loss. Stored weights have been updated!')
            log.info("Epoch {}: Mean train batch loss mse is {}".format(epoch, np.mean(batch_losses)))
        self.store_model_weights(self.working_dir)
        log.info("Training done!")

    def train_agent_on_steps(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
        best_test_mse = 1e10
        mean_mse = 1e10
        generator = iter(train_loader)

        for step in tqdm(range(self.max_train_steps)):
            # run a test batch every n steps
            if not self.steps % self.eval_every_n_steps:
                test_mse = []
                for batch in test_loader:
                    mean_mse = self.evaluate(batch)
                    test_mse.append(mean_mse)
                avrg_test_mse = sum(test_mse) / len(test_mse)
                log.info("Step {}: Mean test mse is {}".format(step, avrg_test_mse))
                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir)
                    log.info('New best test loss. Stored weights have been updated!')
            try:
                batch_loss = self.train_step(next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_loader)
                batch_loss = self.train_step(next(generator))

            # log.info loss every 1000 steps
            if not self.steps % 1000:
                log.info("Step {}: Mean batch loss mse is {}".format(step, batch_loss))
            wandb.log(
                {
                    "loss": batch_loss,
                    "test_loss": avrg_test_mse
                }
            )
            
        self.store_model_weights(self.working_dir)
        log.info("Training done!")

    def train_step(self, batch: dict):
        """
        Performs a single training step using the provided batch of data.

        Args:
            batch (dict): A dictionary containing the training data.
        Returns:
            float: The value of the loss function after the training step.
        Raises:
            None
        """
        # scale data if necessarry, otherwise the scaler will return unchanged values
        state, action, goal = self.process_batch(batch, predict=False)
        
        self.model.train()
        self.model.training = True
        # set up the noise
        noise = torch.randn_like(action)
        # define the sigma values
        sigma = self.make_sample_density()(shape=(len(action),), device=self.device)
        # calculate the loss
        loss = self.model.loss(state, action, goal, noise, sigma)
        # Before the backward pass, zero all the network gradients
        self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        self.optimizer.step()
        self.lr_scheduler.step()
        self.steps += 1
        # update the ema model
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())
        return loss.item()

    @torch.no_grad()
    def evaluate(self, batch: dict):
        """
        Evaluates the model using the provided batch of data and returns the mean squared error (MSE) loss.

        Args:
            batch (dict): A dictionary containing the evaluation data
        Returns:
            float: The total mean squared error (MSE) loss.
        Raises:
            None
        """
        total_mse = 0
        # scale data if necessary, otherwise the scaler will return unchanged values
        state, action, goal = self.process_batch(batch, predict=True)
        # use the EMA model variant
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()
        self.model.training = False
        # get the sigma distribution for sampling based on Karras et al. 2022
        sigmas = get_sigmas_exponential(self.num_sampling_steps, self.sigma_min, self.sigma_max, self.device)

        x = torch.randn_like(action) * self.sigma_max
        # generate the action based on the chosen sampler type 
        x_0 = self.sample_loop(sigmas, x, state, goal, self.sampler_type)
        # x_0 = self.scaler.clip_action(x_0)
        
        if self.pred_last_action_only:
            x_0 = einops.rearrange(x_0, 'b d -> b 1 d')# only train the last timestep 
            
        mse = nn.functional.mse_loss(x_0, action, reduction="none")
        total_mse += mse.mean().item()
        #if get_mean is not None:
        #    print(f'Average STD for the action predictions: {torch.stack(pred_list).std()}')
        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        return total_mse
    
    def reset(self):
        """ Resets the context of the model."""
        self.obs_context.clear()
        self.action_context.clear()

    @torch.no_grad()
    def predict(
        self, 
        batch: dict,
        new_sampler_type=None, 
        get_mean=None, 
        new_sampling_steps=None,
        extra_args=None, 
        noise_scheduler=None
    ) -> torch.Tensor:
        """
        Predicts the output of the model based on the provided batch of data.

        Args:
            batch (dict): A dictionary containing the input data.
            new_sampler_type (str): Optional. The new sampler type to use for sampling actions. Defaults to None.
            get_mean (int): Optional. The number of samples to use for calculating the mean prediction. Defaults to None.
            new_sampling_steps (int): Optional. The new number of sampling steps to use. Defaults to None.
            extra_args: Optional. Additional arguments for the sampling loop. Defaults to None.
            noise_scheduler: Optional. The noise scheduler for the sigma distribution. Defaults to None.
        Returns:
            torch.Tensor: The predicted output of the model.
        Raises:
            None
        """
        noise_scheduler = self.noise_scheduler if noise_scheduler is None else noise_scheduler
        state, goal, _ = self.process_batch(batch, predict=True)
        if len(state.shape) == 2  and self.window_size > 1:
            self.obs_context.append(state)
            input_state = torch.stack(tuple(self.obs_context), dim=1)
        else:
            input_state = state
        if len(goal.shape) == 2 and self.window_size > 1:
            goal = einops.rearrange(goal, 'b d -> 1 b d')
            
        # change sampler type and step size if requested otherwise use self. parameters
        if new_sampler_type is not None:
            sampler_type = new_sampler_type
        else:
            sampler_type = self.sampler_type
        # same with the number of sampling steps 
        if new_sampling_steps is not None:
            n_sampling_steps = new_sampling_steps
        else:
            n_sampling_steps = self.num_sampling_steps

            
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()

        # get the sigma distribution for the desired sampling method
        sigmas = self.get_noise_schedule(n_sampling_steps, noise_scheduler)
        
        # adept for time sequence if chosen
        if self.window_size > 1:
            # depending if we use a single sample or the mean over n samples
            if get_mean is not None:
                x = torch.randn((len(input_state)*get_mean, 1, self.scaler.y_bounds.shape[1]), device=self.device) * self.sigma_max
            else:
                x = torch.randn((len(input_state), 1, self.scaler.y_bounds.shape[1]), device=self.device) * self.sigma_max
                # check if we need to get thew hole action context for the DiffusionGPT model variant
                
                if len(self.action_context) > 0:  
                    previous_a = torch.cat(tuple(self.action_context), dim=1)
                    x = torch.cat([previous_a, x], dim=1)
        else:
            if get_mean is not None:
                x = torch.randn((len(input_state)*get_mean, self.scaler.y_bounds.shape[1]), device=self.device) * self.sigma_max
        
            else:
                x = torch.randn((len(input_state), 1, self.scaler.y_bounds.shape[1]), device=self.device) * self.sigma_max
        
        x_0 = self.sample_loop(sigmas, x, input_state, goal, sampler_type, extra_args)
        
        # only get the last action if we use a sequence model 
        if x_0.size()[1] > 1 and len( x_0.size()) ==3:
            x_0 = x_0[:, -1, :]
        # if we predict a sequence we only want the last predicted action of our transformer model
        
        # scale the final output
        x_0 = self.scaler.clip_action(x_0)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        # if we have an DiffusionGPT we also que the actions
        model_pred = self.scaler.inverse_scale_output(x_0)
        # if self.que_actions or self.pred_last_action_only:
        if len(model_pred.shape) == 2:
            x_0 = einops.rearrange(x_0, 'b d -> b 1 d')
        self.action_context.append(x_0)
        return model_pred
    
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
        Main method to generate samples depending on the chosen sampler type for rollouts
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

    def load_pretrained_model(self, weights_path: str, **kwargs) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """
        self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), self.decay, self.device)
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        torch.save(self.model.state_dict(), os.path.join(store_path, "non_ema_model_state_dict.pth"))
    
    @torch.no_grad()
    def visualize_ode(
        self, 
        state: torch.Tensor, 
        goal, 
        get_mean=1000, 
        new_sampling_steps=None,
        noise_scheduler=None,
    ) -> torch.Tensor:
        """
        Only used for debugging purposes
        """
        if self.use_kde:
            get_mean = 100 if get_mean is None else get_mean

        # same with the number of sampling steps 
        if new_sampling_steps is not None:
            n_sampling_steps = new_sampling_steps
        else:
            n_sampling_steps = self.num_sampling_steps
        
        # scale data if necessary, otherwise the scaler will return unchanged values
        state = self.scaler.scale_input(state)
        goal = self.scaler.scale_input(goal)
        # manage the case for sequence models with multiple obs
        if self.window_size > 1:
            # rearrange from 2d -> sequence
            self.obs_context.append(state) # this automatically manages the number of allowed observations
            input_state = torch.stack(tuple(self.obs_context), dim=1)
        else:
            input_state = state
            
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()
        # get the noise schedule
        sigmas = self.get_noise_schedule(n_sampling_steps, noise_scheduler)
        # create the initial noise
        x = torch.randn((len(input_state)*get_mean,self.scaler.y_bounds.shape[1]), device=self.device) * self.sigma_max

        # repeat the state goal n times to get mean prediction
        state_rpt = torch.repeat_interleave(input_state, repeats=get_mean, dim=0)
        goal_rpt = torch.repeat_interleave(goal, repeats=get_mean, dim=0)
        
        # x = torch.repeat_interleave(x, repeats=get_mean, dim=0)
        # generate the action based on the chosen sampler type
        sampled_actions = [x] 
        x_0 = x
        for i in range(n_sampling_steps):
            simgas_2 = sigmas[i:(i+2)]
            x_0 = sample_ddim(self.model, state_rpt, x_0, goal_rpt, simgas_2, disable=True)
            # x_0 = sample_euler(self.model, state_rpt, x_0, goal_rpt, simgas_2, disable=True)
            sampled_actions.append(x_0)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        model_pred = self.scaler.inverse_scale_output(x_0)

        return sampled_actions
        
    def make_sample_density(self):
        """ 
        Generate a sample density function based on the desired type for training the model
        """
        sd_config = []
        
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  
            scale = self.sigma_sample_density_std 
            return partial(utils.rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        
        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')
    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps
        """
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