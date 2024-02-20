import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import einops 
import numpy as np 

from beso.agents.base_agent import BaseAgent
from beso.networks.scaler.scaler_class import Scaler
from beso.agents.lmp_agents.lmp_agent import LatentPlansAgent

log = logging.getLogger(__name__)


class VqLatentPlansAgent(LatentPlansAgent):
    '''
    Variant of LatentPlansAgent that uses a VQ-VAE plan recognizer with a GPT plan proposal module.
    '''
    def __init__(
            self,
            model: DictConfig,
            input_encoder: DictConfig,
            optimization: DictConfig,
            obs_modalities: list,
            goal_modalities: list,
            target_modality: str,
            lr_scheduler: DictConfig,
            device: str,
            goal_conditioned: bool,
            train_method: str,
            max_train_steps: int,
            max_epochs: int,
            eval_every_n_steps: int,
            window_size: int,
            goal_window_size: int,
            patience: int
    ):
        super().__init__(model, input_encoder, optimization, obs_modalities, goal_modalities, target_modality, lr_scheduler, device, goal_conditioned, train_method, max_train_steps, max_epochs, eval_every_n_steps, window_size, goal_window_size, patience)

        self.optimizer = 0
        
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
            self.store_model_weights(self.working_dir)
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

    def train_agent(self, train_loader, test_loader):
        """
        Train the agent on a given number of epochs or steps
        """
        if self.train_method == 'epochs':
            self.train_agent_on_epochs(train_loader, test_loader, self.epochs)
        elif self.train_method == 'steps':
            self.train_agent_on_steps(train_loader, test_loader, self.max_train_steps)
        else:
            raise ValueError('Either epochs or n_steps must be specified!')
        
        self.fine_tune_agent(train_loader, test_loader, self.epochs)
        
    def fine_tune_agent(self, train_loader, test_loader, epochs):
        """
        Fine tune the agent on a given number of epochs 
        """
        self.model.freeze_model_weights()
        self.step = 0
        interrupt_training = False
        best_test_mse = 1e10
        mean_mse = 1e10
        generator = iter(train_loader)
        if epochs > 200:
            epochs = 100
        for epoch in tqdm(range(epochs)):
            for batch in test_loader:
                test_mse = []
                mean_mse = self.finetune_evaluate(batch)
                test_mse.append(mean_mse)
            avrg_test_mse = sum(test_mse) / len(test_mse)
            interrupt_training, best_test_mse = self.early_stopping(best_test_mse, mean_mse, self.patience, epochs)    
            
            if interrupt_training:
                log.info('Early stopping!')
                break
            
            batch_losses = []
            for batch in train_loader:
                batch_loss = self.finetune_train_step(batch)
                batch_losses.append(batch_loss)
                self.steps += 1
                wandb.log(
                {
                    "fine_tuning/loss": batch_loss,
                    "fine_tuning/test_loss": avrg_test_mse
                }
            )
            wandb.log(
                {
                    'fine_tuning/epoch_loss': np.mean(batch_losses),
                    'fine_tuning/epoch_test_loss': avrg_test_mse,
                    'fine_tuning/epoch': epoch,
                }
            )
            log.info("Epoch {}: Mean test mse is {}".format(epoch, avrg_test_mse))
            # log.info('New best test loss. Stored weights have been updated!')
            log.info("Epoch {}: Mean train batch loss mse is {}".format(epoch, np.mean(batch_losses)))
        self.store_model_weights(self.working_dir)
        log.info("Training done!")
    
    def train_step(self, batch: dict):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()
        state, actions, goal = self.process_batch(batch, predict=False)
        loss = self.model.compute_loss(state, actions, goal)
        return loss
        
    def finetune_train_step(self, batch: dict):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()
        state, actions, goal = self.process_batch(batch, predict=False)
        loss = self.model.compute_finetune_loss(state, actions, goal)
        return loss

    @torch.no_grad()
    def finetune_evaluate(self, batch: dict) -> float:
        """
        Method for evaluating the model on one epoch of data
        """
        state, actions, goal = self.process_batch(batch, predict=True)
        self.model.eval()
        
        mse = self.model.compute_tuning_val_loss(state, actions, goal)
        total_mse = mse.mean(dim=-1).mean().item()
        return total_mse