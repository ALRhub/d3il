import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import einops 
import numpy as np 
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from beso.agents.base_agent import BaseAgent
from beso.networks.scaler.scaler_class import Scaler


log = logging.getLogger(__name__)


class LatentPlansAgent(BaseAgent):
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
        super().__init__(model, input_encoder, optimization, obs_modalities, goal_modalities, target_modality, device, max_train_steps, eval_every_n_steps, max_epochs)

        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler,
            optimizer=self.optimizer
        )
        self.train_method = train_method
        self.gc = goal_conditioned
        # get the window size for prediction
        self.window_size = window_size
        self.goal_window_size = goal_window_size
        self.patience = patience
        self.epochs_no_improvement = 0
        self.latent_plans = []
        self.goal_task_names = []
        self.collect_plans = False
        
    def get_scaler(self, scaler: Scaler):
        """
        Define the scaler from the Workspace class used to scale input and output data if necessary
        """
        self.scaler = scaler
    
    def set_bounds(self, scaler: Scaler):
        """
        Define the bounds for the sampler class
        """
        self.model.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(self.device)
        self.model.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(self.device)
        
        self.model.action_decoder.set_bounds(scaler, self.window_size, self.device)
        
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
        '''
        Method to train the agent on a given number of steps instead of epochs
        '''
        best_test_mse = 1e10
        mean_mse = 1e10
        generator = iter(train_loader)

        for step in tqdm(range(self.max_train_steps)):
            # run a test batch every n steps
            if not self.steps % self.eval_every_n_steps:
                test_mse = []
                for data in test_loader:
                    state, action, mask,goal = data
                    mean_mse = self.evaluate(state, action, mask, goal)
                    test_mse.append(mean_mse)
                avrg_test_mse = sum(test_mse) / len(test_mse)
                log.info("Step {}: Mean test mse is {}".format(step, avrg_test_mse))
                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir)
                    log.info('New best test loss. Stored weights have been updated!')

            try:
                batch_loss = self.train_step(*next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_loader)
                batch_loss = self.train_step(*next(generator))

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
        Executes a single training step on a mini-batch of data
        """
        self.model.train()
        state, actions, goal = self.process_batch(batch, predict=False)
        loss = self.model.compute_loss(state, actions, goal)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, batch: dict) -> float:
        """
        Method for evaluating the model on one epoch of data
        """
        state, actions, goal = self.process_batch(batch, predict=True)
        self.model.eval()
        
        mse = self.model.compute_val_loss(state, actions, goal)
        total_mse = mse.mean(dim=-1).mean().item()
        return total_mse

    @torch.no_grad()
    def predict(self, batch: dict) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        state, goal, goal_task_name = self.process_batch(batch, predict=True)
        self.model.eval()
        if len(state.shape) == 2:
            state = einops.rearrange(state, 'b d -> 1 b d')
        if len(goal.shape) == 2:
            goal = einops.rearrange(goal, 'b d -> 1 b d')
     
        out, latent_plan = self.model.step(state, goal)
        
        model_pred = self.scaler.inverse_scale_output(out)

        if self.collect_plans:
            if self.is_tensor_in_list(latent_plan, self.latent_plans) == False:
                self.latent_plans.append(latent_plan)
                self.goal_task_names.append(goal_task_name)
        return model_pred

    def store_model_weights(self, working_dir: str):
        """
        Stores the model weights in the working directory
        """
        torch.save(self.model.state_dict(), os.path.join(working_dir, "model_state_dict.pth"))
    
    def generate_tnse_plot(self, labels=None):
        """
        Generates a t-SNE plot of the latent plans with additional sub goal labels
        """
        latent_plans = torch.cat(self.latent_plans).squeeze(1).cpu().numpy()

        labels = self.goal_task_names 
        unique_labels = list(set(labels))
        num_labels = len(unique_labels)
        tsne = TSNE(n_components=2, verbose=2, perplexity=10, n_iter=300, learning_rate=200)
        tsne_results = tsne.fit_transform(latent_plans)

        # palette = sns.color_palette("tab10", num_labels)
        # label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}
        # label_colors = [label_to_color[label] for label in labels]

        plt.figure(figsize=(16,10), dpi=300)
        sns.scatterplot(
            x=tsne_results[:,0], y=tsne_results[:,1],
            hue=labels,
            palette=sns.color_palette("tab10", 10),
            legend="full",
            alpha=0.3
        )
        filepath = os.path.join(self.working_dir, "tsne_latent_plans.png")
        plt.savefig(filepath)
        plt.close()

    @staticmethod
    def is_tensor_in_list(tensor, tensor_list):
        for t in tensor_list:
            if torch.all(torch.eq(t, tensor)):
                return True
        return False