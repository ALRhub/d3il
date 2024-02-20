import logging
import os
from typing import Optional

import os
import logging
from collections import deque

import einops
from omegaconf import DictConfig
import hydra
import torch
from tqdm import tqdm
import wandb

from agents.base_agent import BaseAgent

# A logger for this file
log = logging.getLogger(__name__)


class ActAgent(BaseAgent):

    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            eval_every_n_epochs: int,
            goal_conditioned: bool,
            lr_scheduler: DictConfig,
            decay: float,
            # update_ema_every_n_steps: int,
            window_size: int,
            obs_size: int,
            action_seq_size: int,
            goal_window_size: int,
            que_actions: bool = False,
            patience: int = 10,
            kl_loss_factor: float = 1.0
    ):
        super().__init__(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        self.kl_loss_factor = kl_loss_factor

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.eval_model_name = "eval_best_act.pth"
        self.last_model_name = "last_act.pth"

        # define the goal conditioned flag for the model
        self.gc = goal_conditioned
        # define the training method

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.get_params()
        )

        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler,
            optimizer=self.optimizer
        )
        # define the usage of exponential moving average
        self.decay = decay
        # self.update_ema_every_n_steps = update_ema_every_n_steps
        self.patience = patience
        # get the window size for prediction
        self.window_size = window_size
        self.goal_window_size = goal_window_size
        # set up the rolling window contexts
        self.action_seq_size = action_seq_size
        self.obs_size = obs_size
        self.window_size = window_size
        self.obs_context = deque(maxlen=self.obs_size)
        self.goal_context = deque(maxlen=self.goal_window_size)
        # if we use DiffusionGPT we need an action context and use deques to store the actions
        self.action_context = deque(maxlen=self.action_seq_size)
        self.que_actions = que_actions
        self.pred_counter = 0
        self.action_counter = self.action_seq_size
        assert self.window_size == self.action_seq_size + self.obs_size - 1, "window_size does not match the expected value"

        self.act_buffer = torch.zeros(())

    def train_agent(self):
        """
        Train the agent on a given number of epochs
        """
        self.step = 0
        interrupt_training = False
        best_test_mse = 1e10
        mean_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch + 1) % self.eval_every_n_epochs:

                test_mse = []
                for data in self.test_dataloader:
                    state, action, mask = data

                    mean_mse = self.evaluate(state, action)

                    test_mse.append(mean_mse)

                avrg_test_mse = sum(test_mse) / len(test_mse)

                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')

                wandb.log(
                    {
                        "mean_test_loss": avrg_test_mse,
                    }
                )

            batch_losses = []
            for data in self.train_dataloader:
                state, action, mask = data

                batch_loss = self.train_step(state, action)

                batch_losses.append(batch_loss)
                wandb.log(
                    {
                        "training/loss": batch_loss,
                    }
                )

            avrg_train_loss = sum(batch_losses) / len(batch_losses)

            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        log.info("Training done!")

    def train_step(self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        # state, actions, goal = self.process_batch(batch, predict=False)

        state = self.scaler.scale_input(state)
        actions = self.scaler.scale_output(actions)

        a_hat, (mu, logvar) = self.model(state, goal, actions)
        action_loss = torch.mean((a_hat - actions) ** 2)
        total_kld, dimension_wise_kld, mean_kld = self.kl_divergence(mu, logvar)

        total_loss = action_loss + total_kld.mean() * self.kl_loss_factor
        wandb.log(
            {
                "kl_loss": total_kld.mean(),
            }
        )
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return total_loss.item()

    @torch.no_grad()
    def evaluate(self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None) -> float:
        """
        Method for evaluating the model on one epoch of data
        """
        # state, actions, goal = self.process_batch(batch, predict=True)
        self.model.eval()

        state = self.scaler.scale_input(state)
        actions = self.scaler.scale_output(actions)

        a_hat, (mu, logvar) = self.model(state, goal, actions)
        action_loss = torch.mean((a_hat - actions) ** 2)
        total_kld, dimension_wise_kld, mean_kld = self.kl_divergence(mu, logvar)

        total_loss = action_loss + total_kld.mean()

        # a_hat, (mu, logvar) = self.model(state, goal)
        # loss = torch.mean((a_hat - actions) ** 2)
        # total_mse = loss.item()
        return total_loss.item()

    def reset(self):
        """ Resets the context of the model."""
        self.action_counter = self.action_seq_size

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(0)
        state = self.scaler.scale_input(state)

        if self.action_counter == self.action_seq_size:
            self.action_counter = 0

            # state, goal, goal_task_name = self.process_batch(batch, predict=True)

            self.model.eval()
            if len(state.shape) == 2:
                state = einops.rearrange(state, 'b d -> 1 b d')
            # if len(goal.shape) == 2:
            #     goal = einops.rearrange(goal, 'b d -> 1 b d')

            a_hat, (_, _) = self.model(state, goal)  # no action, sample from prior

            a_hat = a_hat.clamp_(self.min_action, self.max_action)

            model_pred = self.scaler.inverse_scale_output(a_hat)
            self.curr_action_seq = model_pred

        next_action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        return next_action.detach().cpu().numpy()

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld