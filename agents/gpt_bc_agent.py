import logging

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import einops
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional
from collections import deque

from agents.base_agent import BaseAgent
import agents.models.bet.utils as utils

log = logging.getLogger(__name__)


class GptPolicy(nn.Module):

    def __init__(self,
                 model: DictConfig,
                 obs_encoder: DictConfig,
                 visual_input: bool = False,
                 device: str = 'cpu'):

        super(GptPolicy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

    def forward(self, inputs):

        # encode state and visual inputs
        # the encoder should be shared by all the baselines

        if self.visual_input:

            agentview_image, in_hand_image, state = inputs

            B, T, C, H, W = agentview_image.size()

            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            state = state.view(B * T, -1)

            # bp_imgs = einops.rearrange(bp_imgs, "B T C H W -> (B T) C H W")
            # inhand_imgs = einops.rearrange(inhand_imgs, "B T C H W -> (B T) C H W")

            obs_dict = {"agentview_image": agentview_image,
                        "in_hand_image": in_hand_image,
                        "robot_ee_pos": state}

            obs = self.obs_encoder(obs_dict)
            obs = obs.view(B, T, -1)

        else:

            obs = self.obs_encoder(inputs)

        # make prediction
        pred = self.model(obs)

        return pred

    def get_params(self):
        return self.parameters()



class Gpt_Agent(BaseAgent):
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
            window_size,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.get_params()
        )

        self.eval_model_name = "eval_best_gpt.pth"
        self.last_model_name = "last_gpt.pth"

        self.window_size = window_size

        self.obs_context = deque(maxlen=self.window_size)

    def train_agent(self):

        best_test_loss = 1e10

        # for step in tqdm(range(self.max_train_steps)):
        for num_epoch in tqdm(range(self.epoch)):

            train_loss = []

            for data in self.train_dataloader:

                observations, action, mask = data

                loss = self.train_step(observations, action)

                train_loss.append(loss)

                wandb.log(
                    {
                        "train_loss": loss
                    }
                )
            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

            ####################################################################
            # evaluate the model
            if not (num_epoch+1) % self.eval_every_n_epochs:

                test_loss = []
                for data in self.test_dataloader:

                    observations, action, mask = data
                    loss = self.evaluate(observations, action)

                    test_loss.append(loss)
                    wandb.log(
                        {
                            "test_loss": loss,
                        }
                    )

                avrg_test_loss = sum(test_loss) / len(test_loss)
                log.info("Epoch {}: Average test loss is {}".format(num_epoch, avrg_test_loss))

                if avrg_test_loss < best_test_loss:
                    best_test_loss = avrg_test_loss
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')

                wandb.log(
                    {
                        "avrg_test_loss": avrg_test_loss,
                    }
                )

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        log.info("Training done!")

    def train_step(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """
        # train the model
        self.model.train()

        obs = self.scaler.scale_input(state)
        act = self.scaler.scale_output(actions)

        out = self.model(obs)

        loss = F.mse_loss(out, act)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()

        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)

        total_mse = 0.0

        out = self.model(state)

        mse = F.mse_loss(out, action)  # , reduction="none")
        total_mse += mse.mean(dim=-1).sum().item()

        return total_mse

    def reset(self):
        """ Resets the context of the model."""
        self.obs_context.clear()

    def predict(self, state, sample=False):

        self.model.eval()

        obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        obs = self.scaler.scale_input(obs)

        # Now, add to history. This automatically handles the case where
        # the history is full.
        self.obs_context.append(obs)

        enc_obs_seq = torch.stack(tuple(self.obs_context), dim=1)  # type: ignore
        # Sample latents from the prior
        actions = self.model(enc_obs_seq)

        actions = actions.clamp_(self.min_action, self.max_action)

        actions = self.scaler.inverse_scale_output(actions)
        actions = actions.detach().cpu().numpy()

        actions = actions[:, -1]

        # actions = einops.rearrange(
        #     actions, "batch 1 action_dim -> batch action_dim"
        # )

        return actions