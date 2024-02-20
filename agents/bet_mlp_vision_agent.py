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

from agents.models.bet.libraries.loss_fn import FocalLoss, soft_cross_entropy
from agents.base_agent import BaseAgent
import agents.models.bet.utils as utils

log = logging.getLogger(__name__)


class Bet_Vision_Policy(nn.Module):
    def __init__(self, model: DictConfig, obs_encoder: DictConfig, visual_input: bool = False, device: str = "cpu"):
        super(Bet_Vision_Policy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

    def get_latent_and_loss(self, inputs, latent):
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

        _, loss, loss_components = self.model.get_latent_and_loss(
            obs_rep=obs,
            target_latents=latent,
            return_loss_components=True,
        )

        return _, loss, loss_components

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
        latents = self.model.generate_latents(obs)

        return latents

    def get_params(self):
        return self.parameters()


class BeTPolicy(nn.Module):

    def __init__(self,
                 model: DictConfig,
                 obs_dim: int = 2,
                 act_dim: int = 2,
                 vocab_size: int = 16,
                 predict_offsets: bool = False,
                 focal_loss_gamma: float = 0.0,
                 offset_loss_scale: int = 1.0,
                 device: str = 'cpu'):

        super(BeTPolicy, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.offset_loss_scale = offset_loss_scale
        self.focal_loss_gamma = focal_loss_gamma

        self.vocab_size = vocab_size

        self.predict_offsets = predict_offsets
        self.model = hydra.utils.instantiate(model, _recursive_=False,
                                             output_dim=self.vocab_size * (1+self.act_dim)).to(device)

    def get_latent_and_loss(self, obs_rep: torch.Tensor, target_latents: torch.Tensor,
                            return_loss_components: bool = False):

        if self.predict_offsets:
            target_latents, target_offsets = target_latents

        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (
                self.vocab_size != 1
        )

        if is_soft_target:
            target_latents = target_latents.view(-1, target_latents.size(-1))
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = FocalLoss(gamma=self.focal_loss_gamma)

        output = self.model(obs_rep)

        logits = output[:, :, :self.vocab_size]
        offsets = output[:, :, self.vocab_size:]

        batch = logits.shape[0]
        seq = logits.shape[1]
        offsets = einops.rearrange(
            offsets,
            "N T (V A) -> (N T) V A",  # N = batch, T = seq
            V=self.vocab_size,
            A=self.act_dim,
        )
        # calculate (optionally soft) cross entropy and offset losses
        class_loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
        # offset loss is only calculated on the target class
        # if soft targets, argmax is considered the target class
        selected_offsets = offsets[
            torch.arange(offsets.size(0)),
            target_latents.argmax(dim=-1).view(-1)
            if is_soft_target
            else target_latents.view(-1),
        ]
        offset_loss = self.offset_loss_scale * F.mse_loss(
            selected_offsets, target_offsets.view(-1, self.act_dim)
        )
        loss = offset_loss + class_loss
        logits = einops.rearrange(logits, "batch seq classes -> seq batch classes")
        offsets = einops.rearrange(
            offsets,
            "(N T) V A -> T N V A",
            # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            N=batch,
            T=seq,
        )
        if return_loss_components:
            return (
                (logits, offsets),
                loss,
                {"offset": offset_loss, "class": class_loss, "total": loss},
            )
        else:
            return (logits, offsets), loss

    def generate_latents(self, seq_obses: torch.Tensor):

        seq, batch, embed = seq_obses.size()

        obs_rep = einops.rearrange(seq_obses, "seq batch embed -> batch seq embed")

        output = self.model(obs_rep)

        if self.predict_offsets:
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size:]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.act_dim,
            )
        else:
            logits = output
        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        if self.predict_offsets:
            sampled_offsets = offsets[
                torch.arange(offsets.shape[0]), sampled_data.flatten()
            ].view(batch, seq, self.act_dim)

            return (sampled_data, sampled_offsets)
        else:
            return sampled_data

    def get_params(self):
        return self.parameters()


class BetMLP_Agent(BaseAgent):
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
            grad_norm_clip,
            window_size,
            obs_encoding_net: DictConfig,
            action_ae: DictConfig,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.state_prior_optimizer = hydra.utils.instantiate(optimization,
                                                             params=self.model.parameters())

        self.eval_model_name = "eval_best_bet_mlp.pth"
        self.last_model_name = "last_bet_mlp.pth"

        self.grad_norm_clip = grad_norm_clip
        self.window_size = window_size

        self.obs_encoding_net = hydra.utils.instantiate(obs_encoding_net).to(self.device)
        self.action_ae = hydra.utils.instantiate(action_ae, _recursive_=False, num_bins=self.model.model.vocab_size).to(self.device)

        self.obs_context = deque(maxlen=self.window_size)

        self.action_ae.fit_model(self.train_dataloader, self.test_dataloader, self.scaler)

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        _keys_to_save = [
            "model",
            "action_ae",
            "obs_encoding_net",
        ]
        payload = {k: self.__dict__[k] for k in _keys_to_save}

        if sv_name is None:
            file_path = os.path.join(store_path, "Bet.pth")
        else:
            file_path = os.path.join(store_path, sv_name)

        with open(file_path, "wb") as f:
            torch.save(payload, f)

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        _keys_to_save = [
            "model",
            "action_ae",
            "obs_encoding_net",
        ]

        if sv_name is None:
            file_path = os.path.join(weights_path, "Bet.pth")
        else:
            file_path = os.path.join(weights_path, sv_name)

        with open(file_path, "rb") as f:
            payload = torch.load(f, map_location=self.device)

        loaded_keys = []
        for k, v in payload.items():
            if k in _keys_to_save:
                loaded_keys.append(k)
                self.__dict__[k] = v.to(self.device)

        if len(loaded_keys) != len(_keys_to_save):
            raise ValueError(
                "Model does not contain the following keys: "
                f"{set(_keys_to_save) - set(loaded_keys)}"
            )

    def train_agent(self):

        self.action_ae.fit_model(self.train_dataloader, self.test_dataloader, self.scaler)

        best_test_loss = 1e10

        # for step in tqdm(range(self.max_train_steps)):
        for num_epoch in tqdm(range(self.epoch)):

            # train the model
            self.model.train()

            train_loss = []
            with utils.eval_mode(self.obs_encoding_net, self.action_ae):
                for data in self.train_dataloader:

                    observations, action, mask = data

                    loss, loss_components = self.train_step(observations, action)

                    train_loss.append(loss.item())

                    wandb.log(
                        {
                            "offset_loss": loss_components['offset'].item(),
                            "class_loss": loss_components['class'].item(),
                            "total_loss": loss_components['total'].item(),
                        }
                    )
                avrg_train_loss = sum(train_loss) / len(train_loss)
                log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

            ####################################################################
            # evaluate the model
            if not (num_epoch+1) % self.eval_every_n_epochs:

                with utils.eval_mode(self.obs_encoding_net, self.action_ae, self.model, no_grad=True):

                    test_loss = []
                    for data in self.test_dataloader:

                        observations, action, mask = data
                        loss, loss_components = self.evaluate(observations, action)

                        test_loss.append(loss.item())
                        wandb.log(
                            {
                                "eval_offset_loss": loss_components['offset'].item(),
                                "eval_class_loss": loss_components['class'].item(),
                                "eval_total_loss": loss_components['total'].item(),
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

    def train_vision_agent(self):

        # train the model
        self.model.train()

        train_loss = []
        with utils.eval_mode(self.obs_encoding_net, self.action_ae):
            for data in self.train_dataloader:
                bp_imgs, inhand_imgs, obs, action, mask = data

                bp_imgs = bp_imgs.to(self.device)
                inhand_imgs = inhand_imgs.to(self.device)

                obs = self.scaler.scale_input(obs)
                action = self.scaler.scale_output(action)

                state = (bp_imgs, inhand_imgs, obs)

                loss, loss_components = self.train_step(state, action)

                train_loss.append(loss.item())

                wandb.log(
                    {
                        "offset_loss": loss_components['offset'].item(),
                        "class_loss": loss_components['class'].item(),
                        "total_loss": loss_components['total'].item(),
                    }
                )


    def train_step(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """

        self.state_prior_optimizer.zero_grad(set_to_none=True)

        # obs = self.scaler.scale_input(state)
        # act = self.scaler.scale_output(actions)

        # enc_obs = self.obs_encoding_net(obs)
        latent = self.action_ae.encode_into_latent(actions)

        _, loss, loss_components = self.model.get_latent_and_loss(inputs=state, latent=latent)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_norm_clip
        )
        self.state_prior_optimizer.step()

        return loss, loss_components

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        Method for evaluating the model on one epoch of data
        """
        obs = self.scaler.scale_input(state)
        act = self.scaler.scale_output(action)

        enc_obs = self.obs_encoding_net(obs)
        latent = self.action_ae.encode_into_latent(act, enc_obs)
        _, loss, loss_components = self.model.get_latent_and_loss(
            obs_rep=enc_obs,
            target_latents=latent,
            return_loss_components=True,
        )

        return loss, loss_components

    # def _setup_action_sampler(self):
    #     def sampler(actions):
    #         idx = np.random.randint(len(actions))
    #         return actions[idx]
    #
    #     self.sampler = sampler

    def predict(self, state, sample=False, if_vision=False):

        with utils.eval_mode(
            self.action_ae, self.obs_encoding_net, self.model, no_grad=True
        ):

            if if_vision:
                bp_image, inhand_image, des_robot_pos = state

                bp_image = torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0).unsqueeze(0)
                inhand_image = torch.from_numpy(inhand_image).to(self.device).float().unsqueeze(0).unsqueeze(0)
                des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0).unsqueeze(0)

                des_robot_pos = self.scaler.scale_input(des_robot_pos)

                obs = (bp_image, inhand_image, des_robot_pos)
            else:
                obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(0)
                obs = self.scaler.scale_input(obs)

            # enc_obs = self.obs_encoding_net(obs)

            latents = self.model(obs)

            if type(latents) is tuple:
                latents, offsets = latents

            action_latents = (latents[:, -1:, :], offsets[:, -1:, :])

            actions = self.action_ae.decode_actions(
                latent_action_batch=action_latents,
            )

            actions = actions.clamp_(self.min_action, self.max_action)

            actions = self.scaler.inverse_scale_output(actions)

            actions = actions.cpu().numpy()

            if sample:
                sampled_action = np.random.randint(len(actions))
                actions = actions[sampled_action]
                # (seq==1, action_dim), since batch dim reduced by sampling
                # actions = einops.rearrange(actions, "1 action_dim -> action_dim")
            else:
                # (batch, seq==1, action_dim)
                actions = einops.rearrange(
                    actions, "batch 1 action_dim -> batch action_dim"
                )

            return actions
        
    def reset(self):
        """ Resets the context of the model."""
        self.obs_context.clear()