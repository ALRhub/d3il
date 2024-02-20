import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class CVAEPolicy(nn.Module):
    def __init__(self, model: DictConfig, obs_encoder: DictConfig, visual_input: bool = False, device: str = "cpu"):
        super(CVAEPolicy, self).__init__()

        self.visual_input = visual_input
        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        self.model = hydra.utils.instantiate(model).to(device)

    def forward(self, state, action):

        # encode state and visual inputs
        # the encoder should be shared by all the baselines

        if self.visual_input:

            agentview_image, in_hand_image, state = state

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

            obs = self.obs_encoder(state)

        # make prediction
        pred = self.model(obs, action)

        return pred

    def predict(self, state):
        if self.visual_input:

            agentview_image, in_hand_image, state = state

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

            obs = self.obs_encoder(state)

        # make prediction
        pred = self.model.predict(obs)

        return pred

    def get_params(self):
        return self.parameters()


class CVAEAgent(BaseAgent):
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
            eval_every_n_epochs: int = 50,
            kl_loss_factor: float = 1.0
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # Define the number of GPUs available
        num_gpus = torch.cuda.device_count()

        # Check if multiple GPUs are available and select the appropriate device
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            self.model = nn.DataParallel(self.model)

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )

        self.eval_model_name = "eval_best_cvae.pth"
        self.last_model_name = "last_cvae.pth"

        self.kl_loss_factor = kl_loss_factor

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

    def train(self):
        if self.model.visual_input:
            self.train_vision_agent()
        else:
            self.train_agent()

    def train_agent(self):
        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch + 1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:
                    state, action, mask = [torch.squeeze(data[i]) for i in range(3)]

                    state = self.scaler.scale_input(state)
                    action = self.scaler.scale_output(action)

                    mean_mse = self.evaluate(state, action)
                    test_mse.append(mean_mse)

                    wandb.log(
                        {
                            "test_loss": mean_mse,
                        }
                    )

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

            train_loss = []
            for data in self.train_dataloader:
                state, action, mask = [torch.squeeze(data[i]) for i in range(3)]

                state = self.scaler.scale_input(state)
                action = self.scaler.scale_output(action)

                batch_loss = self.train_step(state, action)

                train_loss.append(batch_loss)

                wandb.log(
                    {
                        "loss": batch_loss,
                    }
                )

            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_vision_agent(self):

        train_loss = []
        for data in self.train_dataloader:
            bp_imgs, inhand_imgs, obs, action, mask = data

            bp_imgs = bp_imgs.to(self.device)
            inhand_imgs = inhand_imgs.to(self.device)

            obs = self.scaler.scale_input(obs)
            action = self.scaler.scale_output(action)

            state = (bp_imgs, inhand_imgs, obs)

            batch_loss = self.train_step(state, action)

            train_loss.append(batch_loss)

            wandb.log(
                {
                    "loss": batch_loss,
                }
            )

    def train_step(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        action_pred, mean, std = self.model(state, actions)

        mse_loss = F.mse_loss(action_pred, actions)
        # kl divergence part of the training loss
        KL_loss = -0.5 * (1 + torch.log(std.pow(2) + 1e-8) - mean.pow(2) - std.pow(2)).mean()

        wandb.log(
            {
                "mse_loss": mse_loss.item(),
                "kl_loss": KL_loss.item()
            }
        )

        loss = mse_loss + self.kl_loss_factor * KL_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()

        total_loss = 0.0

        action_pred, mean, std = self.model(state, actions)

        loss = F.mse_loss(action_pred, actions)
        # kl divergence part of the training loss
        KL_loss = -0.5 * (1 + torch.log(std.pow(2) + 1e-8) - mean.pow(2) - std.pow(2)).mean()

        loss = loss + self.kl_loss_factor * KL_loss

        total_loss += loss.mean(dim=-1).sum().item()

        return total_loss

    @torch.no_grad()
    def predict(self, state, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()

        if if_vision:
            bp_image, inhand_image, des_robot_pos = state

            bp_image = torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0).unsqueeze(0)
            inhand_image = torch.from_numpy(inhand_image).to(self.device).float().unsqueeze(0).unsqueeze(0)
            des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0).unsqueeze(0)

            des_robot_pos = self.scaler.scale_input(des_robot_pos)

            state = (bp_image, inhand_image, des_robot_pos)
        else:
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            state = self.scaler.scale_input(state)

        out = self.model.predict(state)

        out = out.clamp_(self.min_action, self.max_action)

        model_pred = self.scaler.inverse_scale_output(out)
        return model_pred.detach().cpu().numpy()[0]

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        if sv_name is None:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        else:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        if sv_name is None:
            torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))
            
    def reset(self):
        pass