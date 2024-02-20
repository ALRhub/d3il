from omegaconf import DictConfig
import hydra
import wandb
from tqdm import tqdm
from typing import Optional
from einops import einops

import os
import logging

from agents.models.ibc.ebm_losses import *
from agents.models.ibc.ema_helper.ema import ExponentialMovingAverage
from agents.base_agent import BaseAgent
from agents.models.ibc.samplers.langevin_mcmc import LangevinMCMCSampler
from agents.models.ibc.samplers.noise_sampler import NoiseSampler

# A logger for this file
log = logging.getLogger(__name__)


class IBCPolicy(nn.Module):
    def __init__(self, model: DictConfig, obs_encoder: DictConfig, visual_input: bool = False, device: str = "cpu"):
        super(IBCPolicy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

    def get_embedding(self, inputs):

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

        return obs

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


class IBCAgent(BaseAgent):

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
            eval_every_n_epochs,

            lr_scheduler: DictConfig,
            sampler: DictConfig,
            loss_type: str = "info_nce",
            avrg_e_regularization: float = 0,
            kl_loss_factor: float = 0,
            grad_norm_factor: float = 1,
            use_ema: bool = False,
            decay: float = 0.999,
            update_ema_every_n_steps: int = 1,
            goal_conditioning: bool = True,
            stop_value: int = 1,
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        self.eval_model_name = "eval_best_ibc.pth"
        self.last_model_name = "last_ibc.pth"

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.get_params()
        )

        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), decay, self.device)
        self.sampler = hydra.utils.instantiate(sampler)
        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler,
            optimizer=self.optimizer
        )
        self.use_ema = use_ema
        self.update_ema_every_n_steps = update_ema_every_n_steps
        self.loss_type = loss_type
        self.avrg_e_regularization = avrg_e_regularization
        self.kl_loss_factor = kl_loss_factor
        self.grad_norm_factor = grad_norm_factor
        # if we use Langevin MCMC sampling we use the WGAN norm as an additional loss term
        if isinstance(self.sampler, LangevinMCMCSampler) or isinstance(self.sampler, NoiseSampler):
            self.use_grad_norm = True
        else:
            self.use_grad_norm = False

        self.goal_conditioning = goal_conditioning
        self.stop_value = stop_value

        self.mse_loss = nn.MSELoss()

        self.steps = 0

        self.set_bounds(self.scaler)

    def set_bounds(self, scaler):
        """
        Define the bounds for the sampler class
        """
        self.sampler.get_bounds(scaler)

    def train_agent(self):

        best_test_mse = 1e10
        mean_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            # run a test batch every n epochs
            if not (num_epoch+1) % self.eval_every_n_epochs:

                test_mse = []
                for data in self.test_dataloader:
                    if self.goal_conditioning:
                        state, action, mask, goal = data
                    else:
                        state, action, mask = data
                        goal = None

                    mean_mse = self.evaluate(state, action, goal)
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

            epoch_loss = 0
            mse_neg_loss = 0

            # epoch training
            for i, inputs in enumerate(self.train_dataloader):
                if self.goal_conditioning:
                    state, action, mask, goal = inputs
                else:
                    state, action, mask = inputs
                    goal = None

                batch_loss, loss_info = self.train_step(state, action, goal)  # TODO get mean of loss/grad info
                batch_loss = batch_loss.detach().cpu().numpy()

                epoch_loss += batch_loss
                mse_neg_loss += loss_info['mse_neg_true_examples']
                intern_step = i
                self.next_step = False

            self.steps += 1

            epoch_loss = epoch_loss / intern_step
            mse_neg_loss = mse_neg_loss / intern_step
            log.info("Epoch {}: Mean epoch loss mse is {}".format(num_epoch, epoch_loss))
            log.info("MSE value for negative samples: {}".format(mse_neg_loss))
            loss_info['mse_neg_true_examples'] = mse_neg_loss
            # # log.info loss every x steps
            # if not self.steps % self.eval_every_n_steps or self.steps == 1:
            #     log_step = int(self.steps / self.eval_every_n_steps)
            #     print("logging step: ", log_step)
            #     wandb.log(loss_info, step=log_step)

            if not (num_epoch + 1) % self.eval_every_n_epochs:

                wandb.log(
                    {
                        "epoch_loss": epoch_loss,
                        "test_loss": avrg_test_mse
                    }
                )

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        log.info("Training done!")

    def train_vision_agent(self):
        epoch_loss = 0
        mse_neg_loss = 0

        # epoch training
        for data in self.train_dataloader:
            bp_imgs, inhand_imgs, obs, action, mask = data

            bp_imgs = bp_imgs.to(self.device)
            inhand_imgs = inhand_imgs.to(self.device)

            obs = self.scaler.scale_input(obs)
            action = self.scaler.scale_output(action)

            state = (bp_imgs, inhand_imgs, obs)

            state_embedding = self.model.get_embedding(state)

            batch_loss, loss_info = self.train_step(state_embedding, action)  # TODO get mean of loss/grad info
            batch_loss = batch_loss.detach().cpu().numpy()

            wandb.log({"train_loss": batch_loss})
            wandb.log({"mse_neg_loss": loss_info['mse_neg_true_examples']})

            epoch_loss += batch_loss
            mse_neg_loss += loss_info['mse_neg_true_examples']
            self.next_step = False

        self.steps += 1

    def train_step(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None):
        # move state to the chosen devices
        if goal is not None:
            goal = self.scaler.scale_input(goal)

        # # scale data if necessarry, otherwise the scaler will return unchanged values
        # state = self.scaler.scale_input(state)
        # action = self.scaler.scale_output(action)

        # Generate N negatives, one for each element in the batch B with dimensions D: (B, N, D).
        # use the sampler class chosen in the config

        if isinstance(self.sampler, LangevinMCMCSampler):
            negatives = self.sampler.gen_train_samples(
                state.size(0), self.model.model, state, goal, random_start_points=True
            )
        elif isinstance(self.sampler, NoiseSampler):
            negatives = self.sampler.gen_train_samples(state.size(0), self.model.model, state, action, goal, self.steps)
        else:
            negatives = self.sampler.gen_train_samples(state.size(0), self.model.model, state)
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        mse_loss_value = self.mse_loss(negatives, einops.repeat(action, 'b a n -> (b a) neg n', neg=negatives.shape[1]))

        self.model.train()
        # Merge action and negatives: (B, N+1, D).
        # the action action will be located at (B, 0, D) and the negatives at (B, 1:, D)
        actions = torch.cat([action, negatives], dim=1)
        loss, dict_info = self.compute_loss(state, actions, goal)

        # optimize the model based on the loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        # update the learning rate of the model
        self.lr_scheduler.step()

        # use the EMA model variant
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())
        # also log the gradient norm if its non None
        dict_info['mse_neg_true_examples'] = mse_loss_value
        dict_info['model_learning_rate'] = self.lr_scheduler.get_last_lr()[0]

        return loss, dict_info

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None):

        self.model.eval()

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        # scale data if necessarry, otherwise the scaler will return unchanged values
        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        state = state.to(self.device)
        action = action.to(self.device)
        out = self.sampler.infer(state, self.model.model, goal)

        action = einops.rearrange(action, 'b a n -> (b a) n')
        mse = F.mse_loss(out, action, reduction="none").mean()

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        return mse.detach().item()

    def predict(self, state: torch.Tensor, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with state data using the stochastic optimizer instance to generate
        samples and return the best sample with the lowest energy

        :param state:           torch.Tensor of observations    [B, O] with O: observation dim
        :return                 torch.Tensor with best samples in shape [B, X] with X: action dim
        """
        # scale data if necessarry, otherwise returns unchanged values

        self.model.eval()
        # state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        # state = self.scaler.scale_input(state)

        bp_image, inhand_image, des_robot_pos = state

        bp_image = torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0).unsqueeze(0)
        inhand_image = torch.from_numpy(inhand_image).to(self.device).float().unsqueeze(0).unsqueeze(0)
        des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0).unsqueeze(0)

        des_robot_pos = self.scaler.scale_input(des_robot_pos)

        state = (bp_image, inhand_image, des_robot_pos)

        state = self.model.get_embedding(state)

        if not self.goal_conditioning:
            goal = None

        if goal is not None:
            goal = self.scaler.scale_input(goal)
            goal = goal.to(self.device)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        # if we use Langevin MCMC we still need the gradient therefore model.eval() is not called
        if self.sampler == "DerivativeFreeOptimizer":
            self.model.eval()
            out = self.sampler.infer(state, self.model.model, goal)
        else:
            out = self.sampler.infer(state, self.model.model, goal)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        out = self.scaler.inverse_scale_output(out)
        return out.detach().cpu().numpy()

    def compute_loss(self, state: torch.Tensor, actions: torch.Tensor, goal: Optional[torch.Tensor] = None):
        # next we compute the chosen loss function
        # next we compute the chosen loss function
        state = einops.rearrange(state, 'b a n -> (b a) n')
        if goal is not None:
            goal = einops.rearrange(goal, 'b a n -> (b a) n')

        if self.use_grad_norm:
            _, grad_norm, _ = self.sampler.compute_gradient(self.model.model, state, actions, goal, False)
            grad_norm_loss = compute_gradient_loss(grad_norm)

        if self.loss_type == "info_nce":
            info_nce_loss, loss_dict = compute_info_nce_loss(
                ebm=self.model.model,
                state=state, actions=actions,
                device=self.device,
                avrg_e_regularization=self.avrg_e_regularization,
                goal=goal
            )
            # add inference loss together with the gradient loss if necessary
            if self.use_grad_norm:
                loss = info_nce_loss + self.grad_norm_factor * grad_norm_loss
                loss_dict['grad_loss'] = grad_norm_loss
                loss_dict['overall_grad_norms_avg'] = torch.mean(grad_norm)
            else:
                loss = info_nce_loss

        elif self.loss_type == "cd":
            loss = contrastive_divergence(
                ebm=self.model.model,
                state=state,
                actions=actions,
                avrg_e_regularization=self.avrg_e_regularization
            )

            if self.use_grad_norm:
                loss += self.grad_norm_factor * grad_norm

        elif self.loss_type == "cd_kl":
            loss = contrastive_divergence_kl(
                ebm=self.model.model,
                state=state, actions=actions,
                avrg_e_regularization=self.avrg_e_regularization,
                kl_loss_factor=self.kl_loss_factor
            )

        elif self.loss_type == 'cd_entropy':
            loss = contrastive_divergence_entropy_approx(
                ebm=self.model.model,
                state=state, actions=actions,
            )
        elif self.loss_type == 'autoregressive_info_nce':
            loss = compute_autoregressive_info_nce_loss(
                ebm=self.model.model,
                state=state, actions=actions,
                device=self.device,
                avrg_e_regularization=self.avrg_e_regularization
            )
        else:
            raise ValueError("Not a correct loss type! Please chose another one!")

        return loss, loss_dict

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        if sv_name is None:
            torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        torch.save(self.model.state_dict(), os.path.join(store_path, "non_ema_model_state_dict.pth"))
        
    def reset(self):
        pass