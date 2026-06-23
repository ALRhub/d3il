import os
import logging
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import wandb

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class BC_GMM_Agent(BaseAgent):
    """Behaviour cloning with a Gaussian-Mixture-Model action head.

    The model (``agents.models.gmm.bc_gmm.BC_GMM``) maps an observation to a
    ``torch.distributions`` GMM over actions. Training maximises the GMM
    log-likelihood of the demonstrated actions; at inference we sample from the
    mixture (so the multi-modality of the demonstrations — which D3IL's entropy
    metric rewards — is preserved).
    """

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
    ):
        super().__init__(model=model, trainset=trainset, valset=valset,
                         train_batch_size=train_batch_size, val_batch_size=val_batch_size,
                         num_workers=num_workers, device=device, epoch=epoch,
                         scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # The GMM model returns a torch Distribution, which nn.DataParallel
        # cannot gather, so we do NOT wrap it for multi-GPU. ``visual_input``
        # is not defined on the bare model — set it for BaseAgent.train().
        self.model.visual_input = False

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.get_params()
        )

        self.eval_model_name = "eval_best_bc_gmm.pth"
        self.last_model_name = "last_bc_gmm.pth"

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

    def train_agent(self):
        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch + 1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:
                    state, action, mask = data
                    state = self.scaler.scale_input(state)
                    action = self.scaler.scale_output(action)
                    mean_mse = self.evaluate(state, action)
                    test_mse.append(mean_mse)
                    wandb.log({"test_loss": mean_mse})

                avrg_test_mse = sum(test_mse) / len(test_mse)
                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)
                    wandb.log({"best_model_epochs": num_epoch})
                    log.info('New best test loss. Stored weights have been updated!')

                wandb.log({"mean_test_loss": avrg_test_mse})

            train_loss = []
            for data in self.train_dataloader:
                state, action, mask = data
                state = self.scaler.scale_input(state)
                action = self.scaler.scale_output(action)
                batch_loss = self.train_step(state, action)
                train_loss.append(batch_loss)
                wandb.log({"loss": batch_loss})

            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        log.info("Training done!")

    def train_step(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """Single GMM negative-log-likelihood training step (data already scaled)."""
        self.model.train()
        dist = self.model(state)
        loss = -dist.log_prob(action).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """MSE between the GMM mean prediction and the target (data already scaled)."""
        self.model.eval()
        dist = self.model(state)
        pred = dist.mean
        return F.mse_loss(pred, action).item()

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:
        self.model.eval()
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(0)
        state = self.scaler.scale_input(state)

        dist = self.model(state)
        out = dist.sample()
        out = out.clamp_(self.min_action, self.max_action)

        model_pred = self.scaler.inverse_scale_output(out)
        return model_pred.detach().cpu().numpy()[0]

    def reset(self):
        pass
