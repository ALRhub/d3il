"""D3IL agent wrapping the two-phase PyTorch EBM policy.

PyTorch counterpart of ``agents/ebm_jax_agent.py``: same two-phase schedule
(phase 1 generative flow-matching, phase 2 contrastive EBM) and the same
``predict(...) -> numpy_action`` interface for the D3IL simulators, but built on
torch.nn / torch.optim instead of jax/flax. One class handles both the
state-based and image-based (vision) benchmarks (modality auto-detected from the
phase-2 config).
"""
from __future__ import annotations

import logging
import os
from collections import deque

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

from agents.base_agent import BaseAgent
from agents.models.ebm_torch import EBMPolicy

log = logging.getLogger(__name__)


def _make_optimizer(params, opt_cfg):
    opt_cfg = dict(opt_cfg)
    name = str(opt_cfg.pop("name", "adam")).lower()
    lr = opt_cfg.pop("lr", opt_cfg.pop("learning_rate", 1e-3))
    weight_decay = opt_cfg.pop("weight_decay", 0.0)
    amsgrad = bool(opt_cfg.pop("amsgrad", False))
    # AdaBelief is not in torch core; the JAX port already substituted AdamW, so
    # we match that here for parity.
    if name in ("adamw", "adabelief"):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    return torch.optim.Adam(params, lr=lr, amsgrad=amsgrad)


class EBM_Torch_Agent(BaseAgent):
    """Two-phase EBM policy agent for the D3IL benchmark (state or vision)."""

    # The phase-2 InfoNCE loss with in-batch-inverted negatives feeds the energy
    # net O(B^2) candidates; cap the batch so large dataset-default batches don't
    # OOM (the dataset default 1024 is tuned for the cheap MLP baselines).
    _MAX_EBM_BATCH = 64

    def __init__(
        self,
        phase1: DictConfig,
        phase2: DictConfig,
        trainset: DictConfig,
        valset: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        device: str,
        epoch: int,
        scale_data: bool,
        window_size: int,
        phase1_epochs: int = None,
        phase2_epochs: int = None,
        eval_every_n_epochs: int = 50,
    ):
        from agents.utils.scaler import Scaler

        self.device = device
        self.working_dir = os.getcwd()
        self.scale_data = scale_data
        self.window_size = window_size
        self.eval_every_n_epochs = eval_every_n_epochs
        self.epoch = epoch
        self.phase1_epochs = phase1_epochs if phase1_epochs is not None else max(1, epoch // 2)
        self.phase2_epochs = (phase2_epochs if phase2_epochs is not None
                              else max(1, epoch - self.phase1_epochs))

        self.trainset = hydra.utils.instantiate(trainset)
        self.valset = hydra.utils.instantiate(valset)

        self._phase1_config = hydra.utils.instantiate(phase1)
        self._phase2_config = hydra.utils.instantiate(phase2)

        # window_size==1: a transformer EBM over length-1 sequences is pointless.
        if (window_size <= 1 and self._phase2_config.ebm is not None
                and getattr(self._phase2_config.ebm, "arch", "mlp") == "transformer"):
            self._phase2_config.ebm.arch = "mlp"
            log.info("window_size<=1: using MLP EBM (transformer not needed)")

        # Cap the EBM batch for the O(B^2) in-batch-inverted InfoNCE loss.
        ebm_cfg = self._phase2_config.ebm
        if (ebm_cfg is not None and getattr(ebm_cfg, "loss_type", "ce") == "infonce"
                and getattr(ebm_cfg, "infonce_use_inbatch_inverted", False)
                and train_batch_size > self._MAX_EBM_BATCH):
            log.warning("EBM(torch): capping batch_size %d -> %d for in-batch-inverted InfoNCE.",
                        train_batch_size, self._MAX_EBM_BATCH)
            train_batch_size = self._MAX_EBM_BATCH
            val_batch_size = min(val_batch_size, self._MAX_EBM_BATCH)

        dl_kwargs = dict(num_workers=num_workers, pin_memory=True)
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 4
        self.train_dataloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=train_batch_size, shuffle=True, **dl_kwargs)
        self.test_dataloader = torch.utils.data.DataLoader(
            self.valset, batch_size=val_batch_size, shuffle=False, **dl_kwargs)

        self.scaler = Scaler(self.trainset.get_all_observations(),
                             self.trainset.get_all_actions(), scale_data, device)

        # --- models ---
        self.phase1_model = EBMPolicy(self._phase1_config).to(device)
        self.phase2_model = EBMPolicy(self._phase2_config).to(device)

        self.opt1 = _make_optimizer(self.phase1_model.optim_parameters(), self._phase1_config.optimizer_cfg)
        self.opt2 = _make_optimizer(self.phase2_model.optim_parameters(), self._phase2_config.ebm.optimizer_cfg)
        self._ema1_decay = dict(self._phase1_config.ema).get("decay", 0.999)
        self._ema2_decay = self._phase2_config.ebm.ema_decay

        self.eval_model_name = "eval_best_ebm_torch.pth"
        self.last_model_name = "last_ebm_torch.pth"

        self.vision = bool(self._phase2_config.use_static or self._phase2_config.use_gripper)
        self.visual_input = self.vision
        self.obs_length = self._phase2_config.obs_length
        self.pos_context = deque(maxlen=self.obs_length)
        self.static_context = deque(maxlen=self.obs_length)
        self.gripper_context = deque(maxlen=self.obs_length)

        self._step = 0
        self._epoch_count = 0
        self._injected = False
        self._best_test_loss = float("inf")

        n_params = sum(p.numel() for p in self.phase2_model.optim_parameters())
        log.info("EBM-torch (phase2) trainable parameters: %d (vision=%s)", n_params, self.vision)
        try:
            wandb.log({"model parameters": n_params})
        except Exception:
            pass

    # -- batch unpacking ------------------------------------------------------
    def _unpack_batch(self, data):
        """Return (static, gripper, pos, action) torch tensors on device."""
        if self.vision:
            bp, inhand, obs, action, *_ = data
            static = bp.float().to(self.device)
            gripper = inhand.float().to(self.device)
            pos = self.scaler.scale_input(obs.float().to(self.device))
            action = self.scaler.scale_output(action.float().to(self.device))
            return static, gripper, pos, action
        state, action, *_ = data
        pos = self.scaler.scale_input(state.float().to(self.device))
        action = self.scaler.scale_output(action.float().to(self.device))
        return None, None, pos, action

    # -- single steps ---------------------------------------------------------
    def _phase1_step(self, action):
        self.opt1.zero_grad(set_to_none=True)
        loss, _ = self.phase1_model.compute_phase1_loss(action)
        loss.backward()
        self.opt1.step()
        self.phase1_model.update_policy_ema(self._ema1_decay)
        return loss.item()

    def _phase2_step(self, static, gripper, pos, action):
        self.opt2.zero_grad(set_to_none=True)
        loss, _ = self.phase2_model.compute_phase2_loss(static, gripper, action, pos, step=self._step)
        loss.backward()
        self.opt2.step()
        self.phase2_model.update_ebm_ema(self._ema2_decay)
        return loss.item()

    @torch.no_grad()
    def _val_loss(self, static, gripper, pos, action):
        loss, _ = self.phase2_model.compute_phase2_loss(static, gripper, action, pos, step=self._step)
        return loss.item()

    # -- per-epoch loops ------------------------------------------------------
    def _phase1_epoch(self):
        self.phase1_model.train()
        losses = []
        for data in self.train_dataloader:
            _, _, _, action = self._unpack_batch(data)
            losses.append(self._phase1_step(action))
            self._step += 1
            wandb.log({"phase1/loss": losses[-1]})
        return sum(losses) / max(len(losses), 1)

    def _phase2_epoch(self):
        self.phase2_model.train()
        losses = []
        for data in self.train_dataloader:
            static, gripper, pos, action = self._unpack_batch(data)
            losses.append(self._phase2_step(static, gripper, pos, action))
            self._step += 1
            wandb.log({"phase2/loss": losses[-1]})
        return sum(losses) / max(len(losses), 1)

    def _inject_phase1(self):
        if not self._injected:
            log.info("Injecting trained phase-1 generative model into phase-2 EBM")
            self.phase2_model.load_generative_from_model(self.phase1_model)
            self._injected = True

    def _eval_and_store(self, num_epoch):
        self.phase2_model.eval()
        test_loss = []
        for data in self.test_dataloader:
            static, gripper, pos, action = self._unpack_batch(data)
            test_loss.append(self._val_loss(static, gripper, pos, action))
        avrg = sum(test_loss) / max(len(test_loss), 1)
        log.info("Phase2 epoch %d: mean test loss %.6f", num_epoch, avrg)
        if avrg < self._best_test_loss:
            self._best_test_loss = avrg
            self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)
            wandb.log({"best_test_loss": avrg, "best_model_epochs": num_epoch})
            log.info("New best test loss. Stored weights have been updated!")

    # -- training: state path (run.py) ----------------------------------------
    def train_agent(self):
        log.info("=== Phase 1: training generative flow-matching model ===")
        for num_epoch in tqdm(range(self.phase1_epochs)):
            loss = self._phase1_epoch()
            log.info("Phase1 epoch %d: mean loss %.6f", num_epoch, loss)
        self._inject_phase1()
        log.info("=== Phase 2: training EBM ===")
        for num_epoch in tqdm(range(self.phase2_epochs)):
            if not (num_epoch + 1) % self.eval_every_n_epochs:
                self._eval_and_store(num_epoch)
            loss = self._phase2_epoch()
            log.info("Phase2 epoch %d: mean train loss %.6f", num_epoch, loss)
        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        if not os.path.exists(os.path.join(self.working_dir, self.eval_model_name)):
            self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)
        log.info("Training done!")

    # -- training: vision path (run_vision.py, one epoch per call) -------------
    def train_vision_agent(self):
        if self._epoch_count < self.phase1_epochs:
            loss = self._phase1_epoch()
            log.info("Phase1 epoch %d: mean loss %.6f", self._epoch_count, loss)
        else:
            self._inject_phase1()
            loss = self._phase2_epoch()
            log.info("Phase2 epoch %d: mean loss %.6f", self._epoch_count - self.phase1_epochs, loss)
        self._epoch_count += 1

    # -- BaseAgent abstract methods -------------------------------------------
    def train_step(self, state: torch.Tensor, action: torch.Tensor):
        pos = self.scaler.scale_input(state.float().to(self.device))
        action = self.scaler.scale_output(action.float().to(self.device))
        loss = self._phase2_step(None, None, pos, action)
        self._step += 1
        return loss

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        pos = self.scaler.scale_input(state.float().to(self.device))
        action = self.scaler.scale_output(action.float().to(self.device))
        return self._val_loss(None, None, pos, action)

    # -- inference ------------------------------------------------------------
    def reset(self):
        self.pos_context.clear()
        self.static_context.clear()
        self.gripper_context.clear()

    @staticmethod
    def _window(ctx, length):
        items = list(ctx)
        while len(items) < length:
            items.insert(0, items[0])
        return torch.stack(items, dim=1)

    @torch.no_grad()
    def predict(self, state, goal=None, if_vision=False) -> np.ndarray:
        self.phase2_model.eval()
        if self.vision or if_vision:
            bp_image, inhand_image, des_robot_pos = state
            bp = torch.from_numpy(np.asarray(bp_image)).float().to(self.device).unsqueeze(0)
            inhand = torch.from_numpy(np.asarray(inhand_image)).float().to(self.device).unsqueeze(0)
            pos = torch.from_numpy(np.asarray(des_robot_pos)).float().to(self.device).unsqueeze(0)
            pos = self.scaler.scale_input(pos)
            self.static_context.append(bp)
            self.gripper_context.append(inhand)
            self.pos_context.append(pos)
            traj = self.phase2_model.inference(
                batch_size=1,
                static_obs=self._window(self.static_context, self.obs_length),
                gripper_obs=self._window(self.gripper_context, self.obs_length),
                pos=self._window(self.pos_context, self.obs_length))
        else:
            t_state = torch.from_numpy(np.asarray(state)).float().to(self.device).unsqueeze(0)
            t_state = self.scaler.scale_input(t_state)
            self.pos_context.append(t_state)
            traj = self.phase2_model.inference(
                batch_size=1, pos=self._window(self.pos_context, self.obs_length))

        scaled_action = traj[0, -1].unsqueeze(0)
        scaled_action = torch.clamp(
            scaled_action,
            self.scaler.y_bounds_tensor[0, :].float() * 1.1,
            self.scaler.y_bounds_tensor[1, :].float() * 1.1)
        action = self.scaler.inverse_scale_output(scaled_action)
        return action.detach().cpu().numpy()

    # -- persistence ----------------------------------------------------------
    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        path = os.path.join(store_path, sv_name if sv_name else "ebm_torch_state.pth")
        torch.save(self.phase2_model.state_dict(), path)

    def load_pretrained_model(self, weights_path: str, sv_name=None, **kwargs) -> None:
        path = os.path.join(weights_path, sv_name if sv_name else "ebm_torch_state.pth")
        state = torch.load(path, map_location=self.device)
        self.phase2_model.load_state_dict(state)
        self._injected = True
        log.info("Loaded pre-trained EBM (torch) parameters")

    # -- pickling support for spawn-based multiprocess evaluation -------------
    # The avoiding simulator spawns eval workers; rather than IPC-share the large
    # CUDA model, ship a CPU state_dict and rebuild phase2_model in each worker
    # (only inference is needed there).
    def __getstate__(self):
        import io
        import copy as _copy
        state = self.__dict__.copy()
        # Serialise the model to a single byte blob. ForkingPickler reduces every
        # individual tensor to a shared-memory file descriptor, so a few-hundred-
        # tensor state_dict overflows spawn's fd-passing ("bad value(s) in
        # fds_to_keep"); one opaque blob carries zero fds.
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in self.phase2_model.state_dict().items()}, buf)
        state["_phase2_blob"] = buf.getvalue()
        # Same reason for the scaler: store its tensors as numpy (inline, no fd).
        sc = _copy.copy(self.scaler)
        for k, v in vars(sc).items():
            if torch.is_tensor(v):
                setattr(sc, k, v.detach().cpu().numpy())
        state["scaler"] = sc
        state["_scaler_tensor_keys"] = [k for k, v in vars(self.scaler).items() if torch.is_tensor(v)]
        for k in ("phase1_model", "phase2_model", "opt1", "opt2",
                  "train_dataloader", "test_dataloader", "trainset", "valset"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        import io
        blob = state.pop("_phase2_blob", None)
        tensor_keys = state.pop("_scaler_tensor_keys", [])
        self.__dict__.update(state)
        # Restore the scaler's numpy arrays to device tensors.
        for k in tensor_keys:
            arr = getattr(self.scaler, k)
            setattr(self.scaler, k, torch.as_tensor(arr).to(self.device))
        self.phase1_model = None
        self.phase2_model = EBMPolicy(self._phase2_config).to(self.device)
        if blob is not None:
            sd = torch.load(io.BytesIO(blob), map_location=self.device)
            self.phase2_model.load_state_dict(sd)
        self.phase2_model.eval()
