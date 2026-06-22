"""D3IL agent wrapping the two-phase JAX/Flax-NNX EBM policy (ported from vapo).

The energy-based-model itself (``agents.models.ebm_jax``) is unchanged from
vapo — it is jax + flax.nnx + ml-networks. This agent is the thin D3IL harness
that:

* builds the phase-1 (generative flow-matching) and phase-2 (EBM contrastive)
  ``EBMPolicy`` models from Hydra config,
* trains phase 1 then phase 2, injecting the trained phase-1 generative model
  into phase 2 in-memory,
* exposes ``predict(...) -> numpy_action`` for the D3IL simulators, so the EBM
  policy can be benchmarked against the other baselines.

One class handles BOTH the state-based and image-based (vision) benchmarks; the
modality is auto-detected from the config (``use_static`` / ``use_gripper`` set
⇒ vision). For state tasks the low-dim observation drives the proprioceptive
("pos") encoder; for vision tasks the two cameras drive the image encoders and
the robot state drives the pos encoder — mirroring the vapo EBM policy.
"""
from __future__ import annotations

import logging
import os
import pickle
from collections import deque

# JAX must not pre-grab the whole GPU (D3IL/torch shares the device, and the
# multiprocess simulators may spawn several evaluators). Set before jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

import jax
import jax.numpy as jnp

from agents.base_agent import BaseAgent
from agents.models.ebm_jax import EBMPolicy

log = logging.getLogger(__name__)


def _to_jnp(t) -> jnp.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return jnp.asarray(np.asarray(t).astype(np.float32))


class EBM_JAX_Agent(BaseAgent):
    """Two-phase EBM policy agent for the D3IL benchmark (state or vision)."""

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
        # NOTE: BaseAgent.__init__ is intentionally NOT called — it instantiates
        # a torch ``model`` and moves it to the device, which does not apply to
        # the jax model. We replicate the dataset/scaler setup it provides.
        from agents.utils.scaler import Scaler

        self.device = device
        self.working_dir = os.getcwd()
        self.scale_data = scale_data
        self.window_size = window_size
        self.eval_every_n_epochs = eval_every_n_epochs
        self.epoch = epoch
        # The two phases share the total epoch budget (half/half by default).
        # For the vision path (run_vision.py drives one epoch per call) this
        # keeps phase 2 reachable: the first ``phase1_epochs`` outer epochs run
        # phase 1, the remaining ``phase2_epochs`` run phase 2.
        self.phase1_epochs = phase1_epochs if phase1_epochs is not None else max(1, epoch // 2)
        self.phase2_epochs = (phase2_epochs if phase2_epochs is not None
                              else max(1, epoch - self.phase1_epochs))

        self.trainset = hydra.utils.instantiate(trainset)
        self.valset = hydra.utils.instantiate(valset)

        # pin_memory is unnecessary: batches are copied to host for JAX anyway.
        dl_kwargs = dict(num_workers=num_workers, pin_memory=False)
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 10
        self.train_dataloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=train_batch_size, shuffle=True, **dl_kwargs,
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.valset, batch_size=val_batch_size, shuffle=False, **dl_kwargs,
        )

        self.scaler = Scaler(
            self.trainset.get_all_observations(),
            self.trainset.get_all_actions(),
            scale_data, device,
        )

        # Keep the instantiated config dataclasses so the models can be rebuilt
        # after pickling (spawn-based multiprocess evaluation).
        self._phase1_config = hydra.utils.instantiate(phase1)
        self._phase2_config = hydra.utils.instantiate(phase2)
        self._build_models()

        self.eval_model_name = "eval_best_ebm_jax.pkl"
        self.last_model_name = "last_ebm_jax.pkl"

        # Vision is implied by image encoders being configured in phase 2.
        self.vision = bool(self._phase2_config.use_static or self._phase2_config.use_gripper)
        self.visual_input = self.vision

        self._step = 0
        self._epoch_count = 0
        self._injected = False
        self._best_test_loss = 1e10

        self.obs_length = self._phase2_config.obs_length
        self.policy_length = self._phase2_config.policy_length
        # Rolling observation windows for inference.
        self.pos_context = deque(maxlen=self.obs_length)
        self.static_context = deque(maxlen=self.obs_length)
        self.gripper_context = deque(maxlen=self.obs_length)

        n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(self.phase2_model._params))
        log.info("EBM (phase2) trainable parameters: {} (vision={})".format(n_params, self.vision))
        try:
            wandb.log({"model parameters": n_params})
        except Exception:
            pass

    # -- model construction ---------------------------------------------------

    def _build_models(self):
        self.phase1_model = EBMPolicy(self._phase1_config)
        self.phase2_model = EBMPolicy(self._phase2_config)

    # -- batch unpacking ------------------------------------------------------

    def _unpack_batch(self, data):
        """Return (static_j, gripper_j, pos_j, action_j) for a train/val batch.

        State batch: ``(state, action, mask)``.
        Vision batch: ``(bp_imgs, inhand_imgs, obs, action, mask)`` with images
        already in [0, 1], shape ``(B, T, C, H, W)``.
        """
        if self.vision:
            bp, inhand, obs, action, *_ = data
            static_j = _to_jnp(bp)
            gripper_j = _to_jnp(inhand)
            pos_j = _to_jnp(self.scaler.scale_input(obs))
            action_j = _to_jnp(self.scaler.scale_output(action))
            return static_j, gripper_j, pos_j, action_j
        state, action, *_ = data
        pos_j = _to_jnp(self.scaler.scale_input(state))
        action_j = _to_jnp(self.scaler.scale_output(action))
        return None, None, pos_j, action_j

    # -- single jitted steps --------------------------------------------------

    def _phase1_step(self, action_j):
        m = self.phase1_model
        key = jax.random.PRNGKey(self._step)
        m._params, m._opt_state, m.ema.ema_state, loss, ld = m._jit_step(
            m._params, m._opt_state, m.ema.ema_state,
            None, None, action_j, None, None, None, None, key, None,
        )
        return float(loss)

    def _phase2_step(self, static_j, gripper_j, pos_j, action_j):
        m = self.phase2_model
        key = jax.random.PRNGKey(self._step + 1)
        step = jnp.int32(self._step)
        m._params, m._opt_state, m.ebm_ema.ema_state, loss, ld = m._jit_step(
            m._params, m._opt_state, m.ebm_ema.ema_state,
            static_j, gripper_j, action_j, pos_j, None, None, None, key, None, step,
        )
        return float(loss)

    # -- per-epoch loops (shared by state and vision) -------------------------

    def _phase1_epoch(self):
        losses = []
        for data in self.train_dataloader:
            _, _, _, action_j = self._unpack_batch(data)
            losses.append(self._phase1_step(action_j))
            self._step += 1
            wandb.log({"phase1/loss": losses[-1]})
        return sum(losses) / max(len(losses), 1)

    def _phase2_epoch(self):
        losses = []
        for data in self.train_dataloader:
            static_j, gripper_j, pos_j, action_j = self._unpack_batch(data)
            losses.append(self._phase2_step(static_j, gripper_j, pos_j, action_j))
            self._step += 1
            wandb.log({"phase2/loss": losses[-1]})
        return sum(losses) / max(len(losses), 1)

    def _inject_phase1(self):
        if not self._injected:
            log.info("Injecting trained phase-1 generative model into phase-2 EBM")
            self.phase2_model.load_generative_from_model(self.phase1_model)
            self._injected = True

    # -- training: state path (run.py calls train_agent once) -----------------

    def train_agent(self):
        log.info("=== Phase 1: training generative flow-matching model ===")
        for num_epoch in tqdm(range(self.phase1_epochs)):
            loss = self._phase1_epoch()
            log.info("Phase1 epoch {}: mean loss {:.6f}".format(num_epoch, loss))

        self._inject_phase1()

        log.info("=== Phase 2: training EBM ===")
        for num_epoch in tqdm(range(self.phase2_epochs)):
            if not (num_epoch + 1) % self.eval_every_n_epochs:
                self._eval_and_store(num_epoch)
            loss = self._phase2_epoch()
            log.info("Phase2 epoch {}: mean train loss {:.6f}".format(num_epoch, loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        if not os.path.exists(os.path.join(self.working_dir, self.eval_model_name)):
            self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)
        log.info("Training done!")

    def _eval_and_store(self, num_epoch):
        test_loss = []
        for data in self.test_dataloader:
            static_j, gripper_j, pos_j, action_j = self._unpack_batch(data)
            test_loss.append(self._val_loss(static_j, gripper_j, pos_j, action_j))
        avrg = sum(test_loss) / max(len(test_loss), 1)
        log.info("Phase2 epoch {}: mean test loss {:.6f}".format(num_epoch, avrg))
        if avrg < self._best_test_loss:
            self._best_test_loss = avrg
            self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)
            wandb.log({"best_test_loss": avrg, "best_model_epochs": num_epoch})
            log.info("New best test loss. Stored weights have been updated!")

    # -- training: vision path (run_vision.py calls once per epoch) ------------

    def train_vision_agent(self):
        """One epoch of training, scheduled across the two phases.

        ``run_vision.py`` calls this once per outer epoch and evaluates with the
        simulator in between. The first ``phase1_epochs`` calls train the
        generative model; the rest train the EBM (after a one-time injection).
        """
        if self._epoch_count < self.phase1_epochs:
            loss = self._phase1_epoch()
            log.info("Phase1 epoch {}: mean loss {:.6f}".format(self._epoch_count, loss))
        else:
            self._inject_phase1()
            loss = self._phase2_epoch()
            log.info("Phase2 epoch {}: mean loss {:.6f}".format(
                self._epoch_count - self.phase1_epochs, loss))
        self._epoch_count += 1

    # -- abstract-method implementations --------------------------------------

    def train_step(self, state: torch.Tensor, action: torch.Tensor):
        pos_j = _to_jnp(self.scaler.scale_input(state))
        action_j = _to_jnp(self.scaler.scale_output(action))
        loss = self._phase2_step(None, None, pos_j, action_j)
        self._step += 1
        return loss

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        pos_j = _to_jnp(self.scaler.scale_input(state))
        action_j = _to_jnp(self.scaler.scale_output(action))
        return self._val_loss(None, None, pos_j, action_j)

    def _val_loss(self, static_j, gripper_j, pos_j, action_j):
        m = self.phase2_model
        key = jax.random.PRNGKey(self._step + 7)
        step = jnp.int32(self._step)
        loss_dict = m._jit_val(
            m._params, m.ema.ema_state, m.ebm_ema.ema_state,
            static_j, gripper_j, action_j, pos_j, None, None, None, key, None, step,
        )
        return float(loss_dict["loss"])

    # -- inference ------------------------------------------------------------

    def reset(self):
        self.pos_context.clear()
        self.static_context.clear()
        self.gripper_context.clear()

    @staticmethod
    def _window(ctx, length):
        """Stack a deque into (1, length, ...), left-padding with the oldest."""
        items = list(ctx)
        while len(items) < length:
            items.insert(0, items[0])
        return torch.stack(items, dim=1)

    @torch.no_grad()
    def predict(self, state, goal=None, if_vision=False) -> np.ndarray:
        if self.vision or if_vision:
            bp_image, inhand_image, des_robot_pos = state
            bp = torch.from_numpy(np.asarray(bp_image)).float().to(self.device).unsqueeze(0)
            inhand = torch.from_numpy(np.asarray(inhand_image)).float().to(self.device).unsqueeze(0)
            pos = torch.from_numpy(np.asarray(des_robot_pos)).float().to(self.device).unsqueeze(0)
            pos = self.scaler.scale_input(pos)
            self.static_context.append(bp)
            self.gripper_context.append(inhand)
            self.pos_context.append(pos)
            static_j = _to_jnp(self._window(self.static_context, self.obs_length))
            gripper_j = _to_jnp(self._window(self.gripper_context, self.obs_length))
            pos_j = _to_jnp(self._window(self.pos_context, self.obs_length))
            traj = self.phase2_model.inference(
                batch_size=1, static_obs=static_j, gripper_obs=gripper_j, pos=pos_j)
        else:
            t_state = torch.from_numpy(np.asarray(state)).float().to(self.device).unsqueeze(0)
            t_state = self.scaler.scale_input(t_state)
            self.pos_context.append(t_state)
            pos_j = _to_jnp(self._window(self.pos_context, self.obs_length))
            traj = self.phase2_model.inference(batch_size=1, pos=pos_j)

        traj = np.asarray(traj)
        # The action aligned with the most recent observation is the last step.
        scaled_action = torch.from_numpy(np.array(traj[0, -1])).float().to(self.device).unsqueeze(0)
        scaled_action = torch.clamp(
            scaled_action,
            self.scaler.y_bounds_tensor[0, :].float() * 1.1,
            self.scaler.y_bounds_tensor[1, :].float() * 1.1,
        )
        action = self.scaler.inverse_scale_output(scaled_action)     # (1, action_dim)
        return action.detach().cpu().numpy()

    # -- persistence ----------------------------------------------------------

    @staticmethod
    def _leaves(pytree):
        return [np.asarray(x) for x in jax.tree.leaves(pytree)]

    @staticmethod
    def _restore(pytree_ref, leaves):
        treedef = jax.tree.structure(pytree_ref)
        return jax.tree.unflatten(treedef, [jnp.asarray(x) for x in leaves])

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        m = self.phase2_model
        data = {
            "phase2_params": self._leaves(m._params),
            "phase1_ema": self._leaves(m.ema.ema_state),
            "ebm_ema": self._leaves(m.ebm_ema.ema_state),
        }
        path = os.path.join(store_path, sv_name if sv_name else "ebm_jax_state.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_pretrained_model(self, weights_path: str, sv_name=None, **kwargs) -> None:
        path = os.path.join(weights_path, sv_name if sv_name else "ebm_jax_state.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = self.phase2_model
        m._params = self._restore(m._params, data["phase2_params"])
        m._set_jax_state(m._params)
        m.ema.ema_state = self._restore(m.ema.ema_state, data["phase1_ema"])
        m.ebm_ema.ema_state = self._restore(m.ebm_ema.ema_state, data["ebm_ema"])
        log.info("Loaded pre-trained EBM (jax) parameters")

    # -- pickling support for spawn-based multiprocess evaluation -------------

    def __getstate__(self):
        state = self.__dict__.copy()
        m = self.phase2_model
        state["_saved_weights"] = {
            "phase2_params": self._leaves(m._params),
            "phase1_ema": self._leaves(m.ema.ema_state),
            "ebm_ema": self._leaves(m.ebm_ema.ema_state),
        }
        state.pop("phase1_model", None)
        state.pop("phase2_model", None)
        state.pop("train_dataloader", None)
        state.pop("test_dataloader", None)
        return state

    def __setstate__(self, state):
        saved = state.pop("_saved_weights", None)
        self.__dict__.update(state)
        self.phase2_model = EBMPolicy(self._phase2_config)
        self.phase1_model = None
        if saved is not None:
            m = self.phase2_model
            m._params = self._restore(m._params, saved["phase2_params"])
            m._set_jax_state(m._params)
            m.ema.ema_state = self._restore(m.ema.ema_state, saved["phase1_ema"])
            m.ebm_ema.ema_state = self._restore(m.ebm_ema.ema_state, saved["ebm_ema"])
        self.pos_context = deque(maxlen=self.obs_length)
        self.static_context = deque(maxlen=self.obs_length)
        self.gripper_context = deque(maxlen=self.obs_length)
