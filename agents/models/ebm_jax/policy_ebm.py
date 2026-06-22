"""Two-phase EBM policy in JAX / Flax NNX.

The ``EBMPolicy`` class and all of its helpers are ported verbatim from
``vapo.policy.policy`` (the ``EBMPolicy`` block, lines for the two-phase
energy-based-model policy). The jax / flax.nnx / ml-networks implementation is
kept exactly as in vapo so the model math is identical.

Two harness-level changes were necessary to run inside D3IL (a pure-PyTorch
benchmark) instead of vapo's PyTorch-Lightning trainer:

1.  ``JaxLightningModule`` (a ``pl.LightningModule``) is replaced by the tiny
    ``JaxModuleBase`` below. D3IL's ``EBM_JAX_Agent`` drives the jitted
    ``_jit_step`` / ``inference`` calls directly, so none of Lightning's
    training/validation/checkpoint hooks are needed; they are dropped here.
    Persistence is handled by the agent via the jax pytrees.

2.  The proprioceptive ("pos") encoder takes ``cfg.pos_input_dim`` features
    (the D3IL state width) instead of being hard-wired to ``action_dim``.

Everything else — the flow-matching phase-1 generative model, the phase-2 EBM
contrastive training (CE / InfoNCE), Langevin negatives, inversion, denoising
and inference — is unchanged from vapo.
"""
from __future__ import annotations

from typing import Any

import einops
import jax
import jax.numpy as jnp
import ml_networks as ml
import ml_networks.jax as mlj
import numpy as np
import optax
from flax import nnx

from .backbone import (
    ConditionalUnet1DPolicy,
    DiT,
    EnergyModel,
    EnergyTransformer,
)
from .config import (
    DiTConfig,
    EBMPolicyConfig,
    UNetConfig,
)
from .ema import NnxEMA
from .fm import (
    ConditionalFlowMatcher,
    DDIMScheduler,
    DDPMScheduler,
    OTConditionalFlowMatcher,
    OTDiffusionCoupler,
    OTStableFlowMatcher,
    StableFlowMatcher,
    diffusion_training_target,
)


# ---------------------------------------------------------------------------
# Minimal non-Lightning base
# ---------------------------------------------------------------------------

class JaxModuleBase:
    """Stand-in for vapo's ``JaxLightningModule`` without PyTorch Lightning.

    Provides only the attributes the EBM model relies on outside of the
    Lightning hooks. The D3IL agent owns the training loop and persistence.
    """

    def __init__(self):
        self._jax_step_count = 0

    # No-op logging shims so any leftover ``self.log*`` calls are harmless.
    def log_dict(self, *args, **kwargs):
        pass

    def step_global_step(self):
        pass


# ---------------------------------------------------------------------------
# Image-encoder helpers (only used when static/gripper image obs are enabled;
# D3IL state tasks do not construct these).
# ---------------------------------------------------------------------------

def _flatten_encoder_cfg(enc_cfg):
    from types import SimpleNamespace
    backbone = enc_cfg.backbone
    fc = enc_cfg.full_connection
    return SimpleNamespace(
        channels=tuple(backbone.channels),
        kernel_sizes=tuple(c.kernel_size for c in backbone.conv_cfgs),
        strides=tuple(c.stride for c in backbone.conv_cfgs),
        activation=backbone.conv_cfgs[0].activation,
        norm=backbone.conv_cfgs[0].norm,
        norm_cfg=backbone.conv_cfgs[0].norm_cfg if hasattr(backbone.conv_cfgs[0], 'norm_cfg') else None,
        use_spatial_softmax=isinstance(fc, ml.SpatialSoftmaxConfig),
        spatial_softmax_cfg=fc if isinstance(fc, ml.SpatialSoftmaxConfig) else None,
    )


def _adaptive_avg_pool_nhwc(x: jax.Array, output_size: tuple[int, int]) -> jax.Array:
    oh, ow = output_size
    if oh == 1 and ow == 1:
        return jnp.mean(x, axis=(1, 2), keepdims=True)
    _, H, W, _ = x.shape
    stride_h = H // oh
    stride_w = W // ow
    kernel_h = H - (oh - 1) * stride_h
    kernel_w = W - (ow - 1) * stride_w
    sum_pool = jax.lax.reduce_window(
        x, 0.0, jax.lax.add,
        window_dimensions=(1, kernel_h, kernel_w, 1),
        window_strides=(1, stride_h, stride_w, 1),
        padding='VALID',
    )
    return sum_pool / (kernel_h * kernel_w)


class _PooledEncoder(nnx.Module):
    """Wraps mlj.Encoder with optional adaptive average pooling."""

    def __init__(self, obs_shape_hwc, enc_cfg, *, rngs):
        flat_cfg = _flatten_encoder_cfg(enc_cfg)
        self._encoder = mlj.Encoder(obs_shape_hwc, flat_cfg, rngs=rngs)
        fc = enc_cfg.full_connection if hasattr(enc_cfg, 'full_connection') else None
        if isinstance(fc, ml.AdaptiveAveragePoolingConfig):
            self._pool_size = tuple(fc.output_size)
            ph, pw = self._pool_size
            self.output_dim = self._encoder.output_channels * ph * pw
        else:
            self._pool_size = None
            self.output_dim = self._encoder.output_dim

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self._encoder.conv_layers:
            x = layer(x)
        if self._encoder.spatial_softmax is not None:
            x = self._encoder.spatial_softmax(x)
            return x.reshape(x.shape[0], -1)
        if self._pool_size is not None:
            x = _adaptive_avg_pool_nhwc(x, self._pool_size)
        return x.reshape(x.shape[0], -1)


def _make_image_encoder(cfg, obs_shape_hwc, enc_cfg, *, rngs):
    """Construct an image encoder.

    Two paths are supported in the D3IL port:
      * ``ResNet18EncoderConfig`` -> a ResNet18 + SpatialSoftmax encoder that
        matches the other (PyTorch) baselines' vision backbone.
      * ``ml.EncoderConfig`` -> the legacy ``_PooledEncoder`` (ml-networks conv
        encoder).
    The ACT / affordance ResNet encoders are vapo-specific and not ported."""
    if getattr(cfg, "use_act_encoder", False):
        raise NotImplementedError(
            "use_act_encoder is not supported in the D3IL EBM port; use the "
            "ml-networks conv encoder (EncoderConfig) or ResNet18EncoderConfig."
        )
    from .config import ResNet18EncoderConfig
    if isinstance(enc_cfg, ResNet18EncoderConfig):
        from .resnet_encoder import ResNet18Encoder
        return ResNet18Encoder(obs_shape_hwc, enc_cfg, rngs=rngs)
    return _PooledEncoder(obs_shape_hwc, enc_cfg, rngs=rngs)


def _nchw_to_nhwc(x: jax.Array) -> jax.Array:
    if x.ndim == 5:
        return jnp.transpose(x, (0, 1, 3, 4, 2))
    return jnp.transpose(x, (0, 2, 3, 1))


class _NullEMA:
    """Stand-in for ``NnxEMA`` when there is no phase1 backbone to track."""

    def __init__(self):
        self.ema_state = {}
        self.decay = 1.0

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        pass


def _build_gripper_head(gripper_head_cfg, obs_feat_dim: int,
                        flow_action_dim: int, *, rngs):
    if isinstance(gripper_head_cfg, UNetConfig):
        gripper_head_cfg.input_dim = flow_action_dim
        gripper_head_cfg.cond_dim = obs_feat_dim
        gripper_head_cfg.cond_step = 1
        head = ConditionalUnet1DPolicy(gripper_head_cfg, rngs=rngs)
        head.final_linear = nnx.Conv(
            in_features=gripper_head_cfg.channels[0], out_features=1,
            kernel_size=(1,), rngs=rngs,
        )
        return head, True
    if isinstance(gripper_head_cfg, ml.MLPConfig):
        head = mlj.MLPLayer(
            obs_feat_dim + flow_action_dim, 1, gripper_head_cfg, rngs=rngs,
        )
        return head, False
    raise TypeError(
        f"gripper_head_cfg must be MLPConfig or UNetConfig, got "
        f"{type(gripper_head_cfg).__name__}.")


def _apply_gripper_head(head, is_unet: bool, obs_feat, action_flow):
    B, T, _ = action_flow.shape
    if is_unet:
        t_dummy = jnp.zeros((B,), dtype=jnp.float32)
        cond = obs_feat[:, None, :]
        return head(t_dummy, action_flow, cond)
    obs_t = jnp.broadcast_to(
        obs_feat[:, None, :], (B, T, obs_feat.shape[-1]))
    head_in = jnp.concatenate([obs_t, action_flow], axis=-1).reshape(B * T, -1)
    return head(head_in).reshape(B, T, 1)


def _make_framework(cfg):
    """Create noise scheduler or flow matcher from config."""
    from .config import DiffuserConfig, FlowMatchingConfig
    if isinstance(cfg, DiffuserConfig):
        if cfg.diff_type == "ddpm":
            sched = DDPMScheduler(
                num_train_timesteps=cfg.num_timesteps,
                beta_start=cfg.beta_start, beta_end=cfg.beta_end,
                beta_schedule=cfg.beta_schedule,
                prediction_type=cfg.prediction_type,
                clip_sample=cfg.clip_sample,
                variance_type=cfg.variance_type,
            )
        else:
            sched = DDIMScheduler(
                num_train_timesteps=cfg.num_timesteps,
                beta_start=cfg.beta_start, beta_end=cfg.beta_end,
                beta_schedule=cfg.beta_schedule,
                prediction_type=cfg.prediction_type,
                clip_sample=cfg.clip_sample,
            )
            sched.set_timesteps(cfg.num_inference_steps)
        return sched, False
    elif isinstance(cfg, FlowMatchingConfig):
        if cfg.is_stable:
            if cfg.use_ot:
                fm = OTStableFlowMatcher(
                    cfg.sigma, cfg.lambda_x, cfg.eps,
                    ot_method=cfg.ot_method,
                    sinkhorn_epsilon=cfg.sinkhorn_epsilon,
                )
            else:
                fm = StableFlowMatcher(cfg.sigma, cfg.lambda_x, cfg.eps)
        else:
            if cfg.use_ot:
                fm = OTConditionalFlowMatcher(
                    cfg.sigma,
                    ot_method=cfg.ot_method,
                    sinkhorn_epsilon=cfg.sinkhorn_epsilon,
                )
            else:
                fm = ConditionalFlowMatcher(cfg.sigma)
        return fm, True
    raise ValueError(f"Unknown framework config: {type(cfg)}")


def _make_optimizer(name: str, lr: float, **kwargs):
    name = name.lower()
    max_grad_norm = kwargs.pop("max_grad_norm", None)
    if name == "adam":
        opt = optax.adam(lr)
    elif name in ("adamw", "adabelief"):
        opt = optax.adamw(lr, weight_decay=kwargs.get("weight_decay", 0.0))
    else:
        opt = optax.adam(lr)
    if max_grad_norm is not None:
        opt = optax.chain(optax.clip_by_global_norm(max_grad_norm), opt)
    return opt


# ---------------------------------------------------------------------------
# EBMPolicy (ported verbatim from vapo.policy.policy.EBMPolicy)
# ---------------------------------------------------------------------------

class EBMPolicy(JaxModuleBase):
    """Two-phase EBM policy.

    Phase 1 ("generative"): Train an unconditional diffusion/flow matching model
        with OT coupling always enabled.
    Phase 2 ("ebm"): Freeze the generative model and train an Energy-Based Model
        that maps observation features to initial noise via energy minimization.
    """

    def __init__(self, cfg: EBMPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.ebm_cfg = cfg.ebm  # None in phase 1
        rngs = nnx.Rngs(0)

        self._sep_grip = bool(getattr(cfg, "separate_gripper", False))
        self._flow_action_dim = (
            cfg.action_dim - 1 if self._sep_grip else cfg.action_dim
        )
        if self._sep_grip and cfg.action_dim < 2:
            raise ValueError(
                f"separate_gripper=True requires action_dim >= 2, "
                f"got {cfg.action_dim}."
            )

        # --- Policy backbone (unconditional) ---
        if cfg.phase == "ebm_only":
            self.policy = None
            self.framework = None
            self.framework_cfg = None
            self.is_fm = False
            self.ot_coupler = None
        else:
            assert cfg.policy is not None, (
                "EBMPolicyConfig.policy must be set. For phase 2, set phase1_config "
                "in the YAML so that the training script copies it automatically."
            )
            if isinstance(cfg.policy, UNetConfig):
                cfg.policy.cond_dim = 0
                cfg.policy.cond_step = 0
                if self._sep_grip:
                    cfg.policy.input_dim = self._flow_action_dim
                self.policy = ConditionalUnet1DPolicy(cfg.policy, rngs=rngs)
            elif isinstance(cfg.policy, DiTConfig):
                cfg.policy.cond_step = 0
                if self._sep_grip:
                    cfg.policy.input_dim = self._flow_action_dim
                self.policy = DiT(
                    cfg.pred_obs_action, goal_cond_step=0, pred_variance=False,
                    cfg=cfg.policy, rngs=rngs,
                )
            else:
                raise ValueError(f"EBMPolicy supports UNetConfig or DiTConfig, got {type(cfg.policy)}")

            assert cfg.framework_cfg is not None
            self.framework, self.is_fm = _make_framework(cfg.framework_cfg)
            self.framework_cfg = cfg.framework_cfg

            self.ot_coupler = OTDiffusionCoupler(
                ot_method=cfg.ot_method,
                sinkhorn_epsilon=cfg.sinkhorn_epsilon,
            )

        # --- Phase 2 / EBM-only: observation encoders + EBM ---
        if cfg.phase in ("ebm", "ebm_only"):
            assert cfg.ebm is not None, "EBMConfig is required for phase 'ebm'."
            if cfg.use_static:
                assert not cfg.use_static_dino
                obs_shape_hwc = (cfg.static_obs_shape[1], cfg.static_obs_shape[2], cfg.static_obs_shape[0])
                self.static_encoder = _make_image_encoder(cfg, obs_shape_hwc, cfg.static_encoder, rngs=rngs)
                self.static_obs_transform = nnx.Linear(self.static_encoder.output_dim, cfg.static_obs_dim, rngs=rngs)
            if cfg.use_gripper:
                assert not cfg.use_gripper_dino
                obs_shape_hwc = (cfg.gripper_obs_shape[1], cfg.gripper_obs_shape[2], cfg.gripper_obs_shape[0])
                self.gripper_encoder = _make_image_encoder(cfg, obs_shape_hwc, cfg.gripper_encoder, rngs=rngs)
                self.gripper_obs_transform = nnx.Linear(self.gripper_encoder.output_dim, cfg.gripper_obs_dim, rngs=rngs)
            if cfg.use_pos:
                pos_in = cfg.pos_input_dim if cfg.pos_input_dim is not None else cfg.action_dim
                self.pos_encoder = mlj.MLPLayer(pos_in, cfg.pos_obs_dim, cfg.pos_encoder, rngs=rngs)
            if cfg.use_task_id:
                self.task_embed = nnx.Embed(
                    num_embeddings=cfg.n_tasks,
                    features=cfg.task_embed_dim,
                    rngs=rngs,
                )

            obs_modality_dims = []
            if cfg.use_pos:
                obs_modality_dims.append(cfg.pos_obs_dim)
            if cfg.use_static:
                obs_modality_dims.append(cfg.static_obs_dim)
            if cfg.use_gripper:
                obs_modality_dims.append(cfg.gripper_obs_dim)
            if cfg.use_task_id:
                obs_modality_dims.append(cfg.task_embed_dim)
            obs_per_step = sum(obs_modality_dims)
            obs_feat_dim = obs_per_step * cfg.obs_length
            self._obs_feat_dim = obs_feat_dim

            noise_dim = cfg.policy_length * self._flow_action_dim
            if cfg.ebm.arch == "transformer":
                self.ebm = EnergyTransformer(
                    obs_modality_dims=tuple(obs_modality_dims),
                    noise_token_dim=self._flow_action_dim,
                    obs_steps=cfg.obs_length,
                    noise_steps=cfg.policy_length,
                    hidden_dim=cfg.ebm.hidden_dim,
                    n_layers=cfg.ebm.n_layers,
                    nhead=cfg.ebm.nhead,
                    mlp_ratio=cfg.ebm.mlp_ratio,
                    pos_emb_on_qk=cfg.ebm.pos_emb_on_qk,
                    rngs=rngs,
                )
            else:
                self.ebm = EnergyModel(
                    obs_dim=obs_feat_dim,
                    noise_dim=noise_dim,
                    hidden_dim=cfg.ebm.hidden_dim,
                    n_layers=cfg.ebm.n_layers,
                    rngs=rngs,
                )

            if self._sep_grip:
                if cfg.gripper_head_cfg is None:
                    raise ValueError(
                        "separate_gripper=True but gripper_head_cfg is not "
                        "set. Provide an MLPConfig or a UNetConfig.")
                self.gripper_head, self._gripper_head_is_unet = (
                    _build_gripper_head(
                        cfg.gripper_head_cfg, obs_feat_dim,
                        self._flow_action_dim, rngs=rngs,
                    )
                )

        self._build_for_phase(cfg, rngs)

    def _build_for_phase(self, cfg, rngs):
        if cfg.phase == "generative":
            self._jax_modules = {"policy": self.policy}
        elif cfg.phase in ("ebm", "ebm_only"):
            self._jax_modules = {"ebm": self.ebm}
            if cfg.use_static:
                self._jax_modules["static_encoder"] = self.static_encoder
                self._jax_modules["static_obs_transform"] = self.static_obs_transform
            if cfg.use_gripper:
                self._jax_modules["gripper_encoder"] = self.gripper_encoder
                self._jax_modules["gripper_obs_transform"] = self.gripper_obs_transform
            if cfg.use_pos:
                self._jax_modules["pos_encoder"] = self.pos_encoder
            if self._sep_grip:
                self._jax_modules["gripper_head"] = self.gripper_head
        else:
            raise ValueError(f"Unknown phase: {cfg.phase}")

        if cfg.phase in ("ebm", "ebm_only") and cfg.ebm.optimizer_cfg:
            opt_cfg = dict(cfg.ebm.optimizer_cfg)
        else:
            opt_cfg = dict(cfg.optimizer_cfg)
        opt_name = opt_cfg.pop("name", "adam")
        lr = opt_cfg.pop("lr", opt_cfg.pop("learning_rate", 1e-3))
        self._jax_optimizer = _make_optimizer(opt_name, lr, **opt_cfg)

        params = self._get_jax_params()
        self._opt_state = self._jax_optimizer.init(params)

        ema_cfg = dict(cfg.ema)
        if cfg.phase == "ebm_only":
            self.ema = _NullEMA()
        else:
            self.ema = NnxEMA(self.policy, decay=ema_cfg.get("decay", 0.999))

        if cfg.phase in ("ebm", "ebm_only"):
            self.ebm_ema = NnxEMA(self.ebm, decay=cfg.ebm.ema_decay)

        self._params = params
        self._frozen_params = None

        if cfg.phase == "ebm" and cfg.generative_checkpoint is not None:
            self._load_generative_checkpoint(cfg.generative_checkpoint)

        if cfg.phase == "generative":
            self._jit_step = self._create_jit_step_phase1()
        elif cfg.phase == "ebm":
            self._jit_step = self._create_jit_step_phase2()
            self._jit_val = self._create_jit_val_step_phase2()
        else:  # ebm_only
            self._jit_step = self._create_jit_step_phase_ebm_only()
            self._jit_val = self._create_jit_val_step_phase_ebm_only()
            self._jit_val_inference_single = (
                self._create_jit_val_inference_single_ebm_only()
            )
        self._jit_inference = self._create_jit_inference()
        self._inference_call_count = 0

    def _load_generative_checkpoint(self, ckpt_path):
        """Load a phase 1 checkpoint and freeze generative model params."""
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        jax_state = jax.tree.map(jnp.asarray, ckpt.get("jax_state", {}))
        jax_ema = jax.tree.map(jnp.asarray, ckpt.get("jax_ema_state", {}))
        if "policy" in jax_state:
            graphdef, _ = nnx.split(self.policy)
            self.policy = nnx.merge(graphdef, jax_state["policy"])
        if jax_ema:
            self.ema.ema_state = jax_ema
        _, frozen_state = nnx.split(self.policy)
        self._frozen_policy_state = frozen_state

    def load_generative_from_model(self, phase1_model: "EBMPolicy"):
        """Inject a trained in-memory phase-1 generative model + its EMA.

        D3IL trains phase 1 and phase 2 in one process, so the frozen
        generative model is copied directly from the phase-1 ``EBMPolicy``
        instead of being reloaded from a checkpoint on disk.
        """
        graphdef, _ = nnx.split(self.policy)
        # Use the concrete trained parameters returned by the jitted step
        # (``phase1_model._params["policy"]``); reading ``nnx.split`` on the
        # live module would surface tracers left behind by the jit transform.
        p1_state = phase1_model._params["policy"]
        self.policy = nnx.merge(graphdef, jax.tree.map(jnp.asarray, p1_state))
        self.ema.ema_state = jax.tree.map(jnp.asarray, phase1_model.ema.ema_state)
        _, frozen_state = nnx.split(self.policy)
        self._frozen_policy_state = frozen_state

    # --- JAX param management ---

    def _get_jax_params(self):
        params = {}
        for name, mod in self._jax_modules.items():
            _, state = nnx.split(mod)
            params[name] = state
        return params

    def _get_jax_state(self):
        return self._params

    def _set_jax_state(self, state):
        for name, mod in list(self._jax_modules.items()):
            if name in state:
                graphdef, _ = nnx.split(mod)
                new_mod = nnx.merge(graphdef, state[name])
                self._jax_modules[name] = new_mod
                setattr(self, name, new_mod)

    def _get_opt_state(self):
        return self._opt_state

    def _set_opt_state(self, state):
        self._opt_state = state

    # --- Observation encoding (phase 2) ---

    def _encode_obs_flat(self, static_obs, gripper_obs, pos, task_onehot=None):
        cfg = self.cfg
        use_timestep_major = (
            cfg.ebm is not None and cfg.ebm.arch == "transformer"
        )

        task_emb_step = None
        if cfg.use_task_id and task_onehot is not None:
            task_idx = jnp.argmax(task_onehot, axis=-1).astype(jnp.int32)
            task_emb_step = self.task_embed(task_idx)

        if use_timestep_major:
            step_parts = []
            if cfg.use_pos and pos is not None:
                pos_emb = self.pos_encoder(pos)
                if pos_emb.ndim == 2:
                    pos_emb = pos_emb[:, None, :]
                step_parts.append(pos_emb)
            if cfg.use_static and static_obs is not None:
                B, T = static_obs.shape[:2]
                s = einops.rearrange(_nchw_to_nhwc(static_obs), "b t h w c -> (b t) h w c")
                emb = self.static_obs_transform(self.static_encoder(s))
                if cfg.freeze_static:
                    emb = jax.lax.stop_gradient(emb)
                step_parts.append(einops.rearrange(emb, "(b t) d -> b t d", b=B))
            if cfg.use_gripper and gripper_obs is not None:
                B, T = gripper_obs.shape[:2]
                g = einops.rearrange(_nchw_to_nhwc(gripper_obs), "b t h w c -> (b t) h w c")
                emb = self.gripper_obs_transform(self.gripper_encoder(g))
                if cfg.freeze_gripper:
                    emb = jax.lax.stop_gradient(emb)
                step_parts.append(einops.rearrange(emb, "(b t) d -> b t d", b=B))
            if task_emb_step is not None:
                t = jnp.broadcast_to(
                    task_emb_step[:, None, :],
                    (task_emb_step.shape[0], cfg.obs_length, task_emb_step.shape[-1]),
                )
                step_parts.append(t)
            if step_parts:
                obs_per_step = jnp.concatenate(step_parts, axis=-1)
                return obs_per_step.reshape(obs_per_step.shape[0], -1)
        else:
            parts = []
            if cfg.use_pos and pos is not None:
                pos_emb = self.pos_encoder(pos)
                if pos_emb.ndim == 2:
                    pos_emb = pos_emb[:, None, :]
                parts.append(pos_emb.reshape(pos_emb.shape[0], -1))
            if cfg.use_static and static_obs is not None:
                B, T = static_obs.shape[:2]
                s = einops.rearrange(_nchw_to_nhwc(static_obs), "b t h w c -> (b t) h w c")
                emb = self.static_encoder(s)
                if cfg.freeze_static:
                    emb = jax.lax.stop_gradient(emb)
                emb = self.static_obs_transform(emb)
                parts.append(einops.rearrange(emb, "(b t) d -> b (t d)", b=B))
            if cfg.use_gripper and gripper_obs is not None:
                B, T = gripper_obs.shape[:2]
                g = einops.rearrange(_nchw_to_nhwc(gripper_obs), "b t h w c -> (b t) h w c")
                emb = self.gripper_encoder(g)
                if cfg.freeze_gripper:
                    emb = jax.lax.stop_gradient(emb)
                emb = self.gripper_obs_transform(emb)
                parts.append(einops.rearrange(emb, "(b t) d -> b (t d)", b=B))
            if task_emb_step is not None:
                parts.append(jnp.tile(task_emb_step, (1, cfg.obs_length)))
            if parts:
                return jnp.concatenate(parts, axis=-1)

        B = static_obs.shape[0] if static_obs is not None else (gripper_obs.shape[0] if gripper_obs is not None else 1)
        return jnp.zeros((B, 1))

    # --- Denoising ---

    def _denoise(self, initial_noise, ema_state):
        ema_graphdef, _ = nnx.split(self.policy)
        ema_policy = nnx.merge(ema_graphdef, ema_state)
        trajectory = initial_noise
        B = trajectory.shape[0]

        if self.is_fm and isinstance(self.framework, StableFlowMatcher):
            t = jnp.zeros(B)
            for i in range(self.framework_cfg.num_inference_steps):
                tau = self.framework.compute_tau(t)
                vector = ema_policy(t=tau, x=trajectory, cond=None)
                if i == 0:
                    trajectory = trajectory + vector / self.framework.lambda_x
                    t = t + 1.0 / self.framework.lambda_x
                else:
                    trajectory = trajectory + vector * self.framework.eps
                    t = t + self.framework.eps
        elif self.is_fm:
            nfe = int(self.framework_cfg.num_inference_steps)
            dt = 1.0 / nfe

            def _denoise_body(_, state):
                traj, t = state
                vector = ema_policy(t=t, x=traj, cond=None)
                return (traj + vector * dt, t + dt)

            trajectory, _ = jax.lax.fori_loop(
                0, nfe, _denoise_body, (trajectory, jnp.zeros(B)),
            )
        else:
            for ts in self.framework.timesteps:
                t_batch = jnp.full((B,), ts)
                pred = ema_policy(t=t_batch, x=trajectory, cond=None)
                trajectory = self.framework.step(pred, ts, trajectory).prev_sample

        return trajectory

    def _invert(self, actions, ema_state, num_steps=None):
        if not self.is_fm or isinstance(self.framework, StableFlowMatcher):
            raise NotImplementedError(
                "Inversion currently supports ConditionalFlowMatcher (CFM) "
                "only. For StableFM or DDPM/DDIM phase1, either implement a "
                "scheduler-specific inverse or fall back to loss_type='ce'."
            )

        ema_graphdef, _ = nnx.split(self.policy)
        ema_policy = nnx.merge(ema_graphdef, ema_state)
        x = jax.lax.stop_gradient(actions)
        B = x.shape[0]
        nfe = int(num_steps
                  or getattr(self.ebm_cfg, "invert_num_steps", None)
                  or self.framework_cfg.num_inference_steps)
        dt = 1.0 / nfe

        def _invert_body(_, state):
            x, t = state
            vector = ema_policy(t=t, x=x, cond=None)
            return (x - vector * dt, t - dt)

        x, _ = jax.lax.fori_loop(
            0, nfe, _invert_body, (x, jnp.ones(B)),
        )
        return jax.lax.stop_gradient(x)

    # --- Separate-gripper helpers ---

    def _gripper_head_apply(self, obs_features, traj_flow):
        return _apply_gripper_head(
            self.gripper_head, getattr(self, "_gripper_head_is_unet", False),
            obs_features, traj_flow,
        )

    def _gripper_head_loss(self, obs_features, action_full):
        if not self._sep_grip:
            return jnp.float32(0.0)
        flow = action_full[..., :self._flow_action_dim]
        gt_grip = action_full[..., self._flow_action_dim:self._flow_action_dim + 1]
        pred = self._gripper_head_apply(obs_features, flow)
        return jnp.mean((pred - gt_grip) ** 2)

    def _maybe_append_gripper(self, obs_features, traj_flow):
        if not self._sep_grip:
            return traj_flow
        B = traj_flow.shape[0]
        if obs_features.shape[0] != B:
            obs_features = jnp.broadcast_to(
                obs_features.reshape(1, -1),
                (B, obs_features.shape[-1]),
            )
        grip = self._gripper_head_apply(obs_features, traj_flow)
        return jnp.concatenate([traj_flow, grip], axis=-1)

    # --- Phase 1: Unconditional generative model training ---

    def _compute_phase1_loss(self, action, key):
        if self._sep_grip:
            action = action[..., :self._flow_action_dim]
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1, action.shape)
        if self.cfg.use_uniform:
            noise = jax.random.uniform(key1, action.shape, minval=-1, maxval=1)
        noise = self.ot_coupler.couple(noise, action)

        if self.is_fm:
            t = jax.random.uniform(key2, (action.shape[0],))
            t, x_t, u_t = self.framework.sample_location_and_conditional_flow(noise, action, t)
        else:
            t = jax.random.randint(key2, (action.shape[0],), 0, self.framework_cfg.num_timesteps)
            x_t = self.framework.add_noise(action, noise, t)
            u_t = diffusion_training_target(
                self.framework_cfg.prediction_type,
                action, noise,
                self.framework.alphas_cumprod, t,
            )

        v_t = self.policy(t=t, x=x_t, cond=None)
        loss = jnp.mean(((v_t - u_t) ** 2).sum(axis=[-2, -1]))
        return loss, {"loss": loss}

    def _create_jit_step_phase1(self):
        optimizer = self._jax_optimizer
        ema_decay = self.ema.decay

        @jax.jit
        def jit_step(params, opt_state, ema_state,
                     static_obs, gripper_obs, action, pos,
                     static_goal, gripper_goal, goal_far, key, task_onehot):
            def loss_fn(p):
                self._set_jax_state(p)
                return self._compute_phase1_loss(action, key)
            (loss, ld), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_ema = jax.tree.map(
                lambda e, c: e * ema_decay + c * (1.0 - ema_decay),
                ema_state, new_params["policy"],
            )
            return new_params, new_opt, new_ema, loss, ld

        return jit_step

    # --- Phase 2: EBM training ---

    def _compute_phase2_loss(self, static_obs, gripper_obs, action, pos, key, task_onehot=None):
        obs_features = self._encode_obs_flat(static_obs, gripper_obs, pos, task_onehot=task_onehot)

        B = action.shape[0]
        K = self.ebm_cfg.num_samples_k
        key1, key2 = jax.random.split(key)

        action_flow = (action[..., :self._flow_action_dim]
                       if self._sep_grip else action)
        flow_shape = action_flow.shape

        noise_shape = (B * K, flow_shape[1], flow_shape[2])
        noises = jax.random.normal(key1, noise_shape)

        generated = jax.lax.stop_gradient(
            self._denoise(noises, self.ema.ema_state)
        )
        generated = generated.reshape(B, K, *flow_shape[1:])
        noises = noises.reshape(B, K, -1)

        gt_expanded = action_flow[:, None, :, :]
        mse = jnp.mean((generated - gt_expanded) ** 2, axis=(-2, -1))

        obs_expanded = jnp.broadcast_to(
            obs_features[:, None, :], (B, K, obs_features.shape[-1])
        ).reshape(B * K, -1)
        noises_flat = noises.reshape(B * K, -1)

        energies = self.ebm(obs_expanded, noises_flat).reshape(B, K)

        target_probs = jax.nn.softmax(-mse / self.ebm_cfg.temperature, axis=-1)
        pred_log_probs = jax.nn.log_softmax(-energies / self.ebm_cfg.temperature, axis=-1)
        ce_loss = jnp.mean(-jnp.sum(target_probs * pred_log_probs, axis=-1))

        grip_mse = self._gripper_head_loss(obs_features, action)
        total = ce_loss + float(getattr(self.cfg, "gripper_loss_weight", 1.0)) * grip_mse

        loss_dict = {
            "loss": total,
            "ebm_ce": ce_loss,
            "mse_mean": jnp.mean(mse),
            "mse_min": jnp.mean(jnp.min(mse, axis=-1)),
            "grip_mse": grip_mse,
        }
        return total, loss_dict

    def _langevin_negatives(self, key, obs_features, K_neg, steps, step_size):
        B = obs_features.shape[0]
        T = self.cfg.policy_length
        D = self._flow_action_dim

        key_init, key_steps = jax.random.split(key)
        noise = jax.random.normal(key_init, (B, K_neg, T, D))
        noise_flat = noise.reshape(B * K_neg, -1)
        obs_expanded = jnp.broadcast_to(
            obs_features[:, None, :], (B, K_neg, obs_features.shape[-1]),
        ).reshape(B * K_neg, -1)

        def energy_sum(nf):
            return self.ebm(obs_expanded, nf).sum()

        sqrt_step = jnp.sqrt(2.0 * step_size) * 0.01

        def _lang_body(_, state):
            noise_flat, key_steps = state
            key_steps, sub = jax.random.split(key_steps)
            grad_e = jax.grad(energy_sum)(noise_flat)
            noise_flat = noise_flat - step_size * grad_e
            noise_flat = noise_flat + sqrt_step * jax.random.normal(
                sub, noise_flat.shape)
            return (noise_flat, key_steps)

        noise_flat, _ = jax.lax.fori_loop(
            0, int(steps), _lang_body, (noise_flat, key_steps),
        )

        noise_neg = noise_flat.reshape(B, K_neg, T, D)
        return jax.lax.stop_gradient(noise_neg)

    def _compute_phase2_loss_infonce(self, static_obs, gripper_obs, action, pos,
                                     key, task_onehot=None, step=None):
        obs_features = self._encode_obs_flat(
            static_obs, gripper_obs, pos, task_onehot=task_onehot)

        B = action.shape[0]
        T = self.cfg.policy_length
        D = self._flow_action_dim
        K_neg = int(self.ebm_cfg.infonce_num_negatives)
        key_neg, _key_unused = jax.random.split(key)

        action_for_invert = (action[..., :D] if self._sep_grip else action)

        noise_pos = self._invert(action_for_invert, self.ema.ema_state)

        neg_source = getattr(self.ebm_cfg, "infonce_negative_source", "random")
        warmup_steps = int(getattr(
            self.ebm_cfg, "infonce_negative_warmup_steps", 0))

        def _random_negs(k):
            return jax.random.normal(k, (B, K_neg, T, D))

        def _langevin_negs(k):
            return self._langevin_negatives(
                k, obs_features, K_neg,
                steps=self.ebm_cfg.infonce_langevin_steps,
                step_size=self.ebm_cfg.langevin_step_size,
            )

        if neg_source == "random":
            noise_neg = _random_negs(key_neg)
            in_warmup = jnp.float32(0.0)
        elif neg_source == "langevin":
            if warmup_steps > 0 and step is not None:
                in_warmup = (step < warmup_steps).astype(jnp.float32)
                noise_neg = jax.lax.cond(
                    step < warmup_steps,
                    _random_negs,
                    _langevin_negs,
                    key_neg,
                )
            else:
                noise_neg = _langevin_negs(key_neg)
                in_warmup = jnp.float32(0.0)
        else:
            raise ValueError(
                f"infonce_negative_source must be 'random' or 'langevin', "
                f"got {neg_source!r}.")

        use_inbatch = bool(getattr(
            self.ebm_cfg, "infonce_use_inbatch_inverted", False))
        if use_inbatch and B > 1:
            rows = jnp.arange(B)[:, None]
            cols = jnp.arange(B - 1)[None, :]
            other_idx = jnp.where(cols < rows, cols, cols + 1)
            noise_inbatch = noise_pos[other_idx]
            K_inbatch = B - 1
        else:
            noise_inbatch = None
            K_inbatch = 0

        cand_list = [noise_pos[:, None]]
        if noise_inbatch is not None:
            cand_list.append(noise_inbatch)
        cand_list.append(noise_neg)
        all_noises = jnp.concatenate(cand_list, axis=1)
        K_total = 1 + K_inbatch + K_neg

        obs_expanded = jnp.broadcast_to(
            obs_features[:, None, :], (B, K_total, obs_features.shape[-1])
        ).reshape(B * K_total, -1)
        all_noises_flat = all_noises.reshape(B * K_total, -1)
        energies = self.ebm(obs_expanded, all_noises_flat).reshape(B, K_total)

        logits = -energies / self.ebm_cfg.temperature
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        infonce_loss = -jnp.mean(log_probs[:, 0])

        grip_mse = self._gripper_head_loss(obs_features, action)
        grip_weight = float(getattr(self.cfg, "gripper_loss_weight", 1.0))
        total_loss = infonce_loss + grip_weight * grip_mse

        loss_dict = {
            "loss": total_loss,
            "infonce": infonce_loss,
            "grip_mse": grip_mse,
            "energy_pos": jnp.mean(energies[:, 0]),
            "energy_neg": jnp.mean(energies[:, 1:]),
            "infonce_top1_acc": jnp.mean(
                (jnp.argmin(energies, axis=-1) == 0).astype(jnp.float32),
            ),
            "infonce_margin": (
                jnp.mean(energies[:, 0]) - jnp.mean(energies[:, 1:])
            ),
            "infonce_in_warmup": in_warmup,
            "infonce_inbatch_neg": jnp.float32(K_inbatch),
        }
        return total_loss, loss_dict

    def _compute_phase2_loss_dispatch(self, static_obs, gripper_obs, action,
                                      pos, key, task_onehot=None, step=None):
        loss_type = getattr(self.ebm_cfg, "loss_type", "ce")
        if loss_type == "infonce":
            return self._compute_phase2_loss_infonce(
                static_obs, gripper_obs, action, pos, key,
                task_onehot=task_onehot, step=step,
            )
        return self._compute_phase2_loss(
            static_obs, gripper_obs, action, pos, key,
            task_onehot=task_onehot,
        )

    def _create_jit_step_phase2(self):
        optimizer = self._jax_optimizer
        ebm_ema_decay = self.ebm_cfg.ema_decay

        @jax.jit
        def jit_step(params, opt_state, ebm_ema_state,
                     static_obs, gripper_obs, action, pos,
                     static_goal, gripper_goal, goal_far, key, task_onehot, step):
            def loss_fn(p):
                self._set_jax_state(p)
                return self._compute_phase2_loss_dispatch(
                    static_obs, gripper_obs, action, pos, key,
                    task_onehot=task_onehot, step=step,
                )
            (loss, ld), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_ebm_ema = jax.tree.map(
                lambda e, c: e * ebm_ema_decay + c * (1.0 - ebm_ema_decay),
                ebm_ema_state, new_params["ebm"],
            )
            return new_params, new_opt, new_ebm_ema, loss, ld

        return jit_step

    # --- Validation (loss + visualization samples) ---

    def _ebm_inference_single(self, ema_state, ebm_ema_state,
                              static_obs, gripper_obs, pos, key, task_onehot=None):
        obs_features = self._encode_obs_flat(static_obs, gripper_obs, pos, task_onehot=task_onehot)

        ebm_graphdef, _ = nnx.split(self.ebm)
        ebm_ema_model = nnx.merge(ebm_graphdef, ebm_ema_state)

        N = self.ebm_cfg.inference_num_samples
        noise_shape = (N, self.cfg.policy_length, self._flow_action_dim)
        candidates = jax.random.normal(key, noise_shape)

        obs_expanded = jnp.broadcast_to(obs_features, (N, obs_features.shape[-1]))
        energies = ebm_ema_model(obs_expanded, candidates.reshape(N, -1))

        top_indices = jnp.argsort(energies)[:4]
        top_noises = candidates[top_indices]

        trajectories = self._denoise(top_noises, ema_state)
        trajectories = self._maybe_append_gripper(obs_features, trajectories)
        return trajectories

    def _create_jit_val_step_phase2(self):
        @jax.jit
        def jit_val(params, ema_state, ebm_ema_state,
                    static_obs, gripper_obs, action, pos,
                    static_goal, gripper_goal, goal_far, key, task_onehot, step):
            self._set_jax_state(params)
            _, loss_dict = self._compute_phase2_loss_dispatch(
                static_obs, gripper_obs, action, pos, key,
                task_onehot=task_onehot, step=step,
            )
            return loss_dict
        return jit_val

    # --- EBM-only ablation: loss + JIT step + val step ---

    def _compute_phase_ebm_only_loss(self, static_obs, gripper_obs, action, pos,
                                     key, task_onehot=None, step=None):
        del step
        obs_features = self._encode_obs_flat(
            static_obs, gripper_obs, pos, task_onehot=task_onehot)

        B = action.shape[0]
        T = self.cfg.policy_length
        D = self._flow_action_dim
        K_neg = int(self.ebm_cfg.infonce_num_negatives)

        action_for_ebm = (action[..., :D] if self._sep_grip else action)

        key_noise, _ = jax.random.split(key)
        noise_neg = jax.random.normal(key_noise, (B, K_neg, T, D))

        use_inbatch = bool(getattr(
            self.ebm_cfg, "infonce_use_inbatch_inverted", False))
        if use_inbatch and B > 1:
            rows = jnp.arange(B)[:, None]
            cols = jnp.arange(B - 1)[None, :]
            other_idx = jnp.where(cols < rows, cols, cols + 1)
            action_inbatch = action_for_ebm[other_idx]
            K_inbatch = B - 1
        else:
            action_inbatch = None
            K_inbatch = 0

        cand_list = [action_for_ebm[:, None]]
        if action_inbatch is not None:
            cand_list.append(action_inbatch)
        cand_list.append(noise_neg)
        all_cands = jnp.concatenate(cand_list, axis=1)
        K_total = 1 + K_inbatch + K_neg

        obs_expanded = jnp.broadcast_to(
            obs_features[:, None, :], (B, K_total, obs_features.shape[-1])
        ).reshape(B * K_total, -1)
        all_cands_flat = all_cands.reshape(B * K_total, -1)
        energies = self.ebm(obs_expanded, all_cands_flat).reshape(B, K_total)

        logits = -energies / self.ebm_cfg.temperature
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        infonce_loss = -jnp.mean(log_probs[:, 0])

        grip_mse = self._gripper_head_loss(obs_features, action)
        grip_weight = float(getattr(self.cfg, "gripper_loss_weight", 1.0))
        total_loss = infonce_loss + grip_weight * grip_mse

        loss_dict = {
            "loss": total_loss,
            "infonce": infonce_loss,
            "grip_mse": grip_mse,
            "energy_pos": jnp.mean(energies[:, 0]),
            "energy_neg": jnp.mean(energies[:, 1:]),
            "infonce_top1_acc": jnp.mean(
                (jnp.argmin(energies, axis=-1) == 0).astype(jnp.float32),
            ),
            "infonce_margin": (
                jnp.mean(energies[:, 0]) - jnp.mean(energies[:, 1:])
            ),
            "infonce_inbatch_neg": jnp.float32(K_inbatch),
        }
        return total_loss, loss_dict

    def _create_jit_step_phase_ebm_only(self):
        optimizer = self._jax_optimizer
        ebm_ema_decay = self.ebm_cfg.ema_decay

        @jax.jit
        def jit_step(params, opt_state, ebm_ema_state,
                     static_obs, gripper_obs, action, pos,
                     static_goal, gripper_goal, goal_far, key, task_onehot, step):
            def loss_fn(p):
                self._set_jax_state(p)
                return self._compute_phase_ebm_only_loss(
                    static_obs, gripper_obs, action, pos, key,
                    task_onehot=task_onehot, step=step,
                )
            (loss, ld), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_ebm_ema = jax.tree.map(
                lambda e, c: e * ebm_ema_decay + c * (1.0 - ebm_ema_decay),
                ebm_ema_state, new_params["ebm"],
            )
            return new_params, new_opt, new_ebm_ema, loss, ld

        return jit_step

    def _create_jit_val_step_phase_ebm_only(self):
        @jax.jit
        def jit_val(params, ema_state, ebm_ema_state,
                    static_obs, gripper_obs, action, pos,
                    static_goal, gripper_goal, goal_far, key, task_onehot, step):
            self._set_jax_state(params)
            _, loss_dict = self._compute_phase_ebm_only_loss(
                static_obs, gripper_obs, action, pos, key,
                task_onehot=task_onehot, step=step,
            )
            return loss_dict
        return jit_val

    def _create_jit_val_inference_single_ebm_only(self):
        @jax.jit
        def jit_inf_single(params, ebm_ema_state,
                           static_obs_i, gripper_obs_i, pos_i,
                           key, task_onehot_i):
            self._set_jax_state(params)
            return self._ebm_only_inference_single(
                ebm_ema_state,
                static_obs_i, gripper_obs_i, pos_i,
                key,
                task_onehot=task_onehot_i,
            )
        return jit_inf_single

    def _ebm_only_inference_single(self, ebm_ema_state,
                                   static_obs, gripper_obs, pos, key,
                                   task_onehot=None):
        obs_features = self._encode_obs_flat(
            static_obs, gripper_obs, pos, task_onehot=task_onehot)
        ebm_graphdef, _ = nnx.split(self.ebm)
        ebm_ema_model = nnx.merge(ebm_graphdef, ebm_ema_state)
        N = self.ebm_cfg.inference_num_samples
        candidates = jax.random.normal(
            key, (N, self.cfg.policy_length, self._flow_action_dim))
        obs_expanded = jnp.broadcast_to(obs_features, (N, obs_features.shape[-1]))
        energies = ebm_ema_model(obs_expanded, candidates.reshape(N, -1))
        top_indices = jnp.argsort(energies)[:4]
        trajectories = candidates[top_indices]
        trajectories = self._maybe_append_gripper(obs_features, trajectories)
        return trajectories

    # --- Inference ---

    def _create_jit_inference(self):
        @jax.jit(static_argnums=(2,))
        def jit_inf(params, ema_state, batch_size,
                    static_obs, gripper_obs, pos,
                    static_goal, gripper_goal, goal_far,
                    ebm_ema_state, key, task_onehot):
            self._set_jax_state(params)

            if self.cfg.phase == "generative":
                traj_shape = (batch_size, self.cfg.policy_length, self._flow_action_dim)
                trajectory = jax.random.normal(key, traj_shape)
                trajectory = self._denoise(trajectory, ema_state)
                if self._sep_grip:
                    pad = jnp.zeros((batch_size, self.cfg.policy_length, 1),
                                    dtype=trajectory.dtype)
                    trajectory = jnp.concatenate([trajectory, pad], axis=-1)
                return trajectory, jnp.zeros(1), jnp.zeros(1)

            obs_features = self._encode_obs_flat(static_obs, gripper_obs, pos, task_onehot=task_onehot)
            obs_features = jnp.broadcast_to(obs_features, (1, obs_features.shape[-1]))

            N = max(self.ebm_cfg.inference_num_samples, batch_size)
            key1, key2 = jax.random.split(key)
            cand_shape = (N, self.cfg.policy_length, self._flow_action_dim)
            candidates = jax.random.normal(key1, cand_shape)

            ebm_graphdef, _ = nnx.split(self.ebm)
            ebm_ema_model = nnx.merge(ebm_graphdef, ebm_ema_state)

            obs_expanded = jnp.broadcast_to(
                obs_features, (N, obs_features.shape[-1])
            )
            energies = ebm_ema_model(obs_expanded, candidates.reshape(N, -1))

            top_indices = jnp.argsort(energies)[:batch_size]
            top_cands = candidates[top_indices]

            if self.cfg.phase == "ebm":
                trajectory = self._denoise(top_cands, ema_state)
            else:
                trajectory = top_cands
            trajectory = self._maybe_append_gripper(obs_features, trajectory)
            return trajectory, jnp.zeros(1), jnp.zeros(1)

        return jit_inf

    def _create_jit_inference_langevin(self):
        @jax.jit(static_argnums=(2,))
        def jit_inf_langevin(params, ema_state, batch_size,
                             static_obs, gripper_obs, pos,
                             static_goal, gripper_goal, goal_far,
                             ebm_ema_state, key, task_onehot):
            self._set_jax_state(params)
            obs_features = self._encode_obs_flat(static_obs, gripper_obs, pos, task_onehot=task_onehot)
            obs_features = jnp.broadcast_to(obs_features, (1, obs_features.shape[-1]))

            ebm_graphdef, _ = nnx.split(self.ebm)
            ebm_ema_model = nnx.merge(ebm_graphdef, ebm_ema_state)

            N = max(self.ebm_cfg.inference_num_samples, batch_size)
            key1, key2 = jax.random.split(key)
            candidates = jax.random.normal(
                key1, (N, self.cfg.policy_length, self._flow_action_dim))
            obs_expanded = jnp.broadcast_to(obs_features, (N, obs_features.shape[-1]))
            energies = ebm_ema_model(obs_expanded, candidates.reshape(N, -1))
            top_indices = jnp.argsort(energies)[:batch_size]
            init_noises = candidates[top_indices]

            noise_flat = init_noises.reshape(batch_size, -1)
            obs_batch = jnp.broadcast_to(obs_features, (batch_size, obs_features.shape[-1]))
            step_size = self.ebm_cfg.langevin_step_size

            def energy_sum(nf):
                return ebm_ema_model(obs_batch, nf).sum()

            sqrt_step_inf = jnp.sqrt(2 * step_size) * 0.01

            def _inf_lang_body(_, state):
                noise_flat, key2 = state
                key2, subkey = jax.random.split(key2)
                grad_e = jax.grad(energy_sum)(noise_flat)
                noise_flat = noise_flat - step_size * grad_e
                noise_flat = noise_flat + sqrt_step_inf * jax.random.normal(
                    subkey, noise_flat.shape)
                return (noise_flat, key2)

            noise_flat, _ = jax.lax.fori_loop(
                0, int(self.ebm_cfg.langevin_steps),
                _inf_lang_body, (noise_flat, key2),
            )

            refined = noise_flat.reshape(
                batch_size, self.cfg.policy_length, self._flow_action_dim)
            if self.cfg.phase == "ebm":
                trajectory = self._denoise(refined, ema_state)
            else:
                trajectory = refined
            trajectory = self._maybe_append_gripper(obs_batch, trajectory)
            return trajectory, jnp.zeros(1), jnp.zeros(1)

        return jit_inf_langevin

    def inference(
        self,
        batch_size: int,
        static_obs=None, gripper_obs=None, pos=None,
        static_goal=None, gripper_goal=None, goal_far=None,
        nfe=None, return_emb=False, task_onehot=None,
    ):
        key = jax.random.PRNGKey(self._inference_call_count)
        self._inference_call_count += 1

        ebm_ema_state = self.ebm_ema.ema_state if hasattr(self, "ebm_ema") else None

        if self.cfg.phase in ("ebm", "ebm_only") and self.ebm_cfg.use_langevin:
            if not hasattr(self, "_jit_inference_langevin"):
                self._jit_inference_langevin = self._create_jit_inference_langevin()
            traj, s_emb, g_emb = self._jit_inference_langevin(
                self._params, self.ema.ema_state, batch_size,
                static_obs, gripper_obs, pos,
                static_goal, gripper_goal, goal_far,
                ebm_ema_state, key, task_onehot,
            )
        else:
            traj, s_emb, g_emb = self._jit_inference(
                self._params, self.ema.ema_state, batch_size,
                static_obs, gripper_obs, pos,
                static_goal, gripper_goal, goal_far,
                ebm_ema_state, key, task_onehot,
            )

        if return_emb:
            return traj, s_emb, g_emb
        return traj
