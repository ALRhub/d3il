"""Config dataclasses for the JAX/Flax-NNX EBM policy.

Ported verbatim from ``vapo.policy.config`` (the EBM-relevant subset only).
The only D3IL-specific addition is ``EBMPolicyConfig.pos_input_dim`` which lets
the proprioceptive/state encoder take an observation of arbitrary width
(in vapo it was hard-wired to ``action_dim``); everything else matches the
original two-phase EBM policy exactly so the jax/flax.nnx model is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Union

import ml_networks as ml
from omegaconf import DictConfig, ListConfig


@dataclass
class DiffuserConfig:
    num_timesteps: int
    num_inference_steps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    diff_type: Literal["ddpm", "ddim"] = "ddpm"
    variance_type: Literal["fixed_small", "fixed_large", "learned"] = "fixed_small"

    def __post_init__(self):
        if self.diff_type not in ["ddpm", "ddim"]:
            raise ValueError(f"Invalid diff_type: {self.diff_type}. Must be 'ddpm' or 'ddim'.")
        if self.variance_type not in ["fixed_small", "fixed_large", "learned"]:
            raise ValueError(
                f"Invalid variance_type: {self.variance_type}. Must be 'fixed_small', "
                f"'fixed_large', or 'learned'."
            )
        if "learned" in self.variance_type:
            assert self.diff_type == "ddpm", \
                "Learned variance type is only supported for DDPM. Please set diff_type to 'ddpm'."


@dataclass
class FlowMatchingConfig:
    num_inference_steps: int = 10
    sigma: float = 0.1
    ot_method: Literal["sinkhorn", "exact"] = "exact"
    solver: Literal["dopri5", "tsit5", "rk4", "euler"] = "dopri5"
    sensitivity: str = "adjoint"
    atol: float = 1e-4
    rtol: float = 1e-4
    is_stable: bool = False
    lambda_x: float = 2.5
    eps: float = 0.1
    use_ot: bool = False
    use_ode_solver: bool = False
    sinkhorn_epsilon: float = 0.01

    def __post_init__(self):
        self.eps = min(1 / self.lambda_x, self.eps)


@dataclass
class PosEmbConfig:
    vector: int
    scale: float = 1000.0


@dataclass
class RNNConfig:
    rnn_name: Literal["LSTM", "GRU"]
    hidden_dim: int
    mlp_cfg: ml.MLPConfig


@dataclass
class UNetConfig:
    input_dim: int
    cond_dim: int
    cond_step: int
    time_emb: Union[PosEmbConfig, "ml.SoftmaxTransConfig"]
    channels: Tuple[int, ...]
    conv_cfg: ml.ConvConfig
    cond_cfg: ml.MLPConfig
    cond_pred_scale: bool = False
    nhead: Optional[int] = None
    has_attn: bool = False

    def __post_init__(self):
        if self.has_attn:
            assert self.nhead is not None, "nhead must be specified when has_attn is True."
        if isinstance(self.channels, (list, ListConfig)):
            self.channels = tuple(self.channels)


@dataclass
class TransformerConfig:
    input_dim: int
    cond_dim: int
    cond_step: int
    horizon: int
    d_model: int
    nhead: int
    dim_ff: int
    time_emb: Union[PosEmbConfig, "ml.SoftmaxTransConfig"]
    cond_cfg: ml.MLPConfig
    n_layers: int
    n_enc_layers: int = 0
    dropout: float = 0.1
    hidden_activation: Literal["relu", "gelu"] = "gelu"
    is_causal: bool = False
    encoder_only: bool = False
    goal_for_decoder: bool = False
    time_for_decoder: bool = False
    pos_emb_param: bool = True
    pos_emb_on_qk: bool = False


@dataclass
class DiTConfig:
    input_dim: int
    cond_step: int
    horizon: int
    d_model: int
    nhead: int
    mlp_ratio: float
    time_emb: Union[PosEmbConfig, "ml.SoftmaxTransConfig"]
    cond_cfg: ml.MLPConfig
    n_layers: int
    learn_sigma: bool = True
    use_cross_attention: bool = False
    pos_emb_on_qk: bool = False


@dataclass
class EBMConfig:
    """Configuration for the Energy-Based Model noise predictor."""
    arch: Literal["mlp", "transformer"] = "mlp"
    hidden_dim: int = 256
    n_layers: int = 4
    nhead: int = 8
    mlp_ratio: float = 4.0
    pos_emb_on_qk: bool = False
    activation: str = "gelu"
    num_samples_k: int = 32
    temperature: float = 1.0
    inference_num_samples: int = 256
    use_langevin: bool = False
    langevin_steps: int = 10
    langevin_step_size: float = 0.01
    optimizer_cfg: Union[DictConfig, Dict] = field(default_factory=dict)
    ema_decay: float = 0.999
    loss_type: Literal["ce", "infonce"] = "ce"
    infonce_num_negatives: int = 31
    invert_num_steps: Optional[int] = None
    infonce_negative_source: Literal["random", "langevin"] = "random"
    infonce_langevin_steps: int = 5
    infonce_negative_warmup_steps: int = 0
    infonce_use_inbatch_inverted: bool = False


@dataclass
class EBMPolicyConfig:
    """Top-level config for the two-phase EBM policy system.

    Phase 1 ("generative"): ``policy``, ``framework_cfg``, ``optimizer_cfg``, ``ema``.
    Phase 2 ("ebm"): ``ebm`` + observation encoders + a frozen phase-1 model
        (injected in-memory by the agent, or loaded from ``generative_checkpoint``).
    """
    static_obs_shape: Tuple[int, ...]
    gripper_obs_shape: Tuple[int, ...]
    action_dim: int
    policy_length: int = 32
    obs_length: int = 1
    pred_obs_action: bool = False
    phase: Literal["generative", "ebm", "ebm_only"] = "generative"
    # --- Phase 1: generative model settings ---
    policy: Optional[Union[UNetConfig, DiTConfig]] = None
    framework_cfg: Optional[Union[DiffuserConfig, FlowMatchingConfig]] = None
    optimizer_cfg: Union[DictConfig, Dict] = field(default_factory=dict)
    ema: Union[DictConfig, Dict] = field(default_factory=dict)
    ot_method: str = "exact"
    sinkhorn_epsilon: float = 0.01
    use_uniform: bool = False
    # --- Phase 2: EBM + observation encoder settings ---
    ebm: Optional[EBMConfig] = None
    phase1_config: Optional[str] = None
    generative_checkpoint: Optional[str] = None
    static_encoder: Optional[ml.EncoderConfig] = None
    gripper_encoder: Optional[ml.EncoderConfig] = None
    pos_encoder: Optional[ml.MLPConfig] = None
    obs_dim: Optional[int] = None
    static_obs_dim: Optional[int] = None
    gripper_obs_dim: Optional[int] = None
    pos_obs_dim: Optional[int] = None
    # D3IL addition: width of the state vector fed to the pos encoder.
    # In vapo this was hard-wired to ``action_dim`` (proprioception). For
    # D3IL the conditioning is the task observation of ``obs_dim`` width.
    # ``None`` keeps the original behaviour (falls back to ``action_dim``).
    pos_input_dim: Optional[int] = None
    # --- Separate gripper head ---
    separate_gripper: bool = False
    gripper_head_cfg: Optional[Union[ml.MLPConfig, UNetConfig]] = None
    gripper_loss_weight: float = 1.0
    # --- Shared ---
    loss_cfg: Union[DictConfig, Dict] = field(default_factory=dict)
    freeze_static: bool = False
    freeze_gripper: bool = False
    use_static_dino: bool = False
    use_gripper_dino: bool = False
    use_task_id: bool = False
    n_tasks: int = 34
    task_embed_dim: int = 64
    use_act_encoder: bool = False
    act_encoder_style: str = "act"
    act_encoder_pretrained: Optional[str] = "IMAGENET1K_V1"
    act_encoder_pool: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        self.static_obs_shape = tuple(self.static_obs_shape)
        self.gripper_obs_shape = tuple(self.gripper_obs_shape)
        self.use_static = self.static_encoder is not None
        self.use_gripper = self.gripper_encoder is not None
        self.use_pos = self.pos_encoder is not None
        if self.act_encoder_pool is not None and not isinstance(self.act_encoder_pool, tuple):
            self.act_encoder_pool = tuple(self.act_encoder_pool)
        if self.act_encoder_style not in ("act", "affordance"):
            raise ValueError(
                f"Invalid act_encoder_style={self.act_encoder_style!r}. "
                f"Expected one of {{'act', 'affordance'}}."
            )
        if self.obs_dim is not None:
            if self.static_obs_dim is None:
                self.static_obs_dim = self.obs_dim
            if self.gripper_obs_dim is None:
                self.gripper_obs_dim = self.obs_dim
            if self.pos_obs_dim is None:
                self.pos_obs_dim = self.obs_dim

    def dictcfg2dict(self) -> None:
        self.optimizer_cfg = dict(self.optimizer_cfg)
        self.loss_cfg = dict(self.loss_cfg)
        self.ema = dict(self.ema)
        if self.ebm is not None:
            self.ebm.optimizer_cfg = dict(self.ebm.optimizer_cfg)
        if self.static_encoder is not None:
            self.static_encoder.dictcfg2dict()
        if self.gripper_encoder is not None:
            self.gripper_encoder.dictcfg2dict()
