"""Policy + energy backbones for the PyTorch EBM policy.

A faithful PyTorch port of the subset of ``agents/models/ebm_jax/backbone.py``
used by the D3IL benchmark configs: the DiT flow-matching policy and the MLP /
Transformer energy models (plus the small helpers they need). Channels-first /
``(B, T, C)`` tensors throughout, matching torch conventions.
"""
from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity(x):
    return x


# Module-level (picklable) activation table — lambdas would break the
# spawn-based multiprocess evaluator's pickling of the model.
_ACT_MAP = {
    "mish": F.mish, "Mish": F.mish,
    "relu": F.relu, "ReLU": F.relu,
    "gelu": F.gelu, "GELU": F.gelu,
    "silu": F.silu, "SiLU": F.silu,
    "swish": F.silu, "Swish": F.silu,
    "tanh": torch.tanh, "Tanh": torch.tanh,
    "sigmoid": torch.sigmoid, "Sigmoid": torch.sigmoid,
    "identity": _identity, "Identity": _identity, "none": _identity,
}


def get_activation(name: str):
    return _ACT_MAP.get(name, _identity)


def sinusoidal_positional_encoding(length: int, d_model: int, base: float = 10_000.0) -> torch.Tensor:
    """(length, d_model) sinusoidal positional encoding."""
    position = torch.arange(length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,)
        x = x * self.scale
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class _LinearNormAct(nn.Module):
    """Linear -> (optional norm) -> activation -> (optional dropout).

    Mirrors ml_networks ``LinearNormActivation``; for the benchmark configs
    norm='none' and dropout=0, so this reduces to Linear + activation.
    """

    def __init__(self, in_dim: int, out_dim: int, lin_cfg):
        super().__init__()
        self.norm_first = bool(getattr(lin_cfg, "norm_first", False))
        norm = getattr(lin_cfg, "norm", "none")
        norm_dim = in_dim if self.norm_first else out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bool(getattr(lin_cfg, "bias", True)))
        if norm == "layer":
            self.norm = nn.LayerNorm(norm_dim)
        else:
            self.norm = None
        self.act = get_activation(getattr(lin_cfg, "activation", "Identity"))
        drop = float(getattr(lin_cfg, "dropout", 0.0) or 0.0)
        self.drop = nn.Dropout(drop) if drop > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first and self.norm is not None:
            x = self.norm(x)
        x = self.linear(x)
        if not self.norm_first and self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


def build_mlp(in_dim: int, out_dim: int, cfg) -> nn.Sequential:
    """Mirror ml_networks ``MLPLayer``: in->hidden, (n_layers-1) hidden->hidden,
    hidden->out with ``output_activation`` on the last layer."""
    hidden = cfg.hidden_dim
    layers = [_LinearNormAct(in_dim, hidden, cfg.linear_cfg)]
    for _ in range(cfg.n_layers - 1):
        layers.append(_LinearNormAct(hidden, hidden, cfg.linear_cfg))
    last_cfg = copy.copy(cfg.linear_cfg)
    last_cfg.activation = cfg.output_activation
    layers.append(_LinearNormAct(hidden, out_dim, last_cfg))
    return nn.Sequential(*layers)


def _mha(q_proj, k_proj, v_proj, out_proj, nhead, head_dim, x, pos_emb=None):
    """Multi-head self-attention via scaled_dot_product_attention.

    q/k take ``x + pos_emb`` (when given), v takes ``x`` — matching the JAX model.
    """
    B, T, D = x.shape
    qk_in = x + pos_emb if pos_emb is not None else x
    q = q_proj(qk_in).reshape(B, T, nhead, head_dim).transpose(1, 2)
    k = k_proj(qk_in).reshape(B, T, nhead, head_dim).transpose(1, 2)
    v = v_proj(x).reshape(B, T, nhead, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)  # (B, H, T, head_dim)
    out = out.transpose(1, 2).reshape(B, T, D)
    return out_proj(out)


# ---------------------------------------------------------------------------
# DiT (flow-matching policy backbone)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, n_cond_step: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.nhead = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.mlp_fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.ada_linear = nn.Linear(hidden_size * n_cond_step, 6 * hidden_size)

    def forward(self, x: torch.Tensor, c: torch.Tensor, pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        ada = self.ada_linear(F.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.chunk(6, dim=-1)
        attn = _mha(self.q_proj, self.k_proj, self.v_proj, self.attn_out,
                    self.nhead, self.head_dim, modulate(self.norm1(x), shift_msa, scale_msa), pos_emb)
        x = x + gate_msa.unsqueeze(1) * attn
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp_fc2(F.gelu(self.mlp_fc1(h), approximate="tanh"))
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, n_cond_step: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.ada_linear = nn.Linear(hidden_size * n_cond_step, 2 * hidden_size)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_linear(F.silu(c)).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(self, pred_obs_action: bool, goal_cond_step: int, pred_variance: bool, cfg):
        super().__init__()
        self.cfg = cfg
        self.pred_variance = pred_variance
        T = cfg.horizon + pred_obs_action * cfg.cond_step
        T_cond = cfg.cond_step + 1 + goal_cond_step

        self.input_emb = nn.Linear(cfg.input_dim, cfg.d_model)
        self.time_emb = SinusoidalPosEmb(cfg.time_emb.vector, cfg.time_emb.scale)
        self.time_encoder = build_mlp(cfg.time_emb.vector, cfg.d_model, cfg.cond_cfg)
        self.num_heads = cfg.nhead
        self.register_buffer("pos_embed", sinusoidal_positional_encoding(T, cfg.d_model).unsqueeze(0))

        self.blocks = nn.ModuleList([
            DiTBlock(cfg.d_model, T_cond, cfg.nhead, mlp_ratio=cfg.mlp_ratio)
            for _ in range(cfg.n_layers)
        ])
        out_dim = cfg.input_dim * 2 if pred_variance else cfg.input_dim
        self.final_layer = FinalLayer(cfg.d_model, T_cond, out_dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """x: (B, T, input_dim), t: (B,) in [0,1], cond: (B, *, cond_dim) or None."""
        pos = self.pos_embed
        if self.cfg.pos_emb_on_qk:
            x = self.input_emb(x)
        else:
            x = self.input_emb(x) + pos
            pos = None
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        time_emb = self.time_emb(t).reshape(x.shape[0], -1)
        t_enc = self.time_encoder(time_emb)
        if cond is not None:
            c = torch.cat([t_enc, cond.reshape(cond.shape[0], -1)], dim=-1)
        else:
            c = t_enc
        for block in self.blocks:
            x = block(x, c, pos_emb=pos)
        x = self.final_layer(x, c)
        if self.pred_variance:
            x, var = x.chunk(2, dim=-1)
            var = (torch.tanh(var) + 1) / 2
            x = torch.cat([x, var], dim=-1)
        return x


# ---------------------------------------------------------------------------
# MLP velocity field (plain-MLP phase-1, mirrors the JAX MLPVelocity / toy)
# ---------------------------------------------------------------------------

class MLPVelocity(nn.Module):
    """Plain-MLP velocity field ``v_theta(t, x)`` over a flattened action seq.

    Mirrors the JAX ``MLPVelocity`` (and the toy ``FMModel``): the action
    sequence ``(B, T, D)`` is flattened, the (raw) flow time ``t`` is
    concatenated, and a small MLP regresses the flattened velocity. Same layer
    layout as ``build_mlp`` and the same ``(t, x, cond)`` call signature as DiT
    so the EBMPolicy flow code is used unchanged.
    """

    def __init__(self, cfg):  # MLPPolicyConfig
        super().__init__()
        self.horizon = int(cfg.horizon)
        self.input_dim = int(cfg.input_dim)
        flat = self.horizon * self.input_dim
        self.act = get_activation(cfg.activation)
        self.out_act = get_activation(cfg.output_activation)
        prev = flat + 1  # flattened x + raw scalar time
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(nn.Linear(prev, cfg.hidden_dim))
            prev = cfg.hidden_dim
        self.out = nn.Linear(cfg.hidden_dim, flat)

    def forward(self, t: torch.Tensor, x: torch.Tensor, cond=None) -> torch.Tensor:
        B = x.shape[0]
        xf = x.reshape(B, -1)
        if t.ndim == 0:
            t = t.expand(B)
        tt = t.reshape(B, 1).to(xf.dtype)
        h = torch.cat([xf, tt], dim=-1)
        for layer in self.layers:
            h = self.act(layer(h))
        return self.out_act(self.out(h)).reshape(B, self.horizon, self.input_dim)


# ---------------------------------------------------------------------------
# Energy models
# ---------------------------------------------------------------------------

class EnergyModel(nn.Module):
    """Residual MLP energy net scoring (obs_features, noise) -> scalar."""

    def __init__(self, obs_dim: int, noise_dim: int, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim + noise_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, obs_features: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_features, noise.reshape(noise.shape[0], -1)], dim=-1)
        x = F.gelu(self.input_proj(x))
        for layer, norm in zip(self.layers, self.norms):
            x = x + F.gelu(layer(norm(x)))
        return self.output_proj(x).squeeze(-1)


class _EnergyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp_fc1 = nn.Linear(d_model, mlp_hidden)
        self.mlp_fc2 = nn.Linear(mlp_hidden, d_model)

    def forward(self, x: torch.Tensor, pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        out = _mha(self.q_proj, self.k_proj, self.v_proj, self.attn_out,
                   self.nhead, self.head_dim, h, pos_emb)
        x = x + out
        h = self.norm2(x)
        x = x + self.mlp_fc2(F.gelu(self.mlp_fc1(h), approximate="tanh"))
        return x


class EnergyTransformer(nn.Module):
    """Transformer energy model with per-(modality, timestep) obs tokens and
    per-timestep action tokens; the CLS token output is the scalar energy."""

    def __init__(self, obs_modality_dims: Tuple[int, ...], noise_token_dim: int,
                 obs_steps: int, noise_steps: int, hidden_dim: int = 256,
                 n_layers: int = 4, nhead: int = 8, mlp_ratio: float = 4.0,
                 pos_emb_on_qk: bool = False):
        super().__init__()
        if not obs_modality_dims:
            raise ValueError("EnergyTransformer requires at least one obs modality.")
        d_model = hidden_dim
        self.d_model = d_model
        self.obs_modality_dims = tuple(int(d) for d in obs_modality_dims)
        self.num_modalities = len(self.obs_modality_dims)
        self.obs_total_dim = sum(self.obs_modality_dims)
        self.noise_token_dim = noise_token_dim
        self.obs_steps = obs_steps
        self.noise_steps = noise_steps
        self.pos_emb_on_qk = pos_emb_on_qk

        self.obs_modality_proj = nn.ModuleList([nn.Linear(dim, d_model) for dim in self.obs_modality_dims])
        self.noise_proj = nn.Linear(noise_token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.segment_embed = nn.Embedding(3, d_model)
        max_timesteps = max(obs_steps, noise_steps, 1)
        self.register_buffer("timestep_embed", sinusoidal_positional_encoding(max_timesteps, d_model))
        self.obs_modality_embed = nn.Embedding(self.num_modalities, d_model)
        self.blocks = nn.ModuleList([
            _EnergyTransformerBlock(d_model, nhead, mlp_ratio) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, obs_features: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = obs_features.shape[0]
        M, N, K = self.obs_steps, self.noise_steps, self.num_modalities
        D_total, Dn, d = self.obs_total_dim, self.noise_token_dim, self.d_model
        dev = obs_features.device

        obs_per_step = obs_features.reshape(B, M, D_total)
        modality_tokens = []
        offset = 0
        for k, dim_k in enumerate(self.obs_modality_dims):
            chunk = obs_per_step[..., offset:offset + dim_k]
            modality_tokens.append(self.obs_modality_proj[k](chunk))
            offset += dim_k
        obs_tok = torch.stack(modality_tokens, dim=2).reshape(B, M * K, d)
        noise_tok = self.noise_proj(noise.reshape(B, N, Dn))

        obs_t_ids = torch.arange(M, device=dev).repeat_interleave(K)
        obs_m_ids = torch.arange(K, device=dev).repeat(M)
        noise_t_ids = torch.arange(N, device=dev)
        ts = self.timestep_embed.to(dev)

        obs_seg = self.segment_embed(torch.ones(M * K, dtype=torch.long, device=dev))
        noise_seg = self.segment_embed(torch.full((N,), 2, dtype=torch.long, device=dev))
        cls_seg = self.segment_embed(torch.zeros(1, dtype=torch.long, device=dev))

        obs_pos = obs_seg + ts[obs_t_ids] + self.obs_modality_embed(obs_m_ids)
        noise_pos = noise_seg + ts[noise_t_ids]
        cls_pos = cls_seg + ts[torch.zeros(1, dtype=torch.long, device=dev)]

        cls = self.cls_token.expand(B, 1, d)
        seq = torch.cat([cls, obs_tok, noise_tok], dim=1)
        pos = torch.cat([cls_pos, obs_pos, noise_pos], dim=0)

        if self.pos_emb_on_qk:
            pos = pos.unsqueeze(0)
        else:
            seq = seq + pos
            pos = None

        for block in self.blocks:
            seq = block(seq, pos_emb=pos)
        cls_out = self.norm(seq[:, 0, :])
        return self.output_proj(cls_out).squeeze(-1)
