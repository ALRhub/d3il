"""Policy backbone architectures in JAX/Flax NNX.

All 1D operations use channels-last format (B, T, C) which is JAX native.
Replaces PyTorch nn.Module classes with nnx.Module.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import ml_networks as ml
import ml_networks.jax as mlj
import numpy as np
from einops import rearrange
from flax import nnx

from .config import UNetConfig, TransformerConfig, PosEmbConfig, DiTConfig, RNNConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_activation(name: str):
    _MAP = {
        "mish": jax.nn.mish, "Mish": jax.nn.mish,
        "relu": jax.nn.relu, "ReLU": jax.nn.relu,
        "gelu": jax.nn.gelu, "GELU": jax.nn.gelu,
        "silu": jax.nn.silu, "SiLU": jax.nn.silu,
        "swish": jax.nn.silu, "Swish": jax.nn.silu,
        "tanh": jnp.tanh, "Tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid, "Sigmoid": jax.nn.sigmoid,
        "identity": lambda x: x, "Identity": lambda x: x,
        "none": lambda x: x,
    }
    return _MAP.get(name, lambda x: x)


def sinusoidal_positional_encoding(
    length: int, d_model: int, base: float = 10_000.0
) -> jax.Array:
    """Returns (1, length, d_model) sinusoidal positional encoding."""
    position = jnp.arange(length)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-np.log(base) / d_model))
    pe = jnp.zeros((length, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe[None]  # (1, L, d_model)


def modulate(x, shift, scale):
    return x * (1 + jnp.expand_dims(scale, 1)) + jnp.expand_dims(shift, 1)


# ---------------------------------------------------------------------------
# Sinusoidal Positional Embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nnx.Module):
    def __init__(self, dim: int, scale: float):
        self.dim = dim
        self.scale = scale

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x * self.scale
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class SoftmaxTransformation(nnx.Module):
    """JAX version of ml.SoftmaxTransformation for time embedding."""
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.centers = nnx.Param(jax.random.normal(rngs.params(), (cfg.vector,)))
        self.log_scale = nnx.Param(jnp.zeros((cfg.vector,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x[..., None]
        distances = -((x - self.centers.value) ** 2) * jax.nn.softplus(self.log_scale.value)
        return jax.nn.softmax(distances, axis=-1)


def make_time_emb(cfg, *, rngs: nnx.Rngs):
    """Create time embedding module from config (PosEmbConfig or SoftmaxTransConfig)."""
    if isinstance(cfg, PosEmbConfig):
        return SinusoidalPosEmb(cfg.vector, cfg.scale)
    else:
        return SoftmaxTransformation(cfg, rngs=rngs)


# ---------------------------------------------------------------------------
# 1D Convolution Blocks (channels-last: B, T, C)
# ---------------------------------------------------------------------------

class Conv1dBlock(nnx.Module):
    """Conv1d -> GroupNorm -> Activation -> Dropout (channels-last)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ml.ConvConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm_first = cfg.norm_first
        norm_channels = in_channels if cfg.norm_first else out_channels

        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(cfg.kernel_size,),
            strides=(cfg.stride,),
            padding=((cfg.padding, cfg.padding),),
            input_dilation=(cfg.dilation,),
            feature_group_count=cfg.groups,
            use_bias=cfg.bias,
            rngs=rngs,
        )

        if cfg.norm == "group":
            num_groups = cfg.norm_cfg.get("num_groups", 8)
            self.norm = nnx.GroupNorm(
                num_groups=min(num_groups, norm_channels),
                num_features=norm_channels,
                rngs=rngs,
            )
        elif cfg.norm == "layer":
            self.norm = nnx.LayerNorm(num_features=norm_channels, rngs=rngs)
        else:
            self.norm = None

        self.activation = get_activation(cfg.activation)
        self.dropout_rate = cfg.dropout

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        # x: (B, T, C)
        if self.norm_first and self.norm is not None:
            x = self.norm(x)
        x = self.conv(x)
        if not self.norm_first and self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        if self.dropout_rate > 0 and not deterministic:
            # Simple dropout without RNG tracking (for non-JIT paths)
            pass  # dropout handled at higher level if needed
        return x


class Downsample1d(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=dim, out_features=dim,
            kernel_size=(3,), strides=(2,), padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class Upsample1d(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.conv = nnx.ConvTranspose(
            in_features=dim, out_features=dim,
            kernel_size=(4,), strides=(2,), padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Conditional Residual Block with FiLM
# ---------------------------------------------------------------------------

class ConditionalResidualBlock1D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        conv_cfg: ml.ConvConfig,
        cond_predict_scale: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.first_conv = Conv1dBlock(in_channels, out_channels, conv_cfg, rngs=rngs)
        self.last_conv = Conv1dBlock(out_channels, out_channels, conv_cfg, rngs=rngs)

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_linear = nnx.Linear(cond_dim, cond_channels, rngs=rngs)

        if in_channels != out_channels:
            self.residual_conv = nnx.Conv(
                in_features=in_channels, out_features=out_channels,
                kernel_size=(1,), rngs=rngs,
            )
        else:
            self.residual_conv = None

    def __call__(self, x: jax.Array, cond: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """x: (B, T, C_in), cond: (B, cond_dim)"""
        out = self.first_conv(x, deterministic=deterministic)
        embed = self.cond_linear(cond)  # (B, cond_channels)
        embed = embed[:, None, :]  # (B, 1, cond_channels) for broadcasting

        if self.cond_predict_scale:
            scale = embed[..., :self.out_channels]
            bias = embed[..., self.out_channels:]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.last_conv(out, deterministic=deterministic)
        residual = self.residual_conv(x) if self.residual_conv is not None else x
        return out + residual


# ---------------------------------------------------------------------------
# Attention (1D, channels-last)
# ---------------------------------------------------------------------------

class QKVAttention(nnx.Module):
    def __init__(self, n_heads: int):
        self.n_heads = n_heads

    def __call__(self, qkv: jax.Array) -> jax.Array:
        """qkv: (B, T, 3*C) -> (B, T, C)"""
        B, T, width = qkv.shape
        ch = width // (3 * self.n_heads)
        # Reshape to (B, n_heads, T, 3*ch) then split
        qkv = qkv.reshape(B, T, self.n_heads, 3 * ch).transpose(0, 2, 1, 3)  # (B, H, T, 3*ch)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, H, T, ch)

        scale = 1.0 / jnp.sqrt(jnp.sqrt(ch).astype(q.dtype))
        weight = jnp.einsum("bhtc,bhsc->bhts", q * scale, k * scale)
        weight = jax.nn.softmax(weight - weight.max(axis=-1, keepdims=True), axis=-1)
        out = jnp.einsum("bhts,bhsc->bhtc", weight, v)  # (B, H, T, ch)
        return out.transpose(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, C)


class Attention1d(nnx.Module):
    def __init__(self, channels: int, nhead: int, *, rngs: nnx.Rngs):
        self.channels = channels
        self.num_heads = nhead

        # QKV projection (kernel=1 conv = linear)
        self.norm = nnx.GroupNorm(
            num_groups=min(8, channels),
            num_features=channels,
            rngs=rngs,
        )
        self.qkv = nnx.Linear(channels, channels * 3, rngs=rngs)
        self.attention = QKVAttention(nhead)
        self.proj_out = nnx.Linear(channels, channels, rngs=rngs)

    def __call__(self, x: jax.Array, *args) -> jax.Array:
        """x: (B, T, C) -> (B, T, C)"""
        h = self.norm(x)
        qkv = self.qkv(h)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h


# ---------------------------------------------------------------------------
# Identity placeholder
# ---------------------------------------------------------------------------

class Identity(nnx.Module):
    def __call__(self, x, *args, **kwargs):
        return x


# ---------------------------------------------------------------------------
# Conditional UNet 1D Policy
# ---------------------------------------------------------------------------

class ConditionalUnet1DPolicy(nnx.Module):
    def __init__(self, cfg: UNetConfig, *, rngs: nnx.Rngs):
        all_dims = [cfg.input_dim] + list(cfg.channels)
        start_dim = cfg.channels[0]
        global_cond_dim = cfg.cond_dim * cfg.cond_step

        self.time_emb = make_time_emb(cfg.time_emb, rngs=rngs)
        self.time_encoder = mlj.MLPLayer(
            cfg.time_emb.vector, cfg.time_emb.vector, cfg.cond_cfg, rngs=rngs
        )
        cond_dim = cfg.time_emb.vector + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        # Mid modules
        self.mid_res1 = ConditionalResidualBlock1D(
            mid_dim, mid_dim, cond_dim=cond_dim,
            conv_cfg=cfg.conv_cfg, cond_predict_scale=cfg.cond_pred_scale,
            rngs=rngs,
        )
        self.mid_attn = Attention1d(mid_dim, cfg.nhead, rngs=rngs) if cfg.has_attn else Identity()
        self.mid_res2 = ConditionalResidualBlock1D(
            mid_dim, mid_dim, cond_dim=cond_dim,
            conv_cfg=cfg.conv_cfg, cond_predict_scale=cfg.cond_pred_scale,
            rngs=rngs,
        )

        # Down modules
        down_res1 = []
        down_attn = []
        down_res2 = []
        down_sample = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_res1.append(ConditionalResidualBlock1D(
                dim_in, dim_out, cond_dim=cond_dim,
                conv_cfg=cfg.conv_cfg, cond_predict_scale=cfg.cond_pred_scale,
                rngs=rngs,
            ))
            down_attn.append(
                Attention1d(dim_out, cfg.nhead, rngs=rngs) if cfg.has_attn else Identity()
            )
            down_res2.append(ConditionalResidualBlock1D(
                dim_out, dim_out, cond_dim=cond_dim,
                conv_cfg=cfg.conv_cfg, cond_predict_scale=cfg.cond_pred_scale,
                rngs=rngs,
            ))
            down_sample.append(
                Downsample1d(dim_out, rngs=rngs) if not is_last else Identity()
            )
        self.down_res1 = nnx.List(down_res1)
        self.down_attn = nnx.List(down_attn)
        self.down_res2 = nnx.List(down_res2)
        self.down_sample = nnx.List(down_sample)

        # Up modules
        up_res1 = []
        up_attn = []
        up_res2 = []
        up_sample = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_res1.append(ConditionalResidualBlock1D(
                dim_out * 2, dim_in, cond_dim=cond_dim,
                conv_cfg=cfg.conv_cfg, cond_predict_scale=cfg.cond_pred_scale,
                rngs=rngs,
            ))
            up_attn.append(
                Attention1d(dim_in, cfg.nhead, rngs=rngs) if cfg.has_attn else Identity()
            )
            up_res2.append(ConditionalResidualBlock1D(
                dim_in, dim_in, cond_dim=cond_dim,
                conv_cfg=cfg.conv_cfg, cond_predict_scale=cfg.cond_pred_scale,
                rngs=rngs,
            ))
            up_sample.append(
                Upsample1d(dim_in, rngs=rngs) if not is_last else Identity()
            )
        self.up_res1 = nnx.List(up_res1)
        self.up_attn = nnx.List(up_attn)
        self.up_res2 = nnx.List(up_res2)
        self.up_sample = nnx.List(up_sample)

        # Final conv
        self.final_conv1 = Conv1dBlock(start_dim, start_dim, cfg.conv_cfg, rngs=rngs)
        self.final_linear = nnx.Conv(
            in_features=start_dim, out_features=cfg.input_dim,
            kernel_size=(1,), rngs=rngs,
        )

    def __call__(
        self,
        t: jax.Array,
        x: jax.Array,
        cond: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        x: (B, T, input_dim)
        t: (B,) diffusion step
        cond: (B, cond_step, cond_dim)
        output: (B, T, input_dim)
        """
        # Time embedding
        timesteps = jnp.broadcast_to(t, (x.shape[0],))
        time_feat = self.time_emb(timesteps).reshape(x.shape[0], -1)
        global_feature = self.time_encoder(time_feat)

        if cond is not None:
            # Flatten cond: (B, cond_step, cond_dim) -> (B, cond_step*cond_dim)
            cond_flat = cond.reshape(cond.shape[0], -1)
            global_feature = jnp.concatenate([global_feature, cond_flat], axis=-1)

        # Down path (channels-last: B, T, C)
        h_list = []
        for i in range(len(self.down_res1)):
            x = self.down_res1[i](x, global_feature)
            x = self.down_attn[i](x)
            x = self.down_res2[i](x, global_feature)
            h_list.append(x)
            x = self.down_sample[i](x)

        # Mid
        x = self.mid_res1(x, global_feature)
        x = self.mid_attn(x)
        x = self.mid_res2(x, global_feature)

        # Up path
        for i in range(len(self.up_res1)):
            x = jnp.concatenate([x, h_list.pop()], axis=-1)
            x = self.up_res1[i](x, global_feature)
            x = self.up_attn[i](x)
            x = self.up_res2[i](x, global_feature)
            x = self.up_sample[i](x)

        # Final
        x = self.final_conv1(x)
        x = self.final_linear(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder / Decoder Layers
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm_first = norm_first
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

        # Self-attention via manual QKV
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Feedforward
        self.ff1 = nnx.Linear(d_model, dim_ff, rngs=rngs)
        self.ff2 = nnx.Linear(dim_ff, d_model, rngs=rngs)
        self.act = get_activation(activation)

    def _self_attn(
        self, x: jax.Array, mask: Optional[jax.Array] = None,
        pos_emb: Optional[jax.Array] = None,
    ) -> jax.Array:
        B, T, D = x.shape
        qk_in = x + pos_emb if pos_emb is not None else x
        q = self.q_proj(qk_in).reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(qk_in).reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            attn = attn + mask
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, T, D))

    def __call__(
        self, x: jax.Array, mask: Optional[jax.Array] = None,
        pos_emb: Optional[jax.Array] = None,
    ) -> jax.Array:
        if self.norm_first:
            x = x + self._self_attn(self.norm1(x), mask, pos_emb=pos_emb)
            x = x + self.ff2(self.act(self.ff1(self.norm2(x))))
        else:
            x = self.norm1(x + self._self_attn(x, mask, pos_emb=pos_emb))
            x = self.norm2(x + self.ff2(self.act(self.ff1(x))))
        return x


class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm_first = norm_first
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention
        self.self_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.self_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.self_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.self_out = nnx.Linear(d_model, d_model, rngs=rngs)

        # Cross-attention
        self.cross_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.cross_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.cross_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.cross_out = nnx.Linear(d_model, d_model, rngs=rngs)

        # Feedforward
        self.ff1 = nnx.Linear(d_model, dim_ff, rngs=rngs)
        self.ff2 = nnx.Linear(dim_ff, d_model, rngs=rngs)
        self.act = get_activation(activation)

    def _mha(self, q_in, k_in, v_in, q_proj, k_proj, v_proj, out_proj,
             mask=None, q_pos=None, k_pos=None):
        if q_pos is not None:
            q_in = q_in + q_pos
        if k_pos is not None:
            k_in = k_in + k_pos
        B, Tq, D = q_in.shape
        Tk = k_in.shape[1]
        q = q_proj(q_in).reshape(B, Tq, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = k_proj(k_in).reshape(B, Tk, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = v_proj(v_in).reshape(B, Tk, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            attn = attn + mask
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        return out_proj(out.transpose(0, 2, 1, 3).reshape(B, Tq, D))

    def __call__(
        self,
        tgt: jax.Array,
        memory: jax.Array,
        tgt_mask: Optional[jax.Array] = None,
        tgt_pos: Optional[jax.Array] = None,
        memory_pos: Optional[jax.Array] = None,
    ) -> jax.Array:
        if self.norm_first:
            x = tgt
            h = self.norm1(x)
            x = x + self._mha(h, h, h,
                              self.self_q, self.self_k, self.self_v, self.self_out,
                              tgt_mask, q_pos=tgt_pos, k_pos=tgt_pos)
            x = x + self._mha(self.norm2(x), memory, memory,
                              self.cross_q, self.cross_k, self.cross_v, self.cross_out,
                              q_pos=tgt_pos, k_pos=memory_pos)
            x = x + self.ff2(self.act(self.ff1(self.norm3(x))))
        else:
            x = tgt
            x = self.norm1(x + self._mha(x, x, x,
                                        self.self_q, self.self_k, self.self_v, self.self_out,
                                        tgt_mask, q_pos=tgt_pos, k_pos=tgt_pos))
            x = self.norm2(x + self._mha(x, memory, memory,
                                        self.cross_q, self.cross_k, self.cross_v, self.cross_out,
                                        q_pos=tgt_pos, k_pos=memory_pos))
            x = self.norm3(x + self.ff2(self.act(self.ff1(x))))
        return x


# ---------------------------------------------------------------------------
# Conditional Transformer
# ---------------------------------------------------------------------------

class ConditionalTransformer(nnx.Module):
    def __init__(
        self,
        pred_obs_action: bool,
        cfg: TransformerConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        T = cfg.horizon + pred_obs_action * cfg.cond_step
        T_cond = cfg.cond_step

        self.input_emb = nnx.Linear(cfg.input_dim, cfg.d_model, rngs=rngs)

        if cfg.pos_emb_param:
            self.pos_emb = nnx.Param(jnp.zeros((1, T, cfg.d_model)))
        else:
            self.pos_emb = sinusoidal_positional_encoding(T, cfg.d_model)

        self.time_emb = make_time_emb(cfg.time_emb, rngs=rngs)
        self.time_encoder = mlj.MLPLayer(
            cfg.time_emb.vector, cfg.d_model, cfg.cond_cfg, rngs=rngs
        )

        if cfg.pos_emb_param:
            self.cond_pos_emb = nnx.Param(jnp.zeros((1, T_cond, cfg.d_model)))
        else:
            self.cond_pos_emb = sinusoidal_positional_encoding(T_cond, cfg.d_model)

        self.encoder_only = cfg.encoder_only

        if not cfg.encoder_only:
            self.enc_layers = nnx.List([
                TransformerEncoderLayer(
                    cfg.d_model, cfg.nhead, cfg.dim_ff,
                    dropout=cfg.dropout, activation=cfg.hidden_activation,
                    norm_first=True, rngs=rngs,
                ) for _ in range(cfg.n_enc_layers)
            ])
            self.dec_layers = nnx.List([
                TransformerDecoderLayer(
                    cfg.d_model, cfg.nhead, cfg.dim_ff,
                    dropout=cfg.dropout, activation=cfg.hidden_activation,
                    norm_first=True, rngs=rngs,
                ) for _ in range(cfg.n_layers)
            ])
        else:
            self.enc_layers = nnx.List([
                TransformerEncoderLayer(
                    cfg.d_model, cfg.nhead, cfg.dim_ff,
                    dropout=cfg.dropout, activation=cfg.hidden_activation,
                    norm_first=True, rngs=rngs,
                ) for _ in range(cfg.n_layers)
            ])

        # Causal mask
        if cfg.is_causal:
            mask = jnp.triu(jnp.full((T, T), float("-inf")), k=1)
            self.mask = mask
        else:
            self.mask = None

        self.ln_f = nnx.LayerNorm(num_features=cfg.d_model, rngs=rngs)
        self.head = nnx.Linear(cfg.d_model, cfg.input_dim, rngs=rngs)

        self.T = T
        self.T_cond = T_cond

    def __call__(
        self,
        t: jax.Array,
        x: jax.Array,
        cond: jax.Array,
    ) -> jax.Array:
        """x: (B,T,input_dim), t: (B,), cond: (B,T',cond_dim)"""
        timesteps = jnp.broadcast_to(t, (x.shape[0],))
        time_emb = self.time_emb(timesteps).reshape(x.shape[0], -1)[:, None, :]
        time_emb = self.time_encoder(time_emb)  # (B, 1, d_model)

        input_emb = self.input_emb(x)

        pos_emb = self.pos_emb if not isinstance(self.pos_emb, nnx.Param) else self.pos_emb.value
        cond_pos_emb = self.cond_pos_emb if not isinstance(self.cond_pos_emb, nnx.Param) else self.cond_pos_emb.value

        if self.encoder_only:
            position_emb = pos_emb[:, :self.T, :]
            cond_pe = cond_pos_emb[:, :self.T_cond, :]
            goal_cond, obs_cond = cond[:, :-self.T_cond, :], cond[:, -self.T_cond:, :]

            if self.cfg.pos_emb_on_qk:
                # Build per-token pos_emb for [time(1), goal(?), obs(T_cond), input(T)]
                D = pos_emb.shape[-1]
                goal_len = goal_cond.shape[1]
                parts = [
                    jnp.zeros((1, 1, D)),                 # time token
                    jnp.zeros((1, goal_len, D)),           # goal tokens
                    cond_pe,                               # obs tokens
                    position_emb,                          # input tokens
                ]
                seq_pos = jnp.concatenate(parts, axis=1)
            else:
                input_emb = input_emb + position_emb
                obs_cond = obs_cond + cond_pe
                seq_pos = None

            cond_emb = jnp.concatenate([goal_cond, obs_cond], axis=1)
            token_emb = jnp.concatenate([time_emb, cond_emb, input_emb], axis=1)
            x = token_emb
            for layer in self.enc_layers:
                x = layer(x, self.mask, pos_emb=seq_pos)
            x = x[:, 1:, :]
        else:
            cond_pe = cond_pos_emb[:, :self.T_cond, :]
            goal_cond, obs_cond = cond[:, :-self.T_cond, :], cond[:, -self.T_cond:, :]

            if self.cfg.pos_emb_on_qk:
                D = pos_emb.shape[-1]
                # Build enc_pos matching encoder sequence structure
                enc_pos_parts = []
                if not self.cfg.goal_for_decoder and goal_cond.shape[1] > 0:
                    obs_cond_enc = jnp.concatenate([goal_cond, obs_cond], axis=1)
                    enc_pos_parts.append(jnp.zeros((1, goal_cond.shape[1], D)))
                    enc_pos_parts.append(cond_pe)
                else:
                    obs_cond_enc = obs_cond
                    enc_pos_parts.append(cond_pe)
                if not self.cfg.time_for_decoder:
                    enc_pos_parts.insert(0, jnp.zeros((1, 1, D)))  # time token
                enc_pos = jnp.concatenate(enc_pos_parts, axis=1)
                cond_emb = obs_cond_enc
                if not self.cfg.time_for_decoder:
                    cond_emb = jnp.concatenate([time_emb, cond_emb], axis=1)

                memory = cond_emb
                for layer in self.enc_layers:
                    memory = layer(memory, pos_emb=enc_pos)

                t_len = input_emb.shape[1]
                position_emb = pos_emb[:, :t_len, :]
                x = input_emb  # no pos added
                # Build dec_pos matching decoder sequence structure
                dec_pos_parts = []
                if self.cfg.goal_for_decoder:
                    x = jnp.concatenate([goal_cond, x], axis=1)
                    dec_pos_parts.append(jnp.zeros((1, goal_cond.shape[1], D)))
                if self.cfg.time_for_decoder:
                    x = jnp.concatenate([time_emb, x], axis=1)
                    dec_pos_parts.insert(0, jnp.zeros((1, 1, D)))
                dec_pos_parts.append(position_emb)
                dec_pos = jnp.concatenate(dec_pos_parts, axis=1)

                for layer in self.dec_layers:
                    x = layer(x, memory, self.mask, tgt_pos=dec_pos, memory_pos=enc_pos)
            else:
                obs_cond = obs_cond + cond_pe
                if not self.cfg.goal_for_decoder and goal_cond.shape[1] > 0:
                    obs_cond = jnp.concatenate([goal_cond, obs_cond], axis=1)
                cond_emb = obs_cond
                if not self.cfg.time_for_decoder:
                    cond_emb = jnp.concatenate([time_emb, cond_emb], axis=1)

                memory = cond_emb
                for layer in self.enc_layers:
                    memory = layer(memory)

                t_len = input_emb.shape[1]
                position_emb = pos_emb[:, :t_len, :]
                x = input_emb + position_emb
                if self.cfg.goal_for_decoder:
                    x = jnp.concatenate([goal_cond, x], axis=1)
                if self.cfg.time_for_decoder:
                    x = jnp.concatenate([time_emb, x], axis=1)

                for layer in self.dec_layers:
                    x = layer(x, memory, self.mask)

            if self.cfg.time_for_decoder:
                x = x[:, 1:, :]
            if self.cfg.goal_for_decoder and goal_cond.shape[1] > 0:
                x = x[:, 1:, :]

        x = self.ln_f(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# DiT (Diffusion Transformer)
# ---------------------------------------------------------------------------

class DiTBlock(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        n_cond_step: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                    use_bias=False, use_scale=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                    use_bias=False, use_scale=False, rngs=rngs)

        # Self attention
        self.nhead = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.attn_out = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_fc1 = nnx.Linear(hidden_size, mlp_hidden, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(mlp_hidden, hidden_size, rngs=rngs)

        # adaLN modulation
        self.ada_linear = nnx.Linear(hidden_size * n_cond_step, 6 * hidden_size, use_bias=True, rngs=rngs)

    def _self_attn(self, x, pos_emb=None):
        B, T, D = x.shape
        qk_in = x + pos_emb if pos_emb is not None else x
        q = self.q_proj(qk_in).reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(qk_in).reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        return self.attn_out(out.transpose(0, 2, 1, 3).reshape(B, T, D))

    def __call__(self, x: jax.Array, c: jax.Array, pos_emb: Optional[jax.Array] = None) -> jax.Array:
        ada = jax.nn.silu(c)
        ada = self.ada_linear(ada)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(ada, 6, axis=-1)

        x = x + jnp.expand_dims(gate_msa, 1) * self._self_attn(modulate(self.norm1(x), shift_msa, scale_msa), pos_emb=pos_emb)
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp_fc2(jax.nn.gelu(self.mlp_fc1(h), approximate=True))
        x = x + jnp.expand_dims(gate_mlp, 1) * h
        return x


class FinalLayer(nnx.Module):
    def __init__(self, hidden_size: int, n_cond_step: int, out_dim: int, *, rngs: nnx.Rngs):
        self.norm_final = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                         use_bias=False, use_scale=False, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, out_dim, use_bias=True, rngs=rngs)
        self.ada_linear = nnx.Linear(hidden_size * n_cond_step, 2 * hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        ada = jax.nn.silu(c)
        shift, scale = jnp.split(self.ada_linear(ada), 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nnx.Module):
    def __init__(
        self,
        pred_obs_action: bool,
        goal_cond_step: int,
        pred_variance: bool,
        cfg: DiTConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        self.pred_variance = pred_variance

        T = cfg.horizon + pred_obs_action * cfg.cond_step
        T_cond = cfg.cond_step + 1 + goal_cond_step

        self.input_emb = nnx.Linear(cfg.input_dim, cfg.d_model, rngs=rngs)

        self.time_emb = make_time_emb(cfg.time_emb, rngs=rngs)
        self.time_encoder = mlj.MLPLayer(
            cfg.time_emb.vector, cfg.d_model, cfg.cond_cfg, rngs=rngs
        )
        self.learn_sigma = cfg.learn_sigma
        self.num_heads = cfg.nhead

        # Fixed positional embedding
        self.pos_embed = sinusoidal_positional_encoding(T, cfg.d_model)

        self.blocks = nnx.List([
            DiTBlock(cfg.d_model, T_cond, cfg.nhead, mlp_ratio=cfg.mlp_ratio, rngs=rngs)
            for _ in range(cfg.n_layers)
        ])
        out_dim = cfg.input_dim * 2 if pred_variance else cfg.input_dim
        self.final_layer = FinalLayer(cfg.d_model, T_cond, out_dim, rngs=rngs)

    def __call__(
        self,
        t: jax.Array,
        x: jax.Array,
        cond: jax.Array,
    ) -> jax.Array:
        """x: (B, T, input_dim), t: (B,), cond: (B, *, cond_dim)"""
        pos = self.pos_embed
        if self.cfg.pos_emb_on_qk:
            x = self.input_emb(x)
        else:
            x = self.input_emb(x) + pos  # (B, T, D)
            pos = None
        timesteps = jnp.broadcast_to(t, (x.shape[0],))
        time_emb = self.time_emb(timesteps).reshape(x.shape[0], -1)
        t_enc = self.time_encoder(time_emb)  # (B, D)
        if cond is not None:
            c = jnp.concatenate([t_enc, cond.reshape(cond.shape[0], -1)], axis=-1)
        else:
            c = t_enc

        for block in self.blocks:
            x = block(x, c, pos_emb=pos)
        x = self.final_layer(x, c)

        if self.pred_variance:
            x, var = jnp.split(x, 2, axis=-1)
            var = jnp.tanh(var)
            var = (var + 1) / 2
            x = jnp.concatenate([x, var], axis=-1)
        return x


# ---------------------------------------------------------------------------
# CDiT — Cross-Attention DiT (condition via cross-attention instead of flatten)
# ---------------------------------------------------------------------------

class CDiTBlock(nnx.Module):
    """DiT block with cross-attention conditioning (adaLN-Zero + cross-attn).

    c      = t_enc   (B, D)      — timestep encoding for adaLN modulation
    x_cond = cond    (B, H, D)   — observation condition for cross-attention
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.nhead = num_heads
        self.head_dim = hidden_size // num_heads

        # Layer norms (no affine — modulated by adaLN)
        self.norm1 = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                    use_bias=False, use_scale=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                    use_bias=False, use_scale=False, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                    use_bias=False, use_scale=False, rngs=rngs)
        self.norm_cond = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                        use_bias=False, use_scale=False, rngs=rngs)

        # Self-attention projections
        self.sa_q = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.sa_k = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.sa_v = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.sa_out = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        # Cross-attention projections
        self.ca_q = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.ca_k = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.ca_v = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.ca_out = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_fc1 = nnx.Linear(hidden_size, mlp_hidden, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(mlp_hidden, hidden_size, rngs=rngs)

        # adaLN modulation: input is t_enc (B, D), output is 11 * D
        self.ada_linear = nnx.Linear(hidden_size, 11 * hidden_size, use_bias=True, rngs=rngs)

    def _mha(self, q, k, v):
        """Multi-head attention. q: (B, Tq, D), k/v: (B, Tk, D)."""
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        q = q.reshape(B, Tq, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tk, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tk, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        return out.transpose(0, 2, 1, 3).reshape(B, Tq, -1)

    def __call__(
        self, x: jax.Array, c: jax.Array, x_cond: jax.Array,
        pos_emb: Optional[jax.Array] = None,
        cond_pos_emb: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        x:      (B, T, D)   — noisy action sequence
        c:      (B, D)       — timestep encoding
        x_cond: (B, H, D)   — observation condition
        """
        ada = jax.nn.silu(c)
        ada = self.ada_linear(ada)  # (B, 11*D)
        (shift_msa, scale_msa, gate_msa,
         shift_ca_xcond, scale_ca_xcond,
         shift_ca_x, scale_ca_x, gate_ca_x,
         shift_mlp, scale_mlp, gate_mlp) = jnp.split(ada, 11, axis=-1)

        # 1. Self-attention
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        sa_q_in = h + pos_emb if pos_emb is not None else h
        sa_k_in = sa_q_in
        sa_out = self._mha(self.sa_q(sa_q_in), self.sa_k(sa_k_in), self.sa_v(h))
        sa_out = self.sa_out(sa_out)
        x = x + jnp.expand_dims(gate_msa, 1) * sa_out

        # 2. Cross-attention (query=x, key/value=x_cond)
        q = modulate(self.norm2(x), shift_ca_x, scale_ca_x)
        kv = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        ca_q_in = q + pos_emb if pos_emb is not None else q
        ca_k_in = kv + cond_pos_emb if cond_pos_emb is not None else kv
        ca_out = self._mha(self.ca_q(ca_q_in), self.ca_k(ca_k_in), self.ca_v(kv))
        ca_out = self.ca_out(ca_out)
        x = x + jnp.expand_dims(gate_ca_x, 1) * ca_out

        # 3. MLP
        h = modulate(self.norm3(x), shift_mlp, scale_mlp)
        h = self.mlp_fc2(jax.nn.gelu(self.mlp_fc1(h), approximate=True))
        x = x + jnp.expand_dims(gate_mlp, 1) * h

        return x


class CFinalLayer(nnx.Module):
    """Final layer with cross-attention conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, out_dim: int, *, rngs: nnx.Rngs):
        self.nhead = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm_final = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                         use_bias=False, use_scale=False, rngs=rngs)
        self.norm_cond = nnx.LayerNorm(num_features=hidden_size, epsilon=1e-6,
                                        use_bias=False, use_scale=False, rngs=rngs)

        # Cross-attention
        self.ca_q = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.ca_k = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.ca_v = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.ca_out = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        self.linear = nnx.Linear(hidden_size, out_dim, use_bias=True, rngs=rngs)

        # adaLN: 4 params (shift/scale for x and cond)
        self.ada_linear = nnx.Linear(hidden_size, 4 * hidden_size, use_bias=True, rngs=rngs)

    def _mha(self, q, k, v):
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        q = q.reshape(B, Tq, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tk, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tk, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        return out.transpose(0, 2, 1, 3).reshape(B, Tq, -1)

    def __call__(
        self, x: jax.Array, c: jax.Array, x_cond: jax.Array,
        pos_emb: Optional[jax.Array] = None,
        cond_pos_emb: Optional[jax.Array] = None,
    ) -> jax.Array:
        ada = jax.nn.silu(c)
        shift_norm, scale_norm, shift_cond, scale_cond = jnp.split(self.ada_linear(ada), 4, axis=-1)

        x = modulate(self.norm_final(x), shift_norm, scale_norm)
        kv = modulate(self.norm_cond(x_cond), shift_cond, scale_cond)
        q_in = x + pos_emb if pos_emb is not None else x
        k_in = kv + cond_pos_emb if cond_pos_emb is not None else kv
        x = self._mha(self.ca_q(q_in), self.ca_k(k_in), self.ca_v(kv))
        x = self.ca_out(x)
        return self.linear(x)


class CDiT(nnx.Module):
    """Cross-Attention DiT: condition is kept as (B, H, D) and processed via cross-attention."""

    def __init__(
        self,
        pred_obs_action: bool,
        goal_cond_step: int,
        pred_variance: bool,
        cfg,
        *,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        self.pred_variance = pred_variance

        T = cfg.horizon + pred_obs_action * cfg.cond_step

        self.input_emb = nnx.Linear(cfg.input_dim, cfg.d_model, rngs=rngs)

        self.time_emb = make_time_emb(cfg.time_emb, rngs=rngs)
        self.time_encoder = mlj.MLPLayer(
            cfg.time_emb.vector, cfg.d_model, cfg.cond_cfg, rngs=rngs
        )
        self.learn_sigma = cfg.learn_sigma
        self.num_heads = cfg.nhead

        self.pos_embed = sinusoidal_positional_encoding(T, cfg.d_model)

        self.blocks = nnx.List([
            CDiTBlock(cfg.d_model, cfg.nhead, mlp_ratio=cfg.mlp_ratio, rngs=rngs)
            for _ in range(cfg.n_layers)
        ])
        out_dim = cfg.input_dim * 2 if pred_variance else cfg.input_dim
        self.final_layer = CFinalLayer(cfg.d_model, cfg.nhead, out_dim, rngs=rngs)

    def __call__(
        self,
        t: jax.Array,
        x: jax.Array,
        cond: jax.Array,
    ) -> jax.Array:
        """x: (B, T, input_dim), t: (B,), cond: (B, H, D)"""
        pos = self.pos_embed
        if self.cfg.pos_emb_on_qk:
            x = self.input_emb(x)
        else:
            x = self.input_emb(x) + pos  # (B, T, D)
            pos = None
        timesteps = jnp.broadcast_to(t, (x.shape[0],))
        time_emb = self.time_emb(timesteps).reshape(x.shape[0], -1)
        t_enc = self.time_encoder(time_emb)  # (B, D)

        # cond stays as (B, H, D) — no flatten
        for block in self.blocks:
            x = block(x, t_enc, cond, pos_emb=pos)
        x = self.final_layer(x, t_enc, cond, pos_emb=pos)

        if self.pred_variance:
            x, var = jnp.split(x, 2, axis=-1)
            var = jnp.tanh(var)
            var = (var + 1) / 2
            x = jnp.concatenate([x, var], axis=-1)
        return x


# ---------------------------------------------------------------------------
# RNN Policy
# ---------------------------------------------------------------------------

class RNNPolicy(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: RNNConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        if cfg.rnn_name == "GRU":
            self.rnn = nnx.GRUCell(
                in_features=input_dim,
                hidden_features=cfg.hidden_dim,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {cfg.rnn_name}. Only GRU is supported in JAX.")

        self.fc = mlj.MLPLayer(cfg.hidden_dim, output_dim, cfg.mlp_cfg, rngs=rngs)
        self.hidden = None

    def reset_hidden(self):
        self.hidden = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (B, input_dim) -> (B, output_dim)"""
        if self.hidden is None:
            self.hidden = jnp.zeros((x.shape[0], self.cfg.hidden_dim))

        self.hidden = self.rnn(x, self.hidden)
        # GRUCell returns (carry, output) where carry=output for GRU
        if isinstance(self.hidden, tuple):
            h = self.hidden[0]
        else:
            h = self.hidden
        return self.fc(h)


# ---------------------------------------------------------------------------
# Energy-Based Model for noise prediction
# ---------------------------------------------------------------------------

class EnergyModel(nnx.Module):
    """Energy-Based Model that scores (obs_features, initial_noise) pairs.

    Architecture: Residual MLP with LayerNorm.
    Input: concatenation of ``[obs_features_flat, initial_noise_flat]``
    Output: scalar energy per sample (lower = better match).
    """

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        input_dim = obs_dim + noise_dim
        self.input_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.layers = nnx.List([
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            for _ in range(n_layers)
        ])
        self.norms = nnx.List([
            nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
            for _ in range(n_layers)
        ])
        self.output_proj = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self, obs_features: jax.Array, noise: jax.Array,
    ) -> jax.Array:
        """
        obs_features: (B, obs_feat_dim)
        noise: (B, noise_dim) — flattened initial noise
        Returns: (B,) energy scalar
        """
        x = jnp.concatenate(
            [obs_features, noise.reshape(noise.shape[0], -1)], axis=-1,
        )
        x = jax.nn.gelu(self.input_proj(x))
        for layer, norm in zip(self.layers, self.norms):
            x = x + jax.nn.gelu(layer(norm(x)))
        return self.output_proj(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Transformer-based Energy Model
# ---------------------------------------------------------------------------

class _EnergyTransformerBlock(nnx.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0, *, rngs: nnx.Rngs):
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.attn_out = nnx.Linear(d_model, d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp_fc1 = nnx.Linear(d_model, mlp_hidden, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(mlp_hidden, d_model, rngs=rngs)

    def __call__(self, x: jax.Array, pos_emb: Optional[jax.Array] = None) -> jax.Array:
        """
        x: (B, T, D)
        pos_emb: (1, T, D) or None — added to q and k when pos_emb_on_qk is used.
        """
        # Self-attention (pre-norm). Use jax.nn.dot_product_attention so XLA
        # picks a stable attention kernel (cuDNN / FlashAttention) instead of
        # the einsum-based path, which the Triton GEMM autotuner can fail to
        # configure for long sequences with small head_dim.
        h = self.norm1(x)
        B, T, D = h.shape
        qk_in = h + pos_emb if pos_emb is not None else h
        q = self.q_proj(qk_in).reshape(B, T, self.nhead, self.head_dim)
        k = self.k_proj(qk_in).reshape(B, T, self.nhead, self.head_dim)
        v = self.v_proj(h).reshape(B, T, self.nhead, self.head_dim)
        out = jax.nn.dot_product_attention(q, k, v)  # (B, T, H, D_head)
        out = self.attn_out(out.reshape(B, T, D))
        x = x + out
        # MLP (pre-norm)
        h = self.norm2(x)
        x = x + self.mlp_fc2(jax.nn.gelu(self.mlp_fc1(h), approximate=True))
        return x


class EnergyTransformer(nnx.Module):
    """Transformer-based Energy Model with per-(modality, timestep) obs tokens
    and per-timestep action tokens.

    Sequence layout::

        [CLS]
        [obs_{modality=0, t=0}] [obs_{modality=1, t=0}] ... [obs_{K-1, 0}]
        [obs_{0, t=1}] ...                              [obs_{K-1, M-1}]
        [noise_{t=0}] [noise_{t=1}] ...                 [noise_{N-1}]

    where K = ``len(obs_modality_dims)`` (e.g. 2 for {pos, gripper}),
    M = ``obs_steps``, N = ``noise_steps`` (= policy_length).
    Sequence length is ``L = 1 + K*M + N``.

    Three additive positional embeddings identify each token:

    * **Segment PE** — CLS / obs / noise.
    * **Timestep PE** — sinusoidal index over timesteps (0..M-1 for obs,
      0..N-1 for noise; CLS gets 0).
    * **Modality PE** — learnable, one entry per obs modality.

    The CLS token output is projected to a scalar energy.
    """

    def __init__(
        self,
        obs_modality_dims: Tuple[int, ...],
        noise_token_dim: int,
        obs_steps: int,
        noise_steps: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        pos_emb_on_qk: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        if not obs_modality_dims:
            raise ValueError(
                "EnergyTransformer requires at least one obs modality "
                "(obs_modality_dims must be non-empty)."
            )
        d_model = hidden_dim
        self.d_model = d_model
        self.obs_modality_dims = tuple(int(d) for d in obs_modality_dims)
        self.num_modalities = len(self.obs_modality_dims)
        self.obs_total_dim = sum(self.obs_modality_dims)
        self.noise_token_dim = noise_token_dim
        self.obs_steps = obs_steps
        self.noise_steps = noise_steps
        self.pos_emb_on_qk = pos_emb_on_qk

        # Per-modality value projection (each modality's per-step embedding
        # → d_model). One token per (modality, timestep).
        self.obs_modality_proj = nnx.List([
            nnx.Linear(dim, d_model, rngs=rngs) for dim in self.obs_modality_dims
        ])
        # Per-timestep action projection (all action dims at a step → 1 token).
        self.noise_proj = nnx.Linear(noise_token_dim, d_model, rngs=rngs)

        # Learnable CLS token
        self.cls_token = nnx.Param(
            jax.random.normal(rngs(), (1, 1, d_model)) * 0.02,
        )

        # Segment embedding (CLS=0, obs=1, noise=2)
        self.segment_embed = nnx.Embed(num_embeddings=3, features=d_model, rngs=rngs)

        # Timestep PE (sinusoidal, shared across obs/noise)
        max_timesteps = max(obs_steps, noise_steps, 1)
        self.timestep_embed = sinusoidal_positional_encoding(max_timesteps, d_model)

        # Modality PE (learnable, per obs modality)
        self.obs_modality_embed = nnx.Embed(
            num_embeddings=self.num_modalities, features=d_model, rngs=rngs,
        )

        # Transformer blocks
        self.blocks = nnx.List([
            _EnergyTransformerBlock(d_model, nhead, mlp_ratio, rngs=rngs)
            for _ in range(n_layers)
        ])
        self.norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.output_proj = nnx.Linear(d_model, 1, rngs=rngs)

    def __call__(
        self, obs_features: jax.Array, noise: jax.Array,
    ) -> jax.Array:
        """
        obs_features: (B, obs_feat_dim) where obs_feat_dim = obs_steps * sum(obs_modality_dims)
            laid out timestep-major: modalities concatenated within each step.
        noise: (B, noise_dim) where noise_dim = noise_steps * noise_token_dim
        Returns: (B,) energy scalar
        """
        B = obs_features.shape[0]
        M = self.obs_steps
        N = self.noise_steps
        K = self.num_modalities
        D_total = self.obs_total_dim
        Dn = self.noise_token_dim

        # (B, M, D_total) — split modalities along last axis, project each.
        obs_per_step = obs_features.reshape(B, M, D_total)
        modality_tokens = []  # each (B, M, D)
        offset = 0
        for k, dim_k in enumerate(self.obs_modality_dims):
            chunk = jax.lax.dynamic_slice_in_dim(obs_per_step, offset, dim_k, axis=-1)
            modality_tokens.append(self.obs_modality_proj[k](chunk))
            offset += dim_k
        # Stack per-modality streams: (B, M, K, D) → (B, M*K, D), timestep-major
        # so that the token order is [m=0,m=1,...,m=K-1] at t=0, then t=1, ...
        obs_tok = jnp.stack(modality_tokens, axis=2).reshape(B, M * K, self.d_model)

        # Per-timestep action tokens
        noise_tok = self.noise_proj(noise.reshape(B, N, Dn))  # (B, N, D)

        # --- Positional embeddings ---
        # obs: timestep idx repeated K times per step; modality idx tiled M times
        obs_t_ids = jnp.repeat(jnp.arange(M, dtype=jnp.int32), K)  # (M*K,)
        obs_m_ids = jnp.tile(jnp.arange(K, dtype=jnp.int32), M)    # (M*K,)
        noise_t_ids = jnp.arange(N, dtype=jnp.int32)                # (N,)

        obs_seg = self.segment_embed(jnp.ones(M * K, dtype=jnp.int32))
        noise_seg = self.segment_embed(jnp.full(N, 2, dtype=jnp.int32))
        cls_seg = self.segment_embed(jnp.zeros(1, dtype=jnp.int32))

        obs_pos = (
            obs_seg
            + self.timestep_embed[0, obs_t_ids]
            + self.obs_modality_embed(obs_m_ids)
        )
        noise_pos = noise_seg + self.timestep_embed[0, noise_t_ids]
        cls_pos = cls_seg + self.timestep_embed[0, jnp.zeros(1, dtype=jnp.int32)]

        # CLS token broadcast and full sequence assembly
        cls = jnp.broadcast_to(self.cls_token.value, (B, 1, self.d_model))
        seq = jnp.concatenate([cls, obs_tok, noise_tok], axis=1)  # (B, 1+M*K+N, D)
        pos = jnp.concatenate([cls_pos, obs_pos, noise_pos], axis=0)  # (1+M*K+N, D)

        if self.pos_emb_on_qk:
            # Position info is injected into q/k at every layer
            pos = pos[None]  # (1, L, D)
        else:
            # Position info is added to token embeddings once
            seq = seq + pos
            pos = None

        # Transformer blocks
        for block in self.blocks:
            seq = block(seq, pos_emb=pos)

        # CLS output -> energy
        cls_out = self.norm(seq[:, 0, :])
        return self.output_proj(cls_out).squeeze(-1)
