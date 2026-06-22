"""JAX/Flax-NNX ResNet18 image encoder matching the D3IL baseline.

This reproduces the vision encoder the other (PyTorch) baselines use
(``agents/models/vision/multi_image_obs_encoder.py`` ->
``agents/models/vision/model_getter.py:get_resnet`` -> robomimic ``VisualCore``
with ``ResNet18Conv`` backbone and ``SpatialSoftmax`` pooling) so that the
JAX EBM policy can condition on images through the *same* architecture:

    [B, H, W, C] (in [0, 1])
      -> optional ImageNet normalisation
      -> ResNet18Conv  (BatchNorm replaced by GroupNorm(C // divisor), no conv bias)
         3x96x96 -> 512x3x3
      -> 1x1 Conv(512 -> num_kp)                 (the robomimic SpatialSoftmax input conv)
      -> SpatialSoftmax soft-argmax -> [B, num_kp, 2]
      -> flatten -> [B, num_kp*2]
      -> Linear(num_kp*2 -> feature_dim)         (VisualCore feature projection)

Channels-last (NHWC) throughout, matching the rest of the ebm_jax backbone.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import ml_networks as ml
import ml_networks.jax as mlj
from flax import nnx

from .config import ResNet18EncoderConfig

# ImageNet statistics (same constants torchvision/baseline use), NHWC layout.
_IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
_IMAGENET_STD = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)


def _group_norm(channels: int, divisor: int, *, rngs: nnx.Rngs) -> nnx.GroupNorm:
    # Baseline replaces every BatchNorm2d with GroupNorm(channels // 16, channels).
    num_groups = max(1, channels // divisor)
    return nnx.GroupNorm(num_groups=num_groups, num_features=channels, rngs=rngs)


def _conv(in_c: int, out_c: int, kernel: int, stride: int, padding: int,
          *, rngs: nnx.Rngs) -> nnx.Conv:
    # ResNet convs have no bias (BN/GN provides the shift).
    return nnx.Conv(
        in_features=in_c,
        out_features=out_c,
        kernel_size=(kernel, kernel),
        strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        use_bias=False,
        rngs=rngs,
    )


class BasicBlock(nnx.Module):
    """torchvision ResNet ``BasicBlock`` (two 3x3 convs + identity/downsample)."""

    def __init__(self, in_c: int, out_c: int, stride: int, divisor: int,
                 *, rngs: nnx.Rngs):
        self.conv1 = _conv(in_c, out_c, 3, stride, 1, rngs=rngs)
        self.gn1 = _group_norm(out_c, divisor, rngs=rngs)
        self.conv2 = _conv(out_c, out_c, 3, 1, 1, rngs=rngs)
        self.gn2 = _group_norm(out_c, divisor, rngs=rngs)

        if stride != 1 or in_c != out_c:
            self.down_conv = _conv(in_c, out_c, 1, stride, 0, rngs=rngs)
            self.down_gn = _group_norm(out_c, divisor, rngs=rngs)
        else:
            self.down_conv = None
            self.down_gn = None

    def __call__(self, x: jax.Array) -> jax.Array:
        identity = x
        out = jax.nn.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.down_conv is not None:
            identity = self.down_gn(self.down_conv(x))
        return jax.nn.relu(out + identity)


class ResNet18Conv(nnx.Module):
    """ResNet18 conv trunk (no avgpool / fc), BatchNorm -> GroupNorm, NHWC.

    For a 3x96x96 input the output is 512x3x3, matching the baseline backbone.
    """

    _CHANNELS = (64, 128, 256, 512)
    _BLOCKS = (2, 2, 2, 2)
    _STRIDES = (1, 2, 2, 2)

    def __init__(self, in_channels: int, divisor: int, *, rngs: nnx.Rngs):
        # Stem: 7x7 s2 conv -> GN -> ReLU -> 3x3 s2 maxpool.
        self.conv1 = _conv(in_channels, 64, 7, 2, 3, rngs=rngs)
        self.gn1 = _group_norm(64, divisor, rngs=rngs)

        blocks = []
        prev_c = 64
        for channels, n_blocks, stride in zip(
            self._CHANNELS, self._BLOCKS, self._STRIDES
        ):
            for b in range(n_blocks):
                s = stride if b == 0 else 1
                blocks.append(BasicBlock(prev_c, channels, s, divisor, rngs=rngs))
                prev_c = channels
        # nnx.List so the module tracks the block parameters (plain python
        # lists are not registered as pytree children in flax.nnx).
        self.blocks = nnx.List(blocks)
        self.output_channels = prev_c

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.gn1(self.conv1(x)))
        # maxpool k3 s2 p1
        x = nnx.max_pool(
            x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )
        for block in self.blocks:
            x = block(x)
        return x


class ResNet18Encoder(nnx.Module):
    """ResNet18Conv + SpatialSoftmax + linear projection (baseline-equivalent).

    ``obs_shape_hwc`` is ``(H, W, C)``; ``output_dim`` exposes the feature width
    so the caller (``EBMPolicy``) can size its ``*_obs_transform`` linear layer
    exactly as it does for the conv encoder.
    """

    def __init__(self, obs_shape_hwc, cfg: ResNet18EncoderConfig, *, rngs: nnx.Rngs):
        in_c = obs_shape_hwc[2]
        self.imagenet_norm = cfg.imagenet_norm
        self.backbone = ResNet18Conv(in_c, cfg.group_norm_divisor, rngs=rngs)
        # robomimic SpatialSoftmax begins with a 1x1 conv to ``num_kp`` channels.
        self.keypoint_conv = nnx.Conv(
            in_features=self.backbone.output_channels,
            out_features=cfg.num_kp,
            kernel_size=(1, 1),
            rngs=rngs,
        )
        self.spatial_softmax = mlj.SpatialSoftmax(
            ml.SpatialSoftmaxConfig(temperature=cfg.temperature, is_argmax=False)
        )
        self.linear = nnx.Linear(cfg.num_kp * 2, cfg.feature_dim, rngs=rngs)
        self.output_dim = cfg.feature_dim

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, H, W, C), assumed in [0, 1] (matches baseline preprocessing).
        if self.imagenet_norm:
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = self.backbone(x)            # (B, h, w, 512)
        x = self.keypoint_conv(x)       # (B, h, w, num_kp)
        kp = self.spatial_softmax(x)    # (B, num_kp, 2)
        kp = kp.reshape(kp.shape[0], -1)  # (B, num_kp*2)
        return self.linear(kp)          # (B, feature_dim)
