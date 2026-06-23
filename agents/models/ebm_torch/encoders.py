"""Image encoders for the PyTorch EBM policy.

``ResNet18Encoder`` matches the D3IL baseline vision encoder (torchvision-style
ResNet18 conv trunk with BatchNorm->GroupNorm, no conv bias, + SpatialSoftmax
keypoint pooling + a linear projection), mirroring the JAX
``agents/models/ebm_jax/resnet_encoder.py``. Inputs are NCHW in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.models.ebm_jax.config import ResNet18EncoderConfig

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _gn(channels: int, divisor: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, channels // divisor), channels)


def _conv(in_c, out_c, k, s, p):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)


class _BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, divisor):
        super().__init__()
        self.conv1 = _conv(in_c, out_c, 3, stride, 1)
        self.gn1 = _gn(out_c, divisor)
        self.conv2 = _conv(out_c, out_c, 3, 1, 1)
        self.gn2 = _gn(out_c, divisor)
        if stride != 1 or in_c != out_c:
            self.down_conv = _conv(in_c, out_c, 1, stride, 0)
            self.down_gn = _gn(out_c, divisor)
        else:
            self.down_conv = None

    def forward(self, x):
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.down_conv is not None:
            identity = self.down_gn(self.down_conv(x))
        return F.relu(out + identity)


class ResNet18Conv(nn.Module):
    """ResNet18 conv trunk (no avgpool/fc); 3x96x96 -> 512x3x3."""

    _CHANNELS = (64, 128, 256, 512)
    _BLOCKS = (2, 2, 2, 2)
    _STRIDES = (1, 2, 2, 2)

    def __init__(self, in_channels: int, divisor: int):
        super().__init__()
        self.conv1 = _conv(in_channels, 64, 7, 2, 3)
        self.gn1 = _gn(64, divisor)
        blocks = []
        prev = 64
        for ch, n, stride in zip(self._CHANNELS, self._BLOCKS, self._STRIDES):
            for b in range(n):
                blocks.append(_BasicBlock(prev, ch, stride if b == 0 else 1, divisor))
                prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.output_channels = prev

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x


class SpatialSoftmax(nn.Module):
    """Soft-argmax keypoint pooling: (B, C, H, W) -> (B, C, 2) in [-1, 1]."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, W, device=x.device),
            torch.linspace(-1.0, 1.0, H, device=x.device),
            indexing="xy",
        )
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        feat = (x / self.temperature).reshape(B, C, H * W)
        sm = F.softmax(feat, dim=-1)
        exp_x = torch.sum(pos_x * sm, dim=-1, keepdim=True)
        exp_y = torch.sum(pos_y * sm, dim=-1, keepdim=True)
        return torch.cat([exp_x, exp_y], dim=-1)  # (B, C, 2)


class ResNet18Encoder(nn.Module):
    """ResNet18Conv + 1x1 keypoint conv + SpatialSoftmax + linear projection."""

    def __init__(self, obs_shape_chw, cfg: ResNet18EncoderConfig):
        super().__init__()
        in_c = obs_shape_chw[0]
        self.imagenet_norm = cfg.imagenet_norm
        self.backbone = ResNet18Conv(in_c, cfg.group_norm_divisor)
        self.keypoint_conv = nn.Conv2d(self.backbone.output_channels, cfg.num_kp, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax(cfg.temperature)
        self.linear = nn.Linear(cfg.num_kp * 2, cfg.feature_dim)
        self.output_dim = cfg.feature_dim
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).reshape(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) in [0, 1]
        if self.imagenet_norm:
            x = (x - self._mean) / self._std
        x = self.backbone(x)
        x = self.keypoint_conv(x)
        kp = self.spatial_softmax(x)              # (B, num_kp, 2)
        return self.linear(kp.reshape(kp.shape[0], -1))


class SharedResNetEncoder(nn.Module):
    """Per-camera image encoder that reuses the *exact* baseline vision module.

    For a fair comparison with the other PyTorch baselines, this wraps the
    shared ``agents.models.vision.model_getter.get_resnet`` (robomimic
    ``VisualCore``: ResNet18Conv + SpatialSoftmax(num_kp=32) + Linear) and
    applies the same BatchNorm->GroupNorm(//16) swap and ImageNet normalisation
    that ``MultiImageObsEncoder`` (use_group_norm=True, imagenet_norm=True)
    applies for every baseline. The result is byte-identical to the encoder the
    baselines use, so the conv backbone is not a confound in the comparison.
    """

    def __init__(self, obs_shape_chw, cfg: ResNet18EncoderConfig):
        super().__init__()
        import torchvision
        from agents.models.vision.model_getter import get_resnet
        from agents.models.common.pytorch_util import replace_submodules

        c, h, w = obs_shape_chw
        model = get_resnet([c, h, w], output_size=cfg.feature_dim)
        # BatchNorm -> GroupNorm, exactly as MultiImageObsEncoder(use_group_norm=True).
        model = replace_submodules(
            root_module=model,
            predicate=lambda m: isinstance(m, nn.BatchNorm2d),
            func=lambda m: nn.GroupNorm(
                num_groups=max(1, m.num_features // cfg.group_norm_divisor),
                num_channels=m.num_features),
        )
        self.rgb_model = model
        self.output_dim = cfg.feature_dim
        self.imagenet_norm = bool(cfg.imagenet_norm)
        if self.imagenet_norm:
            self.normalizer = torchvision.transforms.Normalize(
                mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) in [0, 1]
        if self.imagenet_norm:
            x = self.normalizer(x)
        return self.rgb_model(x)


def make_image_encoder(cfg, obs_shape_chw, enc_cfg) -> nn.Module:
    """Build the image encoder. Uses the shared baseline vision backbone
    (``get_resnet`` + GroupNorm + ImageNet norm) so the EBM's image encoder is
    identical to the other baselines."""
    if isinstance(enc_cfg, ResNet18EncoderConfig):
        return SharedResNetEncoder(obs_shape_chw, enc_cfg)
    raise NotImplementedError(
        "The PyTorch EBM image encoder only supports ResNet18EncoderConfig; "
        f"got {type(enc_cfg).__name__}. Use ResNet18EncoderConfig in the vision config."
    )
