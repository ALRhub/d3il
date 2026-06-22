"""JAX / Flax-NNX two-phase EBM policy, ported from vapo-aif for D3IL.

The neural-network implementation (backbone, flow matching, energy model) is
kept identical to ``vapo.policy`` — it uses ml-networks + jax + flax.nnx. Only
the training/eval harness is adapted to D3IL's PyTorch benchmark interface.
"""
from .config import (
    DiffuserConfig,
    DiTConfig,
    EBMConfig,
    EBMPolicyConfig,
    FlowMatchingConfig,
    PosEmbConfig,
    UNetConfig,
)
from .policy_ebm import EBMPolicy

__all__ = [
    "EBMPolicy",
    "EBMPolicyConfig",
    "EBMConfig",
    "DiTConfig",
    "UNetConfig",
    "PosEmbConfig",
    "DiffuserConfig",
    "FlowMatchingConfig",
]
