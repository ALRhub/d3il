"""Conditional flow matching + mini-batch OT coupling (PyTorch).

Port of the flow-matching subset of ``agents/models/ebm_jax/fm.py`` used by the
D3IL benchmark (``FlowMatchingConfig`` with ``is_stable=False`` and exact OT).
Diffusion (DDPM/DDIM) and StableFlowMatcher are not needed by the benchmark
configs and are intentionally omitted.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class ConditionalFlowMatcher:
    """Conditional Flow Matching: straight-line interpolation x_t and target u_t."""

    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor,
                                             t: Optional[torch.Tensor] = None):
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        t_expand = t.reshape(-1, *([1] * (x0.ndim - 1)))
        x_t = t_expand * x1 + (1 - t_expand) * x0
        u_t = x1 - x0
        return t, x_t, u_t


class OTPlanSampler:
    """Mini-batch exact Optimal Transport coupling.

    Returns a permutation ``perm`` of x0 indices so that ``x0[perm]`` is paired
    optimally with ``x1`` (squared-Euclidean cost, scipy linear_sum_assignment).
    The OT assignment is discrete, so it runs on detached CPU numpy.
    """

    def __init__(self, method: str = "exact", sinkhorn_epsilon: float = 0.01):
        self.method = method
        self.sinkhorn_epsilon = sinkhorn_epsilon

    @torch.no_grad()
    def get_permutation(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        from scipy.optimize import linear_sum_assignment
        a = x0.reshape(x0.shape[0], -1).detach().cpu().numpy()
        b = x1.reshape(x1.shape[0], -1).detach().cpu().numpy()
        a_sq = np.sum(a * a, axis=1, keepdims=True)
        b_sq = np.sum(b * b, axis=1, keepdims=True).T
        cost = a_sq + b_sq - 2.0 * (a @ b.T)
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.empty(x0.shape[0], dtype=np.int64)
        perm[col_ind] = row_ind
        return torch.from_numpy(perm).to(x0.device)


class OTConditionalFlowMatcher(ConditionalFlowMatcher):
    """CFM with mini-batch OT coupling of the noise to the data."""

    def __init__(self, sigma: float = 0.0, ot_method: str = "exact", sinkhorn_epsilon: float = 0.01):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method=ot_method, sinkhorn_epsilon=sinkhorn_epsilon)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        perm = self.ot_sampler.get_permutation(x0, x1)
        x0 = x0[perm]
        return super().sample_location_and_conditional_flow(x0, x1, t)


class OTDiffusionCoupler:
    """Framework-agnostic OT coupling: return OT-permuted noise for given data."""

    def __init__(self, ot_method: str = "exact", sinkhorn_epsilon: float = 0.01):
        self.ot_sampler = OTPlanSampler(method=ot_method, sinkhorn_epsilon=sinkhorn_epsilon)

    def couple(self, noise: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        perm = self.ot_sampler.get_permutation(noise, data)
        return noise[perm]


def make_framework(cfg):
    """Create a flow matcher from ``FlowMatchingConfig``. Returns (framework, is_fm)."""
    from agents.models.ebm_jax.config import FlowMatchingConfig
    if isinstance(cfg, FlowMatchingConfig):
        if cfg.is_stable:
            raise NotImplementedError(
                "StableFlowMatcher is not ported to the PyTorch EBM (the D3IL "
                "benchmark uses is_stable=False)."
            )
        if cfg.use_ot:
            fm = OTConditionalFlowMatcher(cfg.sigma, ot_method=cfg.ot_method,
                                          sinkhorn_epsilon=cfg.sinkhorn_epsilon)
        else:
            fm = ConditionalFlowMatcher(cfg.sigma)
        return fm, True
    raise NotImplementedError(
        f"The PyTorch EBM only supports FlowMatchingConfig, got {type(cfg).__name__}."
    )
