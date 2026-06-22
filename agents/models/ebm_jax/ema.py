"""Exponential Moving Average for Flax NNX models.

Ported verbatim from ``vapo.training.ema`` (``NnxEMA``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class NnxEMA:
    """Exponential Moving Average of Flax NNX model parameters."""

    def __init__(self, model: nnx.Module, decay: float = 0.999):
        self.decay = decay
        graphdef, state = nnx.split(model)
        self._graphdef = graphdef
        self.ema_state = jax.tree.map(jnp.copy, state)

    def update(self, model: nnx.Module) -> None:
        _, current_state = nnx.split(model)
        self.ema_state = jax.tree.map(
            lambda ema, cur: ema * self.decay + cur * (1.0 - self.decay),
            self.ema_state,
            current_state,
        )

    def apply(self, model: nnx.Module) -> nnx.Module:
        graphdef, _ = nnx.split(model)
        return nnx.merge(graphdef, self.ema_state)

    def state_dict(self):
        return jax.tree.map(lambda x: x, self.ema_state)

    def load_state_dict(self, state):
        self.ema_state = jax.tree.map(jnp.asarray, state)
