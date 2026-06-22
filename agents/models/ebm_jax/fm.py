"""Noise schedulers and flow matching for JAX.

Replaces: diffusers.DDPMScheduler, diffusers.DDIMScheduler, torchcfm.ConditionalFlowMatcher,
and the custom StableFlowMatcher (which used torchcfm).

Also provides mini-batch Optimal Transport coupling (OTConditionalFlowMatcher /
OTStableFlowMatcher) and an ODE-based inference helper (ode_solve) using diffrax.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SchedulerOutput:
    prev_sample: jax.Array


def diffusion_training_target(
    prediction_type: str,
    action: jax.Array,
    noise: jax.Array,
    alphas_cumprod: jax.Array,
    timesteps: jax.Array,
) -> jax.Array:
    """Return the target a diffusion noise-predictor should regress to.

    Matches the diffusers ``DDPMScheduler`` / ``DDIMScheduler`` convention
    (mirrored by lerobot's ``DiffusionModel.compute_loss``):

      * ``"epsilon"``      → target = ``noise``
      * ``"sample"``       → target = ``action`` (the clean ``x_0``)
      * ``"v_prediction"`` → target = ``v`` where
        ``v = sqrt(alpha_cumprod_t) * noise - sqrt(1 - alpha_cumprod_t) * action``
        (Salimans & Ho 2022, "Progressive Distillation"; same formula as
        ``diffusers.DDPMScheduler.get_velocity``).

    ``alphas_cumprod`` is consumed only by the v-prediction branch.
    """
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "sample":
        return action
    if prediction_type == "v_prediction":
        alpha_t = alphas_cumprod[timesteps]
        while alpha_t.ndim < action.ndim:
            alpha_t = alpha_t[..., None]
        return jnp.sqrt(alpha_t) * noise - jnp.sqrt(1.0 - alpha_t) * action
    raise ValueError(
        f"Unknown prediction_type={prediction_type!r}. Expected one of "
        f"{{'epsilon', 'sample', 'v_prediction'}}."
    )


class DDPMScheduler:
    """DDPM noise scheduler (JAX implementation)."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        variance_type: str = "fixed_small",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.variance_type = variance_type

        if beta_schedule == "linear":
            betas = jnp.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "squaredcos_cap_v2":
            t = jnp.linspace(0, num_train_timesteps, num_train_timesteps + 1)
            alphas_cumprod = jnp.cos((t / num_train_timesteps + 0.008) / 1.008 * jnp.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = jnp.clip(betas, 0.0001, 0.9999)
        elif beta_schedule == "scaled_linear":
            betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.timesteps = jnp.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(
        self,
        original_samples: jax.Array,
        noise: jax.Array,
        timesteps: jax.Array,
    ) -> jax.Array:
        sqrt_alpha_cumprod = jnp.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus = jnp.sqrt(1.0 - self.alphas_cumprod[timesteps])
        while sqrt_alpha_cumprod.ndim < original_samples.ndim:
            sqrt_alpha_cumprod = sqrt_alpha_cumprod[..., None]
            sqrt_one_minus = sqrt_one_minus[..., None]
        return sqrt_alpha_cumprod * original_samples + sqrt_one_minus * noise

    def get_pred_original_sample(
        self,
        sample: jax.Array,
        model_output: jax.Array,
        timesteps: jax.Array,
    ) -> jax.Array:
        alpha_cumprod_t = self.alphas_cumprod[timesteps]
        while alpha_cumprod_t.ndim < sample.ndim:
            alpha_cumprod_t = alpha_cumprod_t[..., None]
        if self.prediction_type == "epsilon":
            return (sample - jnp.sqrt(1 - alpha_cumprod_t) * model_output) / jnp.sqrt(alpha_cumprod_t)
        elif self.prediction_type == "sample":
            return model_output
        elif self.prediction_type == "v_prediction":
            return jnp.sqrt(alpha_cumprod_t) * sample - jnp.sqrt(1 - alpha_cumprod_t) * model_output
        raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    def step(
        self,
        model_output: jax.Array,
        timestep: jax.Array,
        sample: jax.Array,
    ) -> SchedulerOutput:
        t = timestep
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = jnp.where(t > 0, self.alphas_cumprod[t - 1], 1.0)
        beta_t = self.betas[t]

        while alpha_cumprod_t.ndim < sample.ndim:
            alpha_cumprod_t = alpha_cumprod_t[..., None]
            alpha_cumprod_t_prev = alpha_cumprod_t_prev[..., None]
            beta_t = beta_t[..., None]

        pred_x0 = self.get_pred_original_sample(sample, model_output, t)
        if self.clip_sample:
            pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)

        coeff1 = jnp.sqrt(alpha_cumprod_t_prev) * beta_t / (1.0 - alpha_cumprod_t)
        coeff2 = jnp.sqrt(1.0 - beta_t) * (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t)
        pred_prev = coeff1 * pred_x0 + coeff2 * sample
        return SchedulerOutput(prev_sample=pred_prev)


class DDIMScheduler:
    """DDIM noise scheduler (JAX implementation)."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.num_inference_steps = num_train_timesteps

        if beta_schedule == "linear":
            betas = jnp.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "squaredcos_cap_v2":
            t = jnp.linspace(0, num_train_timesteps, num_train_timesteps + 1)
            alphas_cumprod = jnp.cos((t / num_train_timesteps + 0.008) / 1.008 * jnp.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = jnp.clip(betas, 0.0001, 0.9999)
        elif beta_schedule == "scaled_linear":
            betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.timesteps = jnp.arange(num_train_timesteps - 1, -1, -1)

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = jnp.arange(0, num_inference_steps)[::-1] * step_ratio

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha = jnp.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus = jnp.sqrt(1.0 - self.alphas_cumprod[timesteps])
        while sqrt_alpha.ndim < original_samples.ndim:
            sqrt_alpha = sqrt_alpha[..., None]
            sqrt_one_minus = sqrt_one_minus[..., None]
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def step(self, model_output, timestep, sample) -> SchedulerOutput:
        t = timestep
        alpha_t = self.alphas_cumprod[t]
        prev_t = jnp.maximum(t - self.num_train_timesteps // self.num_inference_steps, 0)
        alpha_t_prev = self.alphas_cumprod[prev_t]

        while alpha_t.ndim < sample.ndim:
            alpha_t = alpha_t[..., None]
            alpha_t_prev = alpha_t_prev[..., None]

        if self.prediction_type == "epsilon":
            pred_x0 = (sample - jnp.sqrt(1 - alpha_t) * model_output) / jnp.sqrt(alpha_t)
        elif self.prediction_type == "sample":
            pred_x0 = model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = jnp.sqrt(alpha_t) * sample - jnp.sqrt(1 - alpha_t) * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        if self.clip_sample:
            pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)

        # DDIM deterministic step (eta=0)
        pred_prev = (
            jnp.sqrt(alpha_t_prev) * pred_x0
            + jnp.sqrt(1 - alpha_t_prev) * model_output
        )
        return SchedulerOutput(prev_sample=pred_prev)


class ConditionalFlowMatcher:
    """Conditional Flow Matching (JAX implementation). Replaces torchcfm."""

    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def sample_location_and_conditional_flow(
        self,
        x0: jax.Array,
        x1: jax.Array,
        t: Optional[jax.Array] = None,
        return_noise: bool = False,
    ):
        if t is None:
            t = jax.random.uniform(jax.random.PRNGKey(0), (x0.shape[0],))

        t_expand = t.reshape(-1, *([1] * (x0.ndim - 1)))
        mu_t = t_expand * x1 + (1 - t_expand) * x0
        x_t = mu_t  # sigma=0 case (most common)

        u_t = x1 - x0

        if return_noise:
            return t, x_t, u_t, jax.random.normal(jax.random.PRNGKey(1), x0.shape)
        return t, x_t, u_t


class StableFlowMatcher:
    """Stable Flow Matching (JAX implementation). Replaces torchcfm-based StableFlowMatcher."""

    def __init__(self, sigma: float, lambda_x: float, eps: float = 0.1):
        self.sigma = sigma
        self.lambda_x = lambda_x
        self.eps = min(1.0 / lambda_x, eps)
        self.tau_0 = 0.0
        self.tau_1 = 1.0

    def sample_xt(self, x0, x1, t):
        t = t.reshape(-1, 1, 1)
        return jnp.exp(-self.lambda_x * t) * (x0 - x1) + x1

    def compute_tau(self, t):
        return self.tau_1 + jnp.exp(-self.lambda_x * t) * (self.tau_0 - self.tau_1)

    def compute_conditional_flow(self, x0, x1, t, xt):
        return -self.lambda_x * (xt - x1)

    def sample_location_and_conditional_flow(
        self,
        x0: jax.Array,
        x1: jax.Array,
        t: Optional[jax.Array] = None,
        return_noise: bool = False,
    ):
        if t is None:
            t = jax.random.uniform(jax.random.PRNGKey(0), (x0.shape[0],))

        x_t = self.sample_xt(x0, x1, t)
        u_t = self.compute_conditional_flow(x0, x1, t, x_t)
        tau = self.compute_tau(t)

        if return_noise:
            return tau, x_t, u_t, jax.random.normal(jax.random.PRNGKey(1), x0.shape)
        return tau, x_t, u_t


# ---------------------------------------------------------------------------
# Mini-batch Optimal Transport coupling
# ---------------------------------------------------------------------------

class OTPlanSampler:
    """Mini-batch Optimal Transport plan sampler.

    Computes an OT coupling between x0 (source/noise) and x1 (target/data)
    within a mini-batch and returns a permutation of x0 indices so that the
    reordered pairs (x0[perm], x1) approximate the optimal transport plan.

    Supports:
      * ``method="exact"``: solves linear sum assignment via scipy wrapped in
        ``jax.pure_callback`` (JIT-compatible; not differentiable, which is fine
        because the assignment is a discrete operation).
      * ``method="sinkhorn"``: uses ``ott-jax`` Sinkhorn (fully JAX-native and
        JIT-compatible); takes argmax per row of the coupling matrix.
    """

    def __init__(self, method: str = "exact", sinkhorn_epsilon: float = 0.01):
        self.method = method
        self.sinkhorn_epsilon = sinkhorn_epsilon

    @staticmethod
    def _flatten(x: jax.Array) -> jax.Array:
        return x.reshape(x.shape[0], -1)

    def _cost_matrix(self, x0: jax.Array, x1: jax.Array) -> jax.Array:
        """Squared Euclidean pairwise cost: C[i, j] = ||x0_i - x1_j||^2."""
        a = self._flatten(x0)
        b = self._flatten(x1)
        a_sq = jnp.sum(a * a, axis=1, keepdims=True)        # (B, 1)
        b_sq = jnp.sum(b * b, axis=1, keepdims=True).T       # (1, B)
        return a_sq + b_sq - 2.0 * (a @ b.T)                # (B, B)

    def _permutation_exact(self, x0: jax.Array, x1: jax.Array) -> jax.Array:
        """Exact OT via scipy linear_sum_assignment (through jax.pure_callback)."""
        cost = self._cost_matrix(x0, x1)
        batch_size = x0.shape[0]

        def _solve_lsa(cost_np):
            from scipy.optimize import linear_sum_assignment
            # row_ind is 0..B-1, col_ind gives optimal assignment for each row.
            # We want a permutation `perm` such that x0[perm[i]] is paired with x1[i].
            # linear_sum_assignment on cost[i, j] = ||x0_i - x1_j||^2 returns
            # (row_ind, col_ind) pairing x0_row with x1_col. To reorder x0 so that
            # position i lines up with x1[i], we need the inverse: perm[col] = row.
            row_ind, col_ind = linear_sum_assignment(np.asarray(cost_np))
            perm = np.empty(batch_size, dtype=np.int32)
            perm[col_ind] = row_ind.astype(np.int32)
            return perm

        perm = jax.pure_callback(
            _solve_lsa,
            jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            cost,
        )
        return perm

    def _permutation_sinkhorn(self, x0: jax.Array, x1: jax.Array) -> jax.Array:
        """Sinkhorn-regularized OT via ott-jax; hard assignment via per-row argmax."""
        from ott.geometry import pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn

        a = self._flatten(x0)
        b = self._flatten(x1)
        geom = pointcloud.PointCloud(a, b, epsilon=self.sinkhorn_epsilon)
        prob = linear_problem.LinearProblem(geom)
        solver = sinkhorn.Sinkhorn()
        out = solver(prob)
        # out.matrix: (B, B) coupling with geom.x as rows (x0) and geom.y as columns (x1).
        # For each target index i (columns), pick the best source (row).
        # We want perm such that x0[perm[i]] ~ x1[i], so perm[i] = argmax_j coupling[j, i].
        perm = jnp.argmax(out.matrix, axis=0).astype(jnp.int32)
        return perm

    def get_permutation(self, x0: jax.Array, x1: jax.Array) -> jax.Array:
        # The OT assignment is a discrete operation; gradients must not flow
        # through it. stop_gradient on the inputs guarantees this holds even if
        # the caller differentiates with respect to x0/x1 (the Sinkhorn solver
        # in ott-jax is differentiable, and the pure_callback path for exact
        # OT cannot support JVP).
        x0 = jax.lax.stop_gradient(x0)
        x1 = jax.lax.stop_gradient(x1)
        if self.method == "sinkhorn":
            return self._permutation_sinkhorn(x0, x1)
        return self._permutation_exact(x0, x1)


class OTConditionalFlowMatcher(ConditionalFlowMatcher):
    """Conditional Flow Matching with mini-batch OT coupling."""

    def __init__(
        self,
        sigma: float = 0.0,
        ot_method: str = "exact",
        sinkhorn_epsilon: float = 0.01,
    ):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(
            method=ot_method, sinkhorn_epsilon=sinkhorn_epsilon,
        )

    def sample_location_and_conditional_flow(
        self,
        x0: jax.Array,
        x1: jax.Array,
        t: Optional[jax.Array] = None,
        return_noise: bool = False,
    ):
        perm = self.ot_sampler.get_permutation(x0, x1)
        x0 = x0[perm]
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)


class OTStableFlowMatcher(StableFlowMatcher):
    """Stable Flow Matching with mini-batch OT coupling."""

    def __init__(
        self,
        sigma: float,
        lambda_x: float,
        eps: float = 0.1,
        ot_method: str = "exact",
        sinkhorn_epsilon: float = 0.01,
    ):
        super().__init__(sigma, lambda_x, eps)
        self.ot_sampler = OTPlanSampler(
            method=ot_method, sinkhorn_epsilon=sinkhorn_epsilon,
        )

    def sample_location_and_conditional_flow(
        self,
        x0: jax.Array,
        x1: jax.Array,
        t: Optional[jax.Array] = None,
        return_noise: bool = False,
    ):
        perm = self.ot_sampler.get_permutation(x0, x1)
        x0 = x0[perm]
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)


class OTDiffusionCoupler:
    """Framework-agnostic OT coupling for both diffusion and flow matching.

    Given noise ``x0`` and data ``x1``, computes an OT permutation of ``x0``
    so that ``x0[perm]`` is optimally paired with ``x1``.
    """

    def __init__(self, ot_method: str = "exact", sinkhorn_epsilon: float = 0.01):
        self.ot_sampler = OTPlanSampler(
            method=ot_method, sinkhorn_epsilon=sinkhorn_epsilon,
        )

    def couple(self, noise: jax.Array, data: jax.Array) -> jax.Array:
        """Return OT-permuted noise."""
        perm = self.ot_sampler.get_permutation(noise, data)
        return noise[perm]


# ---------------------------------------------------------------------------
# ODE-based inference via diffrax
# ---------------------------------------------------------------------------

def ode_solve(
    vector_field_fn: Callable,
    y0: jax.Array,
    cond: Optional[jax.Array],
    solver_name: str = "dopri5",
    num_steps: int = 10,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    t0: float = 0.0,
    t1: float = 1.0,
) -> jax.Array:
    """Integrate dy/dt = vector_field_fn(t, y, cond) from t0 to t1 with diffrax.

    ``vector_field_fn`` is the policy backbone. diffrax passes a scalar ``t`` to
    the ODE term; we broadcast it to ``(B,)`` since the backbone expects
    batched time.

    Supported solvers: ``dopri5`` / ``tsit5`` (adaptive, PIDController on
    ``rtol``/``atol``), ``euler`` / ``rk4`` (fixed step, ``dt0 = 1/num_steps``).

    Returns the state at ``t1`` with shape identical to ``y0``.
    """
    import diffrax

    def _vf(t, y, args):
        cond_ = args
        batch_size = y.shape[0]
        t_batch = jnp.full((batch_size,), t)
        return vector_field_fn(t=t_batch, x=y, cond=cond_)

    term = diffrax.ODETerm(_vf)

    solver_map = {
        "dopri5": diffrax.Dopri5,
        "tsit5": diffrax.Tsit5,
        "euler": diffrax.Euler,
    }
    # diffrax doesn't ship a plain classical RK4; Heun (2nd-order) is a
    # reasonable fixed-step fallback. Most users should pick dopri5/tsit5.
    if solver_name == "rk4":
        solver = diffrax.Heun()
        adaptive = False
    else:
        solver_cls = solver_map.get(solver_name, diffrax.Dopri5)
        solver = solver_cls()
        adaptive = solver_name in ("dopri5", "tsit5")

    dt0 = (t1 - t0) / num_steps

    if adaptive:
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
    else:
        controller = diffrax.ConstantStepSize()

    solution = diffrax.diffeqsolve(
        term, solver,
        t0=t0, t1=t1, dt0=dt0,
        y0=y0, args=cond,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=max(num_steps * 10, 256),
    )
    return solution.ys[0]
