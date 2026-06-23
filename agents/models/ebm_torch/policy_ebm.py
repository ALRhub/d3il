"""Two-phase EBM policy (PyTorch).

A faithful PyTorch port of ``agents/models/ebm_jax/policy_ebm.py`` covering the
path the D3IL benchmark uses:

  * Phase 1 ("generative"): an unconditional DiT flow-matching model trained with
    mini-batch OT coupling.
  * Phase 2 ("ebm"): a frozen phase-1 model + an Energy-Based Model (MLP or
    Transformer) trained with a CE or InfoNCE contrastive loss, conditioned on a
    pos (state) encoder and/or ResNet18 image encoders.

EMA weights for the policy (phase 1) and the EBM (phase 2) are kept as frozen
copies (``policy_ema`` / ``ebm_ema``) so they move with ``.to(device)`` and are
saved in the state dict; denoise / inversion / inference use these copies.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.models.ebm_jax.config import DiTConfig, EBMPolicyConfig, MLPPolicyConfig
from .backbone import DiT, EnergyModel, EnergyTransformer, MLPVelocity, build_mlp
from .encoders import make_image_encoder
from .fm import OTDiffusionCoupler, make_framework


@torch.no_grad()
def _clone_frozen(module: nn.Module) -> nn.Module:
    """Deep-copy a module, detach all params, and disable grad (an EMA shadow)."""
    ema = copy.deepcopy(module)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema_module: nn.Module, live_module: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema_module.parameters(), live_module.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for b_ema, b in zip(ema_module.buffers(), live_module.buffers()):
        b_ema.copy_(b)


class EBMPolicy(nn.Module):
    def __init__(self, cfg: EBMPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.ebm_cfg = cfg.ebm

        self._sep_grip = bool(getattr(cfg, "separate_gripper", False))
        if self._sep_grip:
            raise NotImplementedError("separate_gripper is not ported to the PyTorch EBM.")
        self._flow_action_dim = cfg.action_dim

        # --- Phase-1 policy backbone (unconditional) ---
        if cfg.phase == "ebm_only":
            self.policy = None
            self.policy_ema = None
            self.framework = None
            self.framework_cfg = None
            self.is_fm = False
            self.ot_coupler = None
        else:
            assert cfg.policy is not None, "EBMPolicyConfig.policy must be set."
            if isinstance(cfg.policy, DiTConfig):
                cfg.policy.cond_step = 0
                self.policy = DiT(cfg.pred_obs_action, goal_cond_step=0, pred_variance=False, cfg=cfg.policy)
            elif isinstance(cfg.policy, MLPPolicyConfig):
                # Plain-MLP velocity field (both-phases-MLP variant).
                self.policy = MLPVelocity(cfg.policy)
            else:
                raise NotImplementedError(
                    "The PyTorch EBM supports DiTConfig or MLPPolicyConfig policies.")
            self.policy_ema = _clone_frozen(self.policy)
            assert cfg.framework_cfg is not None
            self.framework, self.is_fm = make_framework(cfg.framework_cfg)
            self.framework_cfg = cfg.framework_cfg
            self.ot_coupler = OTDiffusionCoupler(ot_method=cfg.ot_method,
                                                 sinkhorn_epsilon=cfg.sinkhorn_epsilon)

        # --- Phase-2: observation encoders + EBM ---
        if cfg.phase in ("ebm", "ebm_only"):
            assert cfg.ebm is not None, "EBMConfig is required for phase 'ebm'."
            if cfg.use_static:
                chw = (cfg.static_obs_shape[0], cfg.static_obs_shape[1], cfg.static_obs_shape[2])
                self.static_encoder = make_image_encoder(cfg, chw, cfg.static_encoder)
                self.static_obs_transform = nn.Linear(self.static_encoder.output_dim, cfg.static_obs_dim)
            if cfg.use_gripper:
                chw = (cfg.gripper_obs_shape[0], cfg.gripper_obs_shape[1], cfg.gripper_obs_shape[2])
                self.gripper_encoder = make_image_encoder(cfg, chw, cfg.gripper_encoder)
                self.gripper_obs_transform = nn.Linear(self.gripper_encoder.output_dim, cfg.gripper_obs_dim)
            if cfg.use_pos:
                pos_in = cfg.pos_input_dim if cfg.pos_input_dim is not None else cfg.action_dim
                self.pos_encoder = build_mlp(pos_in, cfg.pos_obs_dim, cfg.pos_encoder)

            obs_modality_dims = []
            if cfg.use_pos:
                obs_modality_dims.append(cfg.pos_obs_dim)
            if cfg.use_static:
                obs_modality_dims.append(cfg.static_obs_dim)
            if cfg.use_gripper:
                obs_modality_dims.append(cfg.gripper_obs_dim)
            self._obs_feat_dim = sum(obs_modality_dims) * cfg.obs_length

            noise_dim = cfg.policy_length * self._flow_action_dim
            if cfg.ebm.arch == "transformer":
                self.ebm = EnergyTransformer(
                    obs_modality_dims=tuple(obs_modality_dims),
                    noise_token_dim=self._flow_action_dim,
                    obs_steps=cfg.obs_length, noise_steps=cfg.policy_length,
                    hidden_dim=cfg.ebm.hidden_dim, n_layers=cfg.ebm.n_layers,
                    nhead=cfg.ebm.nhead, mlp_ratio=cfg.ebm.mlp_ratio,
                    pos_emb_on_qk=cfg.ebm.pos_emb_on_qk,
                )
            else:
                self.ebm = EnergyModel(
                    obs_dim=self._obs_feat_dim, noise_dim=noise_dim,
                    hidden_dim=cfg.ebm.hidden_dim, n_layers=cfg.ebm.n_layers,
                )
            self.ebm_ema = _clone_frozen(self.ebm)

            # The phase-1 policy is frozen in phase 2 (only its EMA drives denoise).
            if self.policy is not None:
                for p in self.policy.parameters():
                    p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Parameters to optimise for the current phase
    # ------------------------------------------------------------------
    def optim_parameters(self):
        if self.cfg.phase == "generative":
            return list(self.policy.parameters())
        params = list(self.ebm.parameters())
        if self.cfg.use_static:
            params += list(self.static_encoder.parameters()) + list(self.static_obs_transform.parameters())
        if self.cfg.use_gripper:
            params += list(self.gripper_encoder.parameters()) + list(self.gripper_obs_transform.parameters())
        if self.cfg.use_pos:
            params += list(self.pos_encoder.parameters())
        return params

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Phase-1 -> phase-2 injection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_generative_from_model(self, phase1: "EBMPolicy") -> None:
        self.policy.load_state_dict(phase1.policy.state_dict())
        self.policy_ema.load_state_dict(phase1.policy_ema.state_dict())

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _encode_obs_flat(self, static_obs, gripper_obs, pos):
        cfg = self.cfg
        timestep_major = cfg.ebm is not None and cfg.ebm.arch == "transformer"
        parts = []  # each (B, T, d)
        if cfg.use_pos and pos is not None:
            pos_emb = self.pos_encoder(pos)
            if pos_emb.ndim == 2:
                pos_emb = pos_emb[:, None, :]
            parts.append(pos_emb)
        if cfg.use_static and static_obs is not None:
            B, T = static_obs.shape[:2]
            s = static_obs.reshape(B * T, *static_obs.shape[2:])
            emb = self.static_obs_transform(self.static_encoder(s))
            parts.append(emb.reshape(B, T, -1))
        if cfg.use_gripper and gripper_obs is not None:
            B, T = gripper_obs.shape[:2]
            g = gripper_obs.reshape(B * T, *gripper_obs.shape[2:])
            emb = self.gripper_obs_transform(self.gripper_encoder(g))
            parts.append(emb.reshape(B, T, -1))
        if not parts:
            B = pos.shape[0] if pos is not None else 1
            return torch.zeros(B, 1, device=self.device)
        if timestep_major:
            # (B, T, sum_d) -> (B, T*sum_d) timestep-major
            obs_per_step = torch.cat(parts, dim=-1)
            return obs_per_step.reshape(obs_per_step.shape[0], -1)
        # flat: concat each modality's flattened (B, T*d)
        flat = [p.reshape(p.shape[0], -1) for p in parts]
        return torch.cat(flat, dim=-1)

    # ------------------------------------------------------------------
    # Denoising / inversion (use the frozen policy EMA)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _denoise(self, initial_noise: torch.Tensor) -> torch.Tensor:
        traj = initial_noise
        B = traj.shape[0]
        nfe = int(self.framework_cfg.num_inference_steps)
        dt = 1.0 / nfe
        t = torch.zeros(B, device=traj.device)
        for _ in range(nfe):
            vector = self.policy_ema(t=t, x=traj, cond=None)
            traj = traj + vector * dt
            t = t + dt
        return traj

    @torch.no_grad()
    def _invert(self, actions: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        x = actions.detach()
        B = x.shape[0]
        nfe = int(num_steps or getattr(self.ebm_cfg, "invert_num_steps", None)
                  or self.framework_cfg.num_inference_steps)
        dt = 1.0 / nfe
        t = torch.ones(B, device=x.device)
        for _ in range(nfe):
            vector = self.policy_ema(t=t, x=x, cond=None)
            x = x - vector * dt
            t = t - dt
        return x.detach()

    # ------------------------------------------------------------------
    # Phase 1 loss
    # ------------------------------------------------------------------
    def compute_phase1_loss(self, action: torch.Tensor):
        noise = torch.randn_like(action)
        if self.cfg.use_uniform:
            noise = torch.empty_like(action).uniform_(-1, 1)
        noise = self.ot_coupler.couple(noise, action)
        t = torch.rand(action.shape[0], device=action.device)
        t, x_t, u_t = self.framework.sample_location_and_conditional_flow(noise, action, t)
        v_t = self.policy(t=t, x=x_t, cond=None)
        loss = ((v_t - u_t) ** 2).sum(dim=(-2, -1)).mean()
        return loss, {"loss": loss.item()}

    # ------------------------------------------------------------------
    # Phase 2 losses
    # ------------------------------------------------------------------
    def compute_phase2_loss_ce(self, static_obs, gripper_obs, action, pos):
        obs_features = self._encode_obs_flat(static_obs, gripper_obs, pos)
        B = action.shape[0]
        K = self.ebm_cfg.num_samples_k
        T, D = action.shape[1], self._flow_action_dim

        noises = torch.randn(B * K, T, D, device=action.device)
        generated = self._denoise(noises).reshape(B, K, T, D)        # no grad
        noises = noises.reshape(B, K, -1)

        gt = action[:, None, :, :]
        mse = ((generated - gt) ** 2).mean(dim=(-2, -1))             # (B, K)

        obs_expanded = obs_features[:, None, :].expand(B, K, obs_features.shape[-1]).reshape(B * K, -1)
        energies = self.ebm(obs_expanded, noises.reshape(B * K, -1)).reshape(B, K)

        target = F.softmax(-mse / self.ebm_cfg.temperature, dim=-1)
        pred_logp = F.log_softmax(-energies / self.ebm_cfg.temperature, dim=-1)
        ce = (-(target * pred_logp).sum(dim=-1)).mean()
        ld = {"loss": ce.item(), "ebm_ce": ce.item(),
              "mse_mean": mse.mean().item(), "mse_min": mse.min(dim=-1).values.mean().item()}
        return ce, ld

    def _langevin_negatives(self, obs_features, K_neg, steps, step_size):
        B = obs_features.shape[0]
        T, D = self.cfg.policy_length, self._flow_action_dim
        noise = torch.randn(B * K_neg, T * D, device=obs_features.device)
        obs_expanded = obs_features[:, None, :].expand(B, K_neg, obs_features.shape[-1]).reshape(B * K_neg, -1)
        sqrt_step = (2.0 * step_size) ** 0.5 * 0.01
        for _ in range(int(steps)):
            noise = noise.detach().requires_grad_(True)
            energy = self.ebm(obs_expanded, noise).sum()
            grad = torch.autograd.grad(energy, noise)[0]
            noise = noise - step_size * grad
            noise = noise + sqrt_step * torch.randn_like(noise)
        return noise.detach().reshape(B, K_neg, T, D)

    def compute_phase2_loss_infonce(self, static_obs, gripper_obs, action, pos, step=None):
        obs_features = self._encode_obs_flat(static_obs, gripper_obs, pos)
        B = action.shape[0]
        T, D = self.cfg.policy_length, self._flow_action_dim
        K_neg = int(self.ebm_cfg.infonce_num_negatives)

        noise_pos = self._invert(action)                            # (B, T, D), no grad

        neg_source = getattr(self.ebm_cfg, "infonce_negative_source", "random")
        warmup = int(getattr(self.ebm_cfg, "infonce_negative_warmup_steps", 0))
        use_langevin_neg = (neg_source == "langevin"
                            and not (warmup > 0 and step is not None and step < warmup))
        if use_langevin_neg:
            noise_neg = self._langevin_negatives(
                obs_features, K_neg, self.ebm_cfg.infonce_langevin_steps, self.ebm_cfg.langevin_step_size)
        else:
            noise_neg = torch.randn(B, K_neg, T, D, device=action.device)

        cand_list = [noise_pos[:, None]]
        K_inbatch = 0
        if bool(getattr(self.ebm_cfg, "infonce_use_inbatch_inverted", False)) and B > 1:
            rows = torch.arange(B, device=action.device)[:, None]
            cols = torch.arange(B - 1, device=action.device)[None, :]
            other_idx = torch.where(cols < rows, cols, cols + 1)      # (B, B-1)
            cand_list.append(noise_pos[other_idx])                   # (B, B-1, T, D)
            K_inbatch = B - 1
        cand_list.append(noise_neg)
        all_noises = torch.cat(cand_list, dim=1)
        K_total = 1 + K_inbatch + K_neg

        obs_expanded = obs_features[:, None, :].expand(B, K_total, obs_features.shape[-1]).reshape(B * K_total, -1)
        energies = self.ebm(obs_expanded, all_noises.reshape(B * K_total, -1)).reshape(B, K_total)

        logp = F.log_softmax(-energies / self.ebm_cfg.temperature, dim=-1)
        loss = -logp[:, 0].mean()
        ld = {"loss": loss.item(), "infonce": loss.item(),
              "energy_pos": energies[:, 0].mean().item(),
              "energy_neg": energies[:, 1:].mean().item(),
              "infonce_top1_acc": (energies.argmin(dim=-1) == 0).float().mean().item()}
        return loss, ld

    def compute_phase2_loss(self, static_obs, gripper_obs, action, pos, step=None):
        if getattr(self.ebm_cfg, "loss_type", "ce") == "infonce":
            return self.compute_phase2_loss_infonce(static_obs, gripper_obs, action, pos, step=step)
        return self.compute_phase2_loss_ce(static_obs, gripper_obs, action, pos)

    # ------------------------------------------------------------------
    # Inference (uses policy_ema for denoise, ebm_ema for scoring)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, batch_size: int, static_obs=None, gripper_obs=None, pos=None):
        dev = self.device
        T, D = self.cfg.policy_length, self._flow_action_dim

        if self.cfg.phase == "generative":
            traj = torch.randn(batch_size, T, D, device=dev)
            return self._denoise(traj)

        obs_features = self._encode_obs_flat(static_obs, gripper_obs, pos)
        obs_features = obs_features[:1]                              # condition on the single obs
        N = max(self.ebm_cfg.inference_num_samples, batch_size)
        candidates = torch.randn(N, T, D, device=dev)
        obs_expanded = obs_features.expand(N, obs_features.shape[-1])
        energies = self.ebm_ema(obs_expanded, candidates.reshape(N, -1))
        top = torch.argsort(energies)[:batch_size]
        top_cands = candidates[top]

        if self.cfg.phase in ("ebm", "ebm_only") and self.ebm_cfg.use_langevin:
            top_cands = self._langevin_inference(obs_features, top_cands)

        if self.cfg.phase == "ebm":
            return self._denoise(top_cands)
        return top_cands

    @torch.no_grad()
    def _langevin_inference(self, obs_features, init_noises):
        batch = init_noises.shape[0]
        noise = init_noises.reshape(batch, -1)
        obs_batch = obs_features.expand(batch, obs_features.shape[-1])
        step_size = self.ebm_cfg.langevin_step_size
        sqrt_step = (2 * step_size) ** 0.5 * 0.01
        for _ in range(int(self.ebm_cfg.langevin_steps)):
            with torch.enable_grad():
                nf = noise.detach().requires_grad_(True)
                energy = self.ebm_ema(obs_batch, nf).sum()
                grad = torch.autograd.grad(energy, nf)[0]
            noise = noise - step_size * grad + sqrt_step * torch.randn_like(noise)
        return noise.reshape(init_noises.shape)

    # ------------------------------------------------------------------
    # EMA update (called by the agent after an optimiser step)
    # ------------------------------------------------------------------
    def update_policy_ema(self, decay: float):
        update_ema(self.policy_ema, self.policy, decay)

    def update_ebm_ema(self, decay: float):
        update_ema(self.ebm_ema, self.ebm, decay)
