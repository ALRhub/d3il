# EBM Policy (JAX / Flax-NNX) — D3IL port

Two-phase Energy-Based-Model policy ported from `vapo-aif` so it can be
benchmarked against the other D3IL baselines. The neural-network
implementation (backbone, flow matching, energy model) is **kept identical to
vapo** — it is `ml-networks` + `jax` + `flax.nnx`. Only the training/eval
harness is adapted to D3IL's PyTorch benchmark interface.

## Files

| file | origin |
|------|--------|
| `backbone.py` | verbatim copy of `vapo/policy/backbone.py` (DiT / UNet / `EnergyModel` / `EnergyTransformer`) |
| `fm.py` | verbatim copy of `vapo/policy/fm.py` (flow matching, schedulers, OT coupling) |
| `ema.py` | `NnxEMA` from `vapo/training/ema.py` |
| `config.py` | the EBM-relevant config dataclasses from `vapo/policy/config.py` |
| `policy_ebm.py` | `EBMPolicy` + helpers from `vapo/policy/policy.py` |
| `../../ebm_jax_agent.py` | D3IL `BaseAgent` wrapper — handles **both** state and vision |

## State and vision

One agent class (`EBM_JAX_Agent`) covers both benchmarks; the modality is
auto-detected from the phase-2 config:

* **State** (`configs/agents/ebm_jax_agent.yaml`) — the low-dim observation
  drives the proprioceptive ("pos") encoder. Trained/eval'd via `run.py`.
* **Vision** (`configs/agents/ebm_jax_vision_agent.yaml`) — the two cameras
  (agentview + in-hand, `[3,96,96]`) drive the ml-networks conv encoders and the
  robot state drives the pos encoder, exactly like the vapo EBM policy. Trained/
  eval'd via `run_vision.py` (which calls `train_vision_agent()` once per epoch,
  so the two phases are scheduled across epochs: first `phase1_epochs` epochs
  train the generative model, the rest train the EBM).

Benchmark scripts exist for every task, `scripts/<task>/ebm_jax_benchmark.sh`:
state — `avoiding`, `pushing`, `sorting` (sorting_2), `sorting_4`, `sorting_6`,
`stacking`, `aligning`; vision — `aligning_vision`, `sorting_4_vision`,
`sorting_6_vision`, `stacking_vision`.

(Note: `stacking_vision_config.yaml` ships with an uncommented
`override hydra/launcher: submitit_slurm` default that requires the
`hydra-submitit-launcher` plugin — this affects every agent's stacking_vision
run, not just this one.)

## What changed from vapo (and why)

The **model math is unchanged**. Only the harness differs:

1. **No PyTorch-Lightning.** vapo's `EBMPolicy` subclassed `JaxLightningModule`.
   D3IL drives the jitted `_jit_step` / `inference` calls directly from
   `EBM_JAX_Agent`, so the Lightning training/validation/checkpoint hooks are
   replaced by a tiny `JaxModuleBase`. Persistence is handled by the agent
   (pickle of the numpy parameter leaves).
2. **In-memory phase-1 → phase-2 hand-off.** Instead of resolving a phase-1
   checkpoint path on disk, the agent trains phase 1 then injects the frozen
   generative model + its EMA straight into the phase-2 model
   (`EBMPolicy.load_generative_from_model`).
3. **`pos_input_dim`.** The proprioceptive ("pos") encoder takes the D3IL state
   width (`obs_dim`) instead of being hard-wired to `action_dim`. D3IL state
   observations condition the EBM through this encoder; image encoders are
   unused for the state benchmarks.

## How it maps onto D3IL

* `policy_length = obs_length = window_size`. The phase-1 flow model is trained
  on the action window; the EBM is conditioned on the observation window.
* `predict(obs)` accumulates an `obs_length` window, runs EBM-guided inference
  (sample `inference_num_samples` candidates → score with the EBM EMA → take the
  lowest-energy → denoise through the frozen phase-1 model), and returns the
  action aligned with the most recent observation as `(1, action_dim)` — the
  shape the D3IL simulators index with `[0]`.

## Running

```bash
# state task
bash scripts/avoiding/ebm_jax_benchmark.sh
# vision task
bash scripts/aligning_vision/ebm_jax_benchmark.sh

# or directly:
python run.py --config-name=avoiding_config agents=ebm_jax_agent \
       agent_name=ebm_jax window_size=1 seed=0
MUJOCO_GL=egl python run_vision.py --config-name=aligning_vision_config \
       agents=ebm_jax_vision_agent agent_name=ebm_jax_vision window_size=1 seed=0
```

Useful overrides (`configs/agents/ebm_jax_agent.yaml`):

* `agents.phase2.ebm.loss_type=infonce` — InfoNCE contrastive loss (OT-inverted
  positive + random/Langevin negatives) instead of the default soft-target CE.
* `agents.phase2.ebm.arch=transformer` — transformer EBM instead of the MLP.
* `agents.phase2.ebm.use_langevin=true` — Langevin refinement at inference
  (matches the vapo phase-2 example; slower).
* `+agents.phase1_epochs=...` / `+agents.phase2_epochs=...` — train the two
  phases for different numbers of epochs (default: both `epoch`).
