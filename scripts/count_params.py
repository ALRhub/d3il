#!/usr/bin/env python
"""Count and compare the number of model parameters for each model_name
that can be launched via scripts/<task_name>/<model_name>_benchmark.sh.

For every ``*.sh`` under the requested task folder(s) this script:
  1. parses the ``python run.py ...`` command to recover the Hydra
     ``--config-name`` and the relevant overrides (e.g. ``agents=...``,
     ``window_size=...``, ``agents.model.*=...``),
  2. composes the corresponding Hydra config,
  3. instantiates only ``cfg.agents.model`` (exactly what BaseAgent counts),
  4. counts its parameters via ``get_params()`` (falling back to
     ``parameters()``),
and finally prints a per-task comparison table sorted by parameter count.

Usage:
    python scripts/count_params.py                # every task folder
    python scripts/count_params.py avoiding       # one task folder
    python scripts/count_params.py avoiding sorting
"""

import os
import re
import sys
import shlex
import logging
import warnings

import torch
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

# --- repo layout -----------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

# run.py lives at the repo root, so `import agents...` works there. When this
# tool is launched as scripts/count_params.py the repo root is not on the path.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Keep the noise down: this is a measurement tool, not a training run.
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

# run.py registers this resolver; we need it too because some configs use
# ${add:${window_size}, 1} for transformer block sizes.
if not OmegaConf.has_resolver("add"):
    OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))


def parse_sh(path):
    """Extract (config_name, overrides) from a benchmark .sh file.

    Returns None if the file has no usable ``run.py`` invocation.
    """
    config_name = None
    overrides = []

    with open(path) as f:
        raw = f.read()

    # Drop commented-out lines, then glue line-continuations back together so
    # the whole `python run.py ... \` block becomes one logical command.
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("#")]
    command = " ".join(ln.rstrip("\\").strip() for ln in lines)

    tokens = shlex.split(command)
    # State-based scripts call run.py; vision scripts call run_vision.py
    # (often prefixed with `MUJOCO_GL=egl`). Only look at tokens after it.
    entry = next((i for i, t in enumerate(tokens)
                  if t in ("run.py", "run_vision.py")), None)
    if entry is None:
        return None
    tokens = tokens[entry + 1:]

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--config-name="):
            config_name = tok.split("=", 1)[1]
        elif tok == "--config-name":
            i += 1
            config_name = tokens[i]
        elif tok == "--multirun" or tok.startswith("--"):
            pass  # flags like --multirun are irrelevant for a single compose
        elif tok.startswith("seed="):
            pass  # a seed list would break a non-multirun compose; default is fine
        elif "=" in tok:
            overrides.append(tok)
        i += 1

    return config_name, overrides


def count_params(task, sh_name, config_name, overrides):
    """Compose the config, instantiate cfg.agents.model, return param count."""
    # Force CPU so the tool works on machines without a GPU, and so model
    # __init__ that allocates tensors on ${device} does not fail.
    ov = [o for o in overrides if not o.startswith("device=")] + ["device=cpu"]

    # Some .sh files override a key at the wrong nesting level (e.g. bet's
    # `agents.model.vocab_size` instead of `agents.model.model.vocab_size`).
    # Hydra rejects overriding a missing struct key; drop such overrides so
    # composition succeeds and the config default is used for the param count.
    cfg = None
    for _ in range(len(ov) + 1):
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
                cfg = compose(config_name=config_name, overrides=ov)
            break
        except ConfigCompositionException as e:
            m = re.search(r"Could not override '([^']+)'", str(e))
            if not m:
                raise
            bad = m.group(1)
            ov = [o for o in ov if o.split("=", 1)[0] != bad]

    if "model" not in cfg.agents:
        raise KeyError("config has no 'agents.model' (non-torch / custom agent)")

    model = hydra.utils.instantiate(cfg.agents.model)

    if hasattr(model, "get_params"):
        params = list(model.get_params())
    else:
        params = list(model.parameters())
    return sum(p.numel() for p in params)


def human(n):
    for unit, div in (("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.3f}{unit}"
    return str(n)


def process_task(task):
    task_dir = os.path.join(SCRIPTS_DIR, task)
    if not os.path.isdir(task_dir):
        print(f"[skip] not a directory: {task_dir}")
        return

    sh_files = sorted(f for f in os.listdir(task_dir) if f.endswith(".sh"))
    if not sh_files:
        print(f"[skip] no .sh files in {task_dir}")
        return

    results = []  # (model_name, agents_override, params or None, error)
    for sh in sh_files:
        path = os.path.join(task_dir, sh)
        parsed = parse_sh(path)
        if parsed is None:
            results.append((sh, "?", None, "no run.py command"))
            continue
        config_name, overrides = parsed
        agents = next((o.split("=", 1)[1] for o in overrides if o.startswith("agents=")), "?")
        if config_name is None:
            results.append((sh, agents, None, "no --config-name"))
            continue
        try:
            n = count_params(task, sh, config_name, overrides)
            results.append((sh, agents, n, None))
        except Exception as e:  # noqa: BLE001 - report and keep going
            results.append((sh, agents, None, f"{type(e).__name__}: {e}"))

    # --- print a comparison table -----------------------------------------
    ok = [r for r in results if r[2] is not None]
    bad = [r for r in results if r[2] is None]
    ok.sort(key=lambda r: r[2], reverse=True)

    print(f"\n=== task: {task}  ({len(ok)} ok / {len(bad)} failed) ===")
    name_w = max([len(r[1]) for r in results] + [10])
    print(f"  {'model_name':<{name_w}}  {'params':>12}  {'(human)':>10}")
    print(f"  {'-' * name_w}  {'-' * 12}  {'-' * 10}")
    for sh, agents, n, _ in ok:
        print(f"  {agents:<{name_w}}  {n:>12,}  {human(n):>10}")
    for sh, agents, n, err in bad:
        print(f"  {agents:<{name_w}}  {'ERROR':>12}  {'':>10}  <- {err}")


def main():
    args = sys.argv[1:]
    if args:
        tasks = args
    else:
        tasks = sorted(
            d for d in os.listdir(SCRIPTS_DIR)
            if os.path.isdir(os.path.join(SCRIPTS_DIR, d))
        )
    for task in tasks:
        process_task(task)


if __name__ == "__main__":
    main()
