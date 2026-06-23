#!/usr/bin/env python
"""Train + evaluate a full grid of (task x method x seed) and produce
paper-ready artifacts: a metrics CSV (raw + summary) and figures (PDF).

It reuses the per-task benchmark scripts (``scripts/<task>/<method>_benchmark.sh``)
so every baseline runs with its intended, tuned hyper-parameters. Each run is
launched once per seed (the ``--multirun`` sweep is stripped) on a chosen GPU,
and each run appends its evaluation metrics to a shared long-form CSV via the
``PAPER_CSV`` hook in run.py / run_vision.py.

Examples
--------
# Everything (state + vision), all discovered methods, seeds 0-5, on GPU 2:
python scripts/paper_benchmark.py --gpu 2

# Only avoiding + pushing, only ddpm_encdec and ebm_jax, seeds 0-2, GPU 1:
python scripts/paper_benchmark.py --gpu 1 --tasks avoiding pushing \
       --methods ddpm_encdec ebm_jax --seeds 0 1 2

# Just (re)build the CSV summary + figures from already-collected runs:
python scripts/paper_benchmark.py --collect-only
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# task key (== scripts/<dir>) -> modality. The key is also the scripts subdir.
STATE_TASKS = ["avoiding", "pushing", "sorting", "sorting_4", "sorting_6", "stacking", "aligning"]
VISION_TASKS = ["aligning_vision", "sorting_4_vision", "sorting_6_vision", "stacking_vision"]
MODALITY = {**{t: "state" for t in STATE_TASKS}, **{t: "image" for t in VISION_TASKS}}

# Headline metrics to plot (skipped automatically when a task doesn't report them).
PLOT_METRICS = ["successes", "entropy", "score"]


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

def discover_methods(tasks):
    methods = set()
    for t in tasks:
        for f in glob.glob(os.path.join(REPO, "scripts", t, "*_benchmark.sh")):
            methods.add(os.path.basename(f)[: -len("_benchmark.sh")])
    return sorted(methods)


def build_command(task, method, seed):
    """Read the benchmark script, strip the --multirun sweep, pin a single seed."""
    path = os.path.join(REPO, "scripts", task, f"{method}_benchmark.sh")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    text = re.sub(r"--multirun\s+", "", text)          # single run, not a sweep
    text = re.sub(r"seed=[0-9,]+", f"seed={seed}", text)  # one seed
    return text


def already_done(csv_path, task, method, seed):
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                if (row["task"] == task and row["method"] == method
                        and int(row["seed"]) == int(seed)):
                    return True
    except Exception:
        return False
    return False


def filter_ebm_backend(methods, backend):
    """Keep only the requested EBM implementation. Non-EBM baselines are always
    kept. ``ebm_jax*`` are the JAX implementations (DiT+transformer and MLP);
    ``ebm_torch*`` are the PyTorch implementation(s)."""
    if backend == "both":
        return methods
    out = []
    for m in methods:
        if m.startswith("ebm_jax"):
            if backend == "jax":
                out.append(m)
        elif m.startswith("ebm_torch"):
            if backend == "torch":
                out.append(m)
        else:
            out.append(m)  # non-EBM baseline -> always run
    return out


def run_grid(args, csv_path):
    tasks = resolve_tasks(args)
    methods = args.methods if args.methods != ["all"] else discover_methods(tasks)
    methods = filter_ebm_backend(methods, args.ebm_backend)
    log_dir = os.path.join(args.out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    combos = [(t, m, s) for t in tasks for m in methods for s in args.seeds]
    print(f"[paper] {len(combos)} runs: {len(tasks)} tasks x {len(methods)} methods "
          f"x {len(args.seeds)} seeds on GPU {args.gpu}")
    print(f"[paper] ebm_backend={args.ebm_backend}")
    print(f"[paper] tasks={tasks}")
    print(f"[paper] methods={methods}")

    for i, (task, method, seed) in enumerate(combos, 1):
        tag = f"{task}/{method}/seed{seed}"
        cmd = build_command(task, method, seed)
        if cmd is None:
            print(f"[{i}/{len(combos)}] SKIP {tag} (no benchmark script)")
            continue
        if args.skip_existing and already_done(csv_path, task, method, seed):
            print(f"[{i}/{len(combos)}] SKIP {tag} (already in CSV)")
            continue

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["WANDB_MODE"] = args.wandb_mode
        env["PAPER_CSV"] = csv_path
        env["MUJOCO_GL"] = args.mujoco_gl

        print(f"[{i}/{len(combos)}] RUN  {tag}")
        if args.dry_run:
            print("    " + cmd.replace("\n", " ").strip())
            continue
        logf = os.path.join(log_dir, f"{task}_{method}_seed{seed}.log")
        with open(logf, "w") as out:
            ret = subprocess.run(cmd, shell=True, cwd=REPO, env=env,
                                 stdout=out, stderr=subprocess.STDOUT)
        status = "ok" if ret.returncode == 0 else f"FAILED (rc={ret.returncode})"
        print(f"    -> {status} | log: {logf}")


def resolve_tasks(args):
    if args.tasks != ["all"]:
        return args.tasks
    if args.modality == "state":
        return STATE_TASKS
    if args.modality == "image":
        return VISION_TASKS
    return STATE_TASKS + VISION_TASKS


# --------------------------------------------------------------------------- #
# Aggregation + plotting
# --------------------------------------------------------------------------- #

def aggregate_and_plot(args, csv_path):
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[paper] aggregation needs pandas+matplotlib: {e}")
        return

    if not os.path.exists(csv_path):
        print(f"[paper] no metrics CSV yet at {csv_path}; nothing to aggregate.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[paper] metrics CSV is empty.")
        return
    # De-duplicate (keep the last run for a given task/method/seed/metric).
    df = df.drop_duplicates(subset=["task", "modality", "method", "seed", "metric"], keep="last")

    # ---- summary: mean / std / n over seeds ----
    g = (df.groupby(["modality", "task", "method", "metric"])["value"]
           .agg(mean="mean", std="std", n="count").reset_index())
    g["std"] = g["std"].fillna(0.0)
    long_path = os.path.join(args.out, "summary_long.csv")
    g.to_csv(long_path, index=False)

    # wide: one row per (modality, task, method), columns {metric}_mean/{metric}_std
    wide = g.pivot_table(index=["modality", "task", "method"], columns="metric",
                         values=["mean", "std"])
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index().sort_values(["modality", "task", "method"])
    wide_path = os.path.join(args.out, "summary.csv")
    wide.to_csv(wide_path, index=False)
    print(f"[paper] wrote {long_path}")
    print(f"[paper] wrote {wide_path}")

    # ---- figures: one PDF per (modality, metric) ----
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for modality in sorted(g["modality"].unique()):
        for metric in PLOT_METRICS:
            sub = g[(g["modality"] == modality) & (g["metric"] == metric)]
            if sub.empty:
                continue
            tasks = sorted(sub["task"].unique())
            methods = sorted(sub["method"].unique())
            x = np.arange(len(tasks))
            width = 0.8 / max(len(methods), 1)

            fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(tasks)), 4.5))
            for j, method in enumerate(methods):
                means, errs = [], []
                for t in tasks:
                    r = sub[(sub["task"] == t) & (sub["method"] == method)]
                    means.append(float(r["mean"].iloc[0]) if len(r) else np.nan)
                    errs.append(float(r["std"].iloc[0]) if len(r) else 0.0)
                ax.bar(x + j * width, means, width, yerr=errs, capsize=3,
                       label=method)
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(tasks, rotation=20, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} ({modality})  mean ± std over seeds")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            out_pdf = os.path.join(fig_dir, f"{modality}_{metric}.pdf")
            fig.savefig(out_pdf)
            plt.close(fig)
            print(f"[paper] wrote {out_pdf}")


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default="0", help="GPU id for CUDA_VISIBLE_DEVICES (e.g. 2)")
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="task keys (scripts/<task>) or 'all'")
    p.add_argument("--methods", nargs="+", default=["all"],
                   help="method names (e.g. ddpm_encdec bc beso ebm_jax) or 'all'")
    p.add_argument("--ebm-backend", choices=["jax", "torch", "both"], default="both",
                   help="which EBMPolicy implementation to include: 'jax' "
                        "(ebm_jax/ebm_jax_mlp), 'torch' (ebm_torch), or 'both'. "
                        "Non-EBM baselines always run.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--modality", choices=["state", "image", "both"], default="both",
                   help="which task set to use when --tasks all")
    p.add_argument("--out", default=os.path.join(REPO, "results", "paper"),
                   help="output directory for CSV + figures")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                   help="W&B mode (metrics CSV needs online or offline; disabled records nothing)")
    p.add_argument("--mujoco-gl", default="egl", help="MUJOCO_GL backend (egl for headless)")
    p.add_argument("--collect-only", action="store_true",
                   help="skip running; only (re)build summary CSV + figures")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip (task,method,seed) combos already present in the CSV")
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = p.parse_args()

    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "metrics.csv")

    if args.wandb_mode == "disabled" and not args.collect_only:
        print("[paper] WARNING: --wandb-mode disabled records no metrics; "
              "use online or offline.")

    if not args.collect_only:
        run_grid(args, csv_path)
    aggregate_and_plot(args, csv_path)


if __name__ == "__main__":
    main()
