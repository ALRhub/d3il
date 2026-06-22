#!/usr/bin/env python
"""Compare trained methods on the D3IL *avoiding* task by overlaying their
end-effector trajectories over the obstacle field (the canonical D3IL figure
that visualises behavioural multimodality).

Inputs are the ``<agent_name>_seed<seed>.npz`` files written by
``simulation.avoiding_sim.Avoiding_Sim`` when ``save_traj_dir`` is set. Run each
method's benchmark with e.g.::

    bash scripts/avoiding/ddpm_encdec_benchmark.sh \\
        simulation.save_traj_dir=viz/avoiding/traj

(then bc, beso, ebm_jax, ... into the same dir), and finally::

    python scripts/plot_avoiding_trajectories.py viz/avoiding/traj -o viz/avoiding/compare.png

Dependencies: numpy + matplotlib only (no mujoco / torch).
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- obstacle field geometry (mirrors envs/.../objects/avoiding_objects.py) ---
_MID = 0.5
_OFF = 0.075
_Y0 = -0.1
_DY = 0.18
_START = (0.525, -0.28)            # init_end_eff_pos[:2]
_FINISH_Y = _Y0 + 2.5 * _DY        # finish line
# (x, y, radius) for each cylinder post
_OBSTACLES = [
    (_MID, _Y0, 0.03),
    (_MID - _OFF, _Y0 + _DY, 0.025),
    (_MID + _OFF, _Y0 + _DY, 0.025),
    (_MID - 2 * _OFF, _Y0 + 2 * _DY, 0.025),
    (_MID, _Y0 + 2 * _DY, 0.025),
    (_MID + 2 * _OFF, _Y0 + 2 * _DY, 0.025),
]


def _draw_field(ax):
    for x, y, r in _OBSTACLES:
        ax.add_patch(Circle((x, y), r, color="firebrick", alpha=0.85, zorder=3))
    ax.axhline(_FINISH_Y, color="green", lw=2, alpha=0.5, zorder=1)
    ax.plot(*_START, marker="*", color="black", markersize=12, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(-0.32, 0.42)
    ax.set_xticks([])
    ax.set_yticks([])


def _load_by_method(traj_dir):
    """Group all npz files by agent_name; concatenate trajectories over seeds."""
    methods = {}
    for path in sorted(glob.glob(os.path.join(traj_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        name = str(d["agent_name"]) if "agent_name" in d else \
            os.path.basename(path).split("_seed")[0]
        entry = methods.setdefault(name, {"c_pos": [], "lengths": [], "modes": [], "succ": []})
        entry["c_pos"].append(d["c_pos"])
        entry["lengths"].append(d["lengths"])
        entry["modes"].append(d["modes"])
        entry["succ"].append(d["successes"])
    for name, e in methods.items():
        e["c_pos"] = np.concatenate(e["c_pos"], axis=0)
        e["lengths"] = np.concatenate(e["lengths"], axis=0)
        e["modes"] = np.concatenate(e["modes"], axis=0)
        e["succ"] = np.concatenate(e["succ"], axis=0)
    return methods


def _mode_ids(modes):
    # same decimal encoding used for the entropy metric in avoiding_sim
    bits = modes.astype(int)
    return bits.dot(1 << np.arange(bits.shape[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", help="directory with <method>_seed<seed>.npz files")
    ap.add_argument("-o", "--out", default="avoiding_compare.png")
    ap.add_argument("--max-traj", type=int, default=80,
                    help="max trajectories drawn per method (for legibility)")
    ap.add_argument("--cols", type=int, default=0, help="grid columns (0 = auto)")
    args = ap.parse_args()

    methods = _load_by_method(args.traj_dir)
    if not methods:
        raise SystemExit(f"No .npz trajectory files found in {args.traj_dir}")

    names = sorted(methods)
    n = len(names)
    cols = args.cols or min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows), squeeze=False)

    cmap = plt.get_cmap("tab20")
    for k, name in enumerate(names):
        ax = axes[k // cols][k % cols]
        e = methods[name]
        _draw_field(ax)
        ids = _mode_ids(e["modes"])
        succ = e["succ"].astype(bool)
        n_draw = min(args.max_traj, e["c_pos"].shape[0])
        for i in range(n_draw):
            L = int(e["lengths"][i])
            if L < 2:
                continue
            xy = e["c_pos"][i, :L]
            if succ[i]:
                color = cmap(int(ids[i]) % 20)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=0.8, alpha=0.7, zorder=2)
            else:
                ax.plot(xy[:, 0], xy[:, 1], color="0.6", lw=0.5, alpha=0.4, zorder=2)
        sr = float(succ.mean()) if succ.size else 0.0
        n_modes = len(np.unique(ids[succ])) if succ.any() else 0
        ax.set_title(f"{name}\nsuccess={sr:.2f}  modes={n_modes}", fontsize=9)

    # hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle("Avoiding: end-effector trajectories by method "
                 "(coloured = success by mode, grey = failure)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}  ({n} methods: {', '.join(names)})")


if __name__ == "__main__":
    main()
