#!/usr/bin/env python
"""Tile per-method rollout videos into a single side-by-side comparison .mp4.

Inputs are the ``<agent_name>_seed<seed>.mp4`` files written by
``simulation.avoiding_sim.Avoiding_Sim`` when ``record_video_dir`` is set. Run
each method's benchmark with e.g.::

    bash scripts/avoiding/ddpm_encdec_benchmark.sh \\
        simulation.record_video_dir=viz/avoiding/video \\
        simulation.record_n_videos=3

(repeat for the other methods into the same dir), then::

    python scripts/montage_videos.py viz/avoiding/video -o viz/avoiding/montage.mp4

One tile per method (first seed found). Shorter clips are padded by holding
their last frame so every tile stays in sync. Dependencies: imageio + opencv.
"""
import argparse
import glob
import os

import numpy as np
import cv2
import imageio


def _read_frames(path):
    reader = imageio.get_reader(path)
    frames = [np.asarray(f)[:, :, :3] for f in reader]
    reader.close()
    return frames


def _label(frame, text, size):
    frame = cv2.resize(frame, (size, size))
    cv2.rectangle(frame, (0, 0), (size, 22), (0, 0, 0), -1)
    cv2.putText(frame, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir", help="directory with <method>_seed<seed>.mp4 files")
    ap.add_argument("-o", "--out", default="montage.mp4")
    ap.add_argument("--size", type=int, default=320, help="per-tile size (px)")
    ap.add_argument("--cols", type=int, default=0, help="grid columns (0 = auto)")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    # one video per method (first seed encountered)
    by_method = {}
    for path in sorted(glob.glob(os.path.join(args.video_dir, "*.mp4"))):
        name = os.path.basename(path).split("_seed")[0]
        by_method.setdefault(name, path)
    if not by_method:
        raise SystemExit(f"No .mp4 files found in {args.video_dir}")

    names = sorted(by_method)
    clips = {n: _read_frames(by_method[n]) for n in names}
    T = max(len(c) for c in clips.values())

    n = len(names)
    cols = args.cols or min(n, 4)
    rows = (n + cols - 1) // cols
    sz = args.size
    blank = np.zeros((sz, sz, 3), dtype=np.uint8)

    writer = imageio.get_writer(args.out, fps=args.fps)
    try:
        for t in range(T):
            tiles = []
            for n_idx, name in enumerate(names):
                frames = clips[name]
                f = frames[min(t, len(frames) - 1)]   # hold last frame when shorter
                tiles.append(_label(f, name, sz))
            # pad grid to rows*cols
            while len(tiles) < rows * cols:
                tiles.append(blank)
            grid = np.vstack([
                np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)
            ])
            writer.append_data(grid)
    finally:
        writer.close()
    print(f"Wrote {args.out}  ({n} methods x {T} frames: {', '.join(names)})")


if __name__ == "__main__":
    main()
