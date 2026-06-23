"""Helpers for collecting benchmark metrics into a paper-ready CSV.

``append_metrics`` is called at the end of run.py / run_vision.py (opt-in via the
``PAPER_CSV`` environment variable) to append the evaluation metrics of a single
(task, modality, method, seed) run to a shared long-form CSV. The metrics are
read from the active W&B run summary (each simulator logs ``Metrics/*`` and
``score`` there), so the same code captures whatever each task reports
(success rate, entropy, KL, distance, ...) uniformly.

The CSV is long/tidy format — one row per metric:

    task, modality, method, seed, window_size, metric, value

``scripts/paper_benchmark.py`` reads this file to build the summary table and
the figures.
"""
from __future__ import annotations

import csv
import os

_FIELDS = ["task", "modality", "method", "seed", "window_size", "metric", "value"]


def _numeric(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # Drop NaN/inf so the CSV stays clean.
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def append_metrics(csv_path, task, modality, method, seed, window_size, summary):
    """Append one row per relevant metric in ``summary`` to ``csv_path``.

    ``summary`` is the W&B run summary (dict-like). Keys starting with
    ``Metrics/`` (prefix stripped) and the bare ``score`` key are recorded.
    """
    rows = []
    for key, val in dict(summary).items():
        if key == "score" or key.startswith("Metrics/"):
            f = _numeric(val)
            if f is None:
                continue
            metric = key[len("Metrics/"):] if key.startswith("Metrics/") else key
            rows.append({
                "task": task, "modality": modality, "method": method,
                "seed": int(seed), "window_size": int(window_size),
                "metric": metric, "value": f,
            })
    if not rows:
        return

    csv_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
