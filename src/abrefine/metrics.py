"""Target-level evaluation and paired bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def target_means(rows: list[dict], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["target_id"]].append(float(row[field]))
    return {target: float(np.mean(values)) for target, values in grouped.items()}


def target_macro(rows: list[dict], field: str) -> float:
    values = target_means(rows, field)
    if not values:
        raise ValueError("cannot evaluate an empty panel")
    return float(np.mean(list(values.values())))


def paired_comparison(
    rows: list[dict], treatment: str, baseline: str, repeats: int = 10_000, seed: int = 20260918
) -> dict:
    treatment_by_target = target_means(rows, treatment)
    baseline_by_target = target_means(rows, baseline)
    if set(treatment_by_target) != set(baseline_by_target):
        raise ValueError("treatment and baseline target identifiers differ")
    targets = sorted(treatment_by_target)
    effects = np.asarray(
        [treatment_by_target[target] - baseline_by_target[target] for target in targets], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(effects), size=(repeats, len(effects)))
    samples = effects[indices].mean(axis=1)
    return {
        "target_macro_difference": float(effects.mean()),
        "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "targets": len(targets),
        "improved_targets": int(np.sum(effects > 0)),
        "tied_targets": int(np.sum(effects == 0)),
        "worsened_targets": int(np.sum(effects < 0)),
    }

