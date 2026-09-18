"""Numerically stable product-of-experts fusion."""

from __future__ import annotations

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def normalize_log_probabilities(values: list[float] | np.ndarray) -> np.ndarray:
    """Convert 20 unnormalized log scores into a probability vector."""
    logits = np.asarray(values, dtype=np.float64)
    if logits.shape != (20,):
        raise ValueError(f"expected 20 amino-acid scores, received shape {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("log scores must be finite")
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def fuse_log_probabilities(
    structure_log_probabilities: list[float] | np.ndarray,
    evolution_log_probabilities: list[float] | np.ndarray,
    weight: float = 0.59,
) -> np.ndarray:
    """Fuse structure and evolution experts with a fixed product of experts.

    q(a) is proportional to p_structure(a) ** (1-weight) times
    p_evolution(a) ** weight. Weight 0 and 1 exactly recover the respective
    experts.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    ps = normalize_log_probabilities(structure_log_probabilities)
    pe = normalize_log_probabilities(evolution_log_probabilities)
    logits = (1.0 - weight) * np.log(ps) + weight * np.log(pe)
    logits -= logits.max()
    result = np.exp(logits)
    return result / result.sum()

