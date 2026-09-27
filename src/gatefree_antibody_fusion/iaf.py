"""Portable NumPy inference for AbRefine Information-Aware Fusion (IAF)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fusion import normalize_log_probabilities


EPS = 1e-12
FEATURE_NAMES = (
    "structure_entropy",
    "framework_entropy",
    "structure_to_framework_kl",
    "framework_information_kl",
)


def _as_probability_matrix(values: np.ndarray | list[float]) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 20:
        raise ValueError(f"expected shape (20,) or (N, 20), received {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("probabilities must be finite and non-negative")
    totals = array.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("each probability vector must have positive mass")
    return array / totals, single


def information_features(
    structure_log_probabilities: np.ndarray | list[float],
    framework_log_probabilities: np.ndarray | list[float],
    empirical_probabilities: np.ndarray | list[float],
) -> np.ndarray:
    """Return [H_S, H_F, KL(P_S||P_F), KL(P_F||P_emp)]."""
    zs = np.asarray(structure_log_probabilities, dtype=np.float64)
    zf = np.asarray(framework_log_probabilities, dtype=np.float64)
    single = zs.ndim == 1
    if single:
        zs, zf = zs[None, :], zf[None, :]
    if zs.shape != zf.shape or zs.ndim != 2 or zs.shape[1] != 20:
        raise ValueError("structure and framework scores must have matching (N, 20) shapes")
    ps = np.stack([normalize_log_probabilities(row) for row in zs])
    pf = np.stack([normalize_log_probabilities(row) for row in zf])
    pemp, _ = _as_probability_matrix(empirical_probabilities)
    if pemp.shape[0] == 1 and ps.shape[0] > 1:
        pemp = np.repeat(pemp, ps.shape[0], axis=0)
    if pemp.shape != ps.shape:
        raise ValueError("empirical probabilities must match the expert batch shape")
    log_ps = np.log(np.clip(ps, EPS, None))
    log_pf = np.log(np.clip(pf, EPS, None))
    hs = -np.sum(ps * log_ps, axis=1)
    hf = -np.sum(pf * log_pf, axis=1)
    disagreement = np.sum(ps * (log_ps - log_pf), axis=1)
    information = np.sum(pf * (log_pf - np.log(np.clip(pemp, EPS, None))), axis=1)
    output = np.column_stack((hs, hf, disagreement, information))
    return output[0] if single else output


@dataclass(frozen=True)
class IAFMember:
    """One frozen 4-8-1 IAF network represented with NumPy arrays."""

    seed: int
    mean: np.ndarray
    scale: np.ndarray
    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: float
    g0: float = 0.59

    @classmethod
    def from_dict(cls, value: dict) -> "IAFMember":
        return cls(
            seed=int(value["seed"]),
            mean=np.asarray(value["scaler_mean"], dtype=np.float64),
            scale=np.asarray(value["scaler_scale"], dtype=np.float64),
            weight1=np.asarray(value["layer1_weight"], dtype=np.float64),
            bias1=np.asarray(value["layer1_bias"], dtype=np.float64),
            weight2=np.asarray(value["layer2_weight"], dtype=np.float64).reshape(-1),
            bias2=float(np.asarray(value["layer2_bias"]).reshape(-1)[0]),
            g0=float(value.get("g0", 0.59)),
        )

    def gate(self, features: np.ndarray) -> np.ndarray:
        x = (features - self.mean) / self.scale
        hidden_linear = x @ self.weight1.T + self.bias1
        hidden = hidden_linear / (1.0 + np.exp(-hidden_linear))
        raw = hidden @ self.weight2 + self.bias2
        g0_logit = np.log(self.g0 / (1.0 - self.g0))
        return 1.0 / (1.0 + np.exp(-(g0_logit + raw)))


class IAFEnsemble:
    """Mean-probability ensemble of frozen IAF networks."""

    def __init__(self, members: list[IAFMember], metadata: dict | None = None) -> None:
        if not members:
            raise ValueError("at least one IAF member is required")
        self.members = members
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path) -> "IAFEnsemble":
        payload = json.loads(Path(path).read_text())
        if payload.get("feature_order") not in (None, list(FEATURE_NAMES)):
            raise ValueError("unsupported IAF feature order")
        return cls([IAFMember.from_dict(row) for row in payload["members"]], payload)

    def refine(
        self,
        structure_log_probabilities: np.ndarray | list[float],
        framework_log_probabilities: np.ndarray | list[float],
        empirical_probabilities: np.ndarray | list[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ensemble probability, mean gate, and four IAF descriptors."""
        zs = np.asarray(structure_log_probabilities, dtype=np.float64)
        zf = np.asarray(framework_log_probabilities, dtype=np.float64)
        single = zs.ndim == 1
        if single:
            zs, zf = zs[None, :], zf[None, :]
        if zs.shape != zf.shape or zs.ndim != 2 or zs.shape[1] != 20:
            raise ValueError("structure and framework scores must have matching (N, 20) shapes")
        ps = np.stack([normalize_log_probabilities(row) for row in zs])
        pf = np.stack([normalize_log_probabilities(row) for row in zf])
        log_ps, log_pf = np.log(np.clip(ps, EPS, None)), np.log(np.clip(pf, EPS, None))
        features = information_features(zs, zf, empirical_probabilities)
        if features.ndim == 1:
            features = features[None, :]
        probabilities, gates = [], []
        for member in self.members:
            gate = member.gate(features)
            logits = (1.0 - gate[:, None]) * log_ps + gate[:, None] * log_pf
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits)
            probabilities.append(probability / probability.sum(axis=1, keepdims=True))
            gates.append(gate)
        probability = np.mean(probabilities, axis=0)
        gate = np.mean(gates, axis=0)
        if single:
            return probability[0], gate[0], features[0]
        return probability, gate, features
