#!/usr/bin/env python3
"""Train the 49-parameter AbRefine IAF ensemble on development antibodies."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


AA = "ACDEFGHIKLMNPQRSTVWY"
EPS = 1e-12
G0 = 0.59
FEATURES = ["structure_entropy", "framework_entropy",
            "structure_to_framework_kl", "framework_information_kl"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def log_softmax(values: np.ndarray) -> np.ndarray:
    if values.shape != (20,) or not np.all(np.isfinite(values)):
        raise ValueError("expected 20 finite amino-acid log scores")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def assemble(structure_path: Path, framework_path: Path) -> list[dict]:
    framework = {row["patch_id"]: row for row in read_jsonl(framework_path)}
    output = []
    for srow in read_jsonl(structure_path):
        erow = framework.get(srow["patch_id"])
        if erow is None:
            raise KeyError(f"missing framework row {srow['patch_id']}")
        for field in ("target_id", "region", "native_amino_acid"):
            if srow[field] != erow[field]:
                raise ValueError(f"metadata mismatch at {srow['patch_id']} / {field}")
        if srow.get("native_CDR_tokens_visible") is not False or erow.get("native_CDR_tokens_visible") is not False:
            raise ValueError(f"native-CDR leakage contract failed at {srow['patch_id']}")
        if "all_cdr_mask_log_probabilities" not in erow:
            raise ValueError(f"framework row is not all-CDR-masked at {srow['patch_id']}")
        output.append({
            "patch_id": srow["patch_id"],
            "target_id": srow["target_id"],
            "region": srow["region"],
            "native": AA.index(srow["native_amino_acid"]),
            "zs": log_softmax(np.asarray(srow["structure_log_probabilities"] if "structure_log_probabilities" in srow else srow["log_probabilities"], dtype=np.float64)),
            "zf": log_softmax(np.asarray(erow["all_cdr_mask_log_probabilities"], dtype=np.float64)),
        })
    if len({row["patch_id"] for row in output}) != len(output):
        raise ValueError("duplicate patch identifiers")
    return output


def region_lengths(rows: list[dict]) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        offset = int(row["patch_id"].rsplit(":", 1)[1])
        key = (row["target_id"], row["region"])
        result[key] = max(result[key], offset + 1)
    return dict(result)


def position_bin(row: dict, lengths: dict[tuple[str, str], int], bins: int = 10) -> int:
    offset = int(row["patch_id"].rsplit(":", 1)[1])
    length = lengths[(row["target_id"], row["region"])]
    relative = 0.5 if length == 1 else offset / (length - 1)
    return min(bins - 1, int(relative * bins))


def empirical_prior(train: list[dict], evaluation: list[dict], leave_target_out: bool,
                    bins: int = 10, pseudocount: float = 0.5) -> np.ndarray:
    train_lengths, eval_lengths = region_lengths(train), region_lengths(evaluation)
    totals: dict[tuple[str, int], np.ndarray] = defaultdict(lambda: np.zeros(20))
    per_target: dict[tuple[str, str, int], np.ndarray] = defaultdict(lambda: np.zeros(20))
    for row in train:
        key = (row["region"], position_bin(row, train_lengths, bins))
        totals[key][row["native"]] += 1
        per_target[(row["target_id"], *key)][row["native"]] += 1
    output = []
    for row in evaluation:
        key = (row["region"], position_bin(row, eval_lengths, bins))
        counts = totals[key].copy()
        if leave_target_out:
            counts -= per_target[(row["target_id"], *key)]
        counts += pseudocount
        output.append(counts / counts.sum())
    return np.asarray(output)


def arrays(rows: list[dict], pemp: np.ndarray) -> dict[str, np.ndarray]:
    zs, zf = np.stack([r["zs"] for r in rows]), np.stack([r["zf"] for r in rows])
    ps, pf = np.exp(zs), np.exp(zf)
    x = np.column_stack((
        -np.sum(ps * zs, axis=1),
        -np.sum(pf * zf, axis=1),
        np.sum(ps * (zs - zf), axis=1),
        np.sum(pf * (zf - np.log(np.clip(pemp, EPS, None))), axis=1),
    ))
    return {
        "x": x, "zs": zs, "zf": zf,
        "native": np.asarray([r["native"] for r in rows], dtype=np.int64),
        "groups": np.asarray([r["target_id"] for r in rows]),
    }


class IAFNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("g0_logit", torch.tensor(math.log(G0 / (1 - G0)), dtype=torch.float32))

    def forward(self, x: torch.Tensor, zs: torch.Tensor, zf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.g0_logit + self.network(x))
        logits = (1 - gate) * zs + gate * zf
        return logits, torch.mean((gate - G0) ** 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def target_macro_loss(logits: torch.Tensor, labels: torch.Tensor, groups: np.ndarray) -> torch.Tensor:
    residue_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
    values = []
    for target in sorted(set(groups.tolist())):
        indices = torch.as_tensor(np.flatnonzero(groups == target), dtype=torch.long)
        values.append(residue_loss[indices].mean())
    return torch.stack(values).mean()


def fit(train: dict[str, np.ndarray], valid: dict[str, np.ndarray] | None,
        regularization: float, seed: int, epochs: int = 800,
        patience: int = 80) -> tuple[IAFNet, np.ndarray, int, float]:
    set_seed(seed)
    mean, scale = train["x"].mean(0), train["x"].std(0)
    scale[scale < 1e-6] = 1.0
    tx = torch.tensor((train["x"] - mean) / scale, dtype=torch.float32)
    ts, tf = torch.tensor(train["zs"], dtype=torch.float32), torch.tensor(train["zf"], dtype=torch.float32)
    ty = torch.tensor(train["native"], dtype=torch.long)
    model = IAFNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    best_state, best_epoch, best_nll, wait = None, 0, float("inf"), 0
    if valid is not None:
        vx = torch.tensor((valid["x"] - mean) / scale, dtype=torch.float32)
        vs, vf = torch.tensor(valid["zs"], dtype=torch.float32), torch.tensor(valid["zf"], dtype=torch.float32)
        vy = torch.tensor(valid["native"], dtype=torch.long)
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        logits, penalty = model(tx, ts, tf)
        loss = target_macro_loss(logits, ty, train["groups"]) + regularization * penalty
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if valid is None:
            continue
        model.eval()
        with torch.no_grad():
            value = float(target_macro_loss(model(vx, vs, vf)[0], vy, valid["groups"]))
        if value < best_nll - 1e-6:
            best_nll, best_epoch = value, epoch
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if valid is not None:
        if best_state is None:
            raise RuntimeError("no validation checkpoint was produced")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
        model.eval()
        with torch.no_grad():
            best_nll = float(target_macro_loss(model(tx, ts, tf)[0], ty, train["groups"]))
    return model, np.stack((mean, scale)), best_epoch, best_nll


def grouped_folds(groups: np.ndarray, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = {target: int(np.sum(groups == target)) for target in sorted(set(groups.tolist()))}
    assignments, sizes = [[] for _ in range(folds)], [0] * folds
    for target in sorted(counts, key=lambda value: (-counts[value], value)):
        fold = min(range(folds), key=lambda i: (sizes[i], len(assignments[i]), i))
        assignments[fold].append(target); sizes[fold] += counts[target]
    indices = np.arange(len(groups))
    return [(indices[~np.isin(groups, targets)], indices[np.isin(groups, targets)]) for targets in assignments]


def subset(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in data.items()}


def predict(model: IAFNet, scaler: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    x = torch.tensor((data["x"] - scaler[0]) / scaler[1], dtype=torch.float32)
    zs, zf = torch.tensor(data["zs"], dtype=torch.float32), torch.tensor(data["zf"], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x, zs, zf)[0], dim=1).numpy()


def target_macro_metrics(probability: np.ndarray, native: np.ndarray,
                         groups: np.ndarray) -> tuple[float, float]:
    nll = -np.log(np.clip(probability[np.arange(len(native)), native], EPS, None))
    correct = (probability.argmax(1) == native).astype(float)
    targets = sorted(set(groups.tolist()))
    return (
        float(np.mean([nll[groups == target].mean() for target in targets])),
        float(np.mean([correct[groups == target].mean() for target in targets])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-expert", choices=("AntiFold", "AbMPNN"), required=True)
    parser.add_argument("--regularization-grid", type=float, nargs="+", default=(0.0, 0.01, 0.1, 1.0))
    args = parser.parse_args()
    rows = assemble(args.structure, args.framework)
    groups = np.asarray([row["target_id"] for row in rows])
    folds = grouped_folds(groups)
    candidates = []
    for regularization in args.regularization_grid:
        fold_epochs = []
        oof = np.zeros((len(rows), 20), dtype=np.float64)
        for fold, (train_idx, valid_idx) in enumerate(folds):
            train_rows, valid_rows = [rows[i] for i in train_idx], [rows[i] for i in valid_idx]
            train = arrays(train_rows, empirical_prior(train_rows, train_rows, True))
            valid = arrays(valid_rows, empirical_prior(train_rows, valid_rows, False))
            model, scaler, epoch, _ = fit(train, valid, regularization, 1000 + 31 * fold)
            oof[valid_idx] = predict(model, scaler, valid)
            fold_epochs.append(epoch)
        oof_nll, oof_recovery = target_macro_metrics(
            oof, np.asarray([row["native"] for row in rows]), groups
        )
        candidates.append({"regularization": regularization,
                           "oof_target_macro_nll": oof_nll,
                           "oof_target_macro_recovery": oof_recovery,
                           "fold_best_epochs": fold_epochs,
                           "full_training_epochs": int(np.median(fold_epochs))})
    selected = min(candidates, key=lambda value: (
        value["oof_target_macro_nll"], -value["oof_target_macro_recovery"], value["regularization"]
    ))
    full = arrays(rows, empirical_prior(rows, rows, True))
    members = []
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in (41, 42, 43):
        model, scaler, _, train_nll = fit(full, None, selected["regularization"], seed,
                                         epochs=max(20, selected["full_training_epochs"]), patience=10_000)
        state = model.state_dict()
        torch.save({"state_dict": state, "scaler": scaler, "kind": "iaf",
                    "seed": seed, "selection": selected}, args.output / f"iaf_seed{seed}.pt")
        members.append({
            "seed": seed, "g0": G0,
            "scaler_mean": scaler[0].tolist(), "scaler_scale": scaler[1].tolist(),
            "layer1_weight": state["network.0.weight"].numpy().tolist(),
            "layer1_bias": state["network.0.bias"].numpy().tolist(),
            "layer2_weight": state["network.2.weight"].numpy().tolist(),
            "layer2_bias": state["network.2.bias"].numpy().tolist(),
            "training_target_macro_nll": train_nll,
        })
    payload = {
        "schema_version": 1, "method": "AbRefine Information-Aware Fusion",
        "base_expert": args.base_expert, "architecture": "4-8-1 SiLU",
        "amino_acid_order": AA, "feature_order": FEATURES,
        "aggregation": "mean of member probability vectors",
        "development_targets": len(set(groups.tolist())), "development_residues": len(rows),
        "selection": selected, "candidates": candidates, "members": members,
    }
    (args.output / "iaf_ensemble.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "selection": selected,
                      "model": str(args.output / "iaf_ensemble.json")}, indent=2))


if __name__ == "__main__":
    main()
