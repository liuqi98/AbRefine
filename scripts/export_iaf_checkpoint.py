#!/usr/bin/env python3
"""Export small PyTorch IAF checkpoints to portable, inspectable JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FEATURE_ORDER = [
    "structure_entropy",
    "framework_entropy",
    "structure_to_framework_kl",
    "framework_information_kl",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--base-expert", required=True, choices=("AntiFold", "AbMPNN"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    members = []
    provenance = []
    selections = []
    for path in args.checkpoints:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        state = artifact["state_dict"]
        scaler = np.asarray(artifact["scaler"], dtype=np.float64)
        members.append({
            "seed": int(artifact["seed"]),
            "g0": 0.59,
            "scaler_mean": scaler[0].tolist(),
            "scaler_scale": scaler[1].tolist(),
            "layer1_weight": state["network.0.weight"].detach().cpu().numpy().tolist(),
            "layer1_bias": state["network.0.bias"].detach().cpu().numpy().tolist(),
            "layer2_weight": state["network.2.weight"].detach().cpu().numpy().tolist(),
            "layer2_bias": state["network.2.bias"].detach().cpu().numpy().tolist(),
        })
        provenance.append({"file": path.name, "sha256": sha256(path)})
        selections.append(artifact.get("selection"))
    payload = {
        "schema_version": 1,
        "method": "AbRefine Information-Aware Fusion",
        "base_expert": args.base_expert,
        "amino_acid_order": "ACDEFGHIKLMNPQRSTVWY",
        "architecture": "4-8-1 SiLU",
        "feature_order": FEATURE_ORDER,
        "aggregation": "mean of member probability vectors",
        "development_targets": 62,
        "development_residues": 3077,
        "members": members,
        "selection": selections[0] if selections and all(value == selections[0] for value in selections) else selections,
        "source_checkpoints": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
