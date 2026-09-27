"""Command-line fusion and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fusion import AMINO_ACIDS, fuse_log_probabilities, normalize_log_probabilities
from .iaf import FEATURE_NAMES, IAFEnsemble
from .io import read_unique_jsonl, validate_pair, write_jsonl
from .metrics import paired_comparison, target_macro


def fuse(args: argparse.Namespace) -> None:
    structure = read_unique_jsonl(args.structure)
    evolution = read_unique_jsonl(args.evolution)
    validate_pair(structure, evolution)
    rows = []
    for patch_id in sorted(structure):
        srow, erow = structure[patch_id], evolution[patch_id]
        ps = normalize_log_probabilities(srow["log_probabilities"])
        pe_values = erow.get("all_cdr_mask_log_probabilities", erow.get("log_probabilities"))
        if pe_values is None:
            raise ValueError(f"missing evolutionary log probabilities at {patch_id}")
        pe = normalize_log_probabilities(pe_values)
        q = fuse_log_probabilities(np.log(ps), np.log(pe), args.weight)
        row = {
            "patch_id": patch_id,
            "target_id": srow["target_id"],
            "region": srow["region"],
            "fused_probabilities": q.tolist(),
            "predicted_amino_acid": AMINO_ACIDS[int(q.argmax())],
            "weight": args.weight,
            "native_CDR_tokens_visible": False,
        }
        if "native_amino_acid" in srow:
            native = srow["native_amino_acid"]
            index = AMINO_ACIDS.index(native)
            row.update(
                {
                    "native_amino_acid": native,
                    "structure_correct": int(ps.argmax() == index),
                    "evolution_correct": int(pe.argmax() == index),
                    "fused_correct": int(q.argmax() == index),
                    "structure_nll": float(-np.log(ps[index])),
                    "evolution_nll": float(-np.log(pe[index])),
                    "fused_nll": float(-np.log(q[index])),
                }
            )
        rows.append(row)
    write_jsonl(args.output, rows)
    print(json.dumps({"status": "COMPLETE", "residues": len(rows), "weight": args.weight,
                      "output": str(args.output.resolve())}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    rows = list(read_unique_jsonl(args.input).values())
    required = {"structure_correct", "fused_correct", "structure_nll", "fused_nll"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError("evaluation requires native labels and correctness/NLL fields from `fuse`")
    summary = {
        "targets": len({row["target_id"] for row in rows}),
        "residues": len(rows),
        "structure_recovery": target_macro(rows, "structure_correct"),
        "fused_recovery": target_macro(rows, "fused_correct"),
        "recovery_comparison": paired_comparison(
            rows, "fused_correct", "structure_correct", args.bootstrap, args.seed
        ),
        "structure_nll": target_macro(rows, "structure_nll"),
        "fused_nll": target_macro(rows, "fused_nll"),
        "nll_comparison": paired_comparison(rows, "fused_nll", "structure_nll", args.bootstrap, args.seed + 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


def iaf_fuse(args: argparse.Namespace) -> None:
    """Apply a frozen IAF ensemble to matched per-residue expert outputs."""
    structure = read_unique_jsonl(args.structure)
    evolution = read_unique_jsonl(args.evolution)
    empirical = read_unique_jsonl(args.empirical_prior)
    validate_pair(structure, evolution)
    if set(structure) != set(empirical):
        raise ValueError("structure and empirical-prior patch identifiers differ")
    model = IAFEnsemble.load(args.model)
    rows = []
    for patch_id in sorted(structure):
        srow, erow, prow = structure[patch_id], evolution[patch_id], empirical[patch_id]
        pe_values = erow.get("all_cdr_mask_log_probabilities", erow.get("log_probabilities"))
        if pe_values is None:
            raise ValueError(f"missing framework log probabilities at {patch_id}")
        pemp = prow.get("empirical_probabilities", prow.get("probabilities"))
        if pemp is None:
            raise ValueError(f"missing empirical probabilities at {patch_id}")
        probability, gate, features = model.refine(srow["log_probabilities"], pe_values, pemp)
        ps = normalize_log_probabilities(srow["log_probabilities"])
        pf = normalize_log_probabilities(pe_values)
        row = {
            "patch_id": patch_id,
            "target_id": srow["target_id"],
            "region": srow["region"],
            "fused_probabilities": probability.tolist(),
            "predicted_amino_acid": AMINO_ACIDS[int(probability.argmax())],
            "iaf_gate": float(gate),
            "iaf_features": {name: float(value) for name, value in zip(FEATURE_NAMES, features)},
            "model": str(args.model),
            "native_CDR_tokens_visible": False,
        }
        if "native_amino_acid" in srow:
            native = srow["native_amino_acid"]
            index = AMINO_ACIDS.index(native)
            row.update({
                "native_amino_acid": native,
                "structure_correct": int(ps.argmax() == index),
                "evolution_correct": int(pf.argmax() == index),
                "fused_correct": int(probability.argmax() == index),
                "structure_nll": float(-np.log(ps[index])),
                "evolution_nll": float(-np.log(pf[index])),
                "fused_nll": float(-np.log(probability[index])),
            })
        rows.append(row)
    write_jsonl(args.output, rows)
    print(json.dumps({
        "status": "COMPLETE",
        "method": "Information-Aware Fusion",
        "residues": len(rows),
        "mean_gate": float(np.mean([row["iaf_gate"] for row in rows])),
        "output": str(args.output.resolve()),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fuse_parser = subparsers.add_parser("fuse", help="legacy fixed-PoE fusion")
    fuse_parser.add_argument("--structure", type=Path, required=True)
    fuse_parser.add_argument("--evolution", type=Path, required=True)
    fuse_parser.add_argument("--output", type=Path, required=True)
    fuse_parser.add_argument("--weight", type=float, default=0.59)
    fuse_parser.set_defaults(function=fuse)
    iaf_parser = subparsers.add_parser("iaf", help="apply a frozen Information-Aware Fusion ensemble")
    iaf_parser.add_argument("--structure", type=Path, required=True)
    iaf_parser.add_argument("--evolution", type=Path, required=True)
    iaf_parser.add_argument("--empirical-prior", type=Path, required=True)
    iaf_parser.add_argument("--model", type=Path, required=True)
    iaf_parser.add_argument("--output", type=Path, required=True)
    iaf_parser.set_defaults(function=iaf_fuse)
    eval_parser = subparsers.add_parser("evaluate", help="evaluate a fused JSONL with native labels")
    eval_parser.add_argument("--input", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--bootstrap", type=int, default=10_000)
    eval_parser.add_argument("--seed", type=int, default=20260918)
    eval_parser.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
