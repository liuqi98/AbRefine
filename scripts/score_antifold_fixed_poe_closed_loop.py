#!/usr/bin/env python3
"""Closed-loop official AntiFold decoding with a frozen AntiBERTy PoE prior."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from antifold.antiscripts import get_dataset_dataloader, load_IF1_checkpoint
from antifold.esm.pretrained import _load_IF1_local


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def load_unique(path: Path) -> dict[str, dict]:
    result = {}
    for row in read_jsonl(path):
        if row["patch_id"] in result:
            raise RuntimeError(f"duplicate patch_id: {row['patch_id']}")
        result[row["patch_id"]] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_log(values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values -= values.max()
    result = np.exp(values)
    return result / result.sum()


def product_of_experts(ps: np.ndarray, pe: np.ndarray, weight: float) -> np.ndarray:
    logits = (1.0 - weight) * np.log(np.clip(ps, 1e-12, None))
    logits += weight * np.log(np.clip(pe, 1e-12, None))
    logits -= logits.max()
    result = np.exp(logits)
    return result / result.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Official original AntiFold checkpoint")
    parser.add_argument("--inputs", type=Path, required=True, help="AntiFold input CSV")
    parser.add_argument("--pdb-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True, help="All-CDR-mask AntiBERTy JSONL")
    parser.add_argument("--weight", type=float, default=0.59)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.weight <= 1:
        raise ValueError("weight must lie in [0, 1]")

    manifest_rows = read_jsonl(args.manifest)
    priors = load_unique(args.priors)
    if {row["patch_id"] for row in manifest_rows} != set(priors):
        raise RuntimeError("manifest/prior identifiers differ")
    if any(row.get("native_CDR_tokens_visible") is not False for row in priors.values()):
        raise RuntimeError("AntiBERTy leakage contract violated")

    by_input: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        by_input[row["model_input_name"]].append(row)
    frame = pd.read_csv(args.inputs)
    if set(frame["pdb"].astype(str)) != set(by_input):
        raise RuntimeError("input CSV and manifest targets differ")

    model, _ = _load_IF1_local()
    model = load_IF1_checkpoint(model, str(args.checkpoint)).eval()
    device = torch.device(args.device)
    model.to(device)
    alphabet = model.decoder.dictionary
    aa_token_ids = torch.tensor(
        [alphabet.get_idx(amino_acid) for amino_acid in AMINO_ACIDS], dtype=torch.long, device=device
    )
    mask_token_id = alphabet.get_idx("<mask>")
    _, dataloader = get_dataset_dataloader(
        frame, str(args.pdb_dir), batch_size=1, custom_chain_mode=False, num_threads=0
    )

    outputs = []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            (coords, confidence, _strs, tokens, padding_mask, _loss_masks,
             _res_pos, posinschain_list, _targets) = batch
            name = str(frame.iloc[batch_index]["pdb"])
            position_to_patch = {}
            for row in by_input[name]:
                key = str(row["pdb_posins"]).strip() + str(row["pdb_chain"])
                if key in position_to_patch:
                    raise RuntimeError(f"overlapping CDR labels at {name}:{key}")
                position_to_patch[key] = row
            ordered_positions = [str(value) for value in posinschain_list[0]]
            if {key for key in ordered_positions if key in position_to_patch} != set(position_to_patch):
                raise RuntimeError(f"CDR position mapping failed for {name}")

            encoder_out = model.encoder(coords, padding_mask, confidence)
            generated_tokens = tokens[:, :-1].clone()
            cdr_columns = {
                index + 1: position_to_patch[key]
                for index, key in enumerate(ordered_positions) if key in position_to_patch
            }
            for column in cdr_columns:
                generated_tokens[:, column] = mask_token_id
            incremental_state = {}
            for column in range(1, generated_tokens.shape[1]):
                logits, _ = model.decoder(
                    generated_tokens[:, :column], encoder_out=encoder_out, incremental_state=incremental_state
                )
                if column not in cdr_columns:
                    continue
                meta = cdr_columns[column]
                log_ps = torch.log_softmax(logits[0, aa_token_ids, -1].float(), dim=0).cpu().numpy()
                ps = np.exp(log_ps)
                pe = normalize_log(priors[meta["patch_id"]]["all_cdr_mask_log_probabilities"])
                fused = product_of_experts(ps, pe, args.weight)
                selected = int(np.argmax(fused))
                generated_tokens[0, column] = aa_token_ids[selected]
                outputs.append(
                    {
                        "patch_id": meta["patch_id"], "target_id": meta["target_id"],
                        "region": meta["region"], "native_amino_acid": meta.get("native_amino_acid"),
                        "structure_log_probabilities": log_ps.tolist(),
                        "fused_probabilities": fused.tolist(),
                        "greedy_amino_acid": AMINO_ACIDS[selected],
                        "probability_mode": "closed-loop fixed global PoE",
                        "weight": args.weight, "native_CDR_tokens_visible": False,
                    }
                )
            print(json.dumps({"targets_processed": batch_index + 1, "targets_total": len(frame),
                              "model_input_name": name}), flush=True)

    if len(outputs) != len(manifest_rows) or len({row["patch_id"] for row in outputs}) != len(outputs):
        raise RuntimeError("incomplete or duplicate closed-loop outputs")
    by_id = {row["patch_id"]: row for row in outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(by_id[row["patch_id"]], sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {
        "status": "COMPLETE", "targets": len(frame), "residues": len(outputs),
        "native_CDR_tokens_visible": False, "checkpoint_sha256": sha256(args.checkpoint),
        "probability_mode": "closed-loop fixed global PoE", "weight": args.weight,
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

