#!/usr/bin/env python3
"""Closed-loop official AntiFold greedy decoding with a frozen IAF ensemble."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from antifold.antiscripts import get_dataset_dataloader, load_IF1_checkpoint
from antifold.esm.pretrained import _load_IF1_local
from gatefree_antibody_fusion.iaf import IAFEnsemble


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_unique(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {row["patch_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate patch identifiers in {path}")
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--empirical", type=Path, required=True)
    parser.add_argument("--iaf-model", type=Path, required=True,
                        help="portable models/iaf_antifold.json ensemble")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    priors, empirical = load_unique(args.priors), load_unique(args.empirical)
    patch_ids = {row["patch_id"] for row in manifest_rows}
    if patch_ids != set(priors) or patch_ids != set(empirical):
        raise RuntimeError("manifest/evolution/empirical residue identifiers differ")
    by_input: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        by_input[row["model_input_name"]].append(row)
    frame = pd.read_csv(args.inputs)
    if set(frame["pdb"].astype(str)) != set(by_input):
        raise RuntimeError("input CSV and manifest targets differ")

    device = torch.device(args.device)
    model, _ = _load_IF1_local()
    model = load_IF1_checkpoint(model, str(args.checkpoint)).eval().to(device)
    iaf = IAFEnsemble.load(args.iaf_model)
    alphabet = model.decoder.dictionary
    aa_token_ids = torch.tensor(
        [alphabet.get_idx(aa) for aa in AMINO_ACIDS], dtype=torch.long, device=device
    )
    mask_token_id = alphabet.get_idx("<mask>")
    _, dataloader = get_dataset_dataloader(
        frame, str(args.pdb_dir), batch_size=1, custom_chain_mode=False, num_threads=0
    )

    outputs, generated_receipts = [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            (coords, confidence, _strs, tokens, padding_mask, _loss_masks,
             _res_pos, posinschain_list, _targets) = batch
            name = str(frame.iloc[batch_index]["pdb"])
            position_to_patches: dict[str, list[dict]] = defaultdict(list)
            for row in by_input[name]:
                position_to_patches[
                    str(row["pdb_posins"]).strip() + str(row["pdb_chain"])
                ].append(row)
            ordered = [str(value) for value in posinschain_list[0]]
            if {key for key in ordered if key in position_to_patches} != set(position_to_patches):
                raise RuntimeError(f"CDR position mapping failed for {name}")
            if any(len(values) != 1 for values in position_to_patches.values()):
                raise RuntimeError(f"overlapping CDR labels for {name}")
            encoder_out = model.encoder(coords, padding_mask, confidence)
            generated_tokens = tokens[:, :-1].clone()
            cdr_columns = {
                index + 1: position_to_patches[key][0]
                for index, key in enumerate(ordered) if key in position_to_patches
            }
            for column in cdr_columns:
                generated_tokens[:, column] = mask_token_id
            incremental_state = {}
            for column in range(1, generated_tokens.shape[1]):
                logits, _ = model.decoder(
                    generated_tokens[:, :column], encoder_out=encoder_out,
                    incremental_state=incremental_state,
                )
                if column not in cdr_columns:
                    continue
                meta = cdr_columns[column]
                structure_logp_tensor = torch.log_softmax(
                    logits[0, aa_token_ids, -1].float(), dim=0
                )
                structure_logp = structure_logp_tensor.cpu().numpy()
                evolution_logp = np.asarray(
                    priors[meta["patch_id"]]["all_cdr_mask_log_probabilities"], dtype=np.float64
                )
                empirical_row = empirical[meta["patch_id"]]
                empirical_probability = np.asarray(
                    empirical_row.get("empirical_probability",
                                      empirical_row.get("empirical_probabilities",
                                                        empirical_row.get("probabilities"))),
                    dtype=np.float64,
                )
                fused_probability, gate, features = iaf.refine(
                    structure_logp, evolution_logp, empirical_probability
                )
                selected = int(np.argmax(fused_probability))
                generated_tokens[0, column] = aa_token_ids[selected]
                outputs.append({
                    "schema_version": "antifold-iaf-closed-loop-v1",
                    "patch_id": meta["patch_id"], "target_id": meta["target_id"],
                    "split": meta["split"], "region": meta["region"],
                    "native_amino_acid": meta["native_amino_acid"],
                    "structure_log_probabilities": structure_logp.tolist(),
                    "fused_log_probabilities": np.log(np.clip(fused_probability, 1e-12, None)).tolist(),
                    "iaf_gate": float(gate),
                    "iaf_features": features.tolist(),
                    "greedy_amino_acid": AMINO_ACIDS[selected],
                    "probability_mode": "closed-loop three-seed IAF ensemble",
                    "native_CDR_tokens_visible": False,
                })
            generated_receipts.append({
                "model_input_name": name,
                "generated_sequence_with_gap": "".join(
                    alphabet.get_tok(int(token)) for token in generated_tokens[0, 1:].cpu()
                ),
            })
            print(json.dumps({"targets_processed": batch_index + 1,
                              "targets_total": len(frame), "model_input_name": name}),
                  flush=True)

    if len(outputs) != len(manifest_rows) or len({row["patch_id"] for row in outputs}) != len(outputs):
        raise RuntimeError("incomplete or duplicate closed-loop output")
    by_id = {row["patch_id"]: row for row in outputs}
    write_jsonl(args.output, [by_id[row["patch_id"]] for row in manifest_rows])
    receipt = {
        "schema_version": "antifold-iaf-closed-loop-receipt-v1", "status": "COMPLETE",
        "targets": len(frame), "residues": len(outputs),
        "probability_mode": "closed-loop three-seed IAF ensemble",
        "test_set_tuning": False, "native_CDR_tokens_visible": False,
        "generated_sequences": generated_receipts,
    }
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: receipt[key] for key in ("status", "targets", "residues")},
                     indent=2))


if __name__ == "__main__":
    main()
