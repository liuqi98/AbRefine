#!/usr/bin/env python3
"""Score AntiBERTy while masking every CDR residue on a chain simultaneously."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=Path, required=True, help="Chain JSONL described in docs/DATA_SCHEMA.md")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True, help="Official AntiBERTy repository root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.model_root.resolve()))
    from antiberty import AntiBERTy  # noqa: E402
    from transformers import BertConfig, BertTokenizer  # noqa: E402

    checkpoint = args.model_root / "antiberty/trained_models/AntiBERTy_md_smooth"
    tokenizer = BertTokenizer(
        vocab_file=str(args.model_root / "antiberty/trained_models/vocab.txt"), do_lower_case=False
    )
    model = AntiBERTy(BertConfig.from_json_file(str(checkpoint / "config.json")))
    state = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or set(unexpected) - {"bert.embeddings.position_ids"}:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    device = torch.device(args.device)
    model.to(device).eval()
    aa_ids = tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS))

    chains = read_jsonl(args.chains)
    masked_texts = []
    for row in chains:
        cdr_indices = {int(item["sequence_index"]) for item in row["cdr_outputs"]}
        masked_texts.append(
            " ".join("[MASK]" if index in cdr_indices else amino_acid
                     for index, amino_acid in enumerate(row["sequence"]))
        )

    outputs = []
    with torch.no_grad():
        for start in range(0, len(chains), args.batch_size):
            batch = chains[start:start + args.batch_size]
            encoded = tokenizer(
                masked_texts[start:start + args.batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).prediction_logits[:, :, aa_ids]
            for local_index, row in enumerate(batch):
                for item in row["cdr_outputs"]:
                    values = torch.log_softmax(
                        logits[local_index, int(item["sequence_index"]) + 1], dim=-1
                    )
                    outputs.append(
                        {
                            "patch_id": item["patch_id"], "target_id": row["target_id"],
                            "region": item["region"], "native_amino_acid": item.get("native_amino_acid"),
                            "all_cdr_mask_log_probabilities": values.cpu().tolist(),
                            "native_CDR_tokens_visible": False,
                        }
                    )
            print(json.dumps({"chains_processed": min(start + args.batch_size, len(chains)),
                              "chains_total": len(chains)}), flush=True)

    if len({row["patch_id"] for row in outputs}) != len(outputs):
        raise RuntimeError("duplicate output identifiers")
    order = [item["patch_id"] for row in chains for item in row["cdr_outputs"]]
    by_id = {row["patch_id"]: row for row in outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for patch_id in order:
            handle.write(json.dumps(by_id[patch_id], sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {
        "status": "COMPLETE", "targets": len({row["target_id"] for row in chains}),
        "chains": len(chains), "residues": len(outputs), "device": str(device),
        "native_CDR_tokens_visible": False,
        "checkpoint_sha256": sha256(checkpoint / "pytorch_model.bin"),
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

