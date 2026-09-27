"""JSONL contracts and leakage checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_unique_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            patch_id = row.get("patch_id")
            if not patch_id:
                raise ValueError(f"missing patch_id in {path}:{line_number}")
            if patch_id in rows:
                raise ValueError(f"duplicate patch_id {patch_id!r} in {path}")
            rows[patch_id] = row
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def validate_pair(structure: dict[str, dict], evolution: dict[str, dict]) -> None:
    if set(structure) != set(evolution):
        missing_structure = sorted(set(evolution) - set(structure))[:5]
        missing_evolution = sorted(set(structure) - set(evolution))[:5]
        raise ValueError(
            "expert residue identifiers differ; "
            f"missing from structure={missing_structure}, missing from evolution={missing_evolution}"
        )
    for patch_id in structure:
        srow, erow = structure[patch_id], evolution[patch_id]
        if srow.get("native_CDR_tokens_visible") is not False:
            raise ValueError(f"structure leakage contract missing or violated at {patch_id}")
        if erow.get("native_CDR_tokens_visible") is not False:
            raise ValueError(f"evolution leakage contract missing or violated at {patch_id}")
        for key in ("target_id", "region"):
            if srow.get(key) != erow.get(key):
                raise ValueError(f"{key} mismatch at {patch_id}")
        if "native_amino_acid" in srow and "native_amino_acid" in erow:
            if srow["native_amino_acid"] != erow["native_amino_acid"]:
                raise ValueError(f"native label mismatch at {patch_id}")

