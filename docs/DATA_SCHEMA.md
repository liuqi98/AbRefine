# Data contracts

The lightweight fusion CLI consumes two JSONL files joined by `patch_id`.

## Structure expert

Required fields:

```json
{
  "patch_id": "target:H3:0",
  "target_id": "target",
  "region": "H3",
  "log_probabilities": ["20 finite values in ACDEFGHIKLMNPQRSTVWY order"],
  "native_CDR_tokens_visible": false
}
```

`native_amino_acid` is optional and is used only for offline evaluation.

## Evolution expert

Required fields:

```json
{
  "patch_id": "target:H3:0",
  "target_id": "target",
  "region": "H3",
  "all_cdr_mask_log_probabilities": ["20 finite values in ACDEFGHIKLMNPQRSTVWY order"],
  "native_CDR_tokens_visible": false
}
```

All CDR residues on the same chain must be masked in the AntiBERTy input. Masking only the scored residue while leaving the other native CDR residues visible violates the deployment contract.

## Chain input for AntiBERTy scoring

`scripts/score_antiberty_all_cdr_mask.py` expects one record per antibody chain:

```json
{
  "target_id": "target",
  "sequence": "EVQL...",
  "cdr_outputs": [
    {
      "patch_id": "target:H3:0",
      "sequence_index": 97,
      "region": "H3",
      "native_amino_acid": "A"
    }
  ]
}
```

`sequence_index` is zero based. The scorer constructs one input with every index listed under `cdr_outputs` replaced by `[MASK]`.

## Independent statistical unit

Recovery and NLL are first averaged within `target_id` and then across targets. Residues are not treated as independent bootstrap samples.

