# Data contracts

The lightweight IAF CLI consumes three JSONL files joined by `patch_id`.

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

## Development-derived empirical prior

IAF uses a generic CDR distribution only to measure how informative the
framework-conditioned prediction is. Each record contains:

```json
{
  "patch_id": "target:H3:0",
  "target_id": "target",
  "region": "H3",
  "empirical_probabilities": ["20 non-negative values in ACDEFGHIKLMNPQRSTVWY order"]
}
```

In the reported experiments this prior is estimated exclusively from the
62-antibody development cohort using CDR type and one of ten normalized loop
position bins with a 0.5 pseudocount. The prior is frozen before evaluation.
Native residues from evaluation targets must never be used to construct it.

## IAF output

`abrefine iaf` writes the refined distribution, predicted residue, learned
residue-specific coefficient, and the four information descriptors:

```json
{
  "fused_probabilities": ["20 probabilities"],
  "predicted_amino_acid": "Y",
  "iaf_gate": 0.43,
  "iaf_features": {
    "structure_entropy": 1.2,
    "framework_entropy": 1.6,
    "structure_to_framework_kl": 0.8,
    "framework_information_kl": 0.5
  }
}
```

Native labels and correctness/NLL fields are emitted only when
`native_amino_acid` is present and are used solely for offline evaluation.

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
