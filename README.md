# AbRefine

[![tests](https://github.com/liuqi98/GateFree-Antibody-Prior-Fusion/actions/workflows/tests.yml/badge.svg)](https://github.com/liuqi98/GateFree-Antibody-Prior-Fusion/actions/workflows/tests.yml)

**Framework-conditioned, information-aware refinement for structure-guided antibody CDR design.**

AbRefine combines a frozen antibody structure expert with a frozen antibody
language model while keeping every designed CDR residue hidden. Its
Information-Aware Fusion (IAF) module learns a residue-specific product-of-
experts coefficient from only four distributional descriptors. The released
IAF network has 49 trainable parameters; neither pretrained expert is fine-
tuned.

> Repository status: private research code accompanying the AbRefine
> manuscript. Third-party checkpoints, raw SAbDab structures and native test
> sequences are not redistributed.

## Overview

For residue `i`, the structure expert provides `P_S`, and AntiBERTy provides a
framework-conditioned distribution `P_F` after **all CDR residues on the
corresponding chain are masked simultaneously**. IAF computes

```text
phi_i = [H(P_S), H(P_F), KL(P_S || P_F), KL(P_F || P_emp)]
g_i   = sigmoid(logit(0.59) + MLP_4-8-1(standardize(phi_i)))
P_R(a) proportional to P_S(a)^(1-g_i) * P_F(a)^g_i
```

`P_emp` is a development-only empirical distribution indexed by CDR type and
normalized loop position. The final prediction is the mean of three IAF
members trained with seeds 41, 42 and 43.

The native CDR sequence is never a model input. Native residues are used only
for development-set fitting or retrospective evaluation.

## Main evidence

### Generalization across independent cohorts

![Frozen IAF generalization across three cohorts](docs/figures/cross_cohort_generalization.png)

The same frozen AntiFold-based IAF ensemble improves recovery and native NLL
on all three evaluation cohorts:

| Cohort | Targets | AntiFold | AntiFold + IAF | Recovery change (95% CI) | NLL change (95% CI) |
|---|---:|---:|---:|---:|---:|
| Public benchmark | 359 | 68.91% | **71.17%** | **+2.25 pp** [+1.66, +2.86] | **-0.136** [-0.154, -0.118] |
| Recent external | 100 | 64.85% | **67.93%** | **+3.08 pp** [+2.15, +4.10] | **-0.168** [-0.205, -0.134] |
| Family-disjoint reserve | 75 | 65.51% | **67.29%** | **+1.78 pp** [+0.76, +2.82] | **-0.163** [-0.208, -0.120] |

Values are antibody-macro averages. Confidence intervals use 10,000 paired
target-level bootstrap replicates. The public cohort excludes three exact-PDB
overlaps with development.

### Portability across structural experts

![IAF recovery and fold-back evaluation](docs/figures/cross_expert_fidelity.png)

Under the released 20-sample, `T=0.2` protocols on the recent panel:

| Structural expert | Original recovery | Static PoE | IAF | IAF vs original | IAF vs static |
|---|---:|---:|---:|---:|---:|
| AbMPNN | 56.50% | 64.32% | **65.85%** | **+9.35 pp** | **+1.53 pp** |
| AntiFold | 64.53% | 67.30% | **67.48%** | **+2.95 pp** | +0.18 pp |

IAF significantly reduces native NLL for both experts. Fold-back CDR RMSD does
not show detectable aggregate degradation relative to either original expert.
For AntiFold, IAF's additional recovery over static PoE is not significant;
its incremental benefit is primarily improved likelihood.

### Framework-specific mechanism

![Matched-framework controls](docs/figures/framework_controls.png)

- Matched frameworks exceed exact-mask-matched shuffled frameworks by
  **+5.85 pp** (95% CI +5.26 to +6.47).
- The largest gain occurs where the structural distribution is uncertain and
  the framework distribution is informative: **+7.41 pp**.
- Removing framework information or replacing it with a generic empirical
  prior does not reproduce the matched-framework result.

## Installation

Portable inference requires only NumPy:

```bash
git clone https://github.com/liuqi98/GateFree-Antibody-Prior-Fusion.git
cd GateFree-Antibody-Prior-Fusion
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest -q
```

Running expert-scoring or training scripts additionally requires PyTorch,
Transformers, and local installations of the original expert models:

```bash
python -m pip install -r requirements-models.txt
```

## Quick start: frozen IAF inference

The amino-acid order is `ACDEFGHIKLMNPQRSTVWY`. Input contracts are described
in [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

```bash
abrefine iaf \
  --structure examples/structure.jsonl \
  --evolution examples/evolution.jsonl \
  --empirical-prior examples/empirical_prior.jsonl \
  --model models/iaf_antifold.json \
  --output /tmp/abrefine_predictions.jsonl

abrefine evaluate \
  --input /tmp/abrefine_predictions.jsonl \
  --output /tmp/abrefine_summary.json
```

Two portable model ensembles are included:

- `models/iaf_antifold.json`: primary AntiFold-based IAF, regularization 0.01.
- `models/iaf_abmpnn.json`: AbMPNN portability model, regularization 0.

The JSON files contain the feature scaler and all 4-8-1 network parameters;
inference does not require PyTorch. The legacy fixed-PoE command remains
available as `abrefine fuse` for baseline reproduction.

## Development-only training

`scripts/train_iaf.py` reproduces target-grouped five-fold selection and the
three-seed final fit from matched development expert outputs:

```bash
python scripts/train_iaf.py \
  --structure path/to/development_structure.jsonl \
  --framework path/to/development_antiberty_allmask.jsonl \
  --base-expert AntiFold \
  --output outputs/iaf_antifold
```

The script constructs leave-target-out empirical priors inside development,
selects the anchoring coefficient by out-of-fold target-macro NLL, and exports
both PyTorch checkpoints and a portable JSON ensemble. Complete antibodies,
not individual residues, define every fold.

## Generate the leakage-safe framework distribution

```bash
python scripts/score_antiberty_all_cdr_mask.py \
  --chains path/to/chains.jsonl \
  --model-root path/to/official/AntiBERTy \
  --output outputs/antiberty_all_cdr_mask.jsonl
```

The scorer replaces every annotated CDR position on a chain with `[MASK]` in a
single input. Masking only the current residue while leaving other native CDR
tokens visible is not a valid AbRefine evaluation.

## Closed-loop design

In closed-loop mode, each IAF-refined residue is written back before the
structure expert predicts the next position. The language-model distribution
is computed once from the all-CDR-masked framework and remains fixed. The
reported pooled recovery is 70.02% open loop and 70.10% closed loop; their
paired difference is +0.09 pp (95% CI -0.19 to +0.36).

```bash
python scripts/score_antifold_iaf_closed_loop.py \
  --checkpoint path/to/original_antifold_checkpoint.pt \
  --inputs path/to/antifold_inputs.csv \
  --pdb-dir path/to/paired_HL_pdbs \
  --manifest path/to/residue_manifest.jsonl \
  --priors outputs/antiberty_all_cdr_mask.jsonl \
  --empirical outputs/empirical_prior.jsonl \
  --iaf-model models/iaf_antifold.json \
  --output outputs/closed_loop_iaf.jsonl
```

## Repository layout

```text
src/gatefree_antibody_fusion/
  fusion.py                 Legacy fixed-PoE baseline
  iaf.py                    Portable IAF features and ensemble inference
  io.py                     Leakage and identifier contracts
  metrics.py                Target-macro metrics and paired bootstrap
  cli.py                    `abrefine` command-line interface
models/                     Frozen AntiFold and AbMPNN IAF ensembles
scripts/                    Expert scoring, export and decoding utilities
docs/                       Schemas and manuscript figures
results/                    Frozen machine-readable benchmark summaries
tests/                      Numerical, leakage and statistics tests
```

## Reproducibility rules

- Development/model selection: 62 antibodies and 3,077 CDR residues only.
- Evaluation targets remain intact during splitting and bootstrap resampling.
- The public, recent and family-disjoint cohorts are never used for IAF fitting
  or threshold selection.
- All designed CDR identities are hidden from both experts.
- AntiFold, AbMPNN and AntiBERTy parameters remain frozen.
- The reported IAF prediction averages probability vectors from three seeds.
- Recovery, NLL, open-loop and closed-loop results are reported separately.
- Recovery measures native reconstruction, not binding or therapeutic value.

## Citation

The manuscript is currently under preparation. Until a preprint DOI is
available, cite this repository URL and the exact commit hash used.

## License and third-party assets

No open-source license is granted while the manuscript is under preparation;
all rights are reserved by the repository owner. Third-party models remain
subject to their original licenses and are not redistributed here.
