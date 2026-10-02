# Frozen AbRefine results

All evaluation models and preprocessing parameters were selected using only
the 62-antibody development cohort (3,077 CDR residues). Values are
target-macro averages; intervals are paired 10,000-replicate target bootstrap
intervals.

## Cross-cohort generalization

| Cohort | Targets | Residues | AntiFold | IAF | Change (95% CI) | AntiFold NLL | IAF NLL | NLL change (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Public-359 | 359 | 17,816 | 68.91% | **71.17%** | **+2.25 pp** [+1.66, +2.86] | 1.1709 | **1.0352** | **-0.1357** [-0.1544, -0.1178] |
| Recent-100 | 100 | 5,065 | 64.85% | **67.93%** | **+3.08 pp** [+2.15, +4.10] | 1.3217 | **1.1542** | **-0.1676** [-0.2051, -0.1339] |
| Family-75 | 75 | 3,848 | 65.51% | **67.29%** | **+1.78 pp** [+0.76, +2.82] | 1.3108 | **1.1480** | **-0.1627** [-0.2077, -0.1201] |

## Cross-expert sampled protocol

Recent-100 uses 20 samples per antibody at temperature 0.2. AntiFold samples
independently from fixed open-loop probability tables; AbMPNN updates structural
predictions and gates along each method's sampled trajectory. AbMPNN methods
share random decoding orders and uniform draws, but their prefixes can differ.

| Expert | Original | Static PoE | IAF | IAF vs original (95% CI) | IAF vs static (95% CI) |
|---|---:|---:|---:|---:|---:|
| AbMPNN | 56.50% | 64.32% | **65.85%** | **+9.35 pp** [+7.43, +11.48] | **+1.53 pp** [+0.93, +2.14] |
| AntiFold | 64.53% | 67.30% | **67.48%** | **+2.95 pp** [+2.07, +3.93] | +0.18 pp [-0.22, +0.56] |

Comparative NLL uses distributions renormalized over the same 20 amino acids
at temperature 1. The corrected AbMPNN baseline NLL is 1.4774, versus 1.3161
for AbMPNN+Static and 1.1989 for AbMPNN+IAF (IAF-minus-baseline: -0.2785).
The AntiFold change is -0.1676. AntiFold NLL uses common baseline contexts;
AbMPNN NLL uses each method's own sampled contexts. See
[scoring audit](abmpnn_20aa_rescoring_receipt.json) and
[corrected sampled results](sampling_comparison_20aa.json).

Separate deterministic designs were folded with unrefined ABodyBuilder2 model 1.
Each CDR was independently aligned before backbone RMSD calculation. AbMPNN
uses greedy open-loop designs and AntiFold greedy closed-loop designs. Mean
RMSD changes are 0.8627 to 0.8672 Å and 0.8494 to 0.8486 Å, respectively.

## Mechanistic controls

- Matched minus shuffled framework: +5.85 pp [95% CI +5.26, +6.47].
- High structural uncertainty plus high framework information: +7.41 pp
  [95% CI +5.53, +9.30].
- Full framework exceeds local-only and distant-only context by +1.90 and
  +1.68 pp.
- Combined AntiFold+IAF closed-loop minus open-loop recovery: +0.09 pp
  [95% CI -0.19, +0.36].

IAF improves likelihood relative to static fusion on every frozen cohort.
Its additional top-1 recovery over static fusion is significant on the public
benchmark and for AbMPNN under the sampled protocol, but not for AntiFold on
the recent panel.

## Evaluation definitions

Public-359 excludes PDB IDs present in the development collection. Recent-100
contains recent SAbDab structures; Family-75 is a reserve with heavy-chain and
antigen families disjoint from development. The combined 534 cohort records
represent 533 unique PDB/H/L pairs and 447 PDB entries, not 534 unique antibodies.

+Static denotes fixed-weight AntiBERTy fusion (AntiFold: 0.59; AbMPNN: 0.69),
selected separately on development data. +IAF denotes residue-specific fusion.
For shuffled-framework controls, framework probabilities, information features,
and gates are recomputed while structural predictions and learned parameters
remain fixed. Recovery is averaged over ten donor assignments within each target.
The empirical-prior control instead uses development CDR-type/position frequencies
with its own development-selected static weight (0.43).

Detailed protocols and scoring conventions: [../docs/EVALUATION.md](../docs/EVALUATION.md).
