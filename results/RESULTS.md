# Frozen AbRefine results

All evaluation models and preprocessing parameters were selected using only
the 62-antibody development cohort (3,077 CDR residues). Values are
target-macro averages; intervals are paired 10,000-replicate target bootstrap
intervals.

## Cross-cohort generalization

| Cohort | Targets | Residues | AntiFold | IAF | Change (95% CI) | AntiFold NLL | IAF NLL | NLL change (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Public exact-PDB-disjoint | 359 | 17,816 | 68.91% | **71.17%** | **+2.25 pp** [+1.66, +2.86] | 1.1709 | **1.0352** | **-0.1357** [-0.1544, -0.1178] |
| Recent external | 100 | 5,065 | 64.85% | **67.93%** | **+3.08 pp** [+2.15, +4.10] | 1.3217 | **1.1542** | **-0.1676** [-0.2051, -0.1339] |
| Family-disjoint reserve | 75 | 3,848 | 65.51% | **67.29%** | **+1.78 pp** [+0.76, +2.82] | 1.3108 | **1.1480** | **-0.1627** [-0.2077, -0.1201] |

## Cross-expert sampled protocol

The recent-panel comparison uses 20 samples per antibody at temperature 0.2
and common random numbers.

| Expert | Original | Static PoE | IAF | IAF vs original (95% CI) | IAF vs static (95% CI) |
|---|---:|---:|---:|---:|---:|
| AbMPNN | 56.50% | 64.32% | **65.85%** | **+9.35 pp** [+7.43, +11.48] | **+1.53 pp** [+0.93, +2.14] |
| AntiFold | 64.53% | 67.30% | **67.48%** | **+2.95 pp** [+2.07, +3.93] | +0.18 pp [-0.22, +0.56] |

The corresponding IAF-minus-original native-NLL changes are -0.2825 for
AbMPNN and -0.1676 for AntiFold. Aggregate CDR-local ABodyBuilder2 fold-back
RMSD does not show detectable degradation.

## Mechanistic controls

- Matched minus shuffled framework: +5.85 pp [95% CI +5.26, +6.47].
- High structural uncertainty plus high framework information: +7.41 pp
  [95% CI +5.53, +9.30].
- Full framework exceeds local-only and distant-only context by +1.90 and
  +1.68 pp.
- Pooled open-loop versus closed-loop IAF: +0.09 pp
  [95% CI -0.19, +0.36].

IAF improves likelihood relative to static fusion on every frozen cohort.
Its additional top-1 recovery over static fusion is significant on the public
benchmark and for AbMPNN under the sampled protocol, but not for AntiFold on
the recent panel.
