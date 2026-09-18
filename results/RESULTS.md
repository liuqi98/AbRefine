# Frozen benchmark results

## Primary recent non-RAbD panel

- 100 paired antibodies, 200 chains and 5,065 evaluated CDR residues.
- Release dates: 2024-01-23 through 2026-02-23.
- No exact-PDB, antibody-90%, or antigen-90% overlap with the historical development collection.
- Official original AntiFold only; no continued-CE comparator.

| Method | Recovery | Change vs AntiFold | Native NLL | Perplexity |
|---|---:|---:|---:|---:|
| Official AntiFold | 64.85% | 0.00 pp | 1.3217 | 3.750 |
| AntiBERTy, all-CDR mask | 56.07% | -8.78 pp | 1.5695 | 4.804 |
| Lower-entropy expert | 66.97% | +2.12 pp | 1.2950 | 3.651 |
| Linear mixture | 67.99% | +3.14 pp | 1.2410 | 3.459 |
| CDR-specific PoE | 67.86% | +3.01 pp | 1.1694 | 3.220 |
| Equal PoE | 68.06% | +3.21 pp | 1.1558 | 3.177 |
| Fixed PoE, `g=0.59` | **68.03%** | **+3.18 pp** | 1.1723 | 3.230 |

The equal-PoE result is a test-set ablation. The primary method remains `g=0.59` because it was selected on development data before test evaluation.

## Secondary family-disjoint reserve

| Endpoint | AntiFold | Fixed PoE | Change | 95% CI |
|---|---:|---:|---:|---:|
| Open-loop recovery | 65.51% | 66.98% | +1.47 pp | [+0.07, +2.86] |
| Closed-loop recovery | 65.51% | 66.51% | +1.00 pp | [-0.62, +2.59] |
| Open-loop NLL | 1.3108 | 1.1649 | -0.1459 | [-0.2012, -0.0916] |
| Closed-loop NLL | 1.3108 | 1.1925 | -0.1182 | [-0.1823, -0.0558] |

The reserve contains 75 paired antibodies and 3,848 CDR residues. Heavy-chain antibody-90%, antigen-90%, and joint connected components are disjoint; light-chain families are not fully disjoint. It is not described as temporal post-cutoff data.

