# Gate-free Structural–Evolutionary Prior Fusion for Antibody Inverse Folding

[![tests](https://github.com/liuqi98/GateFree-Antibody-Prior-Fusion/actions/workflows/tests.yml/badge.svg)](https://github.com/liuqi98/GateFree-Antibody-Prior-Fusion/actions/workflows/tests.yml)

A leakage-safe, training-free method that combines the **official original AntiFold** structure distribution with an **all-CDR-mask AntiBERTy** evolutionary prior. The method improves CDR native-sequence recovery without continued-CE fine-tuning or a learned residue gate.

> **Repository status:** research code accompanying an unpublished manuscript. The repository is private. Model checkpoints and raw structures are not redistributed.

## 中文简介

本项目研究固定抗体骨架条件下的 CDR 逆折叠：输入配对重链/轻链骨架和已知 framework，在不知道原生 CDR 序列的条件下生成 CDR。官方原始 AntiFold 作为结构专家，AntiBERTy 在同一条链的全部 CDR 同时被遮蔽时提供进化先验；两个冻结分布使用一个开发集选定的全局权重进行 product-of-experts 融合。方法不重新训练 AntiFold、不使用 continued-CE，也不训练残基级 gate。

在公开的 AntiFold/AbMPNN SAbDab 测试划分上，可重建的 362 个配对 Fab（17,945 个 CDR 残基）中，目标宏平均 recovery 从 **68.89% 提高到开放环 70.33%（+1.44 pp）**；闭环为 **69.67%（+0.77 pp）**，但其置信区间跨零。作为外部确认，在 100 个近期、非 RAbD 抗体上，开放环与闭环分别提升 **+3.18 pp** 和 **+3.17 pp**。因此，结果支持进化先验改善概率与开放环恢复率，但闭环提升具有数据集和 CDR 依赖性。

## Motivation and application

Antibody inverse folding must reconcile two sources of information:

- **Structural compatibility:** which amino acids are consistent with a fixed antibody backbone?
- **Evolutionary plausibility:** which amino acids are typical in antibody sequence context?

This setting is useful for fixed-backbone CDR redesign, generation of candidate CDR sequences, and prioritization before structure prediction, developability filtering or experimental testing. Recovery is a reconstruction metric; it does **not** by itself establish antigen binding or therapeutic value.

## Method

At CDR residue `i`, official AntiFold produces a structure distribution

```text
pS_i(a) = P_AntiFold(c_i = a | backbone, framework, generated CDR prefix)
```

and AntiBERTy produces an evolutionary distribution after masking every CDR residue on that chain:

```text
pE_i(a) = P_AntiBERTy(c_i = a | framework, all CDRs = MASK)
```

The fused distribution is

```text
q_i(a) ∝ pS_i(a)^(1-g) · pE_i(a)^g,
g = 0.59.
```

The scalar `g` was selected on 62 development antibodies and then frozen for all test targets, CDRs, positions and amino-acid types. `g=0` exactly recovers AntiFold and `g=1` exactly recovers AntiBERTy.

Although the weight is global, the effect is implicitly position dependent. A sharp AntiFold distribution is difficult for the prior to overturn, whereas a diffuse high-entropy distribution leaves more room for AntiBERTy to change the ranking.

### Closed-loop decoding

In closed-loop mode, the fused argmax at residue `i` is written into the generated sequence and becomes part of AntiFold's context for residue `i+1`. This tests the deployable autoregressive trajectory instead of only re-ranking probabilities recorded along the original AntiFold trajectory.

## Why the evolutionary prior is not leakage

Using a pretrained antibody language model is not inherently information leakage. The legal deployment contract is:

1. all CDR residues on a chain are masked simultaneously;
2. AntiBERTy sees only the framework and mask tokens;
3. neither expert receives a native CDR identity;
4. native residues are used only for development-set selection and offline evaluation.

Masking only the current residue while leaving other native CDR residues visible is **not** allowed. The CLI checks the explicit `native_CDR_tokens_visible: false` contract and refuses inputs that do not satisfy it.

## Main results

All values are target-macro averages. Confidence intervals use 10,000 paired target-level bootstrap replicates.

| Panel / evaluation | Official AntiFold | Fixed PoE | Difference | 95% CI |
|---|---:|---:|---:|---:|
| Public AntiFold/AbMPNN split, open loop; 362/405 paired Fabs, 17,945 residues | 68.89% | 70.33% | **+1.44 pp** | [+0.67, +2.20] |
| Public AntiFold/AbMPNN split, closed loop | 68.89% | 69.67% | +0.77 pp | [-0.09, +1.64] |
| Recent SAbDab2, open loop; 100 targets, 5,065 residues | 64.85% | 68.03% | **+3.18 pp** | [+2.06, +4.36] |
| Recent SAbDab2, closed loop | 64.85% | 68.02% | **+3.17 pp** | [+1.80, +4.58] |
| Family-disjoint reserve, open loop; 75 targets, 3,848 residues | 65.51% | 66.98% | **+1.47 pp** | [+0.07, +2.86] |
| Family-disjoint reserve, closed loop | 65.51% | 66.51% | +1.00 pp | [-0.62, +2.59] |

The public-split and secondary closed-loop recovery estimates are positive but statistically inconclusive. Their closed-loop native NLL values improve by `-0.089` (95% CI `[-0.117, -0.061]`) and `-0.118` (`[-0.182, -0.056]`), respectively. On the public split, the closed-loop gain is significant for H1 and H2 but not universal across CDRs.

### Mechanistic and robustness results

| Analysis | Result |
|---|---:|
| Low AntiFold entropy | +0.44 pp |
| Middle AntiFold entropy | **+3.61 pp** |
| High AntiFold entropy | **+5.64 pp** |
| Closed-loop wrong → correct | 385 residues |
| Closed-loop correct → wrong | 227 residues |
| Net corrections | **158 residues** |
| ProteinMPNN → ProteinMPNN + AntiBERTy | 35.04% → 61.85% |
| Fusion arithmetic | 0.31–0.35 μs per residue on CPU |

The ProteinMPNN experiment tests cross-base portability; the fused ProteinMPNN result remains below AntiFold and is not presented as a new state of the art.

## Installation

The core fusion and evaluation package requires only NumPy:

```bash
git clone https://github.com/liuqi98/GateFree-Antibody-Prior-Fusion.git
cd GateFree-Antibody-Prior-Fusion
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest -q
```

Running the expert-scoring scripts additionally requires PyTorch, Transformers, and locally installed official AntiFold/AntiBERTy code and checkpoints:

```bash
python -m pip install -r requirements-models.txt
```

This repository does not redistribute third-party model weights. Follow the original projects' licenses and download instructions.

## Quick start from precomputed probabilities

The amino-acid order is `ACDEFGHIKLMNPQRSTVWY`. Input formats are documented in [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

```bash
gatefree-fusion fuse \
  --structure path/to/antifold.jsonl \
  --evolution path/to/antiberty_all_cdr_mask.jsonl \
  --weight 0.59 \
  --output outputs/fused.jsonl

gatefree-fusion evaluate \
  --input outputs/fused.jsonl \
  --output outputs/summary.json
```

A two-residue smoke-test input is included under `examples/`:

```bash
gatefree-fusion fuse --structure examples/structure.jsonl --evolution examples/evolution.jsonl --output /tmp/demo_fused.jsonl
gatefree-fusion evaluate --input /tmp/demo_fused.jsonl --output /tmp/demo_summary.json
```

## Generate the all-CDR-mask AntiBERTy prior

```bash
python scripts/score_antiberty_all_cdr_mask.py \
  --chains path/to/chains.jsonl \
  --model-root path/to/official/AntiBERTy \
  --output outputs/antiberty_all_cdr_mask.jsonl
```

The script creates one masked input per chain with **every CDR position masked at the same time**.

## Closed-loop official AntiFold decoding

Run this inside an environment in which the official `antifold` Python package is importable:

```bash
python scripts/score_antifold_fixed_poe_closed_loop.py \
  --checkpoint path/to/official_original_antifold_checkpoint.pt \
  --inputs path/to/antifold_inputs.csv \
  --pdb-dir path/to/paired_HL_pdbs \
  --manifest path/to/residue_manifest.jsonl \
  --priors outputs/antiberty_all_cdr_mask.jsonl \
  --weight 0.59 \
  --output outputs/fixed_poe_closed_loop.jsonl
```

## Repository layout

```text
src/gatefree_antibody_fusion/       Core PoE, data contracts, statistics and CLI
scripts/                            AntiBERTy scoring and closed-loop AntiFold decoding
docs/DATA_SCHEMA.md                 Input schemas and leakage contract
results/                            Frozen aggregate benchmark results
tests/                              Unit tests
```

## Reproducibility choices

- Official original AntiFold is the primary baseline.
- No continued-CE checkpoint is compared.
- The fusion weight is selected on a separate development set.
- Antibody targets, not residues, are the independent bootstrap units.
- The public AntiFold/AbMPNN split provides community-split comparability; 362/405 paired Fabs are reconstructable locally and coverage is reported explicitly.
- Published AntiFold numbers use full sequence context, whereas this repository's matched comparison hides every native CDR token; these conditioning protocols are not mixed.
- The recent external panel excludes exact PDB, antibody-90%, and antigen-90% overlap with the historical development collection.
- The secondary reserve is family-disjoint but is not claimed as temporal post-cutoff data.
- Open-loop and closed-loop results are reported separately.

## Limitations

- Structure–language-model ensembling is not by itself a new concept; the contribution is the strict no-native-CDR protocol, gate-free transfer result and antibody-specific evaluation.
- Closed-loop top-1 recovery is dataset- and CDR-dependent: its interval crosses zero on the public split and family-disjoint reserve despite consistent likelihood improvements.
- Native recovery does not prove that the native sequence is the only valid design.
- The current evidence does not establish binding, specificity, developability or experimental success.
- Sampled-sequence refolding and novelty–fidelity analysis are natural next steps toward de novo design.

## Citation

The manuscript is currently unpublished. Please cite the repository URL and commit hash until a formal citation is available. A publication citation will be added after acceptance.

## License and third-party assets

No open-source license is granted yet because the manuscript is under preparation. All rights are reserved by the repository owner. AntiFold, AntiBERTy and ProteinMPNN remain subject to their respective licenses; their code and checkpoints are not included here.
