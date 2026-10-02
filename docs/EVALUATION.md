# Evaluation and scoring

## Probability space

Align decoder scores to ACDEFGHIKLMNPQRSTVWY, remove special tokens and X,
and normalize over these 20 amino acids before entropy, KL, gating, fusion,
or NLL computation. Training and inference use the same rule. The core API
rejects 21-column inputs rather than guessing their vocabulary order.

`abrefine.metrics.native_residue_nll(scores, native)` normalizes retained,
untempered scores using log-softmax and returns a context-specific residue
score. It does not compute native-sequence joint likelihood.

If alpha is the full decoder's probability mass on the 20 retained residues,
NLL20 = NLLfull + log(alpha). The AbMPNN scoring correction replays saved
sequences with their original orders and checkpoint, without regeneration or
gate retraining. The released receipt records the replay check; corrected
summary statistics are in results/sampling_comparison_20aa.json.

## Protocol map

| Evaluation | Expert | Decoding | NLL context |
|---|---|---|---|
| Recent-100, 20 samples | AntiFold | Fixed-table open loop | Common baseline prefix |
| Recent-100, 20 samples | AbMPNN | Random-order closed loop | Each method's sampled prefix |
| Three-set greedy comparison | AntiFold | Open loop | Common baseline prefix |
| Feedback comparison | AntiFold | Greedy open/closed loop | Baseline / own prefix |
| Separate fold-back designs | AbMPNN / AntiFold | Greedy open / closed loop | Locally aligned CDR RMSD |

Sampling applies temperature 0.2 to each method's final normalized 20-AA
distribution, after probability-level ensemble averaging for IAF. Features and
gates use untempered expert probabilities. Fixed-table native-label NLL at
temperature 1 is unchanged across samples.

Scores are averaged over residues and samples within each antibody, then
macro-averaged across antibodies. Paired bootstrap uses 10,000 target resamples.
Combined analyses resample within cohorts and do not model shared-PDB dependence.

## Development

The shared anchor 0.59 is selected using AntiFold development recovery and held
fixed across folds. Five-fold validation conditionally selects the gate penalty
from 0, 0.01, 0.1, 1 by target-macro NLL. Priors and feature standardization are
fitted within training folds. Training-target priors exclude the target itself;
validation priors contain only training-fold counts. Final external evaluation
uses frozen all-development preprocessing and models. Optimizer settings and
early stopping are specified in scripts/train_iaf.py.

## Fold-back and controls

Fold-back uses unrefined ABodyBuilder2 model 1. N, CA, C and O atoms are aligned
independently for each CDR, and six CDR RMSDs are averaged per antibody.
This measures local loop shape on separate deterministic designs.

Shuffled donors match chain role, length, and CDR-mask layout. Structural
probabilities, empirical priors, normalization statistics and learned parameters
are fixed; framework probabilities, dependent features and gates are recomputed.
Each donor is evaluated separately, followed by within-target averaging.
Mechanistic strata use the matched-framework descriptors.

The generic-prior control replaces AntiBERTy with development amino-acid
frequencies grouped by CDR type and relative-position bin, using independently
tuned static weight 0.43. It is separate from the prior used in IAF features.
