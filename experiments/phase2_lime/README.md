# Phase 2: LIME Evaluation for Zero-Day Attack Detection

## Objective
Compare local LIME explanations with the global Feature Attention Score (FAS)
ranking obtained in Phase 1 of FAS-XAI-ZDA.

## Model and data
- Model: Phase 1 full-feature attention model (`best_model.keras`)
- Input: 41 features, including the `level` feature
- Output: binary Normal/Attack prediction
- Zero-day attack categories: ftp_write, imap, loadmodule, multihop,
  perl, phf, rootkit, spy
- Zero-day evaluation subset: 54 samples across the eight categories

## LIME evaluation protocol
- Explainer: LimeTabularExplainer
- Mode: classification
- Continuous feature discretization: enabled
- Number of features per explanation: 10
- Perturbation samples per explanation: 3000
- Explanation target: Attack class
- Evaluation: fidelity, runtime, top-10 feature overlap with global FAS,
  Jaccard similarity, and zero-day detection performance

## Multi-sample stability protocol
- Selected samples: 2 per zero-day category (16 total)
- Random seeds: 11, 42, 55
- Perturbation samples per explanation: 3000
- Features compared: top-10 LIME features
- Similarity metric: pairwise Jaccard similarity

## Multi-sample stability results
- Mean Jaccard similarity: 0.520986
- Standard deviation: 0.102255
- Highest class-wise mean stability: loadmodule (0.5842)
- Lowest class-wise mean stability: imap (0.4835)

## Overall zero-day results
- Zero-day samples evaluated: 54
- Correctly detected attacks: 49
- Detection rate: 90.7407%
- Mean LIME fidelity: 0.225647
- Mean top-10 FAS overlap: 1.3889 features (13.8889%)
- Mean Jaccard overlap with FAS top-10: 0.075590

## Generated artifacts
See `reports/phase2_lime/` for metrics, CSV comparisons, and plots.

## Reproducibility note
This documentation records the experiment settings and reported outputs.
The experiment was executed in Google Colab using the project repository,
preprocessed NSL-KDD data, and the Phase 1 model. Dataset files and model
weights should be obtained separately; local data is excluded from Git.
