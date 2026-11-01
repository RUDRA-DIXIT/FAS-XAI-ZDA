# FAS-XAI-ZDA
## Explainable Deep Neural Networks for Zero-Day Attack Detection

Research replication and extension project based on the FAS-XAI-ZDA paper.

## 1. Project Overview

This project investigates feature attention scores and Bidirectional LSTM networks for zero-day network attack detection.

The implementation uses the NSL-KDD dataset and compares four experimental configurations.

## 2. Dataset

- Dataset: NSL-KDD
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/hassan06/nslkdd
- Files: KDDTrain_.txt and KDDTest_.txt

Dataset files are not included in this repository.

## 3. Experimental Configurations

| Experiment | Features | Attention |
|---|---|---|
| 01 | All 41 features | Yes |
| 02 | All 41 features | No |
| 03 | Selected 15 features | Yes |
| 04 | Selected 15 features | No |

All four experiments use 10 training epochs.

## 4. Phase 1 Results

| Experiment | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Full features + Attention | 99.70% | 99.52% | 99.85% | 99.68% |
| Full features, No Attention | 99.74% | 99.76% | 99.68% | 99.72% |
| Top 15 + Attention | 97.64% | 97.06% | 97.91% | 97.48% |
| Top 15, No Attention | 97.59% | 96.91% | 97.95% | 97.43% |

These are results from our reproduced experimental runs, not results reported by the original paper.

## 5. Repository Structure

```text
FAS-XAI-ZDA/
├── src/
├── data/
├── experiments/
│   └── phase1_reproduced/
├── reports/
├── notebooks/
├── phase2/
├── README.md
├── requirements.txt
├── AI_USAGE.md
└── .gitignore
```

## 6. Environment

- Python 3.13.18
- TensorFlow 2.21.0
- NumPy 2.1.3
- Pandas 2.2.3
- Scikit-learn 1.6.1
- Matplotlib 3.10.0
- Seaborn 0.13.2

Install dependencies using `pip install -r requirements.txt`.

## 7. Reproducibility

The repository contains source code, preprocessing implementation, experiment outputs, model checkpoints and evaluation reports.

Dataset preparation and experiment execution instructions will be documented alongside the scripts.

## 8. Project Phases

- Phase 1: Research replication and experimental evaluation.
- Phase 2: Research gap identification and proposed improvements.

The phase2 directory is reserved for extension work.

## 9. AI Usage

Refer to AI_USAGE.md for the project's AI usage disclosure.
