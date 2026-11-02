# FAS-XAI-ZDA
### Explainable Deep Neural Networks for Zero-Day Attack Detection Leveraging Feature Attention Scores

## 1. Project Overview

This project investigates the use of Explainable Artificial Intelligence (XAI) for detecting zero-day cyberattacks using deep learning.

The implementation is based on the research paper:

**FAS-XAI-ZDA: Explainable Deep Neural Networks for Zero-Day Attack Detection Leveraging Feature Attention Scores**

The project explores a Bidirectional Long Short-Term Memory (BiLSTM) network with feature-level attention mechanisms to improve network intrusion detection and provide insight into the features influencing model predictions.

The experimental study evaluates the contribution of feature attention and feature selection through four model configurations.

## 2. Research Objectives

- Implement a BiLSTM-based network intrusion detection model.
- Investigate feature-level attention for interpreting model behaviour.
- Evaluate the effect of attention mechanisms on classification performance.
- Study the effect of selecting the top 15 attention-ranked features.
- Compare full-feature and reduced-feature model configurations.

## 3. Dataset

The experiments use the NSL-KDD dataset, derived from the KDD Cup 1999 intrusion detection dataset.

The dataset contains normal network traffic and different categories of attacks.

The project uses:
- `KDDTrain_.txt`
- `KDDTest_.txt`

The processed dataset contains 41 input features in the full-feature configuration, including the `level` feature.

**Dataset files are not included in this repository.** Users must obtain the dataset separately and place the files in the appropriate data directory.

## 4. Model Architecture

The implemented architecture consists of:

1. Input feature sequence
2. Bidirectional LSTM layer with 128 units
3. Batch Normalization
4. Bidirectional LSTM layer with 64 units
5. Batch Normalization
6. Optional feature-level attention mechanism
7. Flattening layer
8. Dropout
9. Dense layer with ELU activation
10. Sigmoid output layer

The attention mechanism assigns weights across input feature positions, allowing the model to learn their relative importance.

## 5. Experimental Configurations

Four configurations were evaluated:

| Experiment | Features | Attention |
|---|---|---|
| Experiment 1 | Full 41 features | Yes |
| Experiment 2 | Full 41 features | No |
| Experiment 3 | Top 15 selected features | Yes |
| Experiment 4 | Top 15 selected features | No |

All four configurations were trained using 10 epochs and the same primary experimental setup.

## 6. Experimental Results

The following table reports the test-set results obtained from the four new experimental runs.

| Experiment | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Full Features + Attention | 99.70% | 99.52% | 99.85% | 99.68% |
| Full Features without Attention | 99.74% | 99.76% | 99.68% | 99.72% |
| Top 15 Features + Attention | 97.64% | 97.06% | 97.91% | 97.48% |
| Top 15 Features without Attention | 97.59% | 96.91% | 97.95% | 97.43% |

These results represent the current implementation's experimental outputs and should not be interpreted as an exact reproduction of every result reported in the original paper.

## 7. Visualizations and Analysis

The repository contains visualizations generated from the new experimental runs.

### 7.1 Experiment Comparison

Located in `reports/phase1_comparison/`:

- Validation AUC across all four experiments
- Training and validation loss
- Training and validation accuracy

### 7.2 Classification Error Analysis

Located in `reports/error_analysis/`:

- False positive attention visualization
- False negative attention visualization
- Correctly classified normal sample
- Correctly classified attack sample

### 7.3 Feature Attention Analysis

Located in `reports/attention_analysis/`:

- Mean Feature Attention Scores across training attack samples (Experiment 1)
- Mean attention scores across test attack samples (Experiment 3)

### 7.4 Model Calibration

Located in `reports/calibration/`:

- Calibration curve for Experiment 3

The attention visualizations provide an analysis of feature weights for selected samples and aggregated attack samples.

## 8. Repository Structure

```text
FAS-XAI-ZDA/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── eval/
│   ├── xai/
│   └── train_primary.py
│
├── experiments/
│   └── phase1_reproduced/
│       ├── 01_full_features_with_attention/
│       ├── 02_full_features_without_attention/
│       ├── 03_top15_features_with_attention/
│       ├── 04_top15_features_without_attention/
│       ├── feature_attention_ranking.json
│       └── feature_selection.json
│
├── reports/
│   ├── phase1_comparison/
│   ├── error_analysis/
│   ├── attention_analysis/
│   ├── calibration/
│   ├── accuracy_comparison.png
│   └── precision_recall_f1_comparison.png
│
├── data/
├── notebooks/
├── phase2/
│
├── AI_USAGE.md
├── README.md
├── requirements.txt
└── .gitignore
```

## 9. Installation and Execution

### Requirements

- Python
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

### Clone the repository

```bash
git clone https://github.com/RUDRA-DIXIT/FAS-XAI-ZDA.git

cd FAS-XAI-ZDA
```

### Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Dataset preparation

Place the NSL-KDD raw dataset files in:

```text
data/raw/
```

Run the preprocessing pipeline according to the scripts in `src/data/`.

The processed arrays and preprocessing artifacts are required before training.

**Note:** The repository currently contains the experiment checkpoints, metrics and logs, but not the raw or processed dataset files. Verify the preprocessing and training script paths before running the experiments from a fresh clone.

## 10. Reproducibility

The experiments use a fixed random seed of 42.

The repository includes:
- Trained model checkpoints
- Training logs
- Test metrics
- Feature attention rankings
- Confusion matrices
- Experimental visualizations

These artifacts allow inspection of the completed experiments without retraining the models.

## 11. AI Usage Disclosure

AI tools were used as development assistance for understanding the research paper, debugging, code organization, documentation and interpretation of experimental outputs.

Further details are provided in `AI_USAGE.md`.

## 12. References

Abbasi, A. A., Zameer, A., & Raja, M. A. Z. (2026). *FAS-XAI-ZDA: Explainable deep neural networks for zero-day attack detection leveraging feature attention scores*. Applied Soft Computing, 196, 115137.

DOI: https://doi.org/10.1016/j.asoc.2026.115137

---

**Project Repository:** https://github.com/RUDRA-DIXIT/FAS-XAI-ZDA
