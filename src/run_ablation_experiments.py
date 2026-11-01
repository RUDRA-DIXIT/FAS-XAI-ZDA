
from __future__ import annotations
import json, random, shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.models.fas_bilstm import build_model
from src.eval.metrics import calculate_metrics

SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
TOP_K = 15
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "experiments" / "phase1_reproduction_run"

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def load_arrays(folder):
    folder = Path(folder)
    tr = np.load(folder / "train.npz", allow_pickle=False)
    va = np.load(folder / "validation.npz", allow_pickle=False)
    te = np.load(folder / "test_primary.npz", allow_pickle=False)
    return (tr["X"], tr["y"], tr["feature_names"].astype(str).tolist(),
            va["X"], va["y"], te["X"], te["y"])

def train_one(name, Xtr, ytr, Xva, yva, Xte, yte, feature_names,
              use_attention=True):
    run_dir = OUT / name
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    print(f"\n===== {name}: {Xtr.shape[1]} features; attention={use_attention} =====",
          flush=True)

    model, attention_model = build_model(
        Xtr.shape[1], use_attention=use_attention
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_model.keras"),
            monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]
    history = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=2, shuffle=True
    )
    model.save(run_dir / "final_model.keras")

    prob = model.predict(Xte, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    metrics = calculate_metrics(yte, prob)
    metrics.update({
        "experiment": name,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "attention_enabled": use_attention,
        "epochs_requested": EPOCHS,
        "epochs_ran": len(history.history["loss"]),
        "test_split": "primary rare-class holdout from KDDTrain+",
        "threshold_selection": "fixed 0.5; no test-based tuning"
    })
    save_json(run_dir / "test_metrics.json", metrics)

    cm = confusion_matrix(yte, (prob >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        cm, display_labels=["Normal", "Attack"]
    ).plot(ax=ax, values_format="d", colorbar=False)
    fig.tight_layout()
    fig.savefig(run_dir / "plots/confusion_matrix.png", dpi=160)
    plt.close(fig)

    plt.figure()
    plt.hist(prob[yte == 0], bins=30, alpha=.7, label="Normal")
    plt.hist(prob[yte == 1], bins=30, alpha=.7, label="Attack")
    plt.xlabel("Predicted attack probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "plots/prediction_probability_histogram.png", dpi=160)
    plt.close()

    for key in ["loss", "accuracy", "auc"]:
        if key in history.history:
            plt.figure()
            plt.plot(history.history[key], label="train")
            if "val_" + key in history.history:
                plt.plot(history.history["val_" + key], label="validation")
            plt.xlabel("Epoch")
            plt.ylabel(key)
            plt.legend()
            plt.tight_layout()
            plt.savefig(run_dir / f"plots/training_{key}.png", dpi=160)
            plt.close()

    print(json.dumps({
        k: metrics[k] for k in [
            "accuracy", "balanced_accuracy", "precision_attack",
            "recall_attack", "f1_attack", "roc_auc",
            "average_precision", "confusion_matrix"
        ]
    }, indent=2), flush=True)
    return model, attention_model, metrics, prob


def main():
    global OUT

    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"Output directory already contains results: {OUT}. "
            "Choose a new output directory before running experiments."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load the 41-feature configuration. Preprocessing was fit on training only.
    Xtr, ytr, features, Xva, yva, Xte, yte = load_arrays(
        PROJECT_ROOT / "data" / "processed" / "arrays_with_level"
    )

    # Experiment 1: full-feature BiLSTM with attention.
    full_model, att_model, full_metrics, _ = train_one(
        "01_full_features_with_attention",
        Xtr, ytr, Xva, yva, Xte, yte, features, True
    )

    # Compute FAS from TRAINING ATTACK samples only; never rank on the test set.
    attack_idx = np.flatnonzero(ytr == 1)
    attention_batches = []
    for start in range(0, len(attack_idx), BATCH_SIZE):
        batch = Xtr[attack_idx[start:start+BATCH_SIZE]]
        attention_batches.append(
            att_model.predict(batch, batch_size=BATCH_SIZE, verbose=0)
        )
    mean_attention = np.concatenate(attention_batches, axis=0).mean(axis=(0, 2))
    order = np.argsort(mean_attention)[::-1]

    ranking = [
        {"rank": rank + 1, "feature": features[int(i)],
         "mean_attention": float(mean_attention[i])}
        for rank, i in enumerate(order)
    ]
    save_json(OUT / "feature_attention_ranking.json", ranking)

    # Paper threshold is underspecified: record this heuristic as supplementary,
    # and use the paper's 15-feature setting for the main reduced-feature comparison.
    sorted_scores = mean_attention[order]
    if len(sorted_scores) > 1:
        cut = int(np.argmax(sorted_scores[:-1] - sorted_scores[1:]) + 1)
    else:
        cut = 1

    selection = {
        "main_selection_method": "top 15 features by mean training-attack attention",
        "top_15_features": [features[int(i)] for i in order[:TOP_K]],
        "transition_heuristic": "largest adjacent drop in descending mean attention",
        "transition_selected_count": cut,
        "transition_selected_features": [features[int(i)] for i in order[:cut]],
        "caveat": "The paper's transition threshold is not specified unambiguously; this heuristic is not claimed to be the exact paper algorithm."
    }
    save_json(OUT / "feature_selection.json", selection)

    plt.figure(figsize=(13, 5))
    plt.plot(np.arange(1, len(order)+1), sorted_scores, marker=".")
    plt.xticks(np.arange(1, len(order)+1),
               [features[int(i)] for i in order], rotation=90)
    plt.xlabel("Features ranked by mean attention")
    plt.ylabel("Mean attention score on training attack samples")
    plt.tight_layout()
    plt.savefig(OUT / "feature_attention_ranking.png", dpi=160)
    plt.close()

    top_indices = order[:TOP_K]
    reduced_features = [features[int(i)] for i in top_indices]
    save_json(OUT / "top15_feature_names.json", reduced_features)

    # Experiment 2: full features without attention.
    _, _, full_noatt_metrics, _ = train_one(
        "02_full_features_without_attention",
        Xtr, ytr, Xva, yva, Xte, yte, features, False
    )

    # Stage 2: reduced-feature classifier, with and without attention.
    Xtr15, Xva15, Xte15 = Xtr[:, top_indices, :], Xva[:, top_indices, :], Xte[:, top_indices, :]

    _, _, reduced_att_metrics, _ = train_one(
        "03_top15_features_with_attention",
        Xtr15, ytr, Xva15, yva, Xte15, yte, reduced_features, True
    )
    _, _, reduced_noatt_metrics, _ = train_one(
        "04_top15_features_without_attention",
        Xtr15, ytr, Xva15, yva, Xte15, yte, reduced_features, False
    )

    # Compact comparison table in JSON and CSV.
    experiments = [
        ("full_features_with_attention", full_metrics),
        ("full_features_without_attention", full_noatt_metrics),
        ("top15_features_with_attention", reduced_att_metrics),
        ("top15_features_without_attention", reduced_noatt_metrics),
    ]
    keys = ["accuracy", "balanced_accuracy", "precision_attack",
            "recall_attack", "f1_attack", "roc_auc", "average_precision"]
    comparison = []
    for name, m in experiments:
        row = {"experiment": name, "feature_count": m["feature_count"]}
        row.update({k: m.get(k) for k in keys})
        comparison.append(row)
    save_json(OUT / "ablation_comparison.json", comparison)

    import csv
    with (OUT / "ablation_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    print("\n===== ALL FOUR EXPERIMENTS FINISHED =====", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)
    print("Drive results folder:", OUT, flush=True)


if __name__ == "__main__":
    main()
