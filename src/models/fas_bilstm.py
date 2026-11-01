"""BiLSTM classifier with optional feature attention."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


def build_model(
    n_features: int,
    learning_rate: float = 1e-4,
    l2_strength: float = 1e-4,
    dropout: float = 0.3,
    use_attention: bool = True,
) -> tuple[Model, Model | None]:
    """Return classifier and optional feature-attention model."""
    inp = layers.Input(shape=(n_features, 1), name="feature_sequence")

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_strength),
            recurrent_regularizer=regularizers.l2(l2_strength),
        ),
        name="bilstm_128",
    )(inp)
    x = layers.BatchNormalization(name="batch_norm_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_strength),
            recurrent_regularizer=regularizers.l2(l2_strength),
        ),
        name="bilstm_64",
    )(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)

    attention_model = None
    if use_attention:
        logits = layers.Dense(1, activation="tanh", name="attention_logits")(x)
        weights = layers.Softmax(axis=1, name="feature_attention")(logits)
        weighted = layers.Multiply(name="attention_weighted_sequence")([x, weights])
        x = layers.Flatten(name="flatten_weighted_sequence")(weighted)
        attention_model = Model(inp, weights, name="feature_attention_model")
    else:
        x = layers.Flatten(name="flatten_sequence")(x)

    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(
        64,
        activation="elu",
        kernel_regularizer=regularizers.l2(l2_strength),
        name="dense_elu",
    )(x)
    out = layers.Dense(1, activation="sigmoid", name="attack_probability")(x)

    model = Model(
        inp,
        out,
        name="FAS_XAI_ZDA_BiLSTM" if use_attention else "BiLSTM_without_attention",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model, attention_model
