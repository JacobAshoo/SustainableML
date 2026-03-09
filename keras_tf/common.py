"""
common.py
─────────
Shared utilities used by every experiment script.
  • reset_seeds()         – deterministic seeding before each run
  • load_and_preprocess() – CIFAR-10 load, normalise, one-hot encode
  • build_keras_model()   – compiled Sequential CNN (tf.keras .fit style)
  • build_tf_model()      – same architecture, NOT compiled (GradientTape style)
  • evaluate()            – accuracy / macro-precision / macro-recall
  • print_results()       – formatted table + CSV export
"""

import os
import gc
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ── Global hyper-parameters (shared across all experiments) ────────────────
SEED   = 42
EPOCHS = 50
BATCH  = 32


# ── Seeding ────────────────────────────────────────────────────────────────
def reset_seeds(seed: int = SEED) -> None:
    """Reset all RNG sources to make each run independently reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ── Data ───────────────────────────────────────────────────────────────────
def load_and_preprocess():
    """
    Returns
    -------
    X_train      : float32 ndarray (50000, 32, 32, 3) in [0, 1]
    y_cat_train  : float32 ndarray (50000, 10) one-hot
    X_test       : float32 ndarray (10000, 32, 32, 3) in [0, 1]
    y_cat_test   : float32 ndarray (10000, 10) one-hot
    y_test_raw   : int ndarray     (10000, 1)  integer labels
    """
    (X_tr, y_tr), (X_te, y_te) = cifar10.load_data()

    X_train     = X_tr[:50_000].astype("float32") / 255.0
    X_test      = X_te[:10_000].astype("float32") / 255.0
    y_cat_train = to_categorical(y_tr[:50_000], 10)
    y_cat_test  = to_categorical(y_te[:10_000], 10)
    y_test_raw  = y_te[:10_000]

    return X_train, y_cat_train, X_test, y_cat_test, y_test_raw


# ── Model architecture ─────────────────────────────────────────────────────
def _cnn_layers(input_shape=(32, 32, 3)):
    """
    Three convolutional blocks (32→64→128 filters) with BatchNorm, MaxPool,
    Dropout(0.25); followed by Dense(128) + Dropout(0.25) + softmax head.
    Returns a list of Keras layers.
    """
    K = (3, 3)
    return [
        Conv2D(32,  K, activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32,  K, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPool2D(2, 2), Dropout(0.25),

        Conv2D(64,  K, activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64,  K, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPool2D(2, 2), Dropout(0.25),

        Conv2D(128, K, activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, K, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPool2D(2, 2), Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(10,  activation="softmax"),
    ]


def build_keras_model() -> Sequential:
    """Compiled model for use with model.fit() / model.train_on_batch()."""
    model = Sequential(_cnn_layers())
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_tf_model() -> Sequential:
    """Un-compiled model for use with a manual GradientTape training loop."""
    return Sequential(_cnn_layers())


# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, X_test: np.ndarray, y_test_raw: np.ndarray):
    """
    Returns
    -------
    acc   : float – overall accuracy
    prec  : float – macro-averaged precision
    rec   : float – macro-averaged recall
    """
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = y_test_raw.flatten()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,  average="macro", zero_division=0)
    return acc, prec, rec


# ── Results reporting ──────────────────────────────────────────────────────
def print_results(results: list[dict], name: str) -> None:
    """Print a formatted table, mean, std-dev, and save a CSV."""
    import pandas as pd

    df = pd.DataFrame(results)
    csv_name = name.lower().replace(" ", "_") + "_results.csv"

    print(f"\n{'=' * 72}")
    print(f"  Experiment : {name}")
    print(f"  Runs       : {len(df)}")
    print(f"{'=' * 72}")
    print(df.to_string(index=True))
    print(f"\n--- Mean ---")
    print(df.mean(numeric_only=True).to_string())
    print(f"\n--- Std  ---")
    print(df.std(numeric_only=True).to_string())
    print(f"{'=' * 72}\n")

    df.to_csv(csv_name, index=False)
    print(f"  Results saved → {csv_name}")
