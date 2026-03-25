#!/usr/bin/env python3
"""
train_tf_32.py

Train a TensorFlow model on a CSV containing 32 feature columns named f0..f31 and a `label` column.
Saves model to `models/tf_32/model.keras` by default.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Tuple

import numpy as np

try:
    import tensorflow as tf  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("TensorFlow is not installed. Install dependencies first (see requirements.txt).") from e

import pandas as pd


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    if len(feature_cols) != 32:
        raise SystemExit(f"Expected 32 feature columns starting with 'f', found {len(feature_cols)}")
    X = df[feature_cols].astype("float32").to_numpy()
    if "label" not in df.columns:
        raise SystemExit("CSV missing 'label' column")
    y = df["label"].astype("float32").to_numpy()
    return X, y


def split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    if n == 0:
        return X, y, X[:0], y[:0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_n = int(round(n * val_ratio)) if val_ratio > 0 else 0
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def build_model(input_dim: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with f0..f31 and label")
    ap.add_argument("--out", default=os.path.join("models", "tf_32"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min_delta", type=float, default=0.0, help="Minimum change to qualify as an improvement for EarlyStopping")
    args = ap.parse_args()

    X, y = load_csv(args.data)
    n = len(y)
    print(f"[train_tf_32] Loaded {n} samples from: {args.data}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X_train, y_train, X_val, y_val = split_train_val(X, y, float(args.val_split), int(args.seed))
    has_val = len(y_val) > 0

    model = build_model(input_dim=32, lr=float(args.lr))

    callbacks = []
    if args.early_stopping and has_val:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                restore_best_weights=True,
            )
        )

    hist = model.fit(
        X_train,
        y_train,
        epochs=int(args.epochs),
        batch_size=int(args.batch),
        validation_data=(X_val, y_val) if has_val else None,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "model.keras")
    mapping_path = os.path.join(args.out, "feature_mapping.json")
    report_path = os.path.join(args.out, "training_report.json")

    model.save(model_path)

    mapping = {
        "model_type": "tensorflow",
        "input_dim": 32,
        "feature_names": [f"f{i}" for i in range(32)],
        "seed": int(args.seed),
        "samples": int(n),
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    final_metrics = {k: float(v[-1]) for k, v in hist.history.items()}
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}
    report = {
        "samples": int(n),
        "input_dim": 32,
        "epochs_ran": int(len(hist.history.get("loss", []))),
        "batch": int(args.batch),
        "seed": int(args.seed),
        "learning_rate": float(args.lr),
        "final_metrics": final_metrics,
        "history": history,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[train_tf_32] Saved model to: {model_path}")
    print(f"[train_tf_32] Saved mapping to: {mapping_path}")
    print(f"[train_tf_32] Saved report to: {report_path}")


if __name__ == "__main__":
    main()
