#!/usr/bin/env python3
"""
train_catboost_v5.py

Train a CatBoost model on a CSV containing 32 feature columns named f0..f31 and a `label` column.
Saves model to `models/catboost_v5/model.cbm` by default.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Tuple

import numpy as np

try:
    from catboost import CatBoostClassifier
except Exception as e:  # pragma: no cover
    raise SystemExit("CatBoost is not installed. Install dependencies first (see requirements.txt).") from e

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


def split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
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


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Simple AUC implementation (ROC AUC)
    if len(y_true) == 0:
        return float('nan')
    # sort by score descending
    desc = np.argsort(-y_score)
    y_true_sorted = y_true[desc]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float('nan')
    cum_pos = np.cumsum(y_true_sorted == 1)
    ranks = np.arange(1, len(y_true) + 1)
    # Mann-Whitney U based AUC
    auc = (np.sum(cum_pos[y_true_sorted == 0]) )/ (pos * neg)
    return float(auc)


def bce_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(loss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with f0..f31 and label")
    ap.add_argument("--out", default=os.path.join("models", "catboost_v5"))
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--val_split", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=50)
    args = ap.parse_args()

    X, y = load_csv(args.data)
    n = len(y)
    print(f"[train_catboost_v5] Loaded {n} samples from: {args.data}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train, X_val, y_val = split_train_val(X, y, float(args.val_split), int(args.seed))
    has_val = len(y_val) > 0

    model = CatBoostClassifier(
        iterations=int(args.iterations),
        depth=int(args.depth),
        learning_rate=float(args.lr),
        l2_leaf_reg=float(args.l2),
        random_seed=int(args.seed),
        verbose=100,
        task_type="CPU",
    )

    fit_kwargs = {}
    if args.early_stopping and has_val:
        fit_kwargs["eval_set"] = (X_val, y_val)
        fit_kwargs["use_best_model"] = True
        fit_kwargs["early_stopping_rounds"] = int(args.patience)

    model.fit(X_train, y_train, **fit_kwargs)

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "model.cbm")
    mapping_path = os.path.join(args.out, "feature_mapping.json")
    report_path = os.path.join(args.out, "training_report.json")

    model.save_model(model_path)

    mapping = {
        "model_type": "catboost",
        "input_dim": 32,
        "feature_names": [f"f{i}" for i in range(32)],
        "seed": int(args.seed),
        "samples": int(n),
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Collect evaluation history from CatBoost if available
    evals_result = {}
    try:
        evals_result = model.get_evals_result()
    except Exception:
        evals_result = {}

    history: dict = {}
    # Map CatBoost metric names to our report keys
    def _add_history(prefix: str, src: dict) -> None:
        for metric_name, values in src.items():
            key = metric_name.lower()
            if key in ("logloss", "loss"):
                hist_key = f"{prefix}loss" if prefix else "loss"
            elif key == "auc":
                hist_key = f"{prefix}auc" if prefix else "auc"
            else:
                hist_key = f"{prefix}{key}"
            history.setdefault(hist_key, [])
            history[hist_key].extend([float(v) for v in values])

    if isinstance(evals_result, dict):
        # learn/training metrics
        if "learn" in evals_result:
            _add_history("", evals_result.get("learn", {}))
        # validation metrics
        if "validation" in evals_result:
            _add_history("val_", evals_result.get("validation", {}))

    # Determine evaluation set for final metrics
    eval_X = X_val if has_val else X_train
    eval_y = y_val if has_val else y_train
    preds_proba = model.predict_proba(eval_X)
    preds = np.array([p[1] for p in preds_proba], dtype="float32")
    pred_labels = (preds >= 0.5).astype("int")

    acc = float(np.mean(pred_labels == eval_y))
    auc = binary_auc(eval_y.astype("int"), preds)
    loss = bce_loss(eval_y, preds)

    iterations_ran = int(args.iterations)
    try:
        best_it = model.get_best_iteration()
        if best_it is not None:
            # CatBoost returns 0-based best iteration index; iterations_ran should be best+1
            iterations_ran = int(best_it) + 1
    except Exception:
        pass

    # Counts for samples and splits
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    train_positives = int(np.sum(y_train == 1))
    train_negatives = int(np.sum(y_train == 0))
    val_samples = int(len(y_val))
    val_positives = int(np.sum(y_val == 1)) if val_samples > 0 else 0
    val_negatives = int(np.sum(y_val == 0)) if val_samples > 0 else 0

    # class weights similar to catboost_v4: total/(2*count)
    class_weights = [
        float(n) / (2.0 * max(1, negatives)),
        float(n) / (2.0 * max(1, positives)),
    ]

    final_metrics = {
        "acc": acc,
        "auc": auc,
        "loss": loss,
    }

    # compute validation final metrics explicitly if val set present
    if has_val:
        val_preds_proba = model.predict_proba(X_val)
        val_preds = np.array([p[1] for p in val_preds_proba], dtype="float32")
        val_pred_labels = (val_preds >= 0.5).astype("int")
        final_metrics["val_acc"] = float(np.mean(val_pred_labels == y_val))
        final_metrics["val_auc"] = float(binary_auc(y_val.astype("int"), val_preds))
        final_metrics["val_loss"] = float(bce_loss(y_val, val_preds))

    report = {
        "model_type": "catboost",
        "samples": int(n),
        "positives": positives,
        "negatives": negatives,
        "train_samples": int(len(y_train)),
        "train_positives": train_positives,
        "train_negatives": train_negatives,
        "val_samples": val_samples,
        "val_positives": val_positives,
        "val_negatives": val_negatives,
        "input_dim": 32,
        "feature_names": mapping.get("feature_names"),
        "categorical_features": [],
        "cat_feature_indices": [],
        "iterations": int(args.iterations),
        "iterations_ran": iterations_ran,
        "depth": int(args.depth),
        "learning_rate": float(args.lr),
        "l2_leaf_reg": float(args.l2),
        "seed": int(args.seed),
        "validation_split": float(args.val_split),
        "early_stopping": bool(args.early_stopping),
        "patience": int(args.patience),
        "max_samples": 0,
        "sample_order": "none",
        "class_weights": [float(class_weights[0]), float(class_weights[1])],
        "best_iteration": int(max(0, iterations_ran - 1)),
        "final_metrics": final_metrics,
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[train_catboost_v5] Saved model to: {model_path}")
    print(f"[train_catboost_v5] Saved mapping to: {mapping_path}")
    print(f"[train_catboost_v5] Saved report to: {report_path}")


if __name__ == "__main__":
    main()
