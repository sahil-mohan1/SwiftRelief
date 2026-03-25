#!/usr/bin/env python3
"""Evaluate saved TF and CatBoost models on all available feedback samples.

Usage: python scripts/evaluate_on_feedback.py
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict

import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from train_model import fetch_training_rows, build_vocab, vectorize
from train_catboost_model import _build_table


def _binary_accuracy(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    if len(y_true) == 0:
        return 0.0
    preds = (probs >= threshold).astype(float)
    return float(np.mean(preds == y_true.astype(float)))


def _binary_logloss(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    p = np.clip(probs.astype(float), 1e-7, 1.0 - 1.0e-7)
    y = y_true.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = y_true.astype(int)
    p = probs.astype(float)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    auc = (pos_rank_sum - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(max(0.0, min(1.0, auc)))


def load_db_rows(db_path: str) -> list[Dict[str, Any]]:
    db_path = os.path.abspath(db_path)
    con = sqlite3.connect(db_path)
    rows = fetch_training_rows(con, max_samples=0, sample_order="feedback_id_asc")
    con.close()
    return rows


def eval_tf(model_path: str, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if tf is None:
        raise SystemExit("TensorFlow not available in this environment")
    # Prefer using saved feature mapping to ensure input_dim matches the model
    mapping_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "feature_mapping.json")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            vocab = {
                "emergency_type_vocab": mapping.get("emergency_type_vocab", []),
                "source_vocab": mapping.get("source_vocab", []),
                "gender_vocab": mapping.get("gender_vocab", []),
            }
        except Exception:
            vocab = build_vocab(rows)
    else:
        vocab = build_vocab(rows)
    X, y = vectorize(rows, vocab)
    model = tf.keras.models.load_model(model_path)
    expected = int(model.input_shape[1]) if model.input_shape and model.input_shape[1] is not None else X.shape[1]
    if X.shape[1] < expected:
        # pad missing columns with zeros
        pad_width = expected - X.shape[1]
        X = np.concatenate([X, np.zeros((X.shape[0], pad_width), dtype=X.dtype)], axis=1)
    elif X.shape[1] > expected:
        # truncate extra columns
        X = X[:, :expected]
    probs = model.predict(X, verbose=0).reshape(-1)
    return {
        "n": int(len(y)),
        "acc": _binary_accuracy(y, probs),
        "auc": _roc_auc(y, probs),
        "logloss": _binary_logloss(y, probs),
    }


def eval_catboost(model_path: str, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if CatBoostClassifier is None:
        raise SystemExit("CatBoost not available in this environment")
    X_df, y, vocab = _build_table(rows)
    model = CatBoostClassifier()
    model.load_model(model_path)
    probs = model.predict_proba(X_df)[:, 1]
    return {
        "n": int(len(y)),
        "acc": _binary_accuracy(y, probs),
        "auc": _roc_auc(y, probs),
        "logloss": _binary_logloss(y, probs),
    }


def main() -> None:
    db = os.path.join("data", "app.db")
    rows = load_db_rows(db)
    if not rows:
        print("No feedback rows found in DB.")
        return

    out: Dict[str, Any] = {"samples": len(rows)}

    # TF model
    tf_path = os.path.join("models", "tf_v3_repro_104", "model.keras")
    if os.path.exists(tf_path):
        try:
            out["tf_v3_repro_104"] = eval_tf(tf_path, rows)
        except Exception as e:
            out["tf_v3_repro_104_error"] = str(e)
    else:
        out["tf_v3_repro_104_error"] = f"Model not found: {tf_path}"

    # CatBoost model
    cb_path = os.path.join("models", "catboost_v4", "model.cbm")
    if os.path.exists(cb_path):
        try:
            out["catboost_v4"] = eval_catboost(cb_path, rows)
        except Exception as e:
            out["catboost_v4_error"] = str(e)
    else:
        out["catboost_v4_error"] = f"Model not found: {cb_path}"

    # Save report
    rep_path = os.path.join("models", "eval_feedback_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote evaluation report:", rep_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
