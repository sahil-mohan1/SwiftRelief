#!/usr/bin/env python3
"""
evaluate_tf_32.py

Evaluate the trained `models/tf_32/model.keras` on real feedback rows from the SQLite DB.
Saves results to `models/tf_32/tf_300_results.json` by default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Tuple

import numpy as np
try:
    import tensorflow as tf  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("TensorFlow is not installed") from e

import sys
from pathlib import Path

# ensure project root is on sys.path so we can import sibling modules
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from train_model import fetch_training_rows
import train_model_old as old


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_health_profile_json(s: Any) -> dict:
    if not s:
        return {}
    try:
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8", errors="ignore")
        return json.loads(str(s))
    except Exception:
        return {}


# Use the old script's vectorizer to produce the same input schema
def vectorize_32(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    vocab = old.build_vocab(rows)
    return old.vectorize(rows, vocab)


def evaluate_model(model_path: str, X: np.ndarray, y: np.ndarray) -> dict:
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X, batch_size=32, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype("float32")

    bce = tf.keras.losses.binary_crossentropy(y, probs).numpy()
    loss = float(np.mean(bce))

    acc_metric = tf.keras.metrics.BinaryAccuracy()
    acc_metric.update_state(y, preds)
    acc = float(acc_metric.result().numpy())

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y, probs)
    auc = float(auc_metric.result().numpy())

    return {"loss": loss, "acc": acc, "auc": auc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join("data", "app.db"))
    ap.add_argument("--model", default=os.path.join("models", "tf_32", "model.keras"))
    ap.add_argument("--out", default=os.path.join("models", "tf_32", "tf_300_results.json"))
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    db_path = str(args.db)
    # resolve similar to other scripts: allow relative
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(db_path)

    if not os.path.exists(db_path):
        raise SystemExit(f"DB not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
    rows = fetch_training_rows(con, max_samples=int(args.max_samples), sample_order="none")
    con.close()

    n = len(rows)
    if n == 0:
        raise SystemExit("No labeled feedback rows found in DB to evaluate on.")

    X, y = vectorize_32(rows)

    # split similarly to training to provide train/val-like metrics
    X_train, y_train, X_val, y_val = old.split_train_val(X, y, 0.2, int(args.seed))

    results: dict = {}
    results["samples"] = int(n)
    results["positives"] = int(np.sum(y))
    results["negatives"] = int(len(y) - np.sum(y))

    if len(y_train) > 0:
        res_train = evaluate_model(args.model, X_train, y_train)
        results["train"] = res_train

    if len(y_val) > 0:
        res_val = evaluate_model(args.model, X_val, y_val)
        results["val"] = res_val

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[evaluate_tf_32] Wrote results to: {args.out}")


if __name__ == "__main__":
    main()
