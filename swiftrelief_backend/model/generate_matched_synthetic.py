#!/usr/bin/env python3
"""
generate_matched_synthetic.py

Generate synthetic samples that match the 32-d feature schema produced by
`train_model_old.vectorize`. Strategy:
- Load labeled real rows, vectorize via `train_model_old.vectorize`
- For each synthetic sample: bootstrap a real vector, add small Gaussian noise
  to numeric slots, occasionally perturb categorical one-hot blocks to nearby
  categories, and copy the label with a small flip probability.
- Save synthetic CSV and a combined CSV (real + synthetic) suitable for
  `model/train_tf_32.py`.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import csv

import sqlite3
import json
import math

# To avoid importing train_model_old (which requires TF at import time),
# implement the minimal DB fetch + vectorize logic here (copied from train_model_old)


def fetch_training_rows(con: sqlite3.Connection, max_samples: int = 0, sample_order: str = "none"):
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cand_cols = {r["name"] for r in cur.execute("PRAGMA table_info(run_candidates);").fetchall()}
    score_col = "baseline_score" if "baseline_score" in cand_cols else "final_score"

    order_map = {
        "none": "",
        "feedback_id_asc": " ORDER BY f.feedback_id ASC",
        "feedback_id_desc": " ORDER BY f.feedback_id DESC",
        "created_at_asc": " ORDER BY f.created_at ASC, f.feedback_id ASC",
        "created_at_desc": " ORDER BY f.created_at DESC, f.feedback_id DESC",
    }
    order_clause = order_map.get(sample_order, "")
    limit_clause = f" LIMIT {int(max_samples)}" if max_samples and max_samples > 0 else ""

    rows = cur.execute(
        f"""
        SELECT
          f.thumbs AS thumbs,
          f.run_id AS run_id,
          f.hospital_id AS hospital_id,
          f.run_candidate_id AS run_candidate_id,

          rc.distance_km,
          rc.travel_time_min,
          rc.rating,
          rc.user_ratings_total,
          rc.category_match_score,
          rc.confidence_score,
          rc.penalties,
          rc.{score_col} AS baseline_score,
          rc.source AS candidate_source,

          rr.emergency_type,
          rr.offline_mode,
          rr.health_profile_json

        FROM feedback f
        JOIN run_candidates rc ON rc.run_candidate_id = f.run_candidate_id
        JOIN recommendation_runs rr ON rr.run_id = rc.run_id
        WHERE f.run_candidate_id IS NOT NULL
        {order_clause}
        {limit_clause}
        """
    ).fetchall()
    out = [dict(r) for r in rows]
    if not out:
        rows = cur.execute(
            f"""
            SELECT
              f.thumbs AS thumbs,
              f.run_id AS run_id,
              f.hospital_id AS hospital_id,
              NULL AS run_candidate_id,

              rc.distance_km,
              rc.travel_time_min,
              rc.rating,
              rc.user_ratings_total,
              rc.category_match_score,
              rc.confidence_score,
              rc.penalties,
              rc.{score_col} AS baseline_score,
              rc.source AS candidate_source,

              rr.emergency_type,
              rr.offline_mode,
              rr.health_profile_json

            FROM feedback f
            JOIN run_candidates rc ON rc.run_id = f.run_id AND rc.hospital_id = f.hospital_id
            JOIN recommendation_runs rr ON rr.run_id = f.run_id
            {order_clause}
            {limit_clause}
            """
        ).fetchall()
        out = [dict(r) for r in rows]
    return out


def build_vocab(rows):
    emergency_types = sorted({str(r.get("emergency_type") or "").strip().lower() for r in rows} | {"unknown"})
    sources = sorted({str(r.get("candidate_source") or "").strip().lower() for r in rows} | {"unknown"})
    genders = ["Male", "Female", "Other"]
    return {"emergency_type_vocab": emergency_types, "source_vocab": sources, "gender_vocab": genders}


def vectorize(rows, vocab):
    et_vocab = list(vocab["emergency_type_vocab"])
    src_vocab = list(vocab["source_vocab"])
    gender_vocab = list(vocab["gender_vocab"])
    gender_vocab_l = [g.lower() for g in gender_vocab]

    X = []
    y = []

    def _safe_float(x, default=0.0):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _safe_int(x, default=0):
        try:
            if x is None:
                return default
            return int(x)
        except Exception:
            return default

    def _one_hot(idx, n):
        out = [0.0] * n
        if 0 <= idx < n:
            out[idx] = 1.0
        return out

    def _load_health_profile_json(s):
        if not s:
            return {}
        try:
            if isinstance(s, (bytes, bytearray)):
                s = s.decode("utf-8", errors="ignore")
            return json.loads(str(s))
        except Exception:
            return {}

    def vidx(v, key):
        try:
            return v.index(key)
        except Exception:
            return 0

    for r in rows:
        thumbs = int(r.get("thumbs") or 0)
        label = 1.0 if thumbs == 1 else 0.0

        hp = _load_health_profile_json(r.get("health_profile_json"))
        gender = str(hp.get("gender") or "Other").strip().lower()
        g_idx = vidx(gender_vocab_l, gender)

        emergency_type = str(r.get("emergency_type") or "unknown").strip().lower()
        source = str(r.get("candidate_source") or "unknown").strip().lower()

        et_idx = vidx(et_vocab, emergency_type)
        src_idx = vidx(src_vocab, source)

        baseline_score = _safe_float(r.get("baseline_score"), 0.0)
        distance_km = _safe_float(r.get("distance_km"), 1e3)
        rating = _safe_float(r.get("rating"), 0.0)
        user_rating_count = _safe_float(r.get("user_ratings_total"), 0.0)
        category_match_score = _safe_float(r.get("category_match_score"), 0.0)
        confidence_score = _safe_float(r.get("confidence_score"), 0.0)
        penalties = _safe_float(r.get("penalties"), 0.0)

        distance_log = math.log1p(max(distance_km, 0.0))
        count_log = math.log1p(max(user_rating_count, 0.0))

        age = _safe_float(hp.get("age"), 0.0)
        age_norm = max(0.0, min(age / 100.0, 1.5))

        has_asthma_copd = float(_safe_int(hp.get("has_asthma_copd"), 0))
        has_diabetes = float(_safe_int(hp.get("has_diabetes"), 0))
        has_heart_disease = float(_safe_int(hp.get("has_heart_disease"), 0))
        has_stroke_history = float(_safe_int(hp.get("has_stroke_history"), 0))
        has_epilepsy = float(_safe_int(hp.get("has_epilepsy"), 0))
        is_pregnant = float(_safe_int(hp.get("is_pregnant"), 0))

        offline_mode = float(_safe_int(r.get("offline_mode"), 0))

        et_norm = emergency_type
        is_important = float(et_norm in {
            "emergency / trauma",
            "cardiology (heart)",
            "neurology (brain / stroke)",
        })

        vec = [
            baseline_score,
            distance_log,
            rating,
            count_log,
            category_match_score,
            confidence_score,
            penalties,
            age_norm,
            has_asthma_copd,
            has_diabetes,
            has_heart_disease,
            has_stroke_history,
            has_epilepsy,
            is_pregnant,
            offline_mode,
            is_important,
        ]
        vec += _one_hot(et_idx, len(et_vocab))
        vec += _one_hot(src_idx, len(src_vocab))
        vec += _one_hot(g_idx, len(gender_vocab))
        X.append(vec)
        y.append(label)

    Xn = np.asarray(X, dtype="float32")
    yn = np.asarray(y, dtype="float32")
    return Xn, yn


def load_real_vectors(db_path: str):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = fetch_training_rows(con)
    con.close()
    if not rows:
        raise SystemExit("No labeled rows in DB to sample from")
    vocab = build_vocab(rows)
    X, y = vectorize(rows, vocab)
    return X, y, vocab


def synth_from_real(X_real: np.ndarray, y_real: np.ndarray, vocab: dict, n: int, seed: int = 42, noise_scale: float = 0.05, flip_prob: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_real, dim = X_real.shape

    # Determine one-hot block sizes from vocab
    et_n = len(vocab.get("emergency_type_vocab") or [])
    src_n = len(vocab.get("source_vocab") or [])
    g_n = len(vocab.get("gender_vocab") or [])

    # positions: first 16 numeric, then et_n, then src_n, then g_n => should sum to dim
    start_et = 16
    start_src = start_et + et_n
    start_g = start_src + src_n

    out_X = np.zeros((n, dim), dtype="float32")
    out_y = np.zeros((n,), dtype="int32")

    for i in range(n):
        # bootstrap a real vector
        idx = int(rng.integers(0, n_real))
        vec = X_real[idx].copy()

        # numeric jitter for first 16 slots
        noise = rng.normal(loc=0.0, scale=noise_scale, size=(16,))
        vec[:16] = vec[:16] + noise.astype("float32")

        # perturb one-hot blocks: with small prob, switch category
        if et_n > 1 and rng.random() < 0.1:
            # zero out block and sample according to empirical distribution across real rows
            vec[start_et:start_src] = 0.0
            new = int(rng.integers(0, et_n))
            vec[start_et + new] = 1.0

        if src_n > 1 and rng.random() < 0.08:
            vec[start_src:start_g] = 0.0
            new = int(rng.integers(0, src_n))
            vec[start_src + new] = 1.0

        if g_n > 1 and rng.random() < 0.05:
            vec[start_g:start_g + g_n] = 0.0
            new = int(rng.integers(0, g_n))
            vec[start_g + new] = 1.0

        # label copy with small flip probability
        label = int(y_real[idx])
        if rng.random() < flip_prob:
            label = 1 - label

        out_X[i] = vec
        out_y[i] = label

    return out_X, out_y


def save_csv(X: np.ndarray, y: np.ndarray, out_path: str) -> None:
    cols = [f"f{i}" for i in range(X.shape[1])]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols + ["label"])
        for xi, yi in zip(X.tolist(), y.tolist()):
            writer.writerow([str(float(v)) for v in xi] + [str(int(yi))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join("data", "app.db"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out_synth", default=os.path.join("data", "synthetic_matched_300.csv"))
    ap.add_argument("--out_combined", default=os.path.join("data", "combined_real_synth.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise_scale", type=float, default=0.05)
    ap.add_argument("--flip_prob", type=float, default=0.05)
    args = ap.parse_args()

    # load rows directly and vectorize here
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = fetch_training_rows(con)
    con.close()
    vocab = build_vocab(rows)
    X_real, y_real = vectorize(rows, vocab)
    n_real = X_real.shape[0]
    print(f"[generate_matched_synthetic] Loaded {n_real} real vectorized rows (dim={X_real.shape[1]})")

    X_syn, y_syn = synth_from_real(X_real, y_real, vocab, int(args.n), int(args.seed), float(args.noise_scale), float(args.flip_prob))
    save_csv(X_syn, y_syn, args.out_synth)
    print(f"[generate_matched_synthetic] Wrote {len(y_syn)} synthetic samples to {args.out_synth}")

    # create combined CSV: real vectors (as f0..f31) + synthetic
    cols = [f"f{i}" for i in range(X_real.shape[1])]
    os.makedirs(os.path.dirname(args.out_combined) or ".", exist_ok=True)
    with open(args.out_combined, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols + ["label"])
        # real rows
        for xi, yi in zip(X_real.tolist(), y_real.tolist()):
            writer.writerow([str(float(v)) for v in xi] + [str(int(yi))])
        # synthetic rows
        for xi, yi in zip(X_syn.tolist(), y_syn.tolist()):
            writer.writerow([str(float(v)) for v in xi] + [str(int(yi))])
    print(f"[generate_matched_synthetic] Wrote combined CSV with {len(X_real)+len(X_syn)} rows to {args.out_combined}")


if __name__ == "__main__":
    main()
