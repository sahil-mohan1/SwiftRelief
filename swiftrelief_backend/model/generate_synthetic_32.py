#!/usr/bin/env python3
"""
generate_synthetic_32.py

Create synthetic tabular training data with 32 input dimensions and a binary label.
Saves CSV to `data/synthetic_32.csv` by default.
"""
from __future__ import annotations

import argparse
import os
import json
from typing import Tuple

import numpy as np
import pandas as pd


def make_samples(n: int, dims: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # draw features from mixture of normals to add some variety
    X = rng.normal(loc=0.0, scale=1.0, size=(n, dims)).astype("float32")

    # construct a latent linear score with sparse weights so patterns exist
    w = rng.normal(loc=0.0, scale=1.0, size=(dims,)).astype("float32")
    # make half the weights smaller to avoid trivial separability
    w *= (rng.random(dims) * 0.8 + 0.2)
    bias = float(rng.normal(scale=0.2))

    logits = X.dot(w) + bias
    probs = 1.0 / (1.0 + np.exp(-logits))

    # sample labels using Bernoulli(probs), but add label noise
    noise_mask = rng.random(n) < 0.05
    y = (rng.random(n) < probs).astype("int32")
    # flip a few labels for realism
    y[noise_mask] = 1 - y[noise_mask]

    return X, y


def save_csv(X: np.ndarray, y: np.ndarray, out_path: str) -> None:
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="Number of synthetic samples to generate")
    ap.add_argument("--dims", type=int, default=32, help="Number of feature dimensions")
    ap.add_argument("--out", default=os.path.join("data", "synthetic_32.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = make_samples(int(args.n), int(args.dims), int(args.seed))
    save_csv(X, y, args.out)
    print(f"[generate_synthetic_32] Wrote {int(args.n)} samples ({int(args.dims)} dims) to: {args.out}")


if __name__ == "__main__":
    main()
