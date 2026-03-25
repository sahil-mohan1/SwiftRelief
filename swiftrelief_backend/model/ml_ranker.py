
# model/ml_ranker.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# TensorFlow is optional at runtime (demo-friendly).
try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _one_hot(idx: int, n: int) -> List[float]:
    if n <= 0:
        return []
    out = [0.0] * n
    if 0 <= idx < n:
        out[idx] = 1.0
    return out


@dataclass
class RankerConfig:
    baseline_weight: float = 0.8
    model_weight: float = 0.2
    # Convert p_positive into a score contribution comparable to baseline.
    # (p - 0.5) * model_score_scale => roughly [-scale/2, +scale/2]
    model_score_scale: float = 10.0


class TFModelRanker:
    """
    Loads a saved Keras model + feature mapping (produced by train_model.py)
    and provides p_positive predictions for candidates.

    This ranker is designed to be safe for demos:
    - If TF is missing OR model files are absent, it disables itself.
    - If anything fails during prediction, it returns None (caller should fallback).
    """

    def __init__(self, model_dir: str, *, config: Optional[RankerConfig] = None):
        self.model_dir = model_dir
        self.config = config or RankerConfig()
        self.enabled = False
        self.model = None
        self.mapping: Dict[str, Any] = {}

        if tf is None:
            return

        try:
            model_path = os.path.join(model_dir, "model.keras")
            mapping_path = os.path.join(model_dir, "feature_mapping.json")
            if not (os.path.exists(model_path) and os.path.exists(mapping_path)):
                return

            self.mapping = _load_json(mapping_path)
            self.model = tf.keras.models.load_model(model_path)
            self.enabled = True
        except Exception:
            self.enabled = False
            self.model = None
            self.mapping = {}

    def _vectorize_one(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
        """
        Feature vector must match train_model.py exactly.
        All features are numeric floats.
        """
        m = self.mapping

        # categorical vocab
        et_vocab: List[str] = list(m.get("emergency_type_vocab") or [])
        src_vocab: List[str] = list(m.get("source_vocab") or [])
        gender_vocab: List[str] = list(m.get("gender_vocab") or [])

        emergency_type = str(context.get("emergency_type") or "").strip().lower()
        source = str(candidate.get("source") or "").strip().lower()
        gender = str(context.get("gender") or "Other").strip()
        gender_norm = gender.lower()

        def _vocab_index(vocab: List[str], key: str) -> int:
            try:
                return vocab.index(key)
            except Exception:
                return 0  # unknown -> 0

        et_idx = _vocab_index(et_vocab, emergency_type)
        src_idx = _vocab_index(src_vocab, source)
        g_idx = _vocab_index([g.lower() for g in gender_vocab], gender_norm)

        # numeric (scaled / stabilized)
        baseline_score = _safe_float(candidate.get("baseline_score"), 0.0)
        distance_km = _safe_float(candidate.get("distance_km"), 1e3)
        rating = _safe_float(candidate.get("rating"), 0.0)
        user_rating_count = _safe_float(candidate.get("user_rating_count"), 0.0)
        category_match_score = _safe_float(candidate.get("category_match_score"), 0.0)
        confidence_score = _safe_float(candidate.get("confidence_score"), 0.0)
        penalties = _safe_float(candidate.get("penalties"), 0.0)

        # log transforms help small-data stability
        import math
        distance_log = math.log1p(max(distance_km, 0.0))
        count_log = math.log1p(max(user_rating_count, 0.0))

        # user profile (snapshot)
        age = _safe_float(context.get("age"), 0.0)
        age_norm = max(0.0, min(age / 100.0, 1.5))  # allow >1 a bit

        offline_mode = float(_safe_int(context.get("offline_mode"), 0))
        is_important = float(_safe_int(context.get("is_important"), 0))
        include_complications = bool(self.mapping.get("include_health_complication_features", True))

        # vector = numeric + one-hots
        vec: List[float] = [
            baseline_score,
            distance_log,
            rating,
            count_log,
            category_match_score,
            confidence_score,
            penalties,
            age_norm,
            offline_mode,
            is_important,
        ]
        if include_complications:
            vec[8:8] = [
                float(_safe_int(context.get("has_asthma_copd"), 0)),
                float(_safe_int(context.get("has_diabetes"), 0)),
                float(_safe_int(context.get("has_heart_disease"), 0)),
                float(_safe_int(context.get("has_stroke_history"), 0)),
                float(_safe_int(context.get("has_epilepsy"), 0)),
                float(_safe_int(context.get("is_pregnant"), 0)),
            ]
        vec += _one_hot(et_idx, len(et_vocab))
        vec += _one_hot(src_idx, len(src_vocab))
        vec += _one_hot(g_idx, len(gender_vocab))
        return vec

    def predict_proba(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> Optional[List[float]]:
        if not self.enabled or self.model is None or tf is None:
            return None
        try:
            X = [self._vectorize_one(c, context) for c in candidates]
            import numpy as np
            Xn = np.asarray(X, dtype="float32")
            preds = self.model.predict(Xn, verbose=0)
            # preds shape (n,1) or (n,)
            out = [float(p[0]) if hasattr(p, "__len__") else float(p) for p in preds]
            # clamp
            out = [max(0.0, min(1.0, p)) for p in out]
            return out
        except Exception:
            return None

    def blend_score(self, baseline_score: float, p_positive: float) -> float:
        cfg = self.config
        model_component = (float(p_positive) - 0.5) * float(cfg.model_score_scale)
        return float(cfg.baseline_weight) * float(baseline_score) + float(cfg.model_weight) * model_component
