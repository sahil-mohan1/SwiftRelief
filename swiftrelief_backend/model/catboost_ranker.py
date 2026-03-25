from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from model.ml_ranker import RankerConfig


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


class CatBoostModelRanker:
    """
    Loads a saved CatBoost model + feature mapping (from train_catboost_model.py)
    and provides p_positive predictions for candidates.

    Fails open: if CatBoost/pandas/model files are unavailable, ranker disables itself.
    """

    def __init__(self, model_dir: str, *, config: Optional[RankerConfig] = None):
        self.model_dir = model_dir
        self.config = config or RankerConfig()
        self.enabled = False
        self.model = None
        self.mapping: Dict[str, Any] = {}

        if CatBoostClassifier is None or pd is None:
            return

        try:
            model_path = os.path.join(model_dir, "model.cbm")
            mapping_path = os.path.join(model_dir, "feature_mapping.json")
            if not (os.path.exists(model_path) and os.path.exists(mapping_path)):
                return

            self.mapping = _load_json(mapping_path)
            model = CatBoostClassifier()
            model.load_model(model_path)
            self.model = model
            self.enabled = True
        except Exception:
            self.enabled = False
            self.model = None
            self.mapping = {}

    def _row_from_candidate(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        baseline_score = _safe_float(candidate.get("baseline_score"), 0.0)
        distance_km = _safe_float(candidate.get("distance_km"), 1e3)
        rating = _safe_float(candidate.get("rating"), 0.0)
        user_rating_count = _safe_float(candidate.get("user_rating_count"), 0.0)
        category_match_score = _safe_float(candidate.get("category_match_score"), 0.0)
        confidence_score = _safe_float(candidate.get("confidence_score"), 0.0)
        penalties = _safe_float(candidate.get("penalties"), 0.0)

        distance_log = math.log1p(max(distance_km, 0.0))
        count_log = math.log1p(max(user_rating_count, 0.0))

        age = _safe_float(context.get("age"), 0.0)
        age_norm = max(0.0, min(age / 100.0, 1.5))

        emergency_type = str(context.get("emergency_type") or "unknown").strip().lower()
        source = str(candidate.get("source") or "unknown").strip().lower()
        gender = str(context.get("gender") or "Other").strip().title()

        return {
            "baseline_score": baseline_score,
            "distance_log": distance_log,
            "rating": rating,
            "count_log": count_log,
            "category_match_score": category_match_score,
            "confidence_score": confidence_score,
            "penalties": penalties,
            "age_norm": age_norm,
            "offline_mode": float(_safe_int(context.get("offline_mode"), 0)),
            "is_important": float(_safe_int(context.get("is_important"), 0)),
            "emergency_type": emergency_type,
            "candidate_source": source,
            "gender": gender,
        }

    def _feature_columns(self) -> List[str]:
        cols = list(self.mapping.get("feature_names") or [])
        if cols:
            return cols

        cats = list(self.mapping.get("categorical_features") or [])
        nums = list(self.mapping.get("numeric_features") or [])
        if nums or cats:
            return [*nums, *cats]

        return [
            "baseline_score",
            "distance_log",
            "rating",
            "count_log",
            "category_match_score",
            "confidence_score",
            "penalties",
            "age_norm",
            "offline_mode",
            "is_important",
            "emergency_type",
            "candidate_source",
            "gender",
        ]

    def predict_proba(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> Optional[List[float]]:
        if not self.enabled or self.model is None or pd is None:
            return None

        try:
            rows = [self._row_from_candidate(c, context) for c in candidates]
            cols = self._feature_columns()
            X = pd.DataFrame(rows)

            for col in cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[cols]

            probs = self.model.predict_proba(X)
            out = [float(p[1]) for p in probs]
            out = [max(0.0, min(1.0, p)) for p in out]
            return out
        except Exception:
            return None

    def blend_score(self, baseline_score: float, p_positive: float) -> float:
        cfg = self.config
        model_component = (float(p_positive) - 0.5) * float(cfg.model_score_scale)
        return float(cfg.baseline_weight) * float(baseline_score) + float(cfg.model_weight) * model_component
