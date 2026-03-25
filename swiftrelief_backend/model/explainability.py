from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore


class ShapExplainer:
    """Optional SHAP explainer for TFModelRanker candidates.

    - Uses the same vectorization as TFModelRanker to keep explanations faithful.
    - Fails open: if SHAP is unavailable or runtime errors occur, returns no explanation.
    """

    NUMERIC_FEATURES = [
        "baseline_score",
        "distance_log",
        "rating",
        "count_log",
        "category_match_score",
        "confidence_score",
        "penalties",
        "age_norm",
        "has_asthma_copd",
        "has_diabetes",
        "has_heart_disease",
        "has_stroke_history",
        "has_epilepsy",
        "is_pregnant",
        "offline_mode",
        "is_important",
    ]

    HUMAN_LABELS = {
        "baseline_score": "baseline score",
        "distance_log": "distance",
        "rating": "rating",
        "count_log": "rating count",
        "category_match_score": "need match",
        "confidence_score": "data confidence",
        "penalties": "penalty",
        "age_norm": "age",
        "has_asthma_copd": "asthma/COPD profile",
        "has_diabetes": "diabetes profile",
        "has_heart_disease": "heart-disease profile",
        "has_stroke_history": "stroke-history profile",
        "has_epilepsy": "epilepsy profile",
        "is_pregnant": "pregnancy profile",
        "offline_mode": "offline mode",
        "is_important": "critical emergency type",
    }

    HIDDEN_FEATURES = {
        "age_norm",
        "has_asthma_copd",
        "has_diabetes",
        "has_heart_disease",
        "has_stroke_history",
        "has_epilepsy",
        "is_pregnant",
    }
    HIDDEN_PREFIXES = (
        "gender=",
    )
    GENERIC_NAME_PREFIX = "f"

    def __init__(self, ranker: Any, *, top_n: int = 3):
        self.ranker = ranker
        self.top_n = max(1, int(top_n))
        self.background_max_samples = max(1, int(os.getenv("SHAP_BG_MAX_SAMPLES", "16")))
        self.n_permutations = max(1, int(os.getenv("SHAP_N_PERMUTATIONS", "2")))
        self.batch_size = max(1, int(os.getenv("SHAP_BATCH_SIZE", "64")))
        self._explainer = None
        self._explainer_feature_dim: Optional[int] = None
        self.enabled = bool(
            shap is not None
            and np is not None
            and ranker is not None
            and getattr(ranker, "enabled", False)
            and getattr(ranker, "model", None) is not None
        )

    def _feature_names(self) -> List[str]:
        mapping: Dict[str, Any] = getattr(self.ranker, "mapping", {}) or {}
        et_vocab = list(mapping.get("emergency_type_vocab") or [])
        src_vocab = list(mapping.get("source_vocab") or [])
        gender_vocab = list(mapping.get("gender_vocab") or [])

        names = list(self.NUMERIC_FEATURES)
        names += [f"emergency_type={v}" for v in et_vocab]
        names += [f"source={v}" for v in src_vocab]
        names += [f"gender={v}" for v in gender_vocab]
        return names

    def _label_for(self, feature_name: str) -> str:
        if feature_name in self.HUMAN_LABELS:
            return self.HUMAN_LABELS[feature_name]
        if feature_name.startswith("emergency_type="):
            return "emergency type"
        if feature_name.startswith("source="):
            return "source"
        if feature_name.startswith("gender="):
            return "gender"
        return feature_name.replace("_", " ")

    def _predict_fn(self, X: Any) -> Any:
        arr = np.asarray(X, dtype="float32")
        try:
            preds = self.ranker.model(arr, training=False)
        except Exception:
            preds = self.ranker.model.predict(arr, verbose=0)
        return np.asarray(preds).reshape(-1)

    def _get_explainer(self, X: Any) -> Any:
        if shap is None or np is None:
            return None

        dim = int(X.shape[1])
        if self._explainer is not None and self._explainer_feature_dim == dim:
            return self._explainer

        bg = X
        if X.shape[0] > self.background_max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(X.shape[0], size=self.background_max_samples, replace=False)
            bg = X[idx]

        masker = shap.maskers.Independent(bg, max_samples=min(self.background_max_samples, int(bg.shape[0])))
        try:
            explainer = shap.Explainer(self._predict_fn, masker, algorithm="permutation")
        except Exception:
            explainer = shap.Explainer(self._predict_fn, np.mean(X, axis=0, keepdims=True))

        self._explainer = explainer
        self._explainer_feature_dim = dim
        return explainer

    def explain_candidates(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
        if not self.enabled:
            return [None for _ in candidates]
        if not candidates:
            return []

        try:
            X = np.asarray([self.ranker._vectorize_one(c, context) for c in candidates], dtype="float32")  # type: ignore[attr-defined]
            if X.ndim != 2 or X.shape[0] == 0:
                return [None for _ in candidates]

            explainer = self._get_explainer(X)
            if explainer is None:
                return [None for _ in candidates]

            max_evals = self.n_permutations * (2 * int(X.shape[1]) + 1)
            try:
                shap_out = explainer(X, max_evals=max_evals, batch_size=self.batch_size)
            except TypeError:
                shap_out = explainer(X, max_evals=max_evals)

            values = np.asarray(shap_out.values)
            if values.ndim == 3:
                values = values[:, :, 0]
            if values.ndim != 2:
                return [None for _ in candidates]

            base_values = np.asarray(shap_out.base_values).reshape(-1)
            probs = self._predict_fn(X)
            names = self._feature_names()
            if len(names) != values.shape[1]:
                names = [f"f{i}" for i in range(values.shape[1])]

            out: List[Optional[Dict[str, Any]]] = []
            for i in range(values.shape[0]):
                row_vals = values[i]
                row_x = X[i]

                ranked_idx = sorted(range(len(row_vals)), key=lambda j: abs(float(row_vals[j])), reverse=True)
                top_pos: List[Dict[str, Any]] = []
                top_neg: List[Dict[str, Any]] = []

                for j in ranked_idx:
                    contrib = float(row_vals[j])
                    fname = names[j]
                    if fname in self.HIDDEN_FEATURES:
                        continue
                    if any(fname.startswith(prefix) for prefix in self.HIDDEN_PREFIXES):
                        continue
                    # Hide generic placeholder names like f0, f1, ... which aren't informative
                    if isinstance(fname, str) and fname.startswith(self.GENERIC_NAME_PREFIX) and fname[1:].isdigit():
                        continue
                    if contrib >= 0 and len(top_pos) < self.top_n:
                        top_pos.append(
                            {
                                "feature": fname,
                                "label": self._label_for(fname),
                                "value": float(row_x[j]),
                                "shap": contrib,
                            }
                        )
                    elif contrib < 0 and len(top_neg) < 2:
                        top_neg.append(
                            {
                                "feature": fname,
                                "label": self._label_for(fname),
                                "value": float(row_x[j]),
                                "shap": contrib,
                            }
                        )
                    if len(top_pos) >= self.top_n and len(top_neg) >= 2:
                        break

                out.append(
                    {
                        "method": "shap",
                        "base_value": float(base_values[i]) if i < len(base_values) else None,
                        "model_p": float(probs[i]) if i < len(probs) else None,
                        "top_positive": top_pos,
                        "top_negative": top_neg,
                    }
                )
            return out
        except Exception:
            return [None for _ in candidates]
