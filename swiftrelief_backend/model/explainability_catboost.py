from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from catboost import Pool  # type: ignore
except Exception:  # pragma: no cover
    Pool = None  # type: ignore


class CatBoostShapExplainer:
    """CatBoost SHAP explainer using model.get_feature_importance(type='ShapValues')."""

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
        "emergency_type": "emergency type",
        "candidate_source": "source",
        "gender": "gender",
    }

    HIDDEN_FEATURES = {
        "age_norm",
        "gender",
        "has_asthma_copd",
        "has_diabetes",
        "has_heart_disease",
        "has_stroke_history",
        "has_epilepsy",
        "is_pregnant",
    }

    GENERIC_NAME_PREFIX = "f"

    def __init__(self, ranker: Any, *, top_n: int = 3):
        self.ranker = ranker
        self.top_n = max(1, int(top_n))
        self.enabled = bool(
            np is not None
            and pd is not None
            and Pool is not None
            and ranker is not None
            and getattr(ranker, "enabled", False)
            and getattr(ranker, "model", None) is not None
        )

    def _feature_names(self) -> List[str]:
        # Prefer explicit feature columns from the ranker if they look meaningful.
        mapping: Dict[str, Any] = getattr(self.ranker, "mapping", {}) or {}
        # If ranker provides explicit feature_names but they are generic (f0..fN),
        # construct human-readable names from mapping vocabularies and numeric lists.
        raw_names = list(self.ranker._feature_columns() or [])
        is_generic = all((isinstance(n, str) and n.startswith("f") and n[1:].isdigit()) for n in raw_names) if raw_names else True

        if raw_names and not is_generic:
            return raw_names

        # Build names similar to TF explainer: numeric features followed by vocab one-hot features
        nums = list(mapping.get("numeric_features") or [])
        if not nums:
            nums = [
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
            ]

        et_vocab = list(mapping.get("emergency_type_vocab") or [])
        src_vocab = list(mapping.get("source_vocab") or list(mapping.get("source_vocab") or mapping.get("candidate_source_vocab") or []))
        if not src_vocab:
            src_vocab = list(mapping.get("candidate_source_vocab") or [])
        gender_vocab = list(mapping.get("gender_vocab") or [])

        names: List[str] = list(nums)
        names += [f"emergency_type={v}" for v in et_vocab]
        names += [f"candidate_source={v}" for v in src_vocab]
        names += [f"gender={v}" for v in gender_vocab]
        return names

    def _label_for(self, feature_name: str) -> str:
        if feature_name in self.HUMAN_LABELS:
            return self.HUMAN_LABELS[feature_name]
        return feature_name.replace("_", " ")

    def explain_candidates(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
        if not self.enabled:
            return [None for _ in candidates]
        if not candidates:
            return []

        try:
            rows = [self.ranker._row_from_candidate(c, context) for c in candidates]  # type: ignore[attr-defined]
            feature_names = self.ranker._feature_columns()  # type: ignore[attr-defined]
            cat_features = list((getattr(self.ranker, "mapping", {}) or {}).get("categorical_features") or [])

            X = pd.DataFrame(rows)
            for col in feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[feature_names]

            pool = Pool(X, cat_features=cat_features)
            raw_shap = self.ranker.model.get_feature_importance(pool, type="ShapValues")
            values = np.asarray(raw_shap)

            if values.ndim != 2 or values.shape[1] < 2:
                return [None for _ in candidates]

            # CatBoost returns [feature_contrib..., expected_value]
            base_values = values[:, -1]
            contribs = values[:, :-1]
            probs = np.asarray(self.ranker.model.predict_proba(X))[:, 1]

            out: List[Optional[Dict[str, Any]]] = []
            for i in range(contribs.shape[0]):
                row_vals = contribs[i]
                row_x = X.iloc[i]

                ranked_idx = sorted(range(len(row_vals)), key=lambda j: abs(float(row_vals[j])), reverse=True)
                top_pos: List[Dict[str, Any]] = []
                top_neg: List[Dict[str, Any]] = []

                for j in ranked_idx:
                    contrib = float(row_vals[j])
                    fname = feature_names[j] if j < len(feature_names) else f"f{j}"
                    # If mapping used generic f0..fN, try to get better feature names
                    display_names = self._feature_names()
                    if j < len(display_names):
                        fname = display_names[j]
                    # Hide generic feature placeholders like f0, f1... – they're not informative
                    if isinstance(fname, str) and fname.startswith(self.GENERIC_NAME_PREFIX) and fname[1:].isdigit():
                        continue
                    if fname in self.HIDDEN_FEATURES:
                        continue

                    try:
                        raw_value = row_x.iloc[j]
                        value = float(raw_value)
                    except Exception:
                        value = None

                    row = {
                        "feature": fname,
                        "label": self._label_for(fname),
                        "value": value,
                        "shap": contrib,
                    }

                    if contrib >= 0 and len(top_pos) < self.top_n:
                        top_pos.append(row)
                    elif contrib < 0 and len(top_neg) < 2:
                        top_neg.append(row)

                    if len(top_pos) >= self.top_n and len(top_neg) >= 2:
                        break

                out.append(
                    {
                        "method": "catboost_shap",
                        "base_value": float(base_values[i]) if i < len(base_values) else None,
                        "model_p": float(probs[i]) if i < len(probs) else None,
                        "top_positive": top_pos,
                        "top_negative": top_neg,
                    }
                )
            return out
        except Exception:
            return [None for _ in candidates]
