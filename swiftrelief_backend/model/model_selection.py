from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ModelChoice:
    key: str
    model_dir: str
    enabled: bool


# Defaults are fallbacks only; prefer configuring via .env.
ACTIVE_MODEL_FALLBACK = "tf"
TF_MODEL_DIR_FALLBACK = os.path.join("models", "tf_v3")
CATBOOST_MODEL_DIR_FALLBACK = os.path.join("models", "catboost_v2")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_model_dir(raw_path: str) -> str:
    p = str(raw_path or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return str((_project_root() / p).resolve())


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_model_registry() -> Dict[str, ModelChoice]:
    # Backward compatibility: MODEL_DIR (legacy) maps to TF dir.
    tf_dir_raw = os.getenv("TF_MODEL_DIR") or os.getenv("MODEL_DIR") or TF_MODEL_DIR_FALLBACK
    catboost_dir_raw = os.getenv("CATBOOST_MODEL_DIR") or CATBOOST_MODEL_DIR_FALLBACK

    tf_dir = _resolve_model_dir(tf_dir_raw)
    catboost_dir = _resolve_model_dir(catboost_dir_raw)

    return {
        "tf": ModelChoice(
            key="tf",
            model_dir=tf_dir,
            enabled=_truthy(os.getenv("TF_MODEL_ENABLED"), default=True),
        ),
        "catboost": ModelChoice(
            key="catboost",
            model_dir=catboost_dir,
            enabled=_truthy(os.getenv("CATBOOST_MODEL_ENABLED"), default=True),
        ),
    }


def get_active_model_choice() -> ModelChoice:
    requested = str(os.getenv("ACTIVE_MODEL", ACTIVE_MODEL_FALLBACK)).strip().lower() or ACTIVE_MODEL_FALLBACK
    registry = get_model_registry()

    if requested not in registry:
        requested = ACTIVE_MODEL_FALLBACK

    return registry[requested]
