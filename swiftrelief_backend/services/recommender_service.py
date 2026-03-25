# services/recommender_service.py
from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from model.recommend_core import (
    EMERGENCY_TO_DEPT,
    DEPT_KEYWORDS,
    IMPORTANT_TYPES,
    google_maps_link,
    infer_tags_from_name,
    is_all_filter,
    load_hospitals,
    normalize_emergency_type,
    recommend_hospitals,
    wanted_keywords_for,
    infer_emergency_type_from_symptom,
)
try:
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
except Exception:
    google = None
import requests
from model.places_cache import RADIUS_KM, search_nearby_hospitals
from model.nearby_live_cache import get_cached as get_nearby_cached, put_cached as put_nearby_cached
from model.text_filters import is_alt_medicine_hospital, is_alt_tagged
from model.ml_ranker import RankerConfig, TFModelRanker
from model.model_selection import get_active_model_choice
from model.explainability import ShapExplainer
from model.explainability_catboost import CatBoostShapExplainer

import json
import logging

logger = logging.getLogger(__name__)

# Optional local classifier that uses Gemini/LLM and enforces allowed labels
try:
    from classify import classify_for_swiftrelief
except Exception:
    classify_for_swiftrelief = None

from db import (
    create_recommendation_run,
    get_hospital_name_by_place_id,
    get_hospital_specialization_tags_by_place_id,
    get_user_by_id,
    get_user_medical_profile,
    insert_run_candidates,
    upsert_hospital,
)

import csv
from pathlib import Path

# ---- Optional ML ranker ----
_ACTIVE_MODEL_CHOICE = get_active_model_choice()
_ACTIVE_MODEL_KEY = str(_ACTIVE_MODEL_CHOICE.key)
_MODEL_DIR = str(_ACTIVE_MODEL_CHOICE.model_dir)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


# Backward-compatible runtime gate: model remains off unless enabled explicitly.
_USE_MODEL = _truthy(os.getenv('USE_MODEL'), default=False)


def _read_ranker_weights() -> tuple[float, float]:
    """Read blend weights from env with safe defaults.

    Shortcuts:
    - RANKER_BLEND=80:20   -> baseline=0.8, model=0.2
    - RANKER_BLEND=20:80   -> baseline=0.2, model=0.8

    Explicit overrides (optional):
    - BASELINE_WEIGHT=0.8 (or 80)
    - MODEL_WEIGHT=0.2 (or 20)
    """
    baseline_w = 0.8
    model_w = 0.2

    blend = str(os.getenv('RANKER_BLEND', '') or '').strip()
    if ':' in blend:
        left, right = blend.split(':', 1)
        try:
            b = float(left.strip())
            m = float(right.strip())
            if b > 1.0 or m > 1.0:
                total = b + m
                if total > 0:
                    baseline_w = b / total
                    model_w = m / total
            elif b >= 0 and m >= 0 and (b + m) > 0:
                baseline_w = b / (b + m)
                model_w = m / (b + m)
        except Exception:
            pass

    bw_env = os.getenv('BASELINE_WEIGHT')
    mw_env = os.getenv('MODEL_WEIGHT')
    if bw_env is not None and mw_env is not None:
        try:
            b = float(str(bw_env).strip())
            m = float(str(mw_env).strip())
            if b > 1.0 or m > 1.0:
                total = b + m
                if total > 0:
                    baseline_w = b / total
                    model_w = m / total
            elif b >= 0 and m >= 0 and (b + m) > 0:
                baseline_w = b / (b + m)
                model_w = m / (b + m)
        except Exception:
            pass

    return float(baseline_w), float(model_w)


_BASELINE_WEIGHT, _MODEL_WEIGHT = _read_ranker_weights()


def _build_ranker() -> Any:
    cfg = RankerConfig(baseline_weight=_BASELINE_WEIGHT, model_weight=_MODEL_WEIGHT)

    if _ACTIVE_MODEL_KEY == 'tf':
        ranker: Any = TFModelRanker(_MODEL_DIR, config=cfg)
    elif _ACTIVE_MODEL_KEY == 'catboost':
        try:
            from model.catboost_ranker import CatBoostModelRanker  # type: ignore

            ranker = CatBoostModelRanker(_MODEL_DIR, config=cfg)
        except Exception:
            # Step 3 will provide concrete CatBoost runtime ranker.
            ranker = TFModelRanker(_MODEL_DIR, config=cfg)
            ranker.enabled = False
    else:
        ranker = TFModelRanker(_MODEL_DIR, config=cfg)
        ranker.enabled = False

    if not bool(getattr(_ACTIVE_MODEL_CHOICE, 'enabled', False)):
        ranker.enabled = False
    if not _USE_MODEL:
        ranker.enabled = False
    return ranker


_RANKER = _build_ranker()
_MODEL_ENABLED = bool(_RANKER and getattr(_RANKER, 'enabled', False))
_MODEL_SIGNATURE = f"{_ACTIVE_MODEL_KEY}|{os.path.basename(os.path.normpath(_MODEL_DIR)).lower()}|{'1' if _MODEL_ENABLED else '0'}"

_USE_SHAP = str(os.getenv('USE_SHAP', '1')).strip().lower() in {'1', 'true', 'yes', 'on'}
if _ACTIVE_MODEL_KEY == 'tf':
    _EXPLAINER = ShapExplainer(_RANKER, top_n=3)
elif _ACTIVE_MODEL_KEY == 'catboost':
    _EXPLAINER = CatBoostShapExplainer(_RANKER, top_n=3)
else:
    _EXPLAINER = None

if _EXPLAINER is not None and (not _USE_SHAP):
    _EXPLAINER.enabled = False


@lru_cache(maxsize=1)
def _load_df():
    """Load dataset, preferring enriched CSV if it exists."""
    base_csv = os.getenv("HOSPITALS_CSV", os.path.join("data", "hospitals_demo_valid_coords.csv"))
    enriched = os.getenv("HOSPITALS_ENRICHED_CSV", os.path.join("data", "hospitals_enriched.csv"))
    csv_path = enriched if Path(enriched).exists() else base_csv
    return load_hospitals(csv_path)


def _json_safe_number(x: Any) -> Any:
    """Convert NaN/Inf floats to None so JSON + frontend won't break."""
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _parse_specialization_tags_csv(tags_csv: str | None) -> List[str]:
    if not tags_csv:
        return []
    out: List[str] = []
    for t in str(tags_csv).split(','):
        s = t.strip().lower()
        if s and s not in out:
            out.append(s)
    return out


def _normalize_hospital_display_name(name: str) -> str:
    """Normalize known department-level aliases to parent hospital display names."""
    n = (name or "").strip()
    low = n.lower()
    if (
        "department" in low
        and "cardiology" in low
        and "medical college" in low
        and "eranakulam" in low
    ):
        return "Govt Medical College Ernakulam"
    return n


def _dedupe_by_place_id_prefer_operational(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If two hospitals share the same place_id, keep ONLY the OPERATIONAL one.
    If place_id is unique (or missing), keep it (including UNVERIFIED).
    If multiple share the same place_id and none are OPERATIONAL, keep the first.
    """
    best_by_pid: Dict[str, Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []

    for it in items:
        pid = (it.get("place_id") or "").strip()
        if not pid:
            # No place_id => treat as unique (keep)
            ordered.append(it)
            continue

        if pid not in best_by_pid:
            best_by_pid[pid] = it
        else:
            cur = best_by_pid[pid]
            cur_status = (cur.get("business_status") or "").upper()
            new_status = (it.get("business_status") or "").upper()

            # Prefer OPERATIONAL over anything else
            if cur_status != "OPERATIONAL" and new_status == "OPERATIONAL":
                best_by_pid[pid] = it

    # Merge back while preserving original order as much as possible
    seen_pids = set()
    for it in items:
        pid = (it.get("place_id") or "").strip()
        if not pid:
            continue
        if pid in seen_pids:
            continue
        ordered.append(best_by_pid[pid])
        seen_pids.add(pid)

    return ordered


def _name_key(name: str) -> str:
    import re

    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # local copy to avoid extra imports in hot path
    import math

    R = 6371.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2) + math.cos(lat1 * p) * math.cos(lat2 * p) * (math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _dedupe_cross_source(items: List[Dict[str, Any]], *, max_km: float = 0.25) -> List[Dict[str, Any]]:
    """Dedupe across dataset+live: by place_id first, then by proximity+name."""

    # 1) place_id-based (prefer OPERATIONAL)
    items = _dedupe_by_place_id_prefer_operational(items)

    # 2) proximity + name token overlap
    kept: List[Dict[str, Any]] = []
    for it in items:
        try:
            lat = float(it.get("lat"))
            lon = float(it.get("lon"))
        except Exception:
            kept.append(it)
            continue
        nk = _name_key(str(it.get("name", "")))
        toks = set(nk.split())
        dup = False
        for k in kept:
            try:
                klat = float(k.get("lat"))
                klon = float(k.get("lon"))
            except Exception:
                continue
            d = _haversine_km(lat, lon, klat, klon)
            if d > max_km:
                continue
            ktoks = set(_name_key(str(k.get("name", ""))).split())
            if toks and (len(toks & ktoks) / max(len(toks), 1) >= 0.5):
                dup = True
                break
        if not dup:
            kept.append(it)
    return kept


def _score_item(
    emergency_type: str,
    name: str,
    tags: List[str],
    distance_km: float,
    business_status: str,
    rating: float,
    user_rating_count: float,
) -> float:
    """Unified scoring for dataset+live. Higher is better."""
    if is_all_filter(emergency_type):
        match = 0
    else:
        wanted = wanted_keywords_for(emergency_type)
        text = f"{name} {' '.join(tags)}".lower()
        match = _keyword_match_count(text, wanted)

    # small penalty for unknown status; closed hospitals are filtered out earlier
    penalty = 0.0
    if (business_status or "").upper() in {"UNVERIFIED", ""}:
        penalty = 0.5

    et_norm = normalize_emergency_type(emergency_type)

    trust_types = {
        "cardiology (heart)",
        "neurology (brain / stroke)",
        "orthopedics (bones / fracture)",
    }

    if et_norm == "emergency / trauma":
        distance_weight = 2.0
    elif et_norm in trust_types:
        distance_weight = 0.7
    else:
        distance_weight = 1.0

    trust_bonus = 0.0
    if et_norm in trust_types:
        rating_safe = max(0.0, min(float(rating or 0.0), 5.0))
        # Prioritize strong ratings (3.0 -> 0, 5.0 -> 1)
        rating_norm = max(0.0, min((rating_safe - 3.0) / 2.0, 1.0))

        count_safe = max(0.0, float(user_rating_count or 0.0))
        # Large-volume hospitals should stand out for trust-focused categories.
        count_norm = min(math.log1p(count_safe) / math.log1p(50000.0), 1.0)
        count_strength = count_norm ** 1.6

        # Strong trust bonus for cardio/neuro/ortho.
        trust_bonus = (rating_norm * 6.0) + (count_strength * 14.0)
        # Low-volume facilities should not outrank highly trusted hospitals so easily.
        if count_safe < 100:
            trust_bonus -= 2.5
        elif count_safe < 300:
            trust_bonus -= 1.0

    return match * 10.0 - (distance_weight * float(distance_km)) - penalty + trust_bonus


def _keyword_match_count(text: str, keywords: List[str]) -> int:
    """Boundary-aware keyword count to avoid false positives (e.g., 'ent' in 'centre')."""
    t = (text or "").lower()
    if not t or not keywords:
        return 0

    count = 0
    alnum = r"[a-z0-9]"
    for kw in keywords:
        key = str(kw or "").strip().lower()
        if not key:
            continue
        escaped = re.escape(key).replace(r"\ ", r"\\s+")
        pat = rf"(?<!{alnum}){escaped}(?!{alnum})"
        if re.search(pat, t):
            count += 1
    return count


def _medical_adjustment(
    *,
    user_profile: Dict[str, Any] | None,
    emergency_type: str,
    name: str,
    tags: List[str],
    rating: float,
    user_rating_count: float,
) -> float:
    """Rule-based, explainable tweaks using stored medical profile.

    Design goals:
    - Make medical context matter *noticeably* when the user selected "All" (where base score is mostly distance).
    - Avoid hard filters; instead, strongly deprioritize obviously unsuitable facilities (e.g., pediatrics clinics
      for a high-risk senior) unless there are no alternatives.
    - Keep effects bounded so keyword match + distance still matter.
    """
    if not user_profile:
        return 0.0

    try:
        age = user_profile.get("age")
        age_int = int(age) if age is not None else None
    except Exception:
        age_int = None

    child = age_int is not None and age_int < 14
    elderly = age_int is not None and age_int >= 65

    has_heart = bool(user_profile.get("has_heart_disease"))
    has_epilepsy = bool(user_profile.get("has_epilepsy"))
    pregnant = bool(user_profile.get("is_pregnant"))
    gender = str(user_profile.get("gender") or "Other").strip() or "Other"
    gender_norm = gender.lower()

    # Quick text for keyword checks
    text = f"{name} {' '.join(tags or [])}".lower()

    # Hospital "capability" heuristics from name/tags (best-effort only)
    high_cap_kw = (
        "medical college",
        "multi speciality",
        "multispeciality",
        "super speciality",
        "general hospital",
        "district hospital",
        "government",
        "institute",
        "research",
        "critical care",
        "trauma",
        "tertiary",
    )
    clinic_kw = ("clinic", "polyclinic", "dispensary", "nursing home")

    # Pediatric-focused facilities (heuristic)
    pediatric_kw = (
        "pediatric",
        "paediatric",
        "children",
        "child care",
        "childcare",
        "kids",
        "mother & child",
        "mother and child",
        "neonatal",
        "nicu",
    )

    # Women / maternity keywords for "women & children" hospitals (should NOT be penalized like pediatrics-only)
    gyn_kw = (
        "women",
        "woman",
        "ladies",
        "maternity",
        "maternal",
        "obst",
        "obg",
        "ob-g",
        "ob/g",
        "ob-gyn",
        "obgyn",
        "gyne",
        "gyna",
        "gyno",
        "gyn",
        "female",
        "mahila",
    )

    is_high_cap = any(k in text for k in high_cap_kw)
    is_clinic_like = any(k in text for k in clinic_kw)
    is_pediatric_like = any(k in text for k in pediatric_kw)
    is_women_children_like = is_pediatric_like and any(k in text for k in gyn_kw)

    et = normalize_emergency_type(emergency_type).replace("women’s", "women's")

    adj = 0.0

    # Child -> prefer pediatrics when relevant (or when 'All' is used)
    if child:
        if "pediatrics" in et or is_all_filter(emergency_type):
            if "pedi" in text or "children" in text or "child" in text or "mother" in text:
                adj += 1.2
        # For high-stakes categories, lightly penalize clinic-like facilities
        if et in {"emergency / trauma", "cardiology (heart)", "neurology (brain / stroke)", "orthopedics (bones / fracture)"}:
            if is_clinic_like:
                adj -= 0.6

    # Pregnancy -> prefer gynecology/maternity
    if pregnant:
        if "gynecology" in et or is_all_filter(emergency_type):
            if "maternity" in text or "women" in text or "gyn" in text or "obg" in text or "obst" in text:
                adj += 1.2

    # Seniors: apply a small clinic penalty irrespective of gender
    # (hard exclusions are handled separately by tags).
    if elderly and is_clinic_like:
        adj -= 0.6

    # Baseline-only complication boosts (small):
    # 1) Epilepsy + All(No filter): slight preference for neuro-capable hospitals.
    if has_epilepsy and is_all_filter(emergency_type):
        neuro_kw = ("neuro", "neurology", "neurosurgery", "brain", "stroke")
        if ("neurology" in tags) or any(k in text for k in neuro_kw):
            adj += 0.8

    # 2) Heart disease (high risk): small boost for trusted/superspecialty hospitals.
    if has_heart:
        trusted_by_reviews = (float(rating or 0.0) >= 4.0) and (float(user_rating_count or 0.0) >= 200.0)
        if is_high_cap or trusted_by_reviews:
            adj += 0.9

    # Seniors generally should NOT be steered to pediatric-focused *clinics*
    # unless the user explicitly selected pediatrics.
    if elderly and ("pediatrics" not in et):
        if is_pediatric_like and is_clinic_like:
            # Do not penalize "Women & Children" hospitals for female users; they can be relevant.
            if not (is_women_children_like and gender_norm == "female"):
                adj -= 3.6

    # Keep bounded (but allow enough range to beat a few km of distance when "All" is selected)
    if adj > 4.0:
        adj = 4.0
    if adj < -4.0:
        adj = -4.0
    return float(adj)


def _ensure_enriched_dataset(live_items: List[Dict[str, Any]]) -> Tuple[str, int]:
    """Append newly discovered live hospitals into an enriched CSV. Returns (path, added_count)."""
    enriched = os.getenv("HOSPITALS_ENRICHED_CSV", os.path.join("data", "hospitals_enriched.csv"))
    base_csv = os.getenv("HOSPITALS_CSV", os.path.join("data", "hospitals_demo_valid_coords.csv"))
    enriched_path = Path(enriched)
    base_path = Path(base_csv)
    enriched_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Sr_No",
        "Hospital_Name",
        "Address_Original_First_Line",
        "Location",
        "District",
        "State",
        "Pincode",
        "State_ID",
        "District_ID",
        "Latitude",
        "Longitude",
        "coords_valid",
    ]

    # If enriched doesn't exist, initialize from base
    if not enriched_path.exists():
        if base_path.exists():
            enriched_path.write_bytes(base_path.read_bytes())
        else:
            with enriched_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

    # Load existing keys (name+coords)
    existing_rows: List[Dict[str, str]] = []
    with enriched_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            existing_rows.append(row)

    def key_for(row: Dict[str, Any]) -> str:
        name = _name_key(str(row.get("Hospital_Name") or row.get("name") or ""))
        lat = str(row.get("Latitude") or row.get("lat") or "").strip()
        lon = str(row.get("Longitude") or row.get("lon") or "").strip()
        return f"{name}|{lat}|{lon}"

    existing_keys = {key_for(r) for r in existing_rows}

    max_sr = 0
    for r in existing_rows:
        try:
            max_sr = max(max_sr, int(str(r.get("Sr_No") or "0").strip() or "0"))
        except Exception:
            continue

    to_add: List[Dict[str, str]] = []
    for it in live_items:
        # Only add operational/unverified, never closed
        if (it.get("business_status") or "").upper() == "CLOSED_PERMANENTLY":
            continue
        # Never add alternative medicine facilities into the enriched dataset
        if is_alt_medicine_hospital(str(it.get("name") or ""), str(it.get("address") or "")):
            continue
        k = key_for(it)
        if k in existing_keys:
            continue
        existing_keys.add(k)
        max_sr += 1
        to_add.append(
            {
                "Sr_No": str(max_sr),
                "Hospital_Name": str(it.get("name") or "").strip(),
                "Address_Original_First_Line": str(it.get("address") or "").strip(),
                "Location": str(it.get("Location") or "").strip(),
                "District": str(it.get("District") or "").strip(),
                "State": str(it.get("State") or "").strip(),
                "Pincode": str(it.get("Pincode") or "").strip(),
                "State_ID": "",
                "District_ID": "",
                "Latitude": f"{float(it.get('lat')):.7f}",
                "Longitude": f"{float(it.get('lon')):.7f}",
                "coords_valid": "True",
            }
        )

    if to_add:
        with enriched_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            # if file was copied, header exists; if not, safest to ensure
            if f.tell() == 0:
                w.writeheader()
            for row in to_add:
                w.writerow(row)

        # New rows written: ensure future calls reload the enriched dataset
        _load_df.cache_clear()

    return str(enriched_path), len(to_add)



def _forbidden_depts_for(emergency_type: str) -> list[str]:
    et = normalize_emergency_type(emergency_type).replace("women’s", "women's")
    if et == "cardiology (heart)":
        return ["ophthalmology", "pediatrics", "gynecology", "ent", "orthopedics"]
    if et == "neurology (brain / stroke)":
        return ["ophthalmology", "pediatrics", "gynecology", "ent", "orthopedics"]
    if et == "emergency / trauma":
        # Ortho can still be relevant to trauma, so don't forbid it here
        return ["ophthalmology", "pediatrics", "gynecology", "ent"]
    if et == "orthopedics (bones / fracture)":
        # Avoid padding ortho results with clearly unrelated specialty-only facilities.
        return ["ophthalmology", "pediatrics", "gynecology", "ent", "cardiology", "neurology"]
    return []


def _is_general_or_multi(text: str, tags: List[str]) -> bool:
    """Heuristic: treat multi/super/general hospitals as broadly capable.

    These should not be penalized just because their name contains other department keywords.
    """
    t = (text or "").lower()
    if "general" in (tags or []):
        return True
    general_markers = (
        "medical college",
        "multispeciality",
        "multi speciality",
        "super speciality",
        "superspeciality",
        "district hospital",
        "general hospital",
        "government",
        "govt",
    )
    return any(m in t for m in general_markers)


def _is_clinic_or_lab_like(text: str) -> bool:
    """Heuristic: clinics/labs/diagnostics that are usually not suitable for major emergencies."""
    t = (text or "").lower()
    markers = (
        "clinic",
        "polyclinic",
        "dispensary",
        "nursing home",
        "diagnostic",
        "diagnostics",
        "laboratory",
        "lab ",
        " lab",
        "pathology",
        "imaging",
        "scan",
        "radiology",
        "x-ray",
        "xray",
    )
    return any(m in t for m in markers)


def _is_eye_specialized(text: str, tags: List[str]) -> bool:
    """Detect eye-focused facilities (not just hospitals that *have* an eye department)."""
    t = (text or "").lower()
    eye_markers = set(DEPT_KEYWORDS.get("ophthalmology", [])) | {
        "eye",
        "optical",
        "opticals",
        "vision",
        "lasik",
        "retina",
    }
    has_eye = ("ophthalmology" in (tags or [])) or any(m in t for m in eye_markers)
    if not has_eye:
        return False
    # If it clearly looks like a general/multi/super hospital, don't treat it as eye-only.
    if _is_general_or_multi(t, tags):
        return False
    return True


def _apply_specialty_guard(items: List[Dict[str, Any]], emergency_type: str) -> tuple[List[Dict[str, Any]], int]:
    """
    For IMPORTANT types (cardio/trauma/neurology):
    - keep general hospitals, but drop hospitals that look clearly like an unrelated specialty
      (eye/peds/gyn/etc) when they don't match the selected type at all.
    Returns (filtered_items, selected_match_count).
    """
    if is_all_filter(emergency_type):
        return items, 0

    et = normalize_emergency_type(emergency_type).replace("women’s", "women's")
    # Apply guard to high-stakes / strict categories. (Ortho added as strict.)
    strict_types = set(IMPORTANT_TYPES) | {"orthopedics (bones / fracture)"}
    if et not in strict_types:
        return items, 0

    wanted = wanted_keywords_for(emergency_type)
    forbidden_depts = _forbidden_depts_for(emergency_type)
    forbidden_kws: set[str] = set()
    for d in forbidden_depts:
        forbidden_kws.update(DEPT_KEYWORDS.get(d, []))
        forbidden_kws.add(d)

    kept: List[Dict[str, Any]] = []
    sel_count = 0

    for it in items:
        name = str(it.get("name") or "")
        tags = list(it.get("tags") or [])
        text = f"{name} {' '.join(tags)}".lower()

        sel = _keyword_match_count(text, wanted)
        forb = _keyword_match_count(text, list(forbidden_kws))

        if sel > 0:
            sel_count += 1
            kept.append(it)
            continue

        # No selected match.
        # If it strongly looks like an unrelated specialty, drop it.
        # BUT: do not penalize multi/super/general hospitals for containing extra dept keywords.
        if forb > 0:
            if _is_general_or_multi(text, tags) and et != "ent":
                kept.append(it)
                continue
            continue

        # Otherwise keep (general/multispeciality etc.)
        kept.append(it)

    return kept, sel_count


def recommend(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return hospital recommendations along with a run_id for feedback/model training."""

    df = _load_df()
    user_lat = float(payload.get("lat"))
    user_lon = float(payload.get("lon"))
    emergency_type = str(payload.get("emergency_type", "general"))
    # If the client provided a free-text symptom, try inferring an emergency_type.
    inferred = None
    inference_source = None
    try:
        symptom_text = str(payload.get("symptom") or payload.get("symptom_text") or payload.get("query_text") or "").strip()
        if symptom_text:
                # Only override when no explicit non-general emergency_type is provided.
                et_candidate = (emergency_type or "").strip().lower()
                if not et_candidate or et_candidate in {"", "general", "all", "all (no filter)"}:
                    # Use the central inference which does rule->LLM internally
                    try:
                        mapped_label, inference_source = infer_emergency_type_from_symptom(symptom_text)
                        inferred = mapped_label
                        logger.info("recommend: inferred emergency_type from symptom '%s' -> %s (source=%s)", symptom_text, inferred, inference_source)
                        print(f"[Mapping] symptom '{symptom_text}' -> {inferred} (source={inference_source})")
                        if inferred:
                            emergency_type = inferred
                    except Exception:
                        logger.exception("recommend: inference failed; leaving emergency_type as provided")
    except Exception:
        # Fail-open: if inference crashes, keep provided emergency_type.
        pass
    offline_mode = bool(payload.get("offline_mode", False))
    top_k = int(payload.get("top_k", 5))
    shortlist = int(payload.get("shortlist", 20))
    cache_user_key = str(payload.get("_user_id") or "anonymous").strip().lower() or "anonymous"

    et_norm = normalize_emergency_type(emergency_type).replace("women’s", "women's")
    is_important = (et_norm in IMPORTANT_TYPES) and not is_all_filter(emergency_type)

    # Optional: rule-based medical weighting (kept small).
    user_profile: Dict[str, Any] | None = None
    health_profile_json: str | None = None
    try:
        uid_raw = payload.get("_user_id")
        if uid_raw is not None and str(uid_raw).strip() != "":
            user_profile = get_user_medical_profile(int(uid_raw))
            # Snapshot full health profile for training. This must not change when the
            # user edits their profile later.
            u = get_user_by_id(int(uid_raw))
            if u:
                snap = {
                    "age": u.get("age"),
                    "gender": u.get("gender") or "Other",
                    "has_asthma_copd": int(u.get("has_asthma_copd") or 0),
                    "has_diabetes": int(u.get("has_diabetes") or 0),
                    "has_heart_disease": int(u.get("has_heart_disease") or 0),
                    "has_stroke_history": int(u.get("has_stroke_history") or 0),
                    "has_epilepsy": int(u.get("has_epilepsy") or 0),
                    "is_pregnant": int(u.get("is_pregnant") or 0),
                    "other_info": u.get("other_info") or "",
                }
                health_profile_json = json.dumps(snap, ensure_ascii=False)
    except Exception:
        user_profile = None

    # For important categories, fetch a larger candidate pool from the dataset
    ds_top_k = top_k if not is_important else max(top_k * 3, top_k + 10)
    ds_shortlist = shortlist if not is_important else max(shortlist * 2, shortlist)

    # 1) Dataset-based results (verified + filtered)
    ds_results = recommend_hospitals(
        df,
        user_lat=user_lat,
        user_lon=user_lon,
        emergency_type=emergency_type,
        top_k=ds_top_k,
        shortlist=ds_shortlist,
    )

    def _run_live(radius_override_km: float | None) -> List[Dict[str, Any]]:
        """Live Places discovery (online) with on-disk caching."""
        try:
            live = search_nearby_hospitals(
                lat=user_lat,
                lon=user_lon,
                max_results=max(15, top_k * 6),
                emergency_type=emergency_type,
                radius_km=radius_override_km,
            )
        except Exception as e:
            print(f"[Recommend] Live Places failed: {e}")
            live = []

        # Keep online mode truly online-only: do not inject cached nearby fallback here.

        # Cache for offline reuse.
        try:
            if live:
                put_nearby_cached(
                    user_lat,
                    user_lon,
                    emergency_type,
                    live,
                    user_key=cache_user_key,
                    model_signature=_MODEL_SIGNATURE,
                )
        except Exception:
            pass

        # Persist newly found live hospitals to an enriched dataset (same format, IDs blank)
        if live:
            _ensure_enriched_dataset(live)
        return live

    # 2) Live Places discovery around the user (default radius)
    #    - Online: call Places and cache results
    #    - Offline: reuse cached results (if any)
    if offline_mode:
        live_raw = get_nearby_cached(
            user_lat,
            user_lon,
            emergency_type,
            user_key=cache_user_key,
            model_signature=_MODEL_SIGNATURE,
        )
    else:
        live_raw = _run_live(None)

    def _build_combined(live_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        combined: List[Dict[str, Any]] = []
        db_tags_cache: Dict[str, List[str]] = {}
        db_name_cache: Dict[str, str] = {}

        def _tags_for(place_id: str | None, fallback_tags: List[str]) -> List[str]:
            pid = (place_id or "").strip()
            if not pid:
                return list(fallback_tags or [])
            if pid not in db_tags_cache:
                stored = get_hospital_specialization_tags_by_place_id(pid)
                db_tags_cache[pid] = _parse_specialization_tags_csv(stored)
            if db_tags_cache[pid]:
                return list(db_tags_cache[pid])
            return list(fallback_tags or [])

        def _name_for(place_id: str | None, fallback_name: str) -> str:
            pid = (place_id or "").strip()
            if not pid:
                return _normalize_hospital_display_name((fallback_name or "").strip())
            if pid not in db_name_cache:
                db_name_cache[pid] = (get_hospital_name_by_place_id(pid) or "").strip()
            if db_name_cache[pid]:
                return _normalize_hospital_display_name(db_name_cache[pid])
            return _normalize_hospital_display_name((fallback_name or "").strip())

        # Dataset items -> normalized dicts
        for r in ds_results:
            tags = _tags_for(r.place_id, list(r.tags or []))
            display_name = _name_for(r.place_id, r.name)
            combined.append(
                {
                    "sr_no": r.sr_no,
                    "source": "dataset",
                    "name": display_name,
                    "address": r.address,
                    "lat": r.lat,
                    "lon": r.lon,
                    "distance_km": r.distance_km,
                    "tags": tags,
                    "geocode_note": r.geocode_note,
                    "geo_good": r.geo_good,
                    "business_status": r.business_status,
                    "rating": _json_safe_number(r.rating),
                    "user_rating_count": _json_safe_number(r.user_rating_count),
                    "phone": r.phone,
                    "website": r.website,
                    "place_match_mode": r.place_match_mode,
                    "place_id": r.place_id,
                    "maps": google_maps_link(r.lat, r.lon, r.place_id),
                }
            )

        # Live items -> normalized dicts
        for p in live_items:
            bstat = (p.get("business_status") or "UNVERIFIED")
            if str(bstat).upper() == "CLOSED_PERMANENTLY":
                continue
            if is_alt_medicine_hospital(str(p.get("name") or ""), str(p.get("address") or "")):
                continue
            lat = float(p.get("lat"))
            lon = float(p.get("lon"))
            dist = _haversine_km(user_lat, user_lon, lat, lon)
            pid = p.get("place_id")
            inferred_tags = infer_tags_from_name(str(p.get("name") or ""))
            tags = _tags_for(pid, inferred_tags)
            display_name = _name_for(pid, str(p.get("name") or ""))
            combined.append(
                {
                    "sr_no": f"LIVE:{pid}" if pid else "LIVE",
                    "source": "live",
                    "name": display_name,
                    "address": str(p.get("address") or "").strip(),
                    "lat": lat,
                    "lon": lon,
                    "distance_km": dist,
                    "tags": tags,
                    "geocode_note": "live_places",
                    "geo_good": True,
                    "business_status": bstat,
                    "rating": _json_safe_number(p.get("rating")),
                    "user_rating_count": _json_safe_number(p.get("user_rating_count")),
                    "phone": p.get("phone"),
                    "website": p.get("website"),
                    "place_match_mode": "LIVE",
                    "place_id": pid,
                    "maps": google_maps_link(lat, lon, pid),
                }
            )

        return combined

    def _rank_and_trim(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Cross-source dedupe + rank
        items = _dedupe_cross_source(items)

        # Global hard exclusion: never show alternative medicine facilities.
        items = [
            it
            for it in items
            if not (
                is_alt_medicine_hospital(
                    str(it.get("name") or ""),
                    str(it.get("address") or ""),
                )
                or is_alt_tagged(list(it.get("tags") or []))
            )
        ]

        # Hard exclusion: strict removals based on user demographics.
        # 1) Adult users (age >= 18): drop hospitals that are tagged ONLY as pediatrics.
        # 2) Adult male users (age >= 18 and gender male): drop hospitals that are tagged ONLY as gynecology.
        if user_profile:
            try:
                age_raw = user_profile.get("age")
                age_int = int(age_raw) if age_raw is not None else None
            except Exception:
                age_int = None
            gender_norm = str(user_profile.get("gender") or "Other").strip().lower()

            def _norm_tag_set(it: Dict[str, Any]) -> set[str]:
                raw_tags = it.get("tags") or []
                if not isinstance(raw_tags, list):
                    raw_tags = [raw_tags]
                return {str(t).strip().lower() for t in raw_tags if str(t).strip()}

            # Adult -> remove pediatrics-only hospitals
            if age_int is not None and age_int >= 18:
                def _is_only_pediatrics_tag(it: Dict[str, Any]) -> bool:
                    norm_tags = _norm_tag_set(it)
                    return norm_tags == {"pediatrics"}

                items = [it for it in items if not _is_only_pediatrics_tag(it)]

            # Adult male -> remove women/children-only hospitals.
            # This excludes:
            # - gynecology-only
            # - gynecology + pediatrics
            if age_int is not None and age_int >= 18 and gender_norm == "male":
                def _is_women_children_only_tag(it: Dict[str, Any]) -> bool:
                    norm_tags = _norm_tag_set(it)
                    if not norm_tags:
                        return False
                    return ("gynecology" in norm_tags) and norm_tags.issubset({"gynecology", "pediatrics"})

                items = [it for it in items if not _is_women_children_only_tag(it)]

        for it in items:
            base = _score_item(
                emergency_type,
                str(it.get("name") or ""),
                list(it.get("tags") or ["general"]),
                float(it.get("distance_km") or 1e9),
                str(it.get("business_status") or "UNVERIFIED"),
                float(it.get("rating") or 0.0),
                float(it.get("user_rating_count") or 0.0),
            )

            med_adj = _medical_adjustment(
                user_profile=user_profile,
                emergency_type=emergency_type,
                name=str(it.get("name") or ""),
                tags=list(it.get("tags") or []),
                rating=float(it.get("rating") or 0.0),
                user_rating_count=float(it.get("user_rating_count") or 0.0),
            )
            it["baseline_score"] = float(base) + float(med_adj)
            it["score"] = it["baseline_score"]

        # -----------------
        # Hard filters to avoid obviously irrelevant recommendations
        # -----------------
        et_local = normalize_emergency_type(emergency_type).replace("women’s", "women's")

        # 1) Eye-focused hospitals should appear only for Eye and All searches.
        if (not is_all_filter(emergency_type)) and (et_local != "ophthalmology (eye)"):
            items = [
                it
                for it in items
                if not _is_eye_specialized(
                    f"{it.get('name','')} {' '.join(it.get('tags') or [])}",
                    list(it.get("tags") or []),
                )
            ]

        # 2) For high-stakes categories, avoid clinics/labs/diagnostics.
        if et_local in {
            "cardiology (heart)",
            "emergency / trauma",
            "neurology (brain / stroke)",
            "orthopedics (bones / fracture)",
        }:
            filtered: List[Dict[str, Any]] = []
            for it in items:
                text = f"{it.get('name','')} {' '.join(it.get('tags') or [])}".lower()
                tags = list(it.get("tags") or [])
                if _is_clinic_or_lab_like(text) and not _is_general_or_multi(text, tags):
                    continue
                filtered.append(it)
            items = filtered

        items.sort(key=lambda x: (-(x.get("score") or 0), float(x.get("distance_km") or 1e9)))

        # Apply specialty guard (prevents eye/peds/gyn hospitals from padding cardio/trauma/neuro)
        items, sel_count = _apply_specialty_guard(items, emergency_type)

        # Medical post-filter: for high-risk seniors, avoid pediatrics-only clinic-like facilities from appearing
        # in the top list (unless the user explicitly chose Pediatrics, or we don't have enough alternatives).
        # Important: do NOT exclude "Women & Children" hospitals for female users.
        if user_profile and ("pediatrics" not in normalize_emergency_type(emergency_type).lower()):
            try:
                age = user_profile.get("age")
                age_int = int(age) if age is not None else None
            except Exception:
                age_int = None
            elderly = age_int is not None and age_int >= 65

            gender = str(user_profile.get("gender") or "Other").strip() or "Other"
            gender_norm = gender.lower()

            # For senior females, rely on the explicit hard filter above + small clinic penalty in scoring.
            if elderly and gender_norm != "female":
                pediatric_kw = (
                    "pediatric",
                    "paediatric",
                    "children",
                    "child care",
                    "childcare",
                    "kids",
                    "mother & child",
                    "mother and child",
                    "neonatal",
                    "nicu",
                )
                gyn_kw = (
                    "women",
                    "woman",
                    "ladies",
                    "maternity",
                    "maternal",
                    "obst",
                    "obg",
                    "ob-gyn",
                    "obgyn",
                    "gyne",
                    "gyna",
                    "gyn",
                    "mahila",
                )

                def _is_peds_only_clinic(it: Dict[str, Any]) -> bool:
                    text = f"{it.get('name','')} {' '.join(it.get('tags') or [])}".lower()
                    is_peds = any(k in text for k in pediatric_kw)
                    if not is_peds:
                        return False
                    # Senior penalties should apply only to clinic/lab-like facilities.
                    if not _is_clinic_or_lab_like(text):
                        return False
                    is_women_children = is_peds and any(k in text for k in gyn_kw)
                    # Keep women&children for female users
                    if is_women_children and gender_norm == "female":
                        return False
                    return True

                non_peds = [it for it in items if not _is_peds_only_clinic(it)]
                # Only drop peds-only if we still have enough to serve top_k
                if len(non_peds) >= top_k:
                    items = non_peds

        # -----------------
        # ML re-ranking (safe blending; baseline still dominates)
        # -----------------
        # Compute lightweight per-candidate features used both for persistence and ML inference.
        wanted = wanted_keywords_for(emergency_type)
        et_norm2 = normalize_emergency_type(emergency_type).replace('women’s', "women's")
        is_important2 = (et_norm2 in IMPORTANT_TYPES) and not is_all_filter(emergency_type)

        for it in items:
            name = str(it.get('name') or '').strip()
            tags = list(it.get('tags') or [])
            text = f"{name} {' '.join(tags)}".lower()
            sel = 0
            if wanted:
                sel = sum(1 for kw in wanted if kw and kw in text)
            match_score = (sel / max(len(wanted), 1)) if not is_all_filter(emergency_type) else 0.0

            bstat = str(it.get('business_status') or '').upper()
            geo_good = bool(it.get('geo_good', True))
            confidence = 1.0
            if not geo_good:
                confidence -= 0.25
            if bstat in {'UNVERIFIED', ''}:
                confidence -= 0.25
            confidence = max(0.0, min(1.0, confidence))

            penalties = 0.0
            if bstat in {'UNVERIFIED', ''}:
                penalties += 0.5

            it['category_match_score'] = float(match_score)
            # Keep confidence signal, but lower its model influence.
            it['confidence_score'] = float(confidence) * 0.5
            it['penalties'] = float(penalties)

        # Prepare context for model inference (from current user_profile; snapshot is stored separately).
        ctx: Dict[str, Any] = {
            'emergency_type': str(emergency_type or '').strip().lower(),
            'offline_mode': int(bool(offline_mode)),
            'is_important': int(bool(is_important2)),
        }
        if user_profile:
            ctx.update({
                'age': user_profile.get('age'),
                'gender': user_profile.get('gender') or 'Other',
            })

        # Default final_score = baseline_score
        for it in items:
            it['final_score'] = float(it.get('baseline_score') or 0.0)

        used_model = False
        if _RANKER and getattr(_RANKER, 'enabled', False):
            probs = _RANKER.predict_proba(items, ctx)  # list[float] or None
            if probs is not None and len(probs) == len(items):
                used_model = True
                for it, p in zip(items, probs):
                    it['model_p'] = float(p)
                    it['blended_score'] = float(_RANKER.blend_score(float(it.get('baseline_score') or 0.0), float(p)))
                    it['final_score'] = it['blended_score']

        # Cardio/Neuro/Ortho: apply trust boost on final score too, so model blending does not suppress
        # well-rated, high-volume hospitals.
        if et_norm2 in {"cardiology (heart)", "neurology (brain / stroke)", "orthopedics (bones / fracture)"}:
            for it in items:
                rating_safe = max(0.0, min(float(it.get('rating') or 0.0), 5.0))
                rating_norm = max(0.0, min((rating_safe - 3.0) / 2.0, 1.0))
                count_safe = max(0.0, float(it.get('user_rating_count') or 0.0))
                count_norm = min(math.log1p(count_safe) / math.log1p(50000.0), 1.0)
                count_strength = count_norm ** 1.6
                trust_boost = (rating_norm * 6.0) + (count_strength * 14.0)
                if count_safe < 100:
                    trust_boost -= 2.5
                elif count_safe < 300:
                    trust_boost -= 1.0
                it['final_score'] = float(it.get('final_score') or 0.0) + float(trust_boost)

        # Emergency / Trauma: stronger proximity preference so very close hospitals rank higher.
        if normalize_emergency_type(emergency_type) == "emergency / trauma":
            for it in items:
                d = float(it.get('distance_km') or 1e9)
                proximity_bonus = max(0.0, 6.0 - d) * 0.8
                it['final_score'] = float(it.get('final_score') or 0.0) + float(proximity_bonus)

        # Sort by what the user actually sees (final_score).
        items.sort(key=lambda x: (-(x.get('final_score') or 0), float(x.get('distance_km') or 1e9)))
        top_items = items[:top_k]
        if _EXPLAINER and getattr(_EXPLAINER, 'enabled', False) and top_items:
            exp_rows = _EXPLAINER.explain_candidates(top_items, ctx)
            if exp_rows and len(exp_rows) == len(top_items):
                for it, exp in zip(top_items, exp_rows):
                    if exp:
                        it['explain'] = exp

        return top_items, sel_count

    combined = _build_combined(live_raw)
    ranked, sel_count = _rank_and_trim(combined)

    expanded_used = False

    # 3) If important category still doesn't have enough matching hospitals, expand Places radius x2
    #    Only possible online.
    if (not offline_mode) and is_important and sel_count < top_k:
        live_more = _run_live(float(RADIUS_KM) * 2.0)
        if live_more:
            combined = _build_combined(live_raw + live_more)
            ranked, _ = _rank_and_trim(combined)
            expanded_used = True

    # -----------------
    # Persist model-ready rows + return run_id (for feedback)
    # -----------------
    run_id: int | None = None
    try:
        model_version_used = (
            os.path.basename(os.path.normpath(_MODEL_DIR))
            if (_RANKER and getattr(_RANKER, 'enabled', False))
            else 'baseline'
        )
        run_id = create_recommendation_run(
            user_id=str(payload.get("_user_id") or "").strip() or None,
            query_text=str(payload.get("query_text") or "").strip() or None,
            query_lat=user_lat,
            query_lng=user_lon,
            emergency_type=emergency_type,
            radius_km=float(RADIUS_KM) * (2.0 if expanded_used else 1.0),
            used_directions=False,
            expanded_radius=expanded_used,
            offline_mode=offline_mode,
            feature_version=str(payload.get("feature_version") or "v1"),
            model_version=str(model_version_used),
            health_profile_json=health_profile_json,
        )

        # Build candidate feature rows from returned ranked list.
        # (Later, you can expand this to store the full candidate pool instead of top_k.)
        cand_rows: list[dict[str, Any]] = []

        for idx, it in enumerate(ranked, start=1):
            name = str(it.get("name") or "").strip()
            addr = str(it.get("address") or "").strip()
            lat = float(it.get("lat"))
            lon = float(it.get("lon"))
            pid = (it.get("place_id") or "").strip() or None
            tags = list(it.get("tags") or [])

            match_score = float(it.get('category_match_score') or 0.0)
            confidence = float(it.get('confidence_score') or 0.0)
            penalties = float(it.get('penalties') or 0.0)
            baseline_score = it.get('baseline_score')
            blended_score = it.get('blended_score')
            model_p = it.get('model_p')
            final_score = it.get('final_score')
            if final_score is None:
                final_score = baseline_score
            it['score'] = _json_safe_number(final_score)
            hid = upsert_hospital(
                place_id=pid,
                hospital_name=name,
                address=addr,
                latitude=lat,
                longitude=lon,
                source=str(it.get("source") or "dataset"),
                coords_valid=bool(it.get("geo_good", True)),
                is_alternate_medicine=False,
                specialization_tags=",".join(tags) if tags else None,
            )

            # Attach hospital_id back to the response item so frontend can send feedback
            it["hospital_id"] = hid

            cand_rows.append(
                {
                    "hospital_id": hid,
                    "distance_km": _json_safe_number(it.get("distance_km")),
                    "travel_time_min": None,
                    "rating": _json_safe_number(it.get("rating")),
                    "user_ratings_total": _json_safe_number(it.get("user_rating_count")),
                    "category_match_score": float(match_score),
                    "confidence_score": float(confidence),
                    "penalties": float(penalties),
                    "final_score": _json_safe_number(final_score),
                    "baseline_score": _json_safe_number(baseline_score),
                    "model_p": _json_safe_number(model_p),
                    "blended_score": _json_safe_number(blended_score),
                    "rank": int(idx),
                    "source": str(it.get("source") or "dataset"),
                }
            )

        cand_ids = insert_run_candidates(run_id, cand_rows)
        # Attach run_candidate_id so feedback can target the exact feature row.
        for i, it in enumerate(ranked):
            if i < len(cand_ids):
                it["run_candidate_id"] = cand_ids[i]
        # Remove internal scoring fields from API response (demo stays clean).
        for it in ranked:
            for k in ('baseline_score','final_score','model_p','blended_score','category_match_score','confidence_score','penalties'):
                it.pop(k, None)
    except Exception:
        # Logging must never break recommendations.
        pass

    explain_meta = {
        "model_enabled": bool(_RANKER and getattr(_RANKER, 'enabled', False)),
        "shap_enabled": bool(_EXPLAINER and getattr(_EXPLAINER, 'enabled', False)),
        "model_key": _ACTIVE_MODEL_KEY,
        "model_dir": _MODEL_DIR,
        "model_signature": _MODEL_SIGNATURE,
    }
    return {
        "run_id": run_id,
        "results": ranked,
        "explain_meta": explain_meta,
        "inferred_emergency_type": inferred,
        "inference_source": inference_source,
    }