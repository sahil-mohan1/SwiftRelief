"""Simple file cache for nearby live Places results.

Why: In offline mode we cannot call Places. But if the user searched a
location earlier while online, we can reuse the saved nearby hospitals.

Storage: cache/nearby_live_cache.json

Keying: rounded lat/lon + normalized emergency_type
Compatibility lookup: same emergency bucket + nearby distance
TTL: default 14 days (can be overridden by env)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


CACHE_PATH = os.getenv("NEARBY_LIVE_CACHE_PATH", os.path.join("cache", "nearby_live_cache.json"))
TTL_SECONDS = int(os.getenv("NEARBY_LIVE_CACHE_TTL", str(14 * 24 * 3600)))


def _now() -> int:
    return int(time.time())


def _norm_et(s: str) -> str:
    s = (s or "general").strip().lower()
    s = s.replace("\u2019", "'").replace("’", "'")
    s = " ".join(s.split())
    return s


def _canon_et(s: str) -> str:
    et = _norm_et(s)
    txt = f" {et} "

    def _has_any(words: List[str]) -> bool:
        return any(w in txt for w in words)

    if _has_any([" all ", " any ", "everything"]):
        return "all"
    if _has_any(["cardio", "heart"]):
        return "cardiology_heart"
    if _has_any(["emergency", "trauma", "accident", "critical"]):
        return "emergency_trauma"
    if _has_any(["neuro", "stroke", "brain"]):
        return "neurology_brain_stroke"
    if _has_any(["ortho", "orthop", "fracture", "bone"]):
        return "orthopedics_bones_fracture"
    if _has_any(["ophtha", "eye", "vision"]):
        return "ophthalmology_eye"
    if _has_any(["gyn", "gyne", "gyna", "obst", "women", "pregnan", "maternity"]):
        return "obstetrics_gynecology"
    if _has_any(["pedi", "paedi", "child", "children", "nicu", "neonat"]):
        return "pediatrics"
    if _has_any(["ent", "ear", "nose", "throat"]):
        return "ent"
    return et


def _norm_user_key(user_key: str | None) -> str:
    v = str(user_key or "anonymous").strip().lower()
    return v or "anonymous"


def _norm_model_sig(model_signature: str | None) -> str:
    v = str(model_signature or "unknown-model").strip().lower()
    return v or "unknown-model"


def _key(
    lat: float,
    lon: float,
    emergency_type: str,
    *,
    user_key: str | None = None,
    model_signature: str | None = None,
) -> str:
    # 6 decimals (~0.11m) to keep cache reuse effectively exact for user input.
    ll = f"{float(lat):.6f},{float(lon):.6f}"
    return f"{ll}|{_canon_et(emergency_type)}|{_norm_user_key(user_key)}|{_norm_model_sig(model_signature)}"


def _parse_key(k: str) -> Optional[Dict[str, Any]]:
    try:
        parts = str(k).split("|")
        if len(parts) < 2:
            return None
        ll = parts[0]
        et = parts[1]
        lat_s, lon_s = ll.split(",", 1)
        return {
            "lat": float(lat_s),
            "lon": float(lon_s),
            "emergency_type": str(et).strip(),
            "user_key": str(parts[2]).strip() if len(parts) > 2 else "anonymous",
            "model_signature": str(parts[3]).strip() if len(parts) > 3 else "unknown-model",
        }
    except Exception:
        return None


def _fresh(item: Dict[str, Any]) -> bool:
    ts = int(item.get("ts") or 0)
    if not ts:
        return False
    return (_now() - ts) <= TTL_SECONDS


def _results_of(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = item.get("results")
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]
    return []


def _load() -> Dict[str, Any]:
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        return {}
    return {}


def _save(doc: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CACHE_PATH)


def get_cached(
    lat: float,
    lon: float,
    emergency_type: str,
    *,
    user_key: str | None = None,
    model_signature: str | None = None,
) -> List[Dict[str, Any]]:
    doc = _load()
    entries = doc.get("entries")
    if not isinstance(entries, dict):
        return []

    k = _key(
        lat,
        lon,
        emergency_type,
        user_key=user_key,
        model_signature=model_signature,
    )
    item = entries.get(k)
    if isinstance(item, dict) and _fresh(item):
        exact = _results_of(item)
        if exact:
            return exact
    return []


def put_cached(
    lat: float,
    lon: float,
    emergency_type: str,
    results: List[Dict[str, Any]],
    *,
    user_key: str | None = None,
    model_signature: str | None = None,
) -> None:
    doc = _load()
    if not isinstance(doc, dict):
        doc = {}
    if "version" not in doc:
        doc["version"] = 1
    if "entries" not in doc or not isinstance(doc.get("entries"), dict):
        doc["entries"] = {}

    # Store a compact view (only what we need later)
    compact: List[Dict[str, Any]] = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        compact.append(
            {
                "place_id": r.get("place_id"),
                "name": r.get("name"),
                "address": r.get("address"),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "business_status": r.get("business_status"),
                "rating": r.get("rating"),
                "user_rating_count": r.get("user_rating_count"),
                "phone": r.get("phone"),
                "website": r.get("website"),
                "types": r.get("types"),
                "Location": r.get("Location"),
                "District": r.get("District"),
                "State": r.get("State"),
                "Pincode": r.get("Pincode"),
            }
        )

    canonical_et = _canon_et(emergency_type)
    normalized_user = _norm_user_key(user_key)
    normalized_model = _norm_model_sig(model_signature)

    doc["entries"][_key(
        lat,
        lon,
        emergency_type,
        user_key=user_key,
        model_signature=model_signature,
    )] = {
        "ts": _now(),
        "lat": float(lat),
        "lon": float(lon),
        "emergency_type": canonical_et,
        "user_key": normalized_user,
        "model_signature": normalized_model,
        "results": compact,
    }
    _save(doc)
