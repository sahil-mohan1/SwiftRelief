"""Offline place-name -> coordinate lookup + online learning.

Supports the "offline mode" UX:
- When offline, we avoid external geocoding APIs.
- Resolve common place names using:
    1) cache/offline_places_user.json (learned during online usage)
    2) data/offline_places_kerala.json (bundled baseline)

The user file is append-only and lives in cache/ so it can be safely
written at runtime (and later mapped to app-private storage on Android).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


BASE_PATH = os.getenv("OFFLINE_PLACES_BASE", os.path.join("data", "offline_places_kerala.json"))
USER_PATH = os.getenv("OFFLINE_PLACES_USER", os.path.join("cache", "offline_places_user.json"))


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        # Corrupt cache should not crash the app.
        return {}
    return {}


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _iter_places(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    places = doc.get("places")
    if isinstance(places, list):
        return [p for p in places if isinstance(p, dict)]
    return []


def _match_in_doc(qn: str, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for p in _iter_places(doc):
        name = _norm(str(p.get("name") or ""))
        if name and (name == qn or name in qn or qn in name):
            if p.get("lat") is not None and p.get("lng") is not None:
                return p
        alts = p.get("alt") or []
        if isinstance(alts, list):
            for a in alts:
                an = _norm(str(a or ""))
                if an and (an == qn or an in qn or qn in an):
                    if p.get("lat") is not None and p.get("lng") is not None:
                        return p
    return None


def lookup_offline_place(query: str) -> Optional[Dict[str, Any]]:
    """Resolve a place name using user cache first, then bundled baseline."""
    qn = _norm(query)
    if not qn:
        return None

    user = _load_json(USER_PATH)
    hit = _match_in_doc(qn, user)
    if hit:
        return {
            "lat": float(hit["lat"]),
            "lon": float(hit["lng"]),
            "display_name": hit.get("name") or query,
            "provider": "offline_user",
        }

    base = _load_json(BASE_PATH)
    hit = _match_in_doc(qn, base)
    if hit:
        return {
            "lat": float(hit["lat"]),
            "lon": float(hit["lng"]),
            "display_name": hit.get("name") or query,
            "provider": "offline_base",
        }

    return None


def learn_place(query: str, lat: float, lon: float) -> bool:
    """Add a place to cache/offline_places_user.json (if not already present)."""
    q = (query or "").strip()
    qn = _norm(q)
    if not qn:
        return False

    doc = _load_json(USER_PATH)
    if not isinstance(doc, dict):
        doc = {}
    if "version" not in doc:
        doc["version"] = 1
    if "places" not in doc or not isinstance(doc.get("places"), list):
        doc["places"] = []
    places: List[Dict[str, Any]] = doc["places"]

    # de-dupe by name/alt
    for p in places:
        name = _norm(str(p.get("name") or ""))
        if name == qn:
            return False
        alts = p.get("alt") or []
        if isinstance(alts, list) and any(_norm(str(a or "")) == qn for a in alts):
            return False

    # proximity de-dupe: within ~0.5km grid
    for p in places:
        try:
            plat = float(p.get("lat"))
            plon = float(p.get("lng"))
        except Exception:
            continue
        if abs(plat - float(lat)) < 0.005 and abs(plon - float(lon)) < 0.005:
            alts = p.get("alt")
            if not isinstance(alts, list):
                alts = []
            if q and q not in alts and _norm(q) != _norm(p.get("name") or ""):
                alts.append(q)
                p["alt"] = alts
                _save_json(USER_PATH, doc)
                return True
            return False

    places.append({"name": q, "alt": [], "lat": float(lat), "lng": float(lon)})
    _save_json(USER_PATH, doc)
    return True
