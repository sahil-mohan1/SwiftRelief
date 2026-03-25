# model/geocode_cache.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple, List

from dotenv import load_dotenv
import googlemaps

load_dotenv()
CACHE_PATH = "data/geocode_cache.json"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _cache_key(name: str, full_address: str, pincode: str = "") -> str:
    # Stable cache key: name + pincode + address
    return _norm(f"{name} | {pincode} | {full_address}")


def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _tokenize_name(name: str) -> List[str]:
    name = _norm(name)
    tokens = re.findall(r"[a-z0-9]+", name)
    return [t for t in tokens if len(t) >= 3]


def _score_geocode_result(r: Dict[str, Any], hospital_name: str, pincode: str = "") -> int:
    """
    Higher = better. Does NOT discard GEOMETRIC_CENTER.
    Prefers: hospital/health types, pincode match, name match, specific addresses.
    """
    formatted = _norm(r.get("formatted_address") or "")
    types = r.get("types") or []
    types_norm = [t.lower() for t in types]

    geom = r.get("geometry") or {}
    loc_type = (geom.get("location_type") or "").upper()

    score = 0

    # POI signals
    if "hospital" in types_norm:
        score += 12
    if "health" in types_norm:
        score += 5
    if "establishment" in types_norm:
        score += 2
    if "point_of_interest" in types_norm:
        score += 1

    # Pincode match is strong
    pin = str(pincode).strip()
    if pin and pin in formatted:
        score += 7

    # Address contains hospital keyword
    if "hospital" in formatted:
        score += 6

    # Name token hits
    tokens = _tokenize_name(hospital_name)
    token_hits = sum(1 for t in tokens if t in formatted)
    score += min(token_hits, 4) * 2

    # Specific road-ish clues
    if any(x in formatted for x in [" rd", " road", " street", " st ", " ave", " highway", " nh ", " sh "]):
        score += 2

    # location_type preference (but don't discard)
    if loc_type == "ROOFTOP":
        score += 8
    elif loc_type == "RANGE_INTERPOLATED":
        score += 5
    elif loc_type == "GEOMETRIC_CENTER":
        score += 2
    elif loc_type == "APPROXIMATE":
        score -= 2

    return score


def geocode_best_with_cache(
    hospital_name: str,
    full_address: str,
    pincode: str = "",
    region: str = "in",
) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """
    Returns (lat, lon, meta) for best match, using JSON cache to avoid repeat API calls.
    meta includes: formatted_address, location_type, types, score, results_count, place_id.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    # If there's no key, we can't call Google Geocoding. Return None so callers can fallback.
    if not api_key:
        return None

    cache = _load_cache()
    key = _cache_key(hospital_name, full_address, pincode)

    # Cached success
    if key in cache and cache[key].get("lat") is not None and cache[key].get("lon") is not None:
        c = cache[key]
        return float(c["lat"]), float(c["lon"]), dict(c.get("meta", {}))

    # Cached negative
    if key in cache and cache[key].get("meta", {}).get("no_results") is True:
        return None

    gmaps = googlemaps.Client(key=api_key)

    pin = str(pincode).strip()
    # Keep query clean; avoid repeating too many fields
    if pin:
        query = f"{hospital_name}, {full_address}, {pin}, India"
    else:
        query = f"{hospital_name}, {full_address}, India"
    query = re.sub(r"\s+", " ", query).strip().strip(",")

    results = gmaps.geocode(query, region=region)

    if not results:
        cache[key] = {"lat": None, "lon": None, "meta": {"query": query, "no_results": True}}
        _save_cache(cache)
        return None

    best = None
    best_score = -10**9
    for r in results:
        s = _score_geocode_result(r, hospital_name, pin)
        if s > best_score:
            best_score = s
            best = r

    loc = best["geometry"]["location"]
    lat, lon = float(loc["lat"]), float(loc["lng"])

    meta = {
        "query": query,
        "formatted_address": best.get("formatted_address"),
        "types": best.get("types"),
        "location_type": (best.get("geometry") or {}).get("location_type"),
        "score": int(best_score),
        "results_count": len(results),
        "place_id": best.get("place_id"),
    }

    cache[key] = {"lat": lat, "lon": lon, "meta": meta}
    _save_cache(cache)
    return lat, lon, meta
