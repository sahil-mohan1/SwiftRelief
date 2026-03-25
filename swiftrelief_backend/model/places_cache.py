"""
Places API (New) status + info cache using SQLite.

Goal:
- Exclude permanently closed hospitals from recommendations.
- Enrich results with rating/phone/website.
- Minimize API calls using TTL-based SQLite cache (stored in data/app.db).
"""

from __future__ import annotations

import difflib
import math
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import requests

from db import get_db
from model.text_filters import is_alt_medicine_hospital

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
DETAILS_URL_BASE = "https://places.googleapis.com/v1/places/"


def _api_key() -> str:
    key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GMAPS_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_MAPS_API_KEY (or GMAPS_API_KEY) in environment")
    return key


# ---- Tuning ----
RADIUS_KM = float(os.getenv("PLACES_RADIUS_KM", "20"))
THRESHOLD_KM = float(os.getenv("PLACES_THRESHOLD_KM", "20"))
ABBR_MAX_KM = float(os.getenv("PLACES_ABBR_MAX_KM", "5"))
MAX_RESULTS = int(os.getenv("PLACES_MAX_RESULTS", "10"))

MIN_KEY_TOKEN_MATCHES = int(os.getenv("PLACES_MIN_KEY_TOK_MATCHES", "1"))
FUZZY_TOKEN_SIM = float(os.getenv("PLACES_FUZZY_TOKEN_SIM", "0.86"))

# Cache TTLs (seconds)
TTL_OPERATIONAL = int(os.getenv("PLACES_TTL_OPERATIONAL", str(30 * 24 * 3600)))
TTL_CLOSED_PERM = int(os.getenv("PLACES_TTL_CLOSED_PERM", str(180 * 24 * 3600)))
TTL_UNVERIFIED = int(os.getenv("PLACES_TTL_UNVERIFIED", str(7 * 24 * 3600)))
PLACES_HTTP_TIMEOUT = float(os.getenv("PLACES_HTTP_TIMEOUT", "8"))



def _norm_emergency_type(s: str) -> str:
    if s is None:
        return "general"
    s = str(s).strip()
    s = s.replace("\u2019", "'").replace("’", "'")
    s = " ".join(s.split())
    return s.lower()


def _now_ts() -> int:
    return int(time.time())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _radius_km_to_viewport(lat: float, lng: float, radius_km: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    lat_delta = radius_km / 111.0
    cos_lat = math.cos(math.radians(lat))
    lng_delta = radius_km / (111.0 * max(cos_lat, 1e-6))
    low = {"latitude": lat - lat_delta, "longitude": lng - lng_delta}
    high = {"latitude": lat + lat_delta, "longitude": lng + lng_delta}
    return low, high


STOPWORDS = {
    "st", "saint", "s", "the", "and", "of", "near", "road", "rd", "no", "po",
    "hospital", "hsopital", "hosp", "medical", "center", "centre", "clinic",
    "health", "care", "pvt", "private", "ltd", "limited", "trust", "general",
    "govt", "government", "community", "primary", "chc", "phc", "mission",
}


def _normalize(s: str) -> str:
    s = (s or "").lower().replace("`", "'")
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> list[str]:
    out: list[str] = []
    for w in _normalize(s).split():
        if w.isdigit():
            continue
        if w in STOPWORDS:
            continue
        out.append(w)
    return out


def _token_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _key_token_matches(dataset_name: str, candidate_name: str) -> int:
    dt = _tokenize(dataset_name)
    ct = _tokenize(candidate_name)
    if not dt or not ct:
        return 0

    matches = 0
    used: set[int] = set()
    for d in dt:
        best = 0.0
        best_j: Optional[int] = None
        for j, c in enumerate(ct):
            if j in used:
                continue
            s = _token_sim(d, c)
            if s > best:
                best = s
                best_j = j
        if best_j is not None and best >= FUZZY_TOKEN_SIM:
            matches += 1
            used.add(best_j)
    return matches


def _abbr(s: str) -> str:
    """
    Abbreviation extractor ONLY for true initialisms.
    - "P.M.M. Hospital" -> "PMM"
    - "PMM Hospital" -> "PMM"
    Non-initialism names like "West Side Hospital" -> ""
    """
    if not s:
        return ""
    s_up = s.upper()

    initials = re.findall(r"\b([A-Z])\b\.?", s_up)
    if len(initials) >= 2:
        ab = "".join(initials)
        return ab if 2 <= len(ab) <= 6 else ""

    tokens = re.findall(r"\b[A-Z]{2,6}\b", s_up)
    return tokens[0] if tokens else ""


def _is_hospital_type(place_types: list[str] | None) -> bool:
    if not place_types:
        return False
    return ("hospital" in place_types) or ("general_hospital" in place_types)


def _cache_fresh(row: Dict[str, Any]) -> bool:
    age = _now_ts() - int(row.get("last_checked_at") or 0)
    status = row.get("business_status") or "UNVERIFIED"
    if status == "CLOSED_PERMANENTLY":
        return age < TTL_CLOSED_PERM
    if status == "OPERATIONAL":
        return age < TTL_OPERATIONAL
    return age < TTL_UNVERIFIED


def get_cached(sr_no: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        return conn.execute("SELECT * FROM places_cache WHERE sr_no = ?", (str(sr_no),)).fetchone()


def upsert_cached(sr_no: str, data: Dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO places_cache (
                sr_no, place_id, matched_name, matched_address, matched_lat, matched_lng,
                match_mode, match_distance_km, business_status, rating, user_rating_count,
                phone, website, last_checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sr_no) DO UPDATE SET
                place_id=excluded.place_id,
                matched_name=excluded.matched_name,
                matched_address=excluded.matched_address,
                matched_lat=excluded.matched_lat,
                matched_lng=excluded.matched_lng,
                match_mode=excluded.match_mode,
                match_distance_km=excluded.match_distance_km,
                business_status=excluded.business_status,
                rating=excluded.rating,
                user_rating_count=excluded.user_rating_count,
                phone=excluded.phone,
                website=excluded.website,
                last_checked_at=excluded.last_checked_at
            """,
            (
                str(sr_no),
                data.get("place_id"),
                data.get("matched_name"),
                data.get("matched_address"),
                data.get("matched_lat"),
                data.get("matched_lng"),
                data.get("match_mode"),
                data.get("match_distance_km"),
                data.get("business_status"),
                data.get("rating"),
                data.get("user_rating_count"),
                data.get("phone"),
                data.get("website"),
                int(data.get("last_checked_at") or _now_ts()),
            ),
        )


def _places_text_search(
    query: str,
    center_lat: float,
    center_lng: float,
    *,
    restrict: bool = True,
    radius_km: float | None = None,
) -> list[dict]:
    """
    Never crash the whole recommendation flow: on error, log and return [].

    radius_km:
      - None => uses default RADIUS_KM (env-driven)
      - else => overrides the viewport restriction radius
    """
    try:
        key = _api_key()
        rad = float(RADIUS_KM if radius_km is None else radius_km)
        low, high = _radius_km_to_viewport(center_lat, center_lng, rad)

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": ",".join(
                [
                    "places.id",
                    "places.displayName",
                    "places.formattedAddress",
                    "places.location",
                    "places.businessStatus",
                    "places.types",
                ]
            ),
        }

        body = {
            "textQuery": query,
            "maxResultCount": MAX_RESULTS,
        }
        if restrict:
            body["locationRestriction"] = {"rectangle": {"low": low, "high": high}}

        r = requests.post(TEXT_SEARCH_URL, headers=headers, json=body, timeout=PLACES_HTTP_TIMEOUT)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
        if r.status_code != 200:
            print(f"[Places] Text Search failed (restrict={restrict}) HTTP {r.status_code}: {data}")
            return []
        return data.get("places", [])
    except Exception as e:
        print(f"[Places] Text Search exception (restrict={restrict}): {e}")
        return []


def _places_details(place_id: str) -> dict:
    try:
        key = _api_key()
        headers = {
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": ",".join(
                [
                    "id",
                    "displayName",
                    "formattedAddress",
                    "addressComponents",
                    "location",
                    "businessStatus",
                    "rating",
                    "userRatingCount",
                    "types",
                    "websiteUri",
                    "nationalPhoneNumber",
                ]
            ),
        }
        r = requests.get(DETAILS_URL_BASE + place_id, headers=headers, timeout=PLACES_HTTP_TIMEOUT)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
        if r.status_code != 200:
            print(f"[Places] Details failed HTTP {r.status_code}: {data}")
            return {}
        return data
    except Exception as e:
        print(f"[Places] Details exception: {e}")
        return {}


def _uniq_parts(parts: list[str]) -> list[str]:
    seen = set()
    out = []
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        k = " ".join(p.lower().split())
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _build_queries(
    hospital_name: str,
    address_line: str,
    location: str,
    district: str,
    state: str,
    pincode: str,
) -> list[str]:
    name = (hospital_name or "").strip()
    loc = (location or "").strip()
    dist = (district or "").strip()
    st = (state or "").strip()
    pin = (pincode or "").strip()
    addr = (address_line or "").strip()

    # q1: strict, minimal
    q1_parts = _uniq_parts([name, loc or dist, pin, "India"])
    # q2: fallback, richer
    q2_parts = _uniq_parts([name, addr, loc or dist, st, pin, "India"])

    queries = []
    if q1_parts:
        queries.append(", ".join(q1_parts))
    if q2_parts:
        queries.append(", ".join(q2_parts))
    return queries


def verify_place_for_hospital(
    *,
    sr_no: str,
    hospital_name: str,
    address_line: str,
    location: str,
    district: str,
    state: str,
    pincode: str,
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    """
    Verify hospital via Places API (New) and cache results.
    """

    cached = get_cached(str(sr_no))
    if cached and _cache_fresh(cached):
        return cached

    ds_abbr = _abbr(hospital_name)

    queries = _build_queries(hospital_name, address_line, location, district, state, pincode)

    candidates: list[dict] = []
    used_ids: set[str] = set()

    # 1) Restricted search (normal)
    for q in queries:
        for p in _places_text_search(q, lat, lon, restrict=True):
            pid = p.get("id")
            if pid and pid not in used_ids:
                used_ids.add(pid)
                candidates.append(p)
        if candidates:
            break

    # 2) If NOTHING came back, do ONE unrestricted search using the short query (q1)
    if not candidates and queries:
        q = queries[0]
        for p in _places_text_search(q, lat, lon, restrict=False):
            pid = p.get("id")
            if pid and pid not in used_ids:
                used_ids.add(pid)
                candidates.append(p)

    scored = []
    for p in candidates:
        pid = p.get("id")
        pname = (p.get("displayName") or {}).get("text") or ""
        paddr = p.get("formattedAddress")
        bstat = p.get("businessStatus")
        locd = p.get("location") or {}
        plat = locd.get("latitude")
        plng = locd.get("longitude")
        types = p.get("types") or []

        if not pid or plat is None or plng is None:
            continue
        if not _is_hospital_type(types):
            continue

        dist_km = haversine_km(lat, lon, float(plat), float(plng))
        if dist_km > THRESHOLD_KM:
            continue

        km = _key_token_matches(hospital_name, pname)
        cand_abbr = _abbr(pname)
        abbr_ok = bool(ds_abbr) and (ds_abbr == cand_abbr) and (dist_km <= ABBR_MAX_KM)
        passes = (km >= MIN_KEY_TOKEN_MATCHES) or abbr_ok

        scored.append((passes, abbr_ok, km, dist_km, pid, pname, paddr, bstat, float(plat), float(plng)))

    scored.sort(key=lambda x: (not x[0], not x[1], -x[2], x[3]))

    if not scored or not scored[0][0]:
        out = {
            "sr_no": str(sr_no),
            "place_id": None,
            "matched_name": None,
            "matched_address": None,
            "matched_lat": None,
            "matched_lng": None,
            "match_mode": None,
            "match_distance_km": None,
            "business_status": "UNVERIFIED",
            "rating": None,
            "user_rating_count": None,
            "phone": None,
            "website": None,
            "last_checked_at": _now_ts(),
        }
        upsert_cached(str(sr_no), out)
        return out

    passes, abbr_ok, km, dist_km, pid, pname, paddr, bstat, mlat, mlng = scored[0]
    details = _places_details(pid)

    out = {
        "sr_no": str(sr_no),
        "place_id": pid,
        "matched_name": (details.get("displayName") or {}).get("text") or pname,
        "matched_address": details.get("formattedAddress") or paddr,
        "matched_lat": (details.get("location") or {}).get("latitude") or mlat,
        "matched_lng": (details.get("location") or {}).get("longitude") or mlng,
        "match_mode": "ABBR" if abbr_ok else "KEYTOK",
        "match_distance_km": float(dist_km),
        "business_status": (details.get("businessStatus") or bstat or "UNVERIFIED"),
        "rating": details.get("rating"),
        "user_rating_count": details.get("userRatingCount"),
        "phone": details.get("nationalPhoneNumber"),
        "website": details.get("websiteUri"),
        "last_checked_at": _now_ts(),
    }
    upsert_cached(str(sr_no), out)
    return out


def _ac_get(components: list[dict] | None, wanted_type: str) -> str:
    """Extract longText/long_name for a specific address component type."""
    if not components:
        return ""
    for c in components:
        types = c.get("types") or []
        if wanted_type in types:
            return (c.get("longText") or c.get("long_name") or c.get("shortText") or "").strip()
    return ""


def _extract_admin_fields(details: dict) -> dict:
    """Best-effort extraction of locality/district/state/pincode from Places details."""
    comps = details.get("addressComponents") or details.get("address_components") or []

    state = _ac_get(comps, "administrative_area_level_1")
    district = _ac_get(comps, "administrative_area_level_2")
    # Prefer a human-friendly city/locality
    location = (
        _ac_get(comps, "locality")
        or _ac_get(comps, "sublocality")
        or _ac_get(comps, "postal_town")
        or _ac_get(comps, "administrative_area_level_3")
    )
    pincode = _ac_get(comps, "postal_code")

    return {
        "Location": location,
        "District": district,
        "State": state,
        "Pincode": pincode,
    }


def search_nearby_hospitals(
    *,
    lat: float,
    lon: float,
    max_results: int = 15,
    emergency_type: str = "general",
    query: str | None = None,
    radius_km: float | None = None,
) -> list[dict]:
    """
    Discover nearby hospitals directly from Places.

    Returns a list of normalized dicts:
      {
        place_id, name, address, lat, lon, business_status, rating, user_rating_count,
        phone, website, Location, District, State, Pincode, types
      }

    Notes:
    - We only keep candidates that look like hospitals.
    - We exclude CLOSED_PERMANENTLY.
    - We enrich with Place Details for better address components + status.
    """
    # Map emergency/specialization -> Places query.
    # Note: Places "Text Search" works better with natural language than strict type filters.
    et = _norm_emergency_type(emergency_type)
    et = et.replace("women’s", "women's")
    if query is None:
        query = {
            # New dropdown keys (normalized)
            "all (no filter)": "hospital",
            "emergency / trauma": "emergency trauma hospital",
            "cardiology (heart)": "cardiology hospital",
            "neurology (brain / stroke)": "neurology hospital stroke center",
            "orthopedics (bones / fracture)": "orthopedic hospital",
            "ophthalmology (eye)": "eye hospital",
            "gynecology (maternity / women's health)": "gynecology maternity hospital",
            "pediatrics (child care)": "children hospital pediatrics",
            "ent": "ent hospital",

            # Legacy keys
            "cardiac": "cardiology hospital",
            "eye": "eye hospital",
            "ortho": "orthopedic hospital",
            "accident": "emergency trauma hospital",
            "emergency": "emergency hospital",
            "general": "hospital",
        }.get(et, "hospital")

    places = _places_text_search(query, lat, lon, restrict=True, radius_km=radius_km)[: max_results * 2]

    out: list[dict] = []
    used: set[str] = set()

    for p in places:
        pid = p.get("id")
        if not pid or pid in used:
            continue
        used.add(pid)

        types = p.get("types") or []
        if not _is_hospital_type(types):
            continue

        d = _places_details(pid)
        bstat = (d.get("businessStatus") or p.get("businessStatus") or "UNVERIFIED")
        if bstat == "CLOSED_PERMANENTLY":
            continue

        locd = d.get("location") or p.get("location") or {}
        plat = locd.get("latitude")
        plng = locd.get("longitude")
        if plat is None or plng is None:
            continue

        name = (d.get("displayName") or {}).get("text") or (p.get("displayName") or {}).get("text") or ""
        addr = d.get("formattedAddress") or p.get("formattedAddress") or ""

        # Exclude alternative medicine facilities (Ayurveda/Homeopathy/etc.),
        # including chains like "Arya Vaidya".
        if is_alt_medicine_hospital(name, addr):
            continue

        admin = _extract_admin_fields(d)

        out.append(
            {
                "place_id": pid,
                "name": name,
                "address": addr,
                "lat": float(plat),
                "lon": float(plng),
                "business_status": bstat,
                "rating": d.get("rating"),
                "user_rating_count": d.get("userRatingCount"),
                "phone": d.get("nationalPhoneNumber"),
                "website": d.get("websiteUri"),
                "types": d.get("types") or types,
                **admin,
            }
        )

        if len(out) >= max_results:
            break

    return out
