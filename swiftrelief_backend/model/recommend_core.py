# model/recommend_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math
import re
import pandas as pd

from model.text_filters import ALT_MED_RE, is_alt_medicine_hospital

from model.geocode_cache import geocode_best_with_cache
from model.places_cache import verify_place_for_hospital

try:
    from classify import classify_for_swiftrelief
except Exception:
    classify_for_swiftrelief = None


# ---- Your dataset columns ----
COL_SR = "Sr_No"
COL_NAME = "Hospital_Name"
COL_ADDR1 = "Address_Original_First_Line"
COL_LOCATION = "Location"
COL_DISTRICT = "District"
COL_STATE = "State"
COL_PIN = "Pincode"
COL_LAT = "Latitude"
COL_LON = "Longitude"
COL_VALID = "coords_valid"


# ---- Keyword rules ----
def normalize_emergency_type(s: str) -> str:
    """Normalize dropdown values (handles curly quotes) for stable matching."""
    if s is None:
        return "general"
    s = str(s).strip()
    # normalize common curly apostrophes/quotes
    s = s.replace("\u2019", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    s = " ".join(s.split())
    return s.lower()


def is_all_filter(emergency_type: str) -> bool:
    return normalize_emergency_type(emergency_type) in {"all (no filter)", "all", "no filter"}


# IMPORTANT categories: avoid padding with irrelevant specialty hospitals (eye/peds/gyn) just to reach top_k.
IMPORTANT_TYPES = {
    "emergency / trauma",
    "cardiology (heart)",
    "neurology (brain / stroke)",
}


# Emergency type -> keywords used for scoring/matching
EMERGENCY_TO_DEPT = {
    # New dropdown keys (exact labels, normalized to lowercase)
    "all (no filter)": [],
    "emergency / trauma": ["emergency", "trauma", "casualty", "accident", "er", "24/7", "24x7"],
    "cardiology (heart)": ["cardio", "cardiology", "heart", "cardiac", "cardiovascular", "cath", "angioplasty"],
    "neurology (brain / stroke)": ["neuro", "neurology", "brain", "stroke", "neurosurgery", "neuro"],
    "orthopedics (bones / fracture)": ["ortho", "orthopedic", "orthopaedic", "bone", "fracture", "joint", "spine"],
    "ophthalmology (eye)": ["ophthal", "ophthalmology", "eye", "vision", "retina", "lasik"],
    "gynecology (maternity / women's health)": ["gyn", "gyne", "gynecology", "gynaecology", "obgyn", "ob-gyn", "obst", "obstetrics", "maternity", "women", "fertility", "ivf"],
    "pediatrics (child care)": ["pediatric", "paediatric", "pediatrics", "paediatrics", "child", "children", "kids", "nicu", "neonat", "newborn"],
    "ent": ["ent", "ear", "nose", "throat", "otolaryng", "otolaryngology"],

    # Legacy keys (backward compatible)
    "general": ["hospital", "medical", "health centre", "health center", "chc", "multispeciality", "multi speciality"],
    "emergency": ["emergency", "trauma", "casualty", "accident", "er", "24/7", "24x7"],
    "cardiac": ["cardio", "cardiology", "heart", "cardiac", "cardiovascular"],
    "accident": ["emergency", "trauma", "casualty", "accident", "er", "24/7", "24x7"],
    "ortho": ["ortho", "orthopedic", "orthopaedic", "bone", "fracture", "joint", "spine"],
    "eye": ["ophthal", "ophthalmology", "eye", "vision"],
    "maternity": ["gyn", "gynecology", "gynaecology", "obgyn", "obstetrics", "maternity", "women", "fertility"],
}


# Department tags inferred from hospital names
DEPT_KEYWORDS = {
    "cardiology": ["cardio", "cardiology", "heart", "cardiac", "cardiovascular"],
    "neurology": ["neuro", "neurology", "neurosurgery", "brain", "stroke"],
    "orthopedics": ["ortho", "orthopedic", "orthopaedic", "bone", "fracture", "joint", "spine", "knee", "knee pain", "leg", "hip", "sprain", "dislocation"],
    "ophthalmology": ["ophthal", "ophthalmology", "eye", "vision", "retina", "lasik"],
    "gynecology": ["gyn", "gyne", "gynecology", "gynaecology", "obgyn", "ob-gyn", "obst", "obstetrics", "maternity", "women", "fertility", "ivf"],
    "pediatrics": ["pediatric", "paediatric", "pediatrics", "paediatrics", "child", "children", "kids", "nicu", "neonat", "newborn"],
    "ent": ["ent", "ear", "nose", "throat", "otolaryng", "otolaryngology"],

    # Broad/utility tags
    "emergency": ["emergency", "trauma", "24x7", "24/7", "accident", "casualty", "er"],
    "general": [
        "hospital",
        "medical",
        "multispeciality",
        "multi speciality",
        "super speciality",
        "superspeciality",
        "institute",
        "health centre",
        "health center",
        "chc",
    ],
}


def wanted_keywords_for(emergency_type: str) -> list[str]:
    et = normalize_emergency_type(emergency_type)
    # Normalize the one label that may arrive with curly apostrophe in the UI
    et = et.replace("women’s", "women's")
    return EMERGENCY_TO_DEPT.get(et, EMERGENCY_TO_DEPT["general"])


def infer_emergency_type_from_symptom(symptom: str) -> tuple[str, str]:
    """Infer the closest emergency dropdown label from a free-text symptom.

    Strategy:
    - Boundary-aware keyword counting against `EMERGENCY_TO_DEPT` lists.
    - Fallback to department-level keywords from `DEPT_KEYWORDS`.
    - If no rule match is found, call the LLM classifier (when available).
    Returns (label, source) where source is one of: 'rule', 'llm', 'fallback', 'empty'.
    """
    s = (symptom or "").strip().lower()
    if not s:
        return ("general", "empty")

    # boundary-aware matching helper
    alnum = r"[a-z0-9]"
    best: str | None = None
    best_score = 0
    for et, kws in EMERGENCY_TO_DEPT.items():
        if not kws:
            continue
        score = 0
        for kw in kws:
            key = str(kw or "").strip().lower()
            if not key:
                continue
            escaped = re.escape(key).replace(r"\\ ", r"\\\\s+")
            pat = rf"(?<!{alnum}){escaped}(?!{alnum})"
            if re.search(pat, s):
                score += 1
        if score > best_score:
            best_score = score
            best = et

    if best_score > 0 and best is not None:
        return (best, "rule")

    # Fallback: try coarse department keywords -> map to first matching emergency label
    for dept, kws in DEPT_KEYWORDS.items():
        for kw in kws:
            key = str(kw or "").strip().lower()
            if not key:
                continue
            escaped = re.escape(key).replace(r"\\ ", r"\\\\s+")
            pat = rf"(?<!{alnum}){escaped}(?!{alnum})"
            if re.search(pat, s):
                # pick the first emergency_type whose keywords include this dept token
                for et in EMERGENCY_TO_DEPT.keys():
                    if dept in et or dept in (et or ""):
                        return (et, "rule")

    # No confident rule-based mapping found: try LLM classifier if available
    if classify_for_swiftrelief is not None:
        try:
            out = classify_for_swiftrelief(symptom)
            cand = None
            if isinstance(out, dict):
                cand = out.get("department") or out.get("category")
            else:
                cand = str(out or "")

            allowed = list(EMERGENCY_TO_DEPT.keys())
            cand_norm = (cand or "").strip().lower()
            for c in allowed:
                if cand_norm == (c or "").strip().lower():
                    return (c, "llm")
            for c in allowed:
                if cand_norm == normalize_emergency_type(c):
                    return (c, "llm")
            for c in allowed:
                if normalize_emergency_type(c) in cand_norm:
                    return (c, "llm")
        except Exception:
            pass

    # Final fallback: return 'general'
    return ("general", "fallback")


@dataclass
class HospitalResult:
    sr_no: str
    name: str
    address: str
    lat: float
    lon: float
    distance_km: float
    tags: List[str]
    geocode_note: str
    place_id: Optional[str] = None
    geo_good: bool = False
    business_status: str = "UNVERIFIED"
    rating: Optional[float] = None
    user_rating_count: Optional[int] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    place_match_mode: Optional[str] = None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def infer_tags_from_name(name: str) -> List[str]:
    """Infer coarse specialization tags from hospital name.

    Important design choices:
    - Keyword matching uses *token boundaries* (prevents false positives like "ear" in "research").
    - "general" is NOT added just because the name contains "hospital"/"institute".
      We add "general" only when:
        (a) no specialty tags match, OR
        (b) the name clearly indicates multi/super/general capability.
    """

    text = (name or "").lower()

    # Treat only these as specialty tags (general is derived separately).
    specialty_depts = [
        "cardiology",
        "neurology",
        "orthopedics",
        "ophthalmology",
        "gynecology",
        "pediatrics",
        "ent",
        "emergency",
    ]

    # More reliable "general/multi" signals (avoid generic tokens like "hospital" / "institute").
    general_like_markers = [
        "multispeciality",
        "multi speciality",
        "super speciality",
        "superspeciality",
        "medical college",
        "district hospital",
        "general hospital",
        "community health centre",
        "community health center",
        "chc",
        "government hospital",
        "govt hospital",
    ]

    _ALNUM = r"[a-z0-9]"

    def _kw_to_pattern(kw: str) -> re.Pattern:
        # Normalize spaces to flexible whitespace; keep punctuation literal.
        escaped = re.escape((kw or "").lower()).replace(r"\ ", r"\\s+")
        # Require boundaries so short keywords don't match inside other words.
        pat = rf"(?<!{_ALNUM}){escaped}(?!{_ALNUM})"
        return re.compile(pat)

    # Build patterns once per call (small dict, fast enough).
    patterns: dict[str, list[re.Pattern]] = {}
    for dept, kws in DEPT_KEYWORDS.items():
        if dept == "general":
            continue
        patterns[dept] = [_kw_to_pattern(kw) for kw in kws if kw]

    tags: list[str] = []

    # Match specialty departments with boundary-aware patterns.
    for dept in specialty_depts:
        for p in patterns.get(dept, []):
            if p.search(text):
                tags.append(dept)
                break

    # Decide whether to add "general".
    is_general_like = any(_kw_to_pattern(m).search(text) for m in general_like_markers)
    if not tags:
        tags = ["general"]
    elif is_general_like:
        tags.append("general")

    return sorted(set(tags))


def dept_match_score(emergency_type: str, tags: List[str], name: str) -> int:
    # For "All (No filter)" we don't want keyword bias; just distance-based ranking.
    if is_all_filter(emergency_type):
        return 0

    wanted = wanted_keywords_for(emergency_type)
    text = f"{name} {' '.join(tags)}".lower()
    return sum(1 for kw in wanted if kw and kw in text)


def _clean_val(x) -> str:
    s = str(x).strip()
    if not s or s.lower() in ["nan", "none", "0"]:
        return ""
    return s


def build_full_address(row: pd.Series) -> str:
    cols = [COL_ADDR1, COL_LOCATION, COL_DISTRICT, COL_STATE, COL_PIN]
    parts_raw = [_clean_val(row.get(c, "")) for c in cols]
    parts_raw = [p for p in parts_raw if p]

    seen = set()
    parts = []
    for p in parts_raw:
        key = " ".join(p.lower().split())
        if key not in seen:
            seen.add(key)
            parts.append(p)

    return ", ".join(parts)


def load_hospitals(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for c in [COL_NAME, COL_LAT, COL_LON]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")

    if COL_VALID in df.columns:
        valid = df[COL_VALID].astype(str).str.lower().isin(["1", "true", "yes", "y"])
        df = df[valid].copy()

    if COL_SR not in df.columns:
        df[COL_SR] = df.index.astype(str)
    df[COL_SR] = df[COL_SR].astype(str)

    df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
    df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")
    df = df.dropna(subset=[COL_LAT, COL_LON]).copy()

    df[COL_NAME] = df[COL_NAME].fillna("").astype(str)
    df["full_address"] = df.apply(build_full_address, axis=1)
    df["tags"] = df[COL_NAME].apply(infer_tags_from_name)

    return df


def google_maps_link(lat: float, lon: float, place_id: Optional[str] = None) -> str:
    if place_id:
        return f"https://www.google.com/maps/place/?q=place_id:{place_id}"
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


def _geo_confidence(meta: dict, pincode: str) -> bool:
    """
    Decide if geocode is trustworthy enough to use as distance source.
    We do NOT reject GEOMETRIC_CENTER. We use POI signals instead.
    """
    types = meta.get("types") or []
    types_norm = [str(t).lower() for t in types]
    formatted = (meta.get("formatted_address") or "").lower()
    score = int(meta.get("score", 0))

    has_hospital = ("hospital" in types_norm) or ("health" in types_norm)
    pin_ok = bool(pincode) and (str(pincode).strip() in formatted)

    return (has_hospital and (pin_ok or score >= 16)) or (score >= 20)


def _places_center_for_row(r: pd.Series) -> tuple[float, float]:
    """
    Choose the best center for Places verification:
      - Use geocoded coords only if geo_good AND high-precision location type
      - Otherwise use dataset coords
    This avoids mis-matching to the wrong nearby hospital.
    """
    geo_good = bool(r.get("geo_good"))
    geo_note = str(r.get("geo_note") or "")
    # only trust these for centering a Places search
    high_precision = geo_note in {"ROOFTOP", "RANGE_INTERPOLATED"}

    ref_lat = r.get("ref_lat")
    ref_lon = r.get("ref_lon")

    if geo_good and high_precision and ref_lat is not None and ref_lon is not None:
        return float(ref_lat), float(ref_lon)

    return float(r.get(COL_LAT)), float(r.get(COL_LON))


def _name_for_places(hospital_name: str, district: str) -> str:
    """
    Generic names often fail matching unless locality is included.
    Examples: "General Hospital" -> "Ernakulam General Hospital"
    """
    hn = (hospital_name or "").strip()
    hn_norm = " ".join(hn.lower().split())
    if hn_norm in {"general hospital", "district hospital"}:
        d = (district or "").strip()
        if d:
            return f"{d} {hn}"
    return hn


def recommend_hospitals(
    df: pd.DataFrame,
    user_lat: float,
    user_lon: float,
    emergency_type: str,
    top_k: int = 5,
    shortlist: int = 20,
) -> List[HospitalResult]:

    temp = df.copy()

    # Hard exclude alternative-medicine facilities from dataset recommendations.
    # This prevents them from entering shortlist/scoring at all.
    try:
        name_s = temp[COL_NAME].fillna("").astype(str)
        addr_s = temp["full_address"].fillna("").astype(str) if "full_address" in temp.columns else ""
        alt_mask = name_s.str.contains(ALT_MED_RE, na=False)
        if isinstance(addr_s, pd.Series):
            alt_mask = alt_mask | addr_s.str.contains(ALT_MED_RE, na=False)
        temp = temp[~alt_mask].copy()
    except Exception:
        # Fail-open: if anything goes wrong, don't crash recommendations.
        pass

    # Dataset distance (always available)
    temp["dataset_distance_km"] = temp.apply(
        lambda r: haversine_km(user_lat, user_lon, float(r[COL_LAT]), float(r[COL_LON])),
        axis=1
    )

    # 1) shortlist by dataset distance
    temp = temp.sort_values("dataset_distance_km").head(shortlist).copy()

    # 2) geocode only shortlist (cached)
    ref_lats, ref_lons = [], []
    geo_notes, place_ids, geo_good_flags = [], [], []

    for _, r in temp.iterrows():
        name = str(r[COL_NAME])
        addr = str(r["full_address"])
        pin = _clean_val(r.get(COL_PIN, ""))

        g = geocode_best_with_cache(name, addr, pincode=pin, region="in")

        if g is None:
            ref_lats.append(float(r[COL_LAT]))
            ref_lons.append(float(r[COL_LON]))
            geo_notes.append("fallback_dataset")
            place_ids.append(None)
            geo_good_flags.append(False)
            continue

        lat, lon, meta = g
        ref_lats.append(lat)
        ref_lons.append(lon)

        loc_type = str(meta.get("location_type", "geocoded"))
        geo_notes.append(loc_type)
        place_ids.append(meta.get("place_id"))
        geo_good_flags.append(_geo_confidence(meta, pin))

    temp["ref_lat"] = ref_lats
    temp["ref_lon"] = ref_lons
    temp["geo_note"] = geo_notes
    temp["place_id"] = place_ids  # geocode place_id (DO NOT use for Places output)
    temp["geo_good"] = geo_good_flags

    # 3) geocoded distance
    temp["geocoded_distance_km"] = temp.apply(
        lambda r: haversine_km(user_lat, user_lon, float(r["ref_lat"]), float(r["ref_lon"])),
        axis=1
    )

    # 4) final distance used for ranking
    temp["final_distance_km"] = temp.apply(
        lambda r: float(r["geocoded_distance_km"]) if bool(r["geo_good"]) else float(r["dataset_distance_km"]),
        axis=1
    )

    # 5) score + rank
    scores = []
    for _, r in temp.iterrows():
        match = dept_match_score(emergency_type, r["tags"], r[COL_NAME])
        penalty = 1.5 if not bool(r["geo_good"]) else 0.0
        score = match * 10 - float(r["final_distance_km"]) - penalty
        scores.append(score)

    temp["score"] = scores

    # 6) Verify via Places API (New) on a small candidate pool
    candidate_pool = max(top_k * 4, top_k + 10)
    temp = temp.sort_values(["score", "final_distance_km"], ascending=[False, True]).head(candidate_pool).copy()

    place_statuses = []
    place_ratings = []
    place_counts = []
    place_phones = []
    place_websites = []
    place_match_modes = []
    place_place_ids = []

    for _, r in temp.iterrows():
        center_lat, center_lon = _places_center_for_row(r)
        hname = _name_for_places(str(r.get(COL_NAME, "")), str(r.get(COL_DISTRICT, "")))

        info = verify_place_for_hospital(
            sr_no=str(r.get(COL_SR, "")),
            hospital_name=hname,
            address_line=str(r.get(COL_ADDR1, "")),
            location=str(r.get(COL_LOCATION, "")),
            district=str(r.get(COL_DISTRICT, "")),
            state=str(r.get(COL_STATE, "")),
            pincode=str(r.get(COL_PIN, "")),
            lat=center_lat,
            lon=center_lon,
        )

        place_statuses.append(info.get("business_status") or "UNVERIFIED")
        place_ratings.append(info.get("rating"))
        place_counts.append(info.get("user_rating_count"))
        place_phones.append(info.get("phone"))
        place_websites.append(info.get("website"))
        place_match_modes.append(info.get("match_mode"))
        place_place_ids.append(info.get("place_id"))

    temp["business_status"] = place_statuses
    temp["rating"] = place_ratings
    temp["user_rating_count"] = place_counts
    temp["phone"] = place_phones
    temp["website"] = place_websites
    temp["place_match_mode"] = place_match_modes
    temp["places_place_id"] = place_place_ids

    # Drop permanently closed hospitals
    temp = temp[temp["business_status"] != "CLOSED_PERMANENTLY"].copy()

    # Final top_k
    temp = temp.sort_values(["score", "final_distance_km"], ascending=[False, True]).head(top_k)

    results: List[HospitalResult] = []
    for _, r in temp.iterrows():
        # IMPORTANT:
        # - Only use Places place_id if Places actually matched (match_mode set)
        # - Never fall back to geocode place_id (can be unrelated / cause duplicates)
        best_place_id = r.get("places_place_id") if r.get("place_match_mode") else None

        results.append(
            HospitalResult(
                sr_no=str(r.get(COL_SR, "")),
                name=str(r[COL_NAME]),
                address=str(r["full_address"]),
                lat=float(r["ref_lat"]),   # still display the best pin you computed
                lon=float(r["ref_lon"]),
                distance_km=float(r["final_distance_km"]),
                tags=list(r["tags"]),
                geocode_note=str(r["geo_note"]),
                place_id=best_place_id,
                geo_good=bool(r["geo_good"]),
                business_status=str(r.get("business_status") or "UNVERIFIED"),
                rating=r.get("rating"),
                user_rating_count=r.get("user_rating_count"),
                phone=r.get("phone"),
                website=r.get("website"),
                place_match_mode=r.get("place_match_mode"),
            )
        )

    return results


if __name__ == "__main__":
    dfh = load_hospitals("data/hospitals_demo_valid_coords.csv")

    user_lat, user_lon = 10.192676, 76.387587
    emergency = "general"

    out = recommend_hospitals(dfh, user_lat, user_lon, emergency_type=emergency, top_k=5, shortlist=20)

    for i, h in enumerate(out, 1):
        print(f"{i}. {h.name}")
        print(f"   {h.distance_km:.2f} km | tags={h.tags} | geocode={h.geocode_note} | geo_good={h.geo_good}")
        print(f"   {h.address}")
        print(f"   Maps: {google_maps_link(h.lat, h.lon, h.place_id)}\n")
