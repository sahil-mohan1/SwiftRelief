#!/usr/bin/env python3
"""
Places API (New) hospital status check by Sr_No (ONLY input parameter).

Hybrid matching:
PASS if:
  A) key_token_matches >= 1  (rare/meaningful tokens)
  OR
  B) abbreviation matches (e.g., P.M.M -> PMM) AND distance <= ABBR_MAX_KM
Otherwise UNVERIFIED.

This fixes:
- "St Philomina..." -> won't match St Joseph
- "P.M.M. Hsopital" -> will match "PMM Hospital"
"""

import os, sys, csv, json, math, re
import requests
from dotenv import load_dotenv
import difflib

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
DETAILS_URL_BASE = "https://places.googleapis.com/v1/places/"

RADIUS_KM = 20.0
THRESHOLD_KM = 20.0
MAX_RESULTS = 10

MIN_KEY_TOKEN_MATCHES = 1
FUZZY_TOKEN_SIM = 0.86
ABBR_MAX_KM = 5.0   # only trust abbreviation-based match if very close


def load_env():
    load_dotenv()
    api_key = os.getenv("GMAPS_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
    csv_path = os.getenv("HOSPITALS_CSV")
    if not api_key:
        raise RuntimeError("Missing GMAPS_API_KEY in .env (or GOOGLE_MAPS_API_KEY).")
    if not csv_path:
        raise RuntimeError("Missing HOSPITALS_CSV in .env.")
    return api_key, csv_path


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def radius_km_to_viewport(lat, lng, radius_km):
    lat_delta = radius_km / 111.0
    cos_lat = math.cos(math.radians(lat))
    lng_delta = radius_km / (111.0 * max(cos_lat, 1e-6))
    low = {"latitude": lat - lat_delta, "longitude": lng - lng_delta}
    high = {"latitude": lat + lat_delta, "longitude": lng + lng_delta}
    return low, high


def read_hospital_by_srno(csv_path: str, sr_no: str):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        expected = [
            "Sr_No","Hospital_Name","Address_Original_First_Line","Location","District","State","Pincode",
            "State_ID","District_ID","Latitude","Longitude","coords_valid"
        ]
        missing = [c for c in expected if c not in headers]
        if missing:
            raise RuntimeError(f"CSV missing expected columns: {missing}\nFound: {headers}")

        for row in reader:
            if str(row.get("Sr_No", "")).strip() == str(sr_no).strip():
                try:
                    lat = float(str(row.get("Latitude", "")).strip())
                    lng = float(str(row.get("Longitude", "")).strip())
                except ValueError:
                    lat = None; lng = None
                return {
                    "Sr_No": row.get("Sr_No"),
                    "Hospital_Name": (row.get("Hospital_Name") or "").strip(),
                    "Address": (row.get("Address_Original_First_Line") or "").strip(),
                    "Location": (row.get("Location") or "").strip(),
                    "District": (row.get("District") or "").strip(),
                    "State": (row.get("State") or "").strip(),
                    "Pincode": (row.get("Pincode") or "").strip(),
                    "Latitude": lat,
                    "Longitude": lng,
                    "coords_valid": (row.get("coords_valid") or "").strip(),
                }
    return None


def build_query(h):
    parts = [h["Hospital_Name"]]
    for k in ["Location", "District", "State", "Pincode"]:
        v = (h.get(k) or "").strip()
        if v:
            parts.append(v)
    addr = (h.get("Address") or "").strip()
    if addr:
        parts.append(addr)
    parts.append("India")
    return ", ".join([p for p in parts if p])


STOPWORDS = {
    "st","saint","s","the","and","of","near","road","rd","no","po",
    "hospital","hsopital","hosp","medical","center","centre","clinic","health","care",
    "pvt","private","ltd","limited","trust","general","govt","government",
    "community","primary","chc","phc","mission","centre"
}

def normalize(s: str) -> str:
    s = (s or "").lower().replace("`", "'")
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    out = []
    for w in normalize(s).split():
        if w.isdigit():
            continue
        if w in STOPWORDS:
            continue
        out.append(w)
    return out

def token_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def key_token_matches(dataset_name: str, candidate_name: str) -> int:
    dt = tokenize(dataset_name)
    ct = tokenize(candidate_name)
    if not dt or not ct:
        return 0

    matches = 0
    used = set()
    for d in dt:
        best = 0.0
        best_j = None
        for j, c in enumerate(ct):
            if j in used:
                continue
            s = token_sim(d, c)
            if s > best:
                best = s
                best_j = j
        if best_j is not None and best >= FUZZY_TOKEN_SIM:
            matches += 1
            used.add(best_j)
    return matches

def abbr(s: str) -> str:
    """
    Robust abbreviation extractor:
    - "P.M.M. Hsopital" -> "PMM"
    - "P. M. M. Hospital" -> "PMM"
    - "P M M Hospital" -> "PMM"
    - "PMM Hospital" -> "PMM"
    """
    if not s:
        return ""

    s_up = s.upper()

    # 1) Collect single-letter initials that appear as standalone letters (with optional dots)
    # Examples matched: "P.", "M", "M."
    initials = re.findall(r"\b([A-Z])\b\.?", s_up)
    # Join all initials found in sequence; for P.M.M it becomes PMM
    if len(initials) >= 2:
        return "".join(initials)

    # 2) Fallback: find a short all-caps token like PMM, CHC, PHC, etc.
    tokens = re.findall(r"\b[A-Z]{2,6}\b", s_up)
    return tokens[0] if tokens else ""


def is_hospital_type(place_types):
    if not place_types:
        return False
    return ("hospital" in place_types) or ("general_hospital" in place_types)


def places_text_search(api_key, text_query, center_lat, center_lng, radius_km):
    low, high = radius_km_to_viewport(center_lat, center_lng, radius_km)
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join([
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.location",
            "places.businessStatus",
            "places.types",
        ]),
    }
    body = {
        "textQuery": text_query,
        "maxResultCount": MAX_RESULTS,
        "locationRestriction": {"rectangle": {"low": low, "high": high}},
    }
    r = requests.post(TEXT_SEARCH_URL, headers=headers, json=body, timeout=25)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    return r.status_code, data, low, high


def places_details(api_key, place_id):
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join([
            "id","displayName","formattedAddress","location","businessStatus",
            "rating","userRatingCount","types","websiteUri","nationalPhoneNumber",
        ]),
    }
    r = requests.get(DETAILS_URL_BASE + place_id, headers=headers, timeout=25)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    return r.status_code, data


def main():
    if len(sys.argv) != 2:
        print("Usage: python places_test.py <Sr_No>", file=sys.stderr)
        sys.exit(1)

    sr_no = sys.argv[1].strip()
    api_key, csv_path = load_env()

    h = read_hospital_by_srno(csv_path, sr_no)
    if not h:
        print(f"ERROR: Sr_No {sr_no} not found in CSV.", file=sys.stderr)
        sys.exit(2)
    if h["Latitude"] is None or h["Longitude"] is None:
        print(f"ERROR: Sr_No {sr_no} has invalid Latitude/Longitude.", file=sys.stderr)
        sys.exit(3)

    ds_abbr = abbr(h["Hospital_Name"])
    print(f"\nHospital from CSV (Sr_No={h['Sr_No']}):")
    print(f"  Hospital_Name: {h['Hospital_Name']}")
    print(f"  Extracted abbr: {ds_abbr or '(none)'}")
    print(f"  Address: {h['Address']}")
    print(f"  Location/District/State/Pincode: {h['Location']} / {h['District']} / {h['State']} / {h['Pincode']}")
    print(f"  coords_valid: {h['coords_valid']}")
    print(f"  Dataset coords: {h['Latitude']}, {h['Longitude']}")
    print(f"  Search radius: {RADIUS_KM} km | Threshold: {THRESHOLD_KM} km | ABBR_MAX_KM: {ABBR_MAX_KM} km\n")

    query = build_query(h)
    print("[1/2] Text Search (New) with locationRestriction rectangle")
    print(f"  query: {query}\n")

    code, data, low, high = places_text_search(api_key, query, h["Latitude"], h["Longitude"], RADIUS_KM)
    if code != 200:
        print(f"Text Search failed (HTTP {code}):")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        sys.exit(4)

    places = data.get("places", [])
    if not places:
        print("No places returned. -> UNVERIFIED")
        sys.exit(0)

    print(f"Restricted viewport low={low} high={high}\n")

    scored = []
    for p in places:
        pid = p.get("id")
        pname = (p.get("displayName") or {}).get("text") or ""
        paddr = p.get("formattedAddress")
        bstat = p.get("businessStatus")
        loc = p.get("location") or {}
        plat = loc.get("latitude"); plng = loc.get("longitude")
        types = p.get("types") or []

        if pid is None or plat is None or plng is None:
            continue

        dist = haversine_km(h["Latitude"], h["Longitude"], float(plat), float(plng))
        if dist > THRESHOLD_KM:
            continue
        if not is_hospital_type(types):
            continue

        km = key_token_matches(h["Hospital_Name"], pname)

        cand_abbr = abbr(pname)
        abbr_ok = bool(ds_abbr) and (ds_abbr == cand_abbr) and (dist <= ABBR_MAX_KM)

        passes = (km >= MIN_KEY_TOKEN_MATCHES) or abbr_ok

        scored.append((passes, abbr_ok, km, dist, pid, pname, bstat, paddr, cand_abbr))

    if not scored:
        print("No hospital-type candidates within threshold. -> UNVERIFIED")
        sys.exit(0)

    # Sort: PASS first, then abbreviation-ok, then key-token matches, then distance
    scored.sort(key=lambda x: (not x[0], not x[1], -x[2], x[3]))

    print("Candidates (sorted by PASS, abbr_ok, key_matches, distance):")
    for i, (passes, abbr_ok, km, dist, pid, pname, bstat, paddr, cand_abbr) in enumerate(scored, start=1):
        tag = "PASS" if passes else "FAIL"
        ab = "ABBR" if abbr_ok else "-"
        print(f" {i}. [{tag} {ab}] key_matches={km} dist={dist:.2f}km abbr={cand_abbr or '(none)'} | {pname} | {bstat} | {paddr} | id={pid}")

    best = scored[0]
    passes, abbr_ok, km, dist, best_id, best_name, _, _, _ = best

    if not passes:
        print("\n⚠️ No confident match (name tokens/abbr). -> UNVERIFIED\n")
        sys.exit(0)

    print(f"\n[2/2] Place Details (New) for best match:\n  {best_id}\n")
    d_code, d_data = places_details(api_key, best_id)
    if d_code != 200:
        print(f"Place Details failed (HTTP {d_code}):")
        print(json.dumps(d_data, indent=2, ensure_ascii=False))
        sys.exit(5)

    r = d_data
    dname = (r.get("displayName") or {}).get("text")
    daddr = r.get("formattedAddress")
    status = r.get("businessStatus")
    rating = r.get("rating")
    count = r.get("userRatingCount")

    print("Place Details summary:")
    print(f"  name: {dname}")
    print(f"  address: {daddr}")
    print(f"  businessStatus: {status}")
    print(f"  rating: {rating} (count: {count})")
    print(f"  types: {r.get('types')}")
    print(f"  website: {r.get('websiteUri')}")
    print(f"  phone: {r.get('nationalPhoneNumber')}")
    print()

    if status == "CLOSED_PERMANENTLY":
        print("✅ Decision: EXCLUDE (CLOSED_PERMANENTLY)")
    elif status == "CLOSED_TEMPORARILY":
        print("⚠️ Decision: Usually exclude or warn (CLOSED_TEMPORARILY)")
    elif status == "OPERATIONAL":
        print("✅ Decision: OK (OPERATIONAL)")
    else:
        print("ℹ️ Decision: Unknown/missing status -> UNVERIFIED")


if __name__ == "__main__":
    main()
