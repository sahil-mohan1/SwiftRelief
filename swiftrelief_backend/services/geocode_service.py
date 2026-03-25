from __future__ import annotations

import os
from typing import Dict, Optional

import requests


def geocode_place(query: str, region: str = "in") -> Optional[Dict]:
    """Geocode a free-text place name into coordinates.

    Strategy:
    1) If GOOGLE_MAPS_API_KEY is available, use Google Geocoding (most reliable).
    2) Else fallback to OpenStreetMap Nominatim (no key; good for demo).

    Returns:
      { "lat": float, "lon": float, "display_name": str, "provider": str }
    """

    q = (query or "").strip()
    if not q:
        return None

    google_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if google_key:
        # Lazy import so the project still runs without googlemaps installed (but it's in requirements).
        import googlemaps

        gmaps = googlemaps.Client(key=google_key)
        results = gmaps.geocode(q, region=region)
        if not results:
            return None
        best = results[0]
        loc = (best.get("geometry") or {}).get("location") or {}
        lat = loc.get("lat")
        lon = loc.get("lng")
        if lat is None or lon is None:
            return None
        return {
            "lat": float(lat),
            "lon": float(lon),
            "display_name": best.get("formatted_address") or q,
            "provider": "google",
        }

    # Fallback: Nominatim
    # Note: For production you should set a proper User-Agent and consider rate limits.
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1, "countrycodes": region}
    headers = {"User-Agent": "hospital-recommender-demo/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    best = data[0]
    return {
        "lat": float(best["lat"]),
        "lon": float(best["lon"]),
        "display_name": best.get("display_name") or q,
        "provider": "nominatim",
    }
