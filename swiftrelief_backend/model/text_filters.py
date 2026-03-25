"""model/text_filters.py

Centralized text-based filters used across dataset + live Places results.

Right now this focuses on excluding alternative medicine facilities (Ayurveda,
Homeopathy, Siddha, Unani, etc.) from being recommended.
"""

from __future__ import annotations

import re
import unicodedata


# India-focused alternative medicine keywords.
# Notes:
# - Keep patterns fairly specific to avoid accidental false positives.
# - "arya vaidya" is explicitly included as requested.
ALT_MED_PATTERNS: list[str] = [
    r"\bayurved(a|ic)\b",
    r"\bhomeo(pathy|eopathy|pathic)\b",
    r"\bhomoeo(pathy|pathic)\b",
    r"\bsiddha\b",
    r"\bunani\b",
    r"\bnaturopathy\b",
    r"\byoga(\s+therapy)?\b",
    r"\bpanchakarma\b",
    r"\bherbal\b",
    r"\balternat(e|ive)\s+medicine\b",

    # Common brand/style terms that often indicate Ayurveda chains/clinics
    r"\barya\s*vaidya\b",      # e.g., "Arya Vaidya"
    r"\baryavaidya\b",
    r"\bvaidya\s*sala\b",     # e.g., "Vaidya Sala"
]

ALT_MED_RE = re.compile("|".join(ALT_MED_PATTERNS), re.IGNORECASE)

ALT_TAGS: set[str] = {
    "alt",
    "alternative",
    "alternative medicine",
    "ayurveda",
    "ayurvedic",
    "homeopathy",
    "homeopathic",
    "homoeopathy",
    "homoeopathic",
    "siddha",
    "unani",
    "naturopathy",
    "yoga therapy",
    "panchakarma",
    "herbal",
}


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("’", "'")
    return normalized.lower()


def is_alt_medicine_hospital(name: str | None = None, address: str | None = None) -> bool:
    """Return True if the name/address suggests an alternative medicine facility."""
    text = _normalize_text(f"{name or ''} {address or ''}".strip())
    if not text:
        return False
    return bool(ALT_MED_RE.search(text))


def is_alt_tagged(tags: list[str] | None = None) -> bool:
    """Return True when specialization tags indicate alternative medicine."""
    if not tags:
        return False
    for t in tags:
        token = _normalize_text(str(t or "").strip())
        if token in ALT_TAGS:
            return True
    return False
