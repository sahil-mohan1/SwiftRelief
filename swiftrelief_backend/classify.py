from google import genai
from google.genai import types
import json
import re
import logging

try:
    # Prefer canonical allowed labels from the core model mapping when available
    from model.recommend_core import EMERGENCY_TO_DEPT
    ALLOWED_CATEGORIES = list(EMERGENCY_TO_DEPT.keys())
except Exception:
    # Fallback list (keeps labels stable if import isn't available)
    ALLOWED_CATEGORIES = [
        "All (No filter)",
        "Emergency / Trauma",
        "Cardiology (Heart)",
        "Neurology (Brain / Stroke)",
        "Orthopedics (Bones / Fracture)",
        "Ophthalmology (Eye)",
        "Gynecology (Maternity / Women's Health)",
        "Pediatrics (Child Care)",
        "ENT",
        "general",
    ]

client = genai.Client(vertexai=True, project='tokyo-eye-485312-v5', location='global')


def _normalize_label(s: str) -> str:
    return (str(s or "") or "").strip().lower()


def _match_to_allowed(candidate: str) -> str | None:
    if not candidate:
        return None
    cand = str(candidate).strip()
    # exact-match (case-insensitive)
    for c in ALLOWED_CATEGORIES:
        if cand.lower() == c.lower():
            return c
    # normalized match
    cand_norm = _normalize_label(cand)
    for c in ALLOWED_CATEGORIES:
        if cand_norm == _normalize_label(c):
            return c
    # substring match: prefer allowed label
    for c in ALLOWED_CATEGORIES:
        if _normalize_label(c) in cand_norm:
            return c
    return None


def _extract_json_from_text(text: str) -> dict | None:
    if not text:
        return None
    text = str(text).strip()
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find a JSON object substring
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def classify_for_swiftrelief(symptom: str) -> dict:
    """Map free-text `symptom` to an exact allowed category.

    Returns a dict: {"department": <exact label from ALLOWED_CATEGORIES>, "raw": <llm raw text>}.
    Falls back to `general` when uncertain.
    """
    instr = (
        "You must reply with a JSON object containing only the key 'department'.\n"
        "'department' MUST be exactly one of the following labels (no additions, no renaming).\n"
    )
    instr += "AVAILABLE_CATEGORIES:\n"
    for c in ALLOWED_CATEGORIES:
        instr += f"- {c}\n"
    instr += (
        "If unsure, reply with 'general' as the department. Only output valid JSON."
    )

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(
                system_instruction=instr,
                response_mime_type="application/json",
            ),
            contents=symptom,
        )
        text = getattr(response, "text", str(response))
    except Exception as e:
        logging.warning("LLM call failed in classify_for_swiftrelief: %s", e)
        text = ""

    parsed = _extract_json_from_text(text) or {}

    # Candidate department may be in various keys
    cand = parsed.get("department") or parsed.get("category") or parsed.get("label") or text

    mapped = _match_to_allowed(cand) or "general"

    return {"department": mapped, "raw": text}
