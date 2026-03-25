from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from werkzeug.security import check_password_hash, generate_password_hash
import logging

from db import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    init_db,
    update_user_profile,
    upsert_feedback,
)
from model.recommend_core import EMERGENCY_TO_DEPT, infer_emergency_type_from_symptom, normalize_emergency_type
import requests
import glob
import json
try:
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
except Exception:
    google = None
from services.geocode_service import geocode_place
from services.recommender_service import recommend
from model.offline_places import lookup_offline_place, learn_place


def create_app() -> Flask:
    # Load .env if present
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # fall back

    app = Flask(__name__)

    # configure logging for debug of symptom mapping: ensure handlers write to stdout
    import sys

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s"))
    # Allow controlling verbosity via env (DEBUG/INFO/WARNING)
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    if env_level == "DEBUG":
        handler.setLevel(logging.DEBUG)
        root_level = logging.DEBUG
    elif env_level == "WARNING":
        handler.setLevel(logging.WARNING)
        root_level = logging.WARNING
    else:
        handler.setLevel(logging.INFO)
        root_level = logging.INFO

    # attach handler to the Flask app logger if not already present
    if not any(isinstance(h, logging.StreamHandler) for h in app.logger.handlers):
        app.logger.addHandler(handler)
    app.logger.setLevel(root_level)

    # also ensure the root and werkzeug loggers print to stdout so logs show in dev server
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)
    root_logger.setLevel(root_level)
    # Ensure our key modules also honor this level so their debug/info logs appear
    logging.getLogger("services.recommender_service").setLevel(root_level)
    logging.getLogger("model.recommend_core").setLevel(root_level)
    logging.getLogger("werkzeug").setLevel(logging.INFO)

    # Basic config (override via .env)
    app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "dev-change-me")
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = int(os.getenv("JWT_EXPIRE_SECONDS", "86400"))  # 1 day

    JWTManager(app)
    init_db()

    # Canonical dropdown reason codes (keep frontend + backend consistent)
    POS_REASON_CODES = {
        "closest",
        "matches_need",
        "trusted",
        "affordable",
    }

    NEG_REASON_CODES = {
        "too_far_or_slow",
        # Prefer the newer, user-friendly label/code. Keep legacy code for older rows/clients.
        "not_relevant",
        "wrong_department",
        "lacks_facilities",
        "low_rating",
        "seems_invalid_or_duplicate",
        "private_or_expensive",
    }

    ALLOWED_REASON_CODES = POS_REASON_CODES | NEG_REASON_CODES

    # Canonical mapping for merged/legacy positive reason codes.
    # - closest + fastest -> closest
    # - trusted + good_rating -> trusted
    REASON_CODE_CANONICAL: Dict[str, str] = {
        "fastest": "closest",
        "fastest_to_reach": "closest",
        "fastest to reach": "closest",
        "good_rating": "trusted",
        "good rating": "trusted",
    }

    # -----------------
    # Pages (server-rendered)
    # -----------------
    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/login")
    def login_page():
        return render_template("login.html")

    @app.get("/register")
    def register_page():
        return render_template("register.html")

    @app.get("/dashboard")
    def dashboard_page():
        return render_template("dashboard.html")

    @app.get("/recommend")
    def recommend_page():
        return render_template("recommend.html")

    @app.get("/profile")
    def profile_page():
        return render_template("profile.html")

    @app.get("/sw.js")
    def service_worker():
        return send_from_directory(app.static_folder, "sw.js")

    # -----------------
    # API: Auth
    # -----------------
    @app.post("/api/auth/register")
    def api_register():
        data = request.get_json(force=True, silent=True) or {}
        name = str(data.get("name", "")).strip()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))

        gender_raw = str(data.get("gender", "Other") or "Other").strip()
        gender_norm = gender_raw.lower()
        gender = "Other"
        if gender_norm in {"male", "m"}:
            gender = "Male"
        elif gender_norm in {"female", "f", "woman", "women"}:
            gender = "Female"
        elif gender_norm in {"other", "o", "non-binary", "nonbinary"}:
            gender = "Other"
        else:
            # keep safe default if unexpected input
            gender = "Other"

        # Optional structured medical fields
        age_raw = data.get("age")
        conditions = data.get("conditions") or {}
        other_info = str(data.get("other_info", "") or "").strip()

        # Age is compulsory (health profile baseline)
        if age_raw in (None, ""):
            return jsonify({"success": False, "message": "Age is required"}), 400
        try:
            age = int(age_raw)
        except Exception:
            return jsonify({"success": False, "message": "Age must be a number"}), 400
        if age < 0 or age > 120:
            return jsonify({"success": False, "message": "Age must be between 0 and 120"}), 400

        if not email or "@" not in email:
            return jsonify({"success": False, "message": "Valid email required"}), 400
        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400

        if get_user_by_email(email):
            return jsonify({"success": False, "message": "Email already registered"}), 409

        pw_hash = generate_password_hash(password)

        def _b(key: str) -> bool:
            try:
                return bool(conditions.get(key))
            except Exception:
                return False

        user_id = create_user(
            name=name or "User",
            email=email,
            password_hash=pw_hash,
            gender=gender,
            age=age,
            has_asthma_copd=_b("asthma_copd"),
            has_diabetes=_b("diabetes"),
            has_heart_disease=_b("heart_disease"),
            has_stroke_history=_b("stroke_history"),
            has_epilepsy=_b("epilepsy"),
            is_pregnant=_b("pregnant"),
            other_info=other_info or None,
        )

        token = create_access_token(identity=str(user_id))
        return jsonify({"success": True, "token": token, "user": {"id": user_id, "name": name or "User", "email": email}})

    @app.post("/api/auth/login")
    def api_login():
        data = request.get_json(force=True, silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))

        user = get_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"success": False, "message": "Invalid email or password"}), 401

        token = create_access_token(identity=str(user["id"]))
        return jsonify({"success": True, "token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"]}})

    @app.get("/api/auth/me")
    @jwt_required()
    def api_me():
        user_id = int(get_jwt_identity())
        user = get_user_by_id(user_id)
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        return jsonify(
            {
                "success": True,
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                },
            }
        )

    # -----------------
    # API: Profile (view + edit)
    # -----------------
    @app.get("/api/profile")
    @jwt_required()
    def api_profile_get():
        user_id = int(get_jwt_identity())
        user = get_user_by_id(user_id)
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404

        profile = {
            "id": user["id"],
            "name": user.get("name") or "User",
            "email": user.get("email"),
            "gender": user.get("gender") or "Other",
            "age": user.get("age"),
            "conditions": {
                "asthma_copd": int(user.get("has_asthma_copd") or 0),
                "diabetes": int(user.get("has_diabetes") or 0),
                "heart_disease": int(user.get("has_heart_disease") or 0),
                "stroke_history": int(user.get("has_stroke_history") or 0),
                "epilepsy": int(user.get("has_epilepsy") or 0),
                "pregnant": int(user.get("is_pregnant") or 0),
            },
            "other_info": user.get("other_info") or "",
            "created_at": user.get("created_at"),
        }
        return jsonify({"success": True, "profile": profile})

    @app.put("/api/profile")
    @jwt_required()
    def api_profile_put():
        user_id = int(get_jwt_identity())
        data = request.get_json(force=True, silent=True) or {}

        name = str(data.get("name", "")).strip()
        gender_raw = str(data.get("gender", "") or "").strip()
        age_raw = data.get("age")
        conditions = data.get("conditions") or {}
        other_info = str(data.get("other_info", "") or "").strip()

        # Enforce compulsory age + gender
        if age_raw in (None, ""):
            return jsonify({"success": False, "message": "Age is required"}), 400
        try:
            age = int(age_raw)
        except Exception:
            return jsonify({"success": False, "message": "Age must be a number"}), 400
        if age < 0 or age > 120:
            return jsonify({"success": False, "message": "Age must be between 0 and 120"}), 400

        if not gender_raw:
            return jsonify({"success": False, "message": "Gender is required"}), 400
        gender_norm = gender_raw.lower()
        if gender_norm in {"male", "m"}:
            gender = "Male"
        elif gender_norm in {"female", "f", "woman", "women"}:
            gender = "Female"
        else:
            gender = "Other"

        def _b(key: str) -> bool:
            try:
                return bool(conditions.get(key))
            except Exception:
                return False

        update_user_profile(
            user_id=user_id,
            name=name or None,
            gender=gender,
            age=age,
            has_asthma_copd=_b("asthma_copd"),
            has_diabetes=_b("diabetes"),
            has_heart_disease=_b("heart_disease"),
            has_stroke_history=_b("stroke_history"),
            has_epilepsy=_b("epilepsy"),
            is_pregnant=_b("pregnant"),
            other_info=other_info or None,
        )
        return jsonify({"success": True})

    # -----------------
    # API: Geocode place -> coords
    # -----------------
    @app.get("/api/geocode")
    @jwt_required(optional=True)
    def api_geocode():
        q = request.args.get("query", "")
        region = request.args.get("region", "in")
        offline = str(request.args.get("offline", "0")).strip().lower() in {"1", "true", "yes", "on"}

        # Offline mode: only resolve from local place lists (no external calls)
        if offline:
            hit = lookup_offline_place(q)
            if not hit:
                return jsonify({"success": False, "message": "Place not available offline"}), 404
            return jsonify({"success": True, "data": hit})

        try:
            res = geocode_place(q, region=region)
        except Exception as e:
            return jsonify({"success": False, "message": f"Geocode failed: {e}"}), 500
        if not res:
            return jsonify({"success": False, "message": "No results"}), 404

        # Online learning: store the resolved place so it can be used offline later.
        try:
            learn_place(q, float(res.get("lat")), float(res.get("lon")))
        except Exception:
            pass
        return jsonify({"success": True, "data": res})

    # -----------------
    # API: Recommend hospitals
    # -----------------
    @app.post("/api/recommend")
    @jwt_required(optional=True)
    def api_recommend():
        payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        if payload.get("lat") is None or payload.get("lon") is None:
            return jsonify({"success": False, "message": "lat and lon are required"}), 400

        # Attach user_id if logged in so we can log model-training data per account.
        uid = get_jwt_identity()
        if uid is not None:
            payload["_user_id"] = str(uid)
        try:
            results = recommend(payload)
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
        return jsonify({"success": True, "data": results})

    # -----------------
    # API: Feedback (thumbs + dropdown reason)
    # -----------------
    @app.post("/api/feedback")
    @jwt_required(optional=True)
    def api_feedback():
        data = request.get_json(force=True, silent=True) or {}
        run_id = data.get("run_id")
        hospital_id = data.get("hospital_id")
        run_candidate_id = data.get("run_candidate_id")
        thumbs = data.get("thumbs")
        reason_code = str(data.get("reason_code") or "").strip()
        reason_code = REASON_CODE_CANONICAL.get(reason_code, reason_code)

        if run_id is None or hospital_id is None:
            return jsonify({"success": False, "message": "run_id and hospital_id are required"}), 400
        try:
            run_id_int = int(run_id)
            hospital_id_int = int(hospital_id)
        except Exception:
            return jsonify({"success": False, "message": "run_id and hospital_id must be integers"}), 400

        run_candidate_id_int: int | None = None
        if run_candidate_id not in (None, ""):
            try:
                run_candidate_id_int = int(run_candidate_id)
            except Exception:
                return jsonify({"success": False, "message": "run_candidate_id must be an integer"}), 400

        try:
            thumbs_int = int(thumbs)
        except Exception:
            return jsonify({"success": False, "message": "thumbs must be 1 (up) or -1 (down)"}), 400
        if thumbs_int not in (1, -1):
            return jsonify({"success": False, "message": "thumbs must be 1 (up) or -1 (down)"}), 400

        if reason_code not in ALLOWED_REASON_CODES:
            return jsonify({"success": False, "message": "Invalid reason_code"}), 400

        # Enforce correct reason set for thumbs direction
        if thumbs_int == 1 and reason_code not in POS_REASON_CODES:
            return jsonify({"success": False, "message": "Invalid reason_code for thumbs up"}), 400
        if thumbs_int == -1 and reason_code not in NEG_REASON_CODES:
            return jsonify({"success": False, "message": "Invalid reason_code for thumbs down"}), 400

        uid = get_jwt_identity()
        upsert_feedback(
            run_id=run_id_int,
            hospital_id=hospital_id_int,
            user_id=str(uid) if uid is not None else "",
            run_candidate_id=run_candidate_id_int,
            thumbs=thumbs_int,
            reason_code=reason_code,
        )
        return jsonify({"success": True})

    # Health check
    @app.get("/api/health")
    def api_health():
        return jsonify({"success": True, "message": "ok"})
    def _ensure_adc_from_keyfile():
        """If `GOOGLE_APPLICATION_CREDENTIALS` isn't set, try to find a tokyo-eye-*.json key and set it."""
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return
        keys = glob.glob(str(Path(__file__).parent / "tokyo-eye-*.json"))
        if keys:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keys[0]


    def _get_vertex_token() -> str:
        """Return an access token using Application Default Credentials.

        Requires `google-auth` to be available and ADC to be configured (or
        `GOOGLE_APPLICATION_CREDENTIALS` pointing at the service account JSON).
        """
        if 'google' not in globals() or google is None:
            raise RuntimeError("google-auth library not available; cannot authenticate to Vertex")
        _ensure_adc_from_keyfile()
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(GoogleAuthRequest())
        if not creds.token:
            raise RuntimeError("Failed to obtain access token from ADC")
        return creds.token


    def _call_gemini(prompt: str) -> str:
        """Call Vertex/Gemini if configured (preferred), otherwise fall back to API-key route.

        Requires these environment variables for Vertex path:
          - VERTEX_PROJECT
          - VERTEX_LOCATION (default: us-central1)
          - VERTEX_MODEL (model resource name, e.g. "models/gemini-1.5-mini")
        """
        # Prefer Vertex service-account auth when project+model are provided.
        project = os.getenv("VERTEX_PROJECT")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        model = os.getenv("VERTEX_MODEL")

        # If project not provided, try to extract from service account keyfile.
        if not project:
            keyfile = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if keyfile and os.path.exists(keyfile):
                try:
                    with open(keyfile, "r", encoding="utf-8") as fh:
                        kd = json.load(fh)
                        project = kd.get("project_id")
                except Exception:
                    project = None

        if project and model:
            # Use ADC to get a Bearer token
            token = _get_vertex_token()
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/models/{model}:predict"
            payload = {"instances": [{"input": prompt}], "parameters": {"maxOutputTokens": 256, "temperature": 0.0}}
            try:
                app.logger.debug("Vertex predict url=%s payload=%s", url, json.dumps(payload)[:1000])
                r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {token}"}, timeout=15)
            except Exception as e:
                raise RuntimeError(f"Vertex request failed: {e}")
            try:
                j = r.json()
                app.logger.debug("Vertex raw response: %s", json.dumps(j)[:2000])
            except Exception:
                raise RuntimeError(f"Vertex returned non-json response: {r.text[:200]}")

            # Parse common Vertex predict shapes
            # 1) {"predictions": [ {"content": [{"text":"..."}, ...] } ] }
            preds = j.get("predictions")
            if isinstance(preds, list) and preds:
                first = preds[0]
                if isinstance(first, dict):
                    contents = first.get("content") or first.get("contents")
                    if isinstance(contents, list) and contents:
                        # Find first text field
                        for item in contents:
                            if isinstance(item, dict) and item.get("text"):
                                return str(item.get("text"))
                    # Direct field
                    for key in ("text", "output", "content"):
                        if first.get(key):
                            return str(first.get(key))

            # 2) Some models return 'candidates'
            cands = j.get("candidates")
            if isinstance(cands, list) and cands:
                first = cands[0]
                if isinstance(first, dict) and first.get("content"):
                    return str(first.get("content"))

            # Last resort: return entire json as string
            return str(j)

        # Fallback: legacy Generative Language API via API key (if present)
        api_key = os.getenv("GCP_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Vertex not configured and no API key available for fallback")

        url = f"https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key={api_key}"
        body = {"prompt": {"text": prompt}, "temperature": 0.0, "maxOutputTokens": 256}
        try:
            r = requests.post(url, json=body, timeout=10)
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}")
        try:
            j = r.json()
        except Exception:
            raise RuntimeError(f"LLM returned non-json response: {r.text[:200]}")

        cands = j.get("candidates") or j.get("responses") or j.get("output")
        if isinstance(cands, list) and cands:
            first = cands[0]
            for key in ("content", "text", "output"):
                if isinstance(first, dict) and first.get(key):
                    return str(first.get(key))
        out = j.get("output")
        if isinstance(out, dict):
            contents = out.get("contents")
            if isinstance(contents, list) and contents:
                t = contents[0].get("text")
                if t:
                    return str(t)
        if j.get("text"):
            return str(j.get("text"))
        return str(j)

    @app.post("/api/map_symptom")
    def api_map_symptom():
        """Map free-text symptom to one of the canonical emergency categories.

        Request JSON: {"symptom": "...", "use_llm": true}
        Response: {"success": True, "category": "...", "source": "llm"|"rule", "raw": "..."}
        """
        data = request.get_json(force=True, silent=True) or {}
        symptom = str(data.get("symptom", "") or "").strip()
        use_llm = bool(data.get("use_llm", True))

        if not symptom:
            return jsonify({"success": False, "message": "symptom text is required"}), 400

        allowed = list(EMERGENCY_TO_DEPT.keys())

        app.logger.info("map_symptom request: symptom=%s use_llm=%s", symptom, use_llm)

        def _match_to_allowed(candidate: str) -> str | None:
            if not candidate:
                return None
            cand = str(candidate).strip()
            # Try exact match first
            for c in allowed:
                if cand.lower() == c.lower():
                    return c
            # Try normalized match
            cand_norm = normalize_emergency_type(cand)
            for c in allowed:
                if cand_norm == normalize_emergency_type(c):
                    return c
            # If candidate contains an allowed label as substring, prefer exact label
            for c in allowed:
                if normalize_emergency_type(c) in cand_norm:
                    return c
            return None

        # 1) Deterministic rule-based mapping first. Treat 'general' as "no confident match".
        try:
            mapped_label, source = infer_emergency_type_from_symptom(symptom)
            app.logger.info("map_symptom inferred: symptom=%s -> %s (source=%s)", symptom, mapped_label, source)
            print(f"[Mapping] symptom '{symptom}' -> {mapped_label} (source={source})")
            return jsonify({"success": True, "category": mapped_label, "source": source, "raw": symptom})
        except Exception as e:
            app.logger.warning("map_symptom inference error: %s", e)
            return jsonify({"success": True, "category": "general", "source": "fallback", "raw": "general"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "5000")), debug=True)
