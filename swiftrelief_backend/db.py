"""SQLite helpers (no ORM) to keep the stack simple + Android-friendly.

DB file: data/app.db

No ORM keeps dependencies low and makes it easy to later port the client to
Android (Android app calls the same REST endpoints).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


DB_PATH = os.getenv("APP_DB_PATH", os.path.join("data", "app.db"))


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = _dict_factory
    # Safer defaults for multi-request Flask usage
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                gender TEXT NOT NULL DEFAULT 'Other',
                age INTEGER,
                has_asthma_copd INTEGER NOT NULL DEFAULT 0,
                has_diabetes INTEGER NOT NULL DEFAULT 0,
                has_heart_disease INTEGER NOT NULL DEFAULT 0,
                has_stroke_history INTEGER NOT NULL DEFAULT 0,
                has_epilepsy INTEGER NOT NULL DEFAULT 0,
                is_pregnant INTEGER NOT NULL DEFAULT 0,
                other_info TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )

        # Lightweight migration for older DBs: add missing columns if needed.
        # (SQLite doesn't support IF NOT EXISTS for ADD COLUMN reliably across all versions.)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(users);").fetchall()}

        def _add_col(sql: str, name: str) -> None:
            if name not in cols:
                conn.execute(sql)
                cols.add(name)

        _add_col("ALTER TABLE users ADD COLUMN gender TEXT NOT NULL DEFAULT 'Other';", "gender")
        _add_col("ALTER TABLE users ADD COLUMN age INTEGER;", "age")
        _add_col("ALTER TABLE users ADD COLUMN has_asthma_copd INTEGER NOT NULL DEFAULT 0;", "has_asthma_copd")
        _add_col("ALTER TABLE users ADD COLUMN has_diabetes INTEGER NOT NULL DEFAULT 0;", "has_diabetes")
        _add_col("ALTER TABLE users ADD COLUMN has_heart_disease INTEGER NOT NULL DEFAULT 0;", "has_heart_disease")
        _add_col("ALTER TABLE users ADD COLUMN has_stroke_history INTEGER NOT NULL DEFAULT 0;", "has_stroke_history")
        _add_col("ALTER TABLE users ADD COLUMN has_epilepsy INTEGER NOT NULL DEFAULT 0;", "has_epilepsy")
        _add_col("ALTER TABLE users ADD COLUMN is_pregnant INTEGER NOT NULL DEFAULT 0;", "is_pregnant")
        _add_col("ALTER TABLE users ADD COLUMN other_info TEXT;", "other_info")

        # Cache for Places API (New) enrichment + closed-status filtering.
        # Stored in the same DB to keep deployment simple.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS places_cache (
                sr_no TEXT PRIMARY KEY,
                place_id TEXT,
                matched_name TEXT,
                matched_address TEXT,
                matched_lat REAL,
                matched_lng REAL,
                match_mode TEXT,
                match_distance_km REAL,
                business_status TEXT,
                rating REAL,
                user_rating_count INTEGER,
                phone TEXT,
                website TEXT,
                last_checked_at INTEGER
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_places_cache_last_checked ON places_cache(last_checked_at);")

        # -----------------------------
        # Model-ready logging tables
        # -----------------------------

        # Static master record for hospitals (dataset + newly discovered live hospitals).
        # IMPORTANT: do not store query-dependent features like distance/travel_time here.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hospitals (
                hospital_id INTEGER PRIMARY KEY AUTOINCREMENT,

                place_id TEXT,
                hospital_name TEXT NOT NULL,
                address TEXT,
                district TEXT,
                state TEXT,
                pincode TEXT,

                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                coords_valid INTEGER DEFAULT 1,

                is_alternate_medicine INTEGER DEFAULT 0,
                specialization_tags TEXT,

                source TEXT DEFAULT 'dataset',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            """
        )

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_hospitals_place_id
            ON hospitals(place_id)
            WHERE place_id IS NOT NULL AND place_id <> '';
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hospitals_lat_lng ON hospitals(latitude, longitude);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hospitals_name ON hospitals(hospital_name);")

        # One row per recommendation request.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                query_text TEXT,
                query_lat REAL NOT NULL,
                query_lng REAL NOT NULL,
                emergency_type TEXT NOT NULL,
                radius_km REAL,
                used_directions INTEGER DEFAULT 0,
                expanded_radius INTEGER DEFAULT 0,
                offline_mode INTEGER NOT NULL DEFAULT 0,
                feature_version TEXT NOT NULL DEFAULT 'v1',
                model_version TEXT NOT NULL DEFAULT 'baseline',
                health_profile_json TEXT,
                had_any_feedback INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON recommendation_runs(created_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_emergency ON recommendation_runs(emergency_type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_offline ON recommendation_runs(offline_mode);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_model_version ON recommendation_runs(model_version);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_had_feedback ON recommendation_runs(had_any_feedback);")

        # Lightweight migration: add newer columns if DB was created earlier.
        run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(recommendation_runs);").fetchall()}

        def _add_run_col(sql: str, name: str) -> None:
            if name not in run_cols:
                conn.execute(sql)
                run_cols.add(name)

        _add_run_col("ALTER TABLE recommendation_runs ADD COLUMN offline_mode INTEGER NOT NULL DEFAULT 0;", "offline_mode")
        _add_run_col("ALTER TABLE recommendation_runs ADD COLUMN feature_version TEXT NOT NULL DEFAULT 'v1';", "feature_version")
        _add_run_col("ALTER TABLE recommendation_runs ADD COLUMN model_version TEXT NOT NULL DEFAULT 'baseline';", "model_version")
        _add_run_col("ALTER TABLE recommendation_runs ADD COLUMN health_profile_json TEXT;", "health_profile_json")
        _add_run_col("ALTER TABLE recommendation_runs ADD COLUMN had_any_feedback INTEGER NOT NULL DEFAULT 0;", "had_any_feedback")

        # One row per hospital candidate per run: this is where the feature schema lives.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_candidates (
                run_candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                hospital_id INTEGER NOT NULL,

                distance_km REAL,
                travel_time_min REAL,
                rating REAL,
                user_ratings_total INTEGER,
                category_match_score REAL,
                confidence_score REAL,
                penalties REAL DEFAULT 0,
                final_score REAL,
                rank INTEGER,
                source TEXT,

                created_at TEXT DEFAULT (datetime('now')),

                FOREIGN KEY (run_id) REFERENCES recommendation_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_run ON run_candidates(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_hospital ON run_candidates(hospital_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_rank ON run_candidates(run_id, rank);")
        # Lightweight migration: add ML-related columns to run_candidates if missing.
        cand_cols = {row['name'] for row in conn.execute('PRAGMA table_info(run_candidates);').fetchall()}
        def _add_cand_col(sql: str, name: str) -> None:
            if name not in cand_cols:
                conn.execute(sql)
                cand_cols.add(name)
        _add_cand_col('ALTER TABLE run_candidates ADD COLUMN baseline_score REAL;', 'baseline_score')
        _add_cand_col('ALTER TABLE run_candidates ADD COLUMN model_p REAL;', 'model_p')
        _add_cand_col('ALTER TABLE run_candidates ADD COLUMN blended_score REAL;', 'blended_score')

        # -----------------------------
        # Feedback (thumbs + dropdown reason only)
        # -----------------------------
        # One feedback row per (run_id, hospital_id, user_id). Use user_id='' for anonymous.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,

                run_id INTEGER NOT NULL,
                hospital_id INTEGER NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',

                -- 1 = thumbs up, -1 = thumbs down
                thumbs INTEGER NOT NULL CHECK (thumbs IN (-1, 1)),

                -- Dropdown reason (code string)
                reason_code TEXT NOT NULL,

                -- Link feedback to the exact candidate row shown to the user
                run_candidate_id INTEGER,

                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),

                FOREIGN KEY (run_id) REFERENCES recommendation_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_one_per_hospital_run
            ON feedback(run_id, hospital_id, user_id);
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_run ON feedback(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_hospital ON feedback(hospital_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_thumbs ON feedback(thumbs);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_reason ON feedback(reason_code);")

        # Lightweight migration: link feedback to the exact candidate row
        fb_cols = {row["name"] for row in conn.execute("PRAGMA table_info(feedback);").fetchall()}
        if "run_candidate_id" not in fb_cols:
            conn.execute("ALTER TABLE feedback ADD COLUMN run_candidate_id INTEGER;")

        # Helpful join index for (run_id, hospital_id) -> run_candidate_id
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_run_candidate_id ON feedback(run_candidate_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_candidates_run_hospital ON run_candidates(run_id, hospital_id);")

        # Triggers: keep had_any_feedback correct without extra code.
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_feedback_insert_set_run_feedback
            AFTER INSERT ON feedback
            BEGIN
              UPDATE recommendation_runs
              SET had_any_feedback = 1
              WHERE run_id = NEW.run_id;
            END;
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_feedback_delete_recalc_run_feedback
            AFTER DELETE ON feedback
            BEGIN
              UPDATE recommendation_runs
              SET had_any_feedback =
                CASE
                  WHEN EXISTS (SELECT 1 FROM feedback f WHERE f.run_id = OLD.run_id) THEN 1
                  ELSE 0
                END
              WHERE run_id = OLD.run_id;
            END;
            """
        )


def upsert_feedback(
    *,
    run_id: int,
    hospital_id: int,
    user_id: str | None,
    run_candidate_id: int | None = None,
    thumbs: int,
    reason_code: str,
) -> None:
    """Insert or update feedback for a given (run_id, hospital_id, user_id)."""
    uid = (user_id or "").strip()
    with get_db() as conn:
        try:
            conn.execute(
                """
                INSERT INTO feedback(run_id, hospital_id, user_id, run_candidate_id, thumbs, reason_code)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(run_id, hospital_id, user_id)
                DO UPDATE SET run_candidate_id=COALESCE(excluded.run_candidate_id, feedback.run_candidate_id),
                             thumbs=excluded.thumbs,
                             reason_code=excluded.reason_code,
                             updated_at=datetime('now');
                """,
                (int(run_id), int(hospital_id), uid, int(run_candidate_id) if run_candidate_id is not None else None, int(thumbs), (reason_code or "").strip()),
            )
        except sqlite3.OperationalError:
            # If the user's DB was created earlier with a different UNIQUE index (e.g., expression-based),
            # fall back to REPLACE so we still enforce "one per hospital per run".
            conn.execute(
                """
                INSERT OR REPLACE INTO feedback(run_id, hospital_id, user_id, run_candidate_id, thumbs, reason_code, updated_at)
                VALUES(?,?,?,?,?,?,datetime('now'));
                """,
                (int(run_id), int(hospital_id), uid, int(run_candidate_id) if run_candidate_id is not None else None, int(thumbs), (reason_code or "").strip()),
            )


def upsert_hospital(
    *,
    place_id: str | None,
    hospital_name: str,
    address: str | None,
    latitude: float,
    longitude: float,
    source: str,
    coords_valid: bool = True,
    is_alternate_medicine: bool = False,
    specialization_tags: str | None = None,
    district: str | None = None,
    state: str | None = None,
    pincode: str | None = None,
) -> int:
    """Insert/update a hospital static record and return hospital_id.

    Uses place_id as the primary dedupe key when available; otherwise falls back to exact
    (name, lat, lng) match.
    """

    pid = (place_id or "").strip()
    name = (hospital_name or "").strip()
    if not name:
        raise ValueError("hospital_name is required")

    with get_db() as conn:
        if pid:
            row = conn.execute("SELECT hospital_id FROM hospitals WHERE place_id = ?", (pid,)).fetchone()
            if row:
                hid = int(row["hospital_id"])
                conn.execute(
                    """
                    UPDATE hospitals
                    SET hospital_name = hospital_name,
                        address = COALESCE(NULLIF(?, ''), address),
                        latitude = ?,
                        longitude = ?,
                        coords_valid = ?,
                        is_alternate_medicine = ?,
                        specialization_tags = CASE
                            WHEN COALESCE(NULLIF(?, ''), '') = '' THEN specialization_tags
                            WHEN lower(trim(COALESCE(NULLIF(?, ''), ''))) = 'general'
                                 AND COALESCE(lower(trim(specialization_tags)), '') NOT IN ('', 'general')
                            THEN specialization_tags
                            ELSE COALESCE(NULLIF(?, ''), specialization_tags)
                        END,
                        district = COALESCE(NULLIF(?, ''), district),
                        state = COALESCE(NULLIF(?, ''), state),
                        pincode = COALESCE(NULLIF(?, ''), pincode),
                        source = COALESCE(NULLIF(?, ''), source),
                        updated_at = datetime('now')
                    WHERE hospital_id = ?
                    """,
                    (
                        (address or "").strip(),
                        float(latitude),
                        float(longitude),
                        1 if coords_valid else 0,
                        1 if is_alternate_medicine else 0,
                        (specialization_tags or "").strip(),
                        (specialization_tags or "").strip(),
                        (specialization_tags or "").strip(),
                        (district or "").strip(),
                        (state or "").strip(),
                        (pincode or "").strip(),
                        (source or "").strip(),
                        hid,
                    ),
                )
                return hid

        # Fallback dedupe for hospitals without place_id
        row = conn.execute(
            """
            SELECT hospital_id FROM hospitals
            WHERE lower(hospital_name) = lower(?) AND latitude = ? AND longitude = ?
            LIMIT 1
            """,
            (name, float(latitude), float(longitude)),
        ).fetchone()
        if row:
            return int(row["hospital_id"])

        cur = conn.execute(
            """
            INSERT INTO hospitals(
                place_id, hospital_name, address, district, state, pincode,
                latitude, longitude, coords_valid, is_alternate_medicine,
                specialization_tags, source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                pid or None,
                name,
                (address or "").strip() or None,
                (district or "").strip() or None,
                (state or "").strip() or None,
                (pincode or "").strip() or None,
                float(latitude),
                float(longitude),
                1 if coords_valid else 0,
                1 if is_alternate_medicine else 0,
                (specialization_tags or "").strip() or None,
                (source or "dataset").strip() or "dataset",
            ),
        )
        return int(cur.lastrowid)


def get_hospital_specialization_tags_by_place_id(place_id: str | None) -> Optional[str]:
    """Return stored specialization_tags for a place_id, if present."""
    pid = (place_id or "").strip()
    if not pid:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT specialization_tags FROM hospitals WHERE place_id = ? LIMIT 1",
            (pid,),
        ).fetchone()
        if not row:
            return None
        val = row.get("specialization_tags")
        if val is None:
            return None
        s = str(val).strip()
        return s or None


def get_hospital_name_by_place_id(place_id: str | None) -> Optional[str]:
    """Return stored hospital_name for a place_id, if present."""
    pid = (place_id or "").strip()
    if not pid:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT hospital_name FROM hospitals WHERE place_id = ? LIMIT 1",
            (pid,),
        ).fetchone()
        if not row:
            return None
        val = row.get("hospital_name")
        if val is None:
            return None
        s = str(val).strip()
        return s or None


def create_recommendation_run(
    *,
    user_id: str | None,
    query_text: str | None,
    query_lat: float,
    query_lng: float,
    emergency_type: str,
    radius_km: float | None,
    used_directions: bool,
    expanded_radius: bool,
    offline_mode: bool = False,
    feature_version: str = 'v1',
    model_version: str = 'baseline',
    health_profile_json: str | None = None,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO recommendation_runs(
                user_id, query_text, query_lat, query_lng, emergency_type,
                radius_km, used_directions, expanded_radius,
                offline_mode, feature_version, model_version, health_profile_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                (user_id or "").strip() or None,
                (query_text or "").strip() or None,
                float(query_lat),
                float(query_lng),
                (emergency_type or "").strip(),
                float(radius_km) if radius_km is not None else None,
                1 if used_directions else 0,
                1 if expanded_radius else 0,
                1 if offline_mode else 0,
                (feature_version or 'v1').strip() or 'v1',
                (model_version or 'baseline').strip() or 'baseline',
                health_profile_json,
            ),
        )
        return int(cur.lastrowid)


def insert_run_candidates(run_id: int, rows: list[dict[str, Any]]) -> list[int]:
    """Insert candidate rows and return their run_candidate_id values.

    This function is schema-aware: if ML columns (baseline_score/model_p/blended_score) exist,
    they will be stored too. This keeps older DB files compatible.
    """
    if not rows:
        return []
    ids: list[int] = []
    with get_db() as conn:
        cols = {row['name'] for row in conn.execute('PRAGMA table_info(run_candidates);').fetchall()}

        base_cols = [
            'run_id','hospital_id','distance_km','travel_time_min','rating','user_ratings_total',
            'category_match_score','confidence_score','penalties','final_score','rank','source'
        ]
        extra_cols = []
        if 'baseline_score' in cols: extra_cols.append('baseline_score')
        if 'model_p' in cols: extra_cols.append('model_p')
        if 'blended_score' in cols: extra_cols.append('blended_score')

        all_cols = base_cols + extra_cols
        placeholders = ','.join(['?'] * len(all_cols))
        col_sql = ', '.join(all_cols)

        for r in rows:
            values = [
                int(run_id),
                int(r['hospital_id']),
                r.get('distance_km'),
                r.get('travel_time_min'),
                r.get('rating'),
                r.get('user_ratings_total'),
                r.get('category_match_score'),
                r.get('confidence_score'),
                r.get('penalties'),
                r.get('final_score'),
                r.get('rank'),
                r.get('source'),
            ]
            if 'baseline_score' in cols: values.append(r.get('baseline_score'))
            if 'model_p' in cols: values.append(r.get('model_p'))
            if 'blended_score' in cols: values.append(r.get('blended_score'))

            cur = conn.execute(
                f"INSERT INTO run_candidates({col_sql}) VALUES ({placeholders})",
                tuple(values),
            )
            ids.append(int(cur.lastrowid))
    return ids
def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        return conn.execute(
            """
            SELECT id, name, email, password_hash,
                   gender, age, has_asthma_copd, has_diabetes, has_heart_disease, has_stroke_history,
                   has_epilepsy, is_pregnant, other_info,
                   created_at
            FROM users WHERE email = ?
            """,
            (email.strip().lower(),),
        ).fetchone()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        return conn.execute(
            """
            SELECT id, name, email, password_hash,
                   gender, age, has_asthma_copd, has_diabetes, has_heart_disease, has_stroke_history,
                   has_epilepsy, is_pregnant, other_info,
                   created_at
            FROM users WHERE id = ?
            """,
            (int(user_id),),
        ).fetchone()


def update_user_profile(
    *,
    user_id: int,
    name: str | None,
    gender: str,
    age: int,
    has_asthma_copd: bool = False,
    has_diabetes: bool = False,
    has_heart_disease: bool = False,
    has_stroke_history: bool = False,
    has_epilepsy: bool = False,
    is_pregnant: bool = False,
    other_info: str | None = None,
) -> None:
    """Update the user's editable profile fields.

    Note: Past recommendation runs must not change. We handle that by storing a
    health_profile_json snapshot in each recommendation run at creation time.
    """
    with get_db() as conn:
        conn.execute(
            """
            UPDATE users
            SET name = COALESCE(NULLIF(?, ''), name),
                gender = COALESCE(NULLIF(?, ''), gender),
                age = ?,
                has_asthma_copd = ?,
                has_diabetes = ?,
                has_heart_disease = ?,
                has_stroke_history = ?,
                has_epilepsy = ?,
                is_pregnant = ?,
                other_info = NULLIF(?, '')
            WHERE id = ?
            """,
            (
                (name or '').strip() or None,
                (gender or 'Other').strip() or 'Other',
                int(age),
                1 if has_asthma_copd else 0,
                1 if has_diabetes else 0,
                1 if has_heart_disease else 0,
                1 if has_stroke_history else 0,
                1 if has_epilepsy else 0,
                1 if is_pregnant else 0,
                (other_info or '').strip() or None,
                int(user_id),
            ),
        )


def create_user(
    *,
    name: str,
    email: str,
    password_hash: str,
    gender: str = "Other",
    age: int | None = None,
    has_asthma_copd: bool = False,
    has_diabetes: bool = False,
    has_heart_disease: bool = False,
    has_stroke_history: bool = False,
    has_epilepsy: bool = False,
    is_pregnant: bool = False,
    other_info: str | None = None,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO users(
                name, email, password_hash,
                gender,
                age,
                has_asthma_copd, has_diabetes, has_heart_disease, has_stroke_history,
                has_epilepsy, is_pregnant,
                other_info
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                name.strip(),
                email.strip().lower(),
                password_hash,
                (gender or "Other").strip() or "Other",
                int(age) if age is not None else None,
                1 if has_asthma_copd else 0,
                1 if has_diabetes else 0,
                1 if has_heart_disease else 0,
                1 if has_stroke_history else 0,
                1 if has_epilepsy else 0,
                1 if is_pregnant else 0,
                (other_info or "").strip() or None,
            ),
        )
        return int(cur.lastrowid)


def get_user_medical_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Return only the fields needed for rule-based medical weighting."""
    with get_db() as conn:
        return conn.execute(
            """
            SELECT id, gender, age,
                   has_asthma_copd, has_diabetes, has_heart_disease, has_stroke_history,
                   has_epilepsy, is_pregnant
            FROM users WHERE id = ?
            """,
            (int(user_id),),
        ).fetchone()