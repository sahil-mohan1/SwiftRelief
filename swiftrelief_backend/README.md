# Hospital Recommender (Flask + SQLite)

## What is implemented
- Flask web UI (templates) + REST APIs
- User Register/Login using JWT
- Emergency recommendation page:
  - Use live GPS location (browser geolocation)
  - OR enter a place name and geocode to coordinates
  - Calls `/api/recommend` to display recommended hospitals

## Run locally

1) Create venv (optional)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

2) Install deps
```bash
pip install -r requirements.txt
```

3) Optional: set environment variables
Create `.env` in project root:
```ini
# JWT
JWT_SECRET_KEY=change-me
JWT_EXPIRE_SECONDS=86400

# If you have Google key (improves geocoding + hospital geocode refining)
GOOGLE_MAPS_API_KEY=...

# Dataset
HOSPITALS_CSV=data/hospitals_demo_valid_coords.csv
```

4) Start server
```bash
python app.py
```
Open: http://127.0.0.1:5000

## Android-friendly notes
- Keep using the same REST endpoints:
  - `POST /api/auth/register`
  - `POST /api/auth/login`
  - `POST /api/recommend`
  - `GET /api/geocode?query=...`
- Android app can store JWT and send `Authorization: Bearer <token>`.
- UI styling is intentionally minimal; teammate can replace CSS/HTML.

## Offline cache behavior
- Nearby live hospital results are persisted in `cache/nearby_live_cache.json` after online searches.
- Offline mode reuses cached results for compatible users using:
  - same emergency/disease bucket (normalized), and
  - nearby location match (default within 12 km).
- Tunable environment variables:
  - `NEARBY_LIVE_CACHE_TTL` (seconds, default `1209600` = 14 days)
  - `NEARBY_LIVE_CACHE_COMPAT_KM` (default `12`)

## Keep user DB out of GitHub (important)
- The project uses local SQLite at `data/app.db`.
- Do not commit this file because it contains user login/profile data.
- `.gitignore` already excludes `.db` and SQLite sidecar files (`*.db-wal`, `*.db-shm`, etc.).

If `app.db` was ever committed earlier, remove it from git tracking (without deleting your local file):

```bash
git rm --cached data/app.db
git rm --cached data/app.db-wal data/app.db-shm
git commit -m "Stop tracking local database files"
```

### What happens for new clones?
- New clones start without `data/app.db` (this is expected).
- Running `python app.py` calls `init_db()` and auto-creates all required tables.
- Users can register normally via API/UI; no manual DB setup is required.
