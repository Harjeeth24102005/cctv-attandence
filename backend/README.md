# AI CCTV Attendance System - Backend

This folder contains your original `app.py` (unchanged) plus a new
`api.py` that adds a JWT-secured REST API for the companion Flutter app,
on top of the same Flask app, same face-recognition engine, same SQLite
database, and same live camera feed.

## Setup

```bash
pip install -r requirements.txt
cp env.example .env
# edit .env: RTSP camera details, BASIC_AUTH_USER/PASS, JWT_SECRET, etc.
```

If you're upgrading an older `attendance.db`, run the schema migration once
first: `python "migrate_employees (1).py"`.

## Running

Run **`api.py`**, not `app.py`, from now on - it does everything `app.py`
did (browser dashboard at `/`, `/admin`, `/enroll`, live MJPEG at
`/video_feed`) *and* serves the mobile API the Flutter app talks to. Don't
run both files at once; they'd both try to open the same camera.

```bash
python api.py
```

## Mobile API reference (all under `/api/v1`)

Every endpoint below except `/auth/login` and `/health` requires
`Authorization: Bearer <token>` (or `?token=...` for the two image
endpoints, since `<img>`/`Image.network` can't set custom headers).

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/auth/login` | `{username, password}` &rarr; `{token}`. Uses the same `BASIC_AUTH_USER`/`BASIC_AUTH_PASS` as the browser dashboard. |
| GET | `/api/v1/auth/verify` | Confirms a token is still valid. |
| GET | `/api/v1/dashboard` | Today's present/absent/currently-inside counts + recent live matches, in one call. |
| GET | `/api/v1/attendance?date=YYYY-MM-DD` | One day's attendance. |
| GET | `/api/v1/attendance?from=YYYY-MM-DD&to=YYYY-MM-DD&employee_id=ID` | Date-range attendance, optionally for one employee. |
| GET | `/api/v1/movements?date=` / `?from=&to=` | Entry/exit log, same date-or-range pattern. |
| GET | `/api/v1/employees` | Full employee directory with pose/embedding counts + thumbnail. |
| GET | `/api/v1/employees/<person_id>` | One employee + their last 30 days of attendance. |
| POST | `/api/v1/employees` | Create/update an employee's name/department/designation/phone/email. |
| GET | `/api/v1/live/matches` | The recent-recognitions sidebar feed. |
| GET | `/api/v1/live/snapshot` | One current JPEG frame (poll this on a timer - cheap on mobile data). |
| GET | `/api/v1/live/stream` | True MJPEG stream (use on Wi-Fi). |
| POST | `/api/v1/enroll/details` | Step 1 of enrolling someone: `{person_id, name, department, designation, phone, email}`. |
| GET | `/api/v1/enroll/status?person_id=` | How many shots captured per pose so far. |
| POST | `/api/v1/enroll/capture` | `{person_id, pose, image: "data:image/jpeg;base64,..."}` - one guided capture. |
| POST | `/api/v1/enroll/reset` | Clear captures for one pose (or all) to retake. |
| POST | `/api/v1/enroll/finish` | Builds embeddings from all captured photos and makes the person recognizable immediately. |
| GET | `/api/v1/health` | No auth required - basic liveness/info check. |

## Notes

- The original browser dashboard and its `/api/*` routes (Basic Auth) are
  untouched - `api.py` only *adds* routes, all under `/api/v1/*`.
- `JWT_SECRET` must be set to a fixed value in `.env` for real use, or
  every restart logs everyone in the app out (see `env.example`).
