"""
api.py - Mobile REST API layer for the AI CCTV Attendance System.

WHY A SEPARATE FILE INSTEAD OF EDITING app.py
------------------------------------------------------------------------------
This file WRAPS app.py rather than duplicating it. Importing app.py re-uses
everything it already built at module load time: the InsightFace engine,
the SQLite connection, the in-memory face database + embedding matrix, the
live recognition state (match_history / latest_jpeg), and every helper
function (mark_attendance, get_attendance_for_date, the employee cache,
the guided-enrollment quality checks, etc). The browser dashboard routes
app.py defines ("/", "/admin", "/enroll", "/video_feed", "/api/matches",
"/api/attendance", "/api/movements", "/api/employees", "/api/enroll/*")
keep working exactly as before, unchanged, on the SAME Flask app object.

What THIS file adds on top, specifically for a phone app:

  1. JWT login (POST /api/v1/auth/login). A mobile app should log in once
     and attach a bearer token to every request - not pop a native
     "this site requires a username and password" dialog on every screen,
     which is what plain HTTP Basic Auth does on mobile.

  2. A dashboard summary endpoint that pre-aggregates present / absent /
     currently-inside counts server-side, so the phone's home screen is
     one HTTP call instead of it fetching three lists and computing the
     same thing client-side on a slower connection.

  3. Date-range queries for attendance/movements/one employee's history -
     the browser dashboard only ever asked for "today"; a phone app's
     history screen needs a range.

  4. A single-JPEG live snapshot endpoint. An MJPEG stream left open on a
     phone burns mobile data and battery for a screen that isn't always
     in the foreground - polling one JPEG on a timer is far cheaper and
     is what the Flutter app in this project actually does. The MJPEG
     stream is still exposed too (as /api/v1/live/stream) for anyone who
     wants a true live view while on Wi-Fi.

  5. The same guided-enrollment flow (identity details -> capture 5 poses
     x 10 shots -> finish) app.py already built for a browser, re-exposed
     under JWT so HR can enroll a new employee from their phone's camera
     instead of walking them over to a laptop.

RUN THIS FILE INSTEAD OF app.py
------------------------------------------------------------------------------
    python api.py

It starts the camera grab/process threads itself (the same two threads
app.py's own __main__ block started) and serves both the existing browser
dashboard and the new mobile API from a single process on PORT. Don't run
app.py and api.py at the same time - they'd both try to open the same
camera source.
"""

import os
import time
import threading
import signal
import secrets as pysecrets
from datetime import datetime, timedelta
from functools import wraps

import jwt  # PyJWT
from flask import request, jsonify, Response, g
from flask_cors import CORS

import app as core  # noqa: E402  (this import runs app.py's whole module body)


# ==============================================================================
# 1. JWT AUTH
# ==============================================================================
# JWT_SECRET should be set explicitly in .env for production - a random
# fallback is generated so the app still boots for local testing, but every
# restart would then invalidate all issued tokens (everyone gets logged out),
# which is exactly why this is only a fallback and gets logged loudly.
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    JWT_SECRET = pysecrets.token_hex(32)
    core.logger.warning(
        "JWT_SECRET is not set in .env - using a random one-off secret. "
        "Every restart will invalidate all mobile app logins. Set JWT_SECRET "
        "to a fixed random string in .env for real deployments."
    )
JWT_ALGO = "HS256"
JWT_EXPIRY_HOURS = float(os.environ.get("JWT_EXPIRY_HOURS", "12"))

CORS(core.flask_app, resources={r"/api/v1/*": {"origins": "*"}})


def issue_token(username):
    now = datetime.utcnow()
    payload = {
        "sub": username,
        "role": "admin",
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    # PyJWT < 2 returns bytes; PyJWT >= 2 returns str. Normalize to str.
    return token.decode("utf-8") if isinstance(token, bytes) else token


def decode_token(token):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])


def _extract_token():
    """Accepts the token either as 'Authorization: Bearer <token>' (every
    normal API call) or as a '?token=' query param (needed for the couple
    of endpoints, like the live snapshot/stream, that get loaded directly
    by an <img>/Image widget which can't attach custom headers)."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[len("Bearer "):].strip()
    return request.args.get("token")


def require_jwt(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        token = _extract_token()
        if not token:
            return jsonify(ok=False, reason="Missing bearer token."), 401
        try:
            g.user = decode_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify(ok=False, reason="Token expired. Please log in again."), 401
        except jwt.InvalidTokenError:
            return jsonify(ok=False, reason="Invalid token."), 401
        return view_func(*args, **kwargs)
    return wrapped


@core.flask_app.route("/api/v1/auth/login", methods=["POST"])
def api_v1_login():
    payload = request.get_json(force=True, silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""

    valid = bool(
        username
        and pysecrets.compare_digest(username, core.BASIC_AUTH_USER)
        and pysecrets.compare_digest(password, core.BASIC_AUTH_PASS)
    )
    if not valid:
        return jsonify(ok=False, reason="Invalid username or password."), 401

    token = issue_token(username)
    return jsonify(
        ok=True,
        token=token,
        expires_in_hours=JWT_EXPIRY_HOURS,
        username=username,
    )


@core.flask_app.route("/api/v1/auth/verify")
@require_jwt
def api_v1_verify():
    return jsonify(ok=True, user=g.user.get("sub"))


# ==============================================================================
# 2. EXTRA DB QUERIES (date ranges / currently-inside / per-employee history)
# ==============================================================================
def get_attendance_range(date_from, date_to, person_id=None):
    query = """
        SELECT a.person_id, a.attendance_date, a.first_seen_time, a.similarity,
               e.name, e.department, e.designation
        FROM attendance a
        LEFT JOIN employees e ON e.person_id = a.person_id
        WHERE a.attendance_date BETWEEN ? AND ?
    """
    params = [date_from, date_to]
    if person_id:
        query += " AND a.person_id = ?"
        params.append(person_id)
    query += " ORDER BY a.attendance_date DESC, a.first_seen_time ASC"

    with core.db_lock:
        rows = core.db_conn.execute(query, params).fetchall()
    return [
        {
            "id": r[0], "date": r[1], "time": r[2], "similarity": round(r[3], 3),
            "name": r[4] or r[0], "department": r[5] or "", "designation": r[6] or "",
        }
        for r in rows
    ]


def get_movements_range(date_from, date_to, person_id=None):
    query = """
        SELECT m.person_id, m.direction, m.event_date, m.event_time, e.name, e.department
        FROM movement_log m
        LEFT JOIN employees e ON e.person_id = m.person_id
        WHERE m.event_date BETWEEN ? AND ?
    """
    params = [date_from, date_to]
    if person_id:
        query += " AND m.person_id = ?"
        params.append(person_id)
    query += " ORDER BY m.event_date DESC, m.event_time ASC"

    with core.db_lock:
        rows = core.db_conn.execute(query, params).fetchall()
    return [
        {
            "id": r[0], "direction": r[1], "date": r[2], "time": r[3],
            "name": r[4] or r[0], "department": r[5] or "",
        }
        for r in rows
    ]


def get_currently_inside(date_str):
    """A person is 'currently inside' if the LAST movement logged for them
    today was an ENTERING event with no EXITING event after it."""
    with core.db_lock:
        rows = core.db_conn.execute(
            """
            SELECT person_id, direction FROM movement_log
            WHERE event_date = ? AND person_id != 'Not Enrolled'
            ORDER BY event_time ASC
            """,
            (date_str,),
        ).fetchall()
    last_direction = {}
    for person_id, direction in rows:
        last_direction[person_id] = direction
    return [pid for pid, d in last_direction.items() if d == "ENTERING"]


# ==============================================================================
# 3. DASHBOARD
# ==============================================================================
@core.flask_app.route("/api/v1/dashboard")
@require_jwt
def api_v1_dashboard():
    today = datetime.now().strftime("%Y-%m-%d")
    present_today = core.get_attendance_for_date(today)
    inside_ids = get_currently_inside(today)

    all_employees = core.get_all_employee_details()
    total_employees = len(all_employees)
    present_ids = {r["id"] for r in present_today}
    absent = [e for e in all_employees if e["person_id"] not in present_ids]

    with core.state_lock:
        recent_matches = list(core.match_history[:10])

    return jsonify(
        ok=True,
        date=today,
        total_employees=total_employees,
        present_count=len(present_today),
        absent_count=len(absent),
        currently_inside_count=len(inside_ids),
        recent_matches=recent_matches,
        absent_employees=[
            {"person_id": e["person_id"], "name": e["name"], "department": e["department"]}
            for e in absent
        ],
    )


# ==============================================================================
# 4. ATTENDANCE / MOVEMENTS
# ==============================================================================
@core.flask_app.route("/api/v1/attendance")
@require_jwt
def api_v1_attendance():
    date_str = request.args.get("date")
    date_from = request.args.get("from")
    date_to = request.args.get("to")
    person_id = request.args.get("employee_id")

    if date_from or date_to:
        date_from = date_from or "2000-01-01"
        date_to = date_to or datetime.now().strftime("%Y-%m-%d")
        records = get_attendance_range(date_from, date_to, person_id)
        return jsonify(ok=True, from_date=date_from, to_date=date_to, records=records)

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    records = core.get_attendance_for_date(date_str)
    if person_id:
        records = [r for r in records if r["id"] == person_id]
    return jsonify(ok=True, date=date_str, records=records)


@core.flask_app.route("/api/v1/movements")
@require_jwt
def api_v1_movements():
    date_str = request.args.get("date")
    date_from = request.args.get("from")
    date_to = request.args.get("to")
    person_id = request.args.get("employee_id")

    if date_from or date_to:
        date_from = date_from or "2000-01-01"
        date_to = date_to or datetime.now().strftime("%Y-%m-%d")
        records = get_movements_range(date_from, date_to, person_id)
        return jsonify(ok=True, from_date=date_from, to_date=date_to, records=records)

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    records = core.get_movements_for_date(date_str)
    if person_id:
        records = [r for r in records if r["id"] == person_id]
    return jsonify(ok=True, date=date_str, records=records)


# ==============================================================================
# 5. EMPLOYEES
# ==============================================================================
@core.flask_app.route("/api/v1/employees")
@require_jwt
def api_v1_employees():
    return jsonify(ok=True, employees=core.get_all_employee_details())


@core.flask_app.route("/api/v1/employees/<person_id>")
@require_jwt
def api_v1_employee_detail(person_id):
    person_id = core.sanitize_person_id(person_id)
    all_employees = core.get_all_employee_details()
    match = next((e for e in all_employees if e["person_id"] == person_id), None)
    if not match:
        return jsonify(ok=False, reason="Employee not found."), 404

    today = datetime.now().strftime("%Y-%m-%d")
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    history = get_attendance_range(thirty_days_ago, today, person_id)
    return jsonify(ok=True, employee=match, recent_attendance=history)


@core.flask_app.route("/api/v1/employees", methods=["POST"])
@require_jwt
def api_v1_upsert_employee():
    payload = request.get_json(force=True, silent=True) or {}
    person_id = core.sanitize_person_id(payload.get("person_id"))
    name = (payload.get("name") or "").strip()
    if not person_id:
        return jsonify(ok=False, reason="Missing/invalid person_id."), 400
    if not name:
        return jsonify(ok=False, reason="Name is required."), 400

    record = core.upsert_employee(
        person_id, name,
        department=(payload.get("department") or "").strip(),
        designation=(payload.get("designation") or "").strip(),
        phone=(payload.get("phone") or "").strip(),
        email=(payload.get("email") or "").strip(),
    )
    return jsonify(ok=True, person_id=person_id, **record)


# ==============================================================================
# 6. LIVE FEED
# ==============================================================================
@core.flask_app.route("/api/v1/live/matches")
@require_jwt
def api_v1_live_matches():
    with core.state_lock:
        return jsonify(ok=True, matches=list(core.match_history))


@core.flask_app.route("/api/v1/live/snapshot")
@require_jwt
def api_v1_live_snapshot():
    """One current annotated JPEG frame - the Flutter app polls this on a
    timer instead of holding an MJPEG connection open, which is far
    cheaper on mobile data/battery for a screen that isn't always visible."""
    with core.state_lock:
        frame = core.latest_jpeg
    if frame is None:
        return jsonify(ok=False, reason="No frame available yet."), 503
    return Response(frame, mimetype="image/jpeg")


@core.flask_app.route("/api/v1/live/stream")
@require_jwt
def api_v1_live_stream():
    """True MJPEG stream, for when the app is on Wi-Fi and wants continuous
    live video rather than polling snapshots."""
    return Response(core.mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ==============================================================================
# 7. GUIDED ENROLLMENT (phone camera -> this API), mirrors app.py's browser
#    flow one-for-one, just JWT-protected instead of Basic-Auth-protected.
# ==============================================================================
@core.flask_app.route("/api/v1/enroll/details", methods=["POST"])
@require_jwt
def api_v1_enroll_details():
    payload = request.get_json(force=True, silent=True) or {}
    person_id = core.sanitize_person_id(payload.get("person_id"))
    name = (payload.get("name") or "").strip()
    if not person_id:
        return jsonify(ok=False, reason="Missing/invalid person ID."), 400
    if not name:
        return jsonify(ok=False, reason="Name is required."), 400

    record = core.upsert_employee(
        person_id, name,
        department=(payload.get("department") or "").strip(),
        designation=(payload.get("designation") or "").strip(),
        phone=(payload.get("phone") or "").strip(),
        email=(payload.get("email") or "").strip(),
    )
    return jsonify(ok=True, person_id=person_id, **record)


@core.flask_app.route("/api/v1/enroll/status")
@require_jwt
def api_v1_enroll_status():
    person_id = core.sanitize_person_id(request.args.get("person_id", ""))
    if not person_id:
        return jsonify(ok=False, reason="Missing person ID."), 400
    counts = {pose: core.count_existing_captures(person_id, pose) for pose in core.ENROLL_POSES}
    return jsonify(ok=True, person_id=person_id, counts=counts, per_pose_target=core.ENROLL_CAPTURES_PER_POSE)


@core.flask_app.route("/api/v1/enroll/capture", methods=["POST"])
@require_jwt
def api_v1_enroll_capture():
    """Identical validation pipeline to app.py's /api/enroll/capture (blur,
    single-face, detection confidence, minimum face size, then pose
    classification) - duplicated here rather than calling that view
    function directly because that route is already bound to the
    Basic-Auth decorator app.py registered it with."""
    payload = request.get_json(force=True, silent=True) or {}
    person_id = core.sanitize_person_id(payload.get("person_id"))
    pose = payload.get("pose")
    image_data = payload.get("image")

    if not person_id:
        return jsonify(ok=False, reason="Missing/invalid person ID."), 400
    if pose not in core.ENROLL_POSES:
        return jsonify(ok=False, reason="Invalid pose."), 400
    if not image_data:
        return jsonify(ok=False, reason="No image received."), 400

    try:
        frame = core.decode_data_uri_image(image_data)
    except Exception:
        frame = None
    if frame is None:
        return jsonify(ok=False, reason="Could not decode image."), 400

    with core.enroll_lock:
        existing = core.count_existing_captures(person_id, pose)
        if existing >= core.ENROLL_CAPTURES_PER_POSE:
            return jsonify(ok=True, already_complete=True, count=existing, total=core.ENROLL_CAPTURES_PER_POSE)

        cooldown_key = f"enroll:{person_id}:{pose}"
        now = time.time()
        if now - core.last_seen_at.get(cooldown_key, 0) < core.ENROLL_CAPTURE_COOLDOWN:
            return jsonify(ok=False, reason="cooldown", count=existing, total=core.ENROLL_CAPTURES_PER_POSE)

        blur_score = core.compute_blur_score(frame)
        if blur_score < core.ENROLL_BLUR_THRESHOLD:
            return jsonify(
                ok=False, reason=f"Too blurry (score {blur_score:.0f}) - hold steady.",
                count=existing, total=core.ENROLL_CAPTURES_PER_POSE,
            )

        faces = core.face_engine.get(core.enhance_backlit_face(frame))
        if len(faces) == 0:
            return jsonify(ok=False, reason="No face detected.", count=existing, total=core.ENROLL_CAPTURES_PER_POSE)
        if len(faces) > 1:
            return jsonify(
                ok=False, reason="Multiple faces in frame - only the person enrolling should be visible.",
                count=existing, total=core.ENROLL_CAPTURES_PER_POSE,
            )

        face = faces[0]
        if float(face.det_score) < core.ENROLL_MIN_DET_SCORE:
            return jsonify(ok=False, reason="Face not clear enough.", count=existing, total=core.ENROLL_CAPTURES_PER_POSE)

        bbox = face.bbox
        face_w, face_h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
        if face_w < core.ENROLL_MIN_FACE_WIDTH or face_h < core.ENROLL_MIN_FACE_WIDTH:
            return jsonify(ok=False, reason="Move closer to the camera.", count=existing, total=core.ENROLL_CAPTURES_PER_POSE)

        yaw_ratio, pitch_ratio = core.estimate_head_pose(face)
        detected_pose = core.classify_pose(yaw_ratio, pitch_ratio)
        if detected_pose != pose:
            return jsonify(
                ok=False,
                reason=f"Pose mismatch - detected '{detected_pose or 'unclear'}', need '{pose}'.",
                yaw=round(yaw_ratio, 3), pitch=round(pitch_ratio, 3),
                count=existing, total=core.ENROLL_CAPTURES_PER_POSE,
            )

        folder = os.path.join(core.CAPTURES_DIR, person_id)
        os.makedirs(folder, exist_ok=True)
        new_count = existing + 1
        core.cv2.imwrite(os.path.join(folder, f"{pose}_{new_count:02d}.jpg"), frame)
        core.last_seen_at[cooldown_key] = now

        return jsonify(
            ok=True, count=new_count, total=core.ENROLL_CAPTURES_PER_POSE,
            pose_complete=(new_count >= core.ENROLL_CAPTURES_PER_POSE),
            yaw=round(yaw_ratio, 3), pitch=round(pitch_ratio, 3),
        )


@core.flask_app.route("/api/v1/enroll/reset", methods=["POST"])
@require_jwt
def api_v1_enroll_reset():
    payload = request.get_json(force=True, silent=True) or {}
    person_id = core.sanitize_person_id(payload.get("person_id"))
    pose = payload.get("pose")
    if not person_id:
        return jsonify(ok=False, reason="Missing person ID."), 400

    folder = os.path.join(core.CAPTURES_DIR, person_id)
    if not os.path.isdir(folder):
        return jsonify(ok=True, cleared=0)

    poses_to_clear = [pose] if pose in core.ENROLL_POSES else core.ENROLL_POSES
    cleared = 0
    with core.enroll_lock:
        for p in poses_to_clear:
            for f in core.glob.glob(os.path.join(folder, f"{p}_*.jpg")):
                os.remove(f)
                cleared += 1
    return jsonify(ok=True, cleared=cleared)


@core.flask_app.route("/api/v1/enroll/finish", methods=["POST"])
@require_jwt
def api_v1_enroll_finish():
    payload = request.get_json(force=True, silent=True) or {}
    person_id = core.sanitize_person_id(payload.get("person_id"))
    if not person_id:
        return jsonify(ok=False, reason="Missing person ID."), 400

    counts = {pose: core.count_existing_captures(person_id, pose) for pose in core.ENROLL_POSES}
    total_captured = sum(counts.values())
    embeddings = core.build_embeddings_for_person(person_id)

    with core.reload_lock:
        core.face_db[person_id] = embeddings
        new_matrix, new_labels = core.build_embedding_matrix(core.face_db)
        core.emb_matrix, core.emb_labels = new_matrix, new_labels

    core.logger.info(
        f"[mobile] Enrolled '{person_id}': {len(embeddings)}/{total_captured} usable "
        f"embeddings built from {total_captured} captured images."
    )
    return jsonify(
        ok=True, person_id=person_id, counts=counts,
        total_captured=total_captured, embeddings_built=len(embeddings),
    )


# ==============================================================================
# 8. HEALTH
# ==============================================================================
@core.flask_app.route("/api/v1/health")
def api_v1_health():
    return jsonify(
        ok=True,
        server_time=datetime.now().isoformat(timespec="seconds"),
        enrolled_count=len(core.face_db),
        camera_source="Webcam" if core.USE_WEBCAM else f"{core.RTSP_HOST}:{core.RTSP_PORT}",
    )


# ==============================================================================
# 9. STARTUP - starts the same background threads app.py's __main__ would,
#    since importing app.py as a module (above) does NOT run its __main__
#    block.
# ==============================================================================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, core.shutdown)
    signal.signal(signal.SIGTERM, core.shutdown)

    problems = core.validate_config()
    if problems:
        for p in problems:
            core.logger.error(p)
        raise SystemExit(1)

    threading.Thread(target=core.run_forever, args=("grab_loop", core.grab_loop), daemon=True).start()
    threading.Thread(target=core.run_forever, args=("process_loop", core.process_loop), daemon=True).start()

    port = int(os.environ.get("PORT", "8080"))
    try:
        from waitress import serve
        core.logger.info(f"Starting dashboard + mobile API (waitress) on 0.0.0.0:{port}")
        serve(core.flask_app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        core.logger.warning(
            "waitress is not installed - falling back to Flask's development "
            "server, which is NOT suitable for production. Run "
            "'pip install waitress' to use the production server."
        )
        core.flask_app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
