import os
import sys
import glob
import time
import signal
import secrets
import sqlite3
import logging
import threading
import base64
from functools import wraps
from datetime import datetime
from urllib.parse import quote
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request, send_from_directory

import insightface
from insightface.app import FaceAnalysis

try:
    from uniface.spoofing import MiniFASNet
except ImportError:
    MiniFASNet = None  # handled in validate_config() if anti-spoofing is enabled

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional at runtime if the environment variables are
    # already set some other way (systemd EnvironmentFile, Docker env, etc).
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")
DB_PATH = os.path.join(SCRIPT_DIR, "attendance.db")
LOG_PATH = os.path.join(SCRIPT_DIR, "attendance_system.log")
SNAPSHOTS_DIR = os.path.join(SCRIPT_DIR, "attendance_snapshots")

# ==============================================================================
# 0. LOGGING (replaces print() so failures are visible & persisted)
# ==============================================================================
logger = logging.getLogger("attendance")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
logger.addHandler(_console)

_file = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3)
_file.setFormatter(_fmt)
logger.addHandler(_file)


# ==============================================================================
# 1. CONFIGURATION (all from environment / .env â€” nothing sensitive hardcoded)
# ==============================================================================
def _env_bool(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


USE_WEBCAM = _env_bool("USE_WEBCAM", False)
WEBCAM_INDEX = int(os.environ.get("WEBCAM_INDEX", "0"))

RTSP_HOST = os.environ.get("RTSP_HOST", "192.168.50.152")
RTSP_PORT = os.environ.get("RTSP_PORT", "554")
RTSP_USER = os.environ.get("RTSP_USER", "admin")
RTSP_PASS = os.environ.get("RTSP_PASS", "Admin@123")
RTSP_PATH = os.environ.get(
    "RTSP_PATH", "/video/live?channel=1&subtype=0&unicast=true&proto=Onvif"
)

SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.40"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "90"))
MATCH_HISTORY_SIZE = int(os.environ.get("MATCH_HISTORY_SIZE", "25"))
MATCH_COOLDOWN_SECONDS = float(os.environ.get("MATCH_COOLDOWN_SECONDS", "5"))

# Caps how many times per second annotate_frame() runs. Without this, the
# processor loop runs flat-out and will peg the CPU at 100% continuously on
# any machine that can't process frames as fast as the camera produces them
# - which is expected, not a bug, but on a shared laptop (other apps, fan
# noise, battery) that's often not worth it. Lower this to trade detection
# responsiveness for CPU headroom; 0 (or unset) disables throttling
# entirely and processes as fast as possible, same as before.
PROCESS_MAX_FPS = float(os.environ.get("PROCESS_MAX_FPS", "0"))
PROCESS_MIN_INTERVAL = (1.0 / PROCESS_MAX_FPS) if PROCESS_MAX_FPS > 0 else 0.0

# Detector input size. 640x640 is the InsightFace default and is the most
# accurate, but on a CPU-only or under-powered GPU it can be too slow to
# keep up with a live camera, which is what shows up as the video feed
# "lagging" or "loading late" - the processor thread just can't finish
# frames as fast as they arrive. Drop this (e.g. to 320) if you're seeing
# delay and confirm via the provider log below that the bottleneck isn't
# actually a missing GPU provider.
DET_SIZE = int(os.environ.get("DET_SIZE", "640"))

# Lower this if faces at a distance or at an angle (e.g. someone walking
# through a far doorway, not looking straight at the camera) aren't getting
# a detection box at all. InsightFace/SCRFD's default is 0.5 - dropping it
# to ~0.35 catches smaller/angled faces at the cost of a few more false
# positives on non-faces, which is usually the right trade-off for an
# entrance camera where people aren't posing for it.
DET_THRESH = float(os.environ.get("DET_THRESH", "0.5"))

# A 4MP+ camera frame is expensive at EVERY step (CLAHE, color conversion,
# detection, drawing, JPEG encode) - not just detection. Downscaling right
# after capture, before any of that runs, cuts CPU cost across the whole
# pipeline at once. This is a bigger lever than DET_SIZE alone when the
# source frame itself is large. Set to 0 to disable (use full camera
# resolution, same as before). 960 is a reasonable starting point for an
# entrance camera - faces close enough to matter are still plenty large
# enough to detect at that width.
FRAME_RESIZE_WIDTH = int(os.environ.get("FRAME_RESIZE_WIDTH", "0"))


def maybe_downscale(frame):
    if FRAME_RESIZE_WIDTH <= 0 or frame.shape[1] <= FRAME_RESIZE_WIDTH:
        return frame
    scale = FRAME_RESIZE_WIDTH / frame.shape[1]
    new_h = int(frame.shape[0] * scale)
    return cv2.resize(frame, (FRAME_RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

# Forces the FFMPEG backend to use TCP transport and skip its own internal
# frame buffering. Without this, OpenCV/FFMPEG on Windows often keeps
# several seconds of RTSP frames queued up regardless of
# CAP_PROP_BUFFERSIZE, which shows up as the stream looking delayed even
# though frames are being read fine.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

TRACK_IOU_THRESHOLD = float(os.environ.get("TRACK_IOU_THRESHOLD", "0.2"))
TRACK_TIMEOUT_SECONDS = float(os.environ.get("TRACK_TIMEOUT_SECONDS", "4.0"))
TRACK_EMBEDDING_REASSOC_THRESHOLD = float(
    os.environ.get("TRACK_EMBEDDING_REASSOC_THRESHOLD", "0.35")
)
# Fallback re-association test used when IOU fails to match a detection to
# an existing track (see annotate_frame). A small/fast-moving face near the
# door can shift enough between two processed frames that its box barely
# overlaps its own previous position at all, even though it's obviously
# still the same person - IOU alone reads that as "gone, brand-new person"
# and spawns a fresh track (visible as the red box flickering back on and
# a new "Not Enrolled" entry appearing repeatedly). Comparing centroid
# movement against the box's OWN diagonal, rather than a fixed pixel
# distance, scales naturally with how close/far the person is from the
# camera. 1.5 means "the box center moved less than 1.5x its own diagonal
# length" still counts as continuous motion, not a new person.
TRACK_CENTROID_DIST_RATIO = float(os.environ.get("TRACK_CENTROID_DIST_RATIO", "1.5"))
REMATCH_INTERVAL_SECONDS = float(os.environ.get("REMATCH_INTERVAL_SECONDS", "1.0"))

# --- Entry/Exit direction detection (LINE-CROSSING, not door detection) ----
# We cannot run object detection on the glass door itself (reflections/glass
# break most detectors), so direction is inferred purely from how each
# person's face MOVES across the frame relative to a fixed virtual line.
#
# DOOR_LINE_Y is a single pixel row (in the *processed* frame's coordinate
# space, i.e. after any FRAME_RESIZE_WIDTH downscale) that sits between the
# door side of the frame and the interior/desk side. You calibrate this once
# per camera install by watching a few real walk-ins and noting roughly
# where their face crosses vertically.
#
# A track's direction is decided by comparing where its face CENTROID was in
# its earliest few samples vs its latest few samples (not an average across
# the whole track), so a person who enters and then stands still near the
# desk still gets a correct "ENTERING" call from the crossing itself, rather
# than the direction being diluted by a long stationary tail.
#
# Face-size growth/shrinkage is used only as a secondary confirming signal
# (it still generally holds - closer to camera = bigger box - it's just not
# reliable enough alone once someone stops moving), so both signals must
# agree before we log an event. This cuts false positives from someone just
# turning their head or shifting weight without actually crossing.
DOOR_LINE_Y = int(os.environ.get("DOOR_LINE_Y", "340"))
DIRECTION_EDGE_SAMPLES = int(os.environ.get("DIRECTION_EDGE_SAMPLES", "3"))
DIRECTION_MIN_SAMPLES = int(os.environ.get("DIRECTION_MIN_SAMPLES", "4"))
# How many recent (timestamp, area/position) samples to keep per track.
# Older samples are dropped so the trend reflects recent movement, not the
# person's entire time in frame.
DIRECTION_HISTORY_SIZE = int(os.environ.get("DIRECTION_HISTORY_SIZE", "10"))

# --- Guided enrollment (phone/webcam, not the entrance CCTV) --------------
# Goal: 50 clean embeddings per person (10 each of front/left/right/up/down)
# so live matching has enough pose coverage to recognize people quickly at
# whatever angle they happen to approach the entrance camera at. Every shot
# is validated server-side (blur, face size, single face, actual head angle)
# BEFORE it's accepted, so a bad frame never silently becomes a bad
# embedding - "50 embeddings" only helps if all 50 are actually usable.
ENROLL_POSES = ["front", "left", "right", "up", "down"]
ENROLL_CAPTURES_PER_POSE = int(os.environ.get("ENROLL_CAPTURES_PER_POSE", "10"))
ENROLL_MIN_FACE_WIDTH = int(os.environ.get("ENROLL_MIN_FACE_WIDTH", "120"))
ENROLL_BLUR_THRESHOLD = float(os.environ.get("ENROLL_BLUR_THRESHOLD", "60"))
ENROLL_MIN_DET_SCORE = float(os.environ.get("ENROLL_MIN_DET_SCORE", "0.65"))
# Minimum time between two ACCEPTED captures of the same pose, so the 10
# shots are 10 genuinely different moments (slightly different micro-angle,
# blink state, etc.) rather than 10 near-duplicates of the same instant.
ENROLL_CAPTURE_COOLDOWN = float(os.environ.get("ENROLL_CAPTURE_COOLDOWN", "0.35"))

# Head-pose thresholds, expressed as ratios derived from the 5-point face
# landmarks (eyes/nose/mouth) InsightFace already returns per detection -
# no extra model needed. yaw_ratio ~0 = nose centered between eyes (facing
# camera horizontally); pitch_ratio ~0.5 = nose vertically centered between
# eyes and mouth. These are reasonable starting points for a face roughly
# 0.5-1m from the camera - if pose keeps getting misclassified for your
# setup, watch the yaw/pitch debug values the API returns and retune here.
ENROLL_YAW_FRONT_MAX = float(os.environ.get("ENROLL_YAW_FRONT_MAX", "0.12"))
ENROLL_YAW_SIDE_MIN = float(os.environ.get("ENROLL_YAW_SIDE_MIN", "0.20"))
ENROLL_PITCH_FRONT_MIN = float(os.environ.get("ENROLL_PITCH_FRONT_MIN", "0.40"))
ENROLL_PITCH_FRONT_MAX = float(os.environ.get("ENROLL_PITCH_FRONT_MAX", "0.60"))
ENROLL_PITCH_UP_MAX = float(os.environ.get("ENROLL_PITCH_UP_MAX", "0.32"))
ENROLL_PITCH_DOWN_MIN = float(os.environ.get("ENROLL_PITCH_DOWN_MIN", "0.68"))

BASIC_AUTH_USER = os.environ.get("BASIC_AUTH_USER", "")
BASIC_AUTH_PASS = os.environ.get("BASIC_AUTH_PASS", "")

RECONNECT_MIN_DELAY = 2
RECONNECT_MAX_DELAY = 30
MAX_CONSECUTIVE_READ_FAILURES = 30  # ~read failures before we force a reconnect


def build_video_source():
    """Builds the RTSP URL from separate env vars and URL-encodes the
    credentials. Doing this (instead of a single hardcoded RTSP_URL string)
    avoids '@' or ':' inside a password breaking the URL's user:pass@host
    parsing, and keeps credentials out of source control."""
    if USE_WEBCAM:
        return WEBCAM_INDEX

    missing = [n for n, v in (("RTSP_HOST", RTSP_HOST), ("RTSP_USER", RTSP_USER),
                               ("RTSP_PASS", RTSP_PASS)) if not v]
    if missing:
        raise RuntimeError(
            f"Missing required RTSP env vars: {', '.join(missing)}. "
            f"Set them in your .env file (see .env.example)."
        )

    user_enc = quote(RTSP_USER, safe="")
    pass_enc = quote(RTSP_PASS, safe="")
    return f"rtsp://{user_enc}:{pass_enc}@{RTSP_HOST}:{RTSP_PORT}{RTSP_PATH}"


flask_app = Flask(__name__)


# ==============================================================================
# 2. HTTP BASIC AUTH (dashboard shows employee identities, so it's gated)
# ==============================================================================
def require_auth(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        valid = bool(
            auth
            and secrets.compare_digest(auth.username or "", BASIC_AUTH_USER)
            and secrets.compare_digest(auth.password or "", BASIC_AUTH_PASS)
        )
        if not valid:
            return Response(
                "Authentication required.", 401,
                {"WWW-Authenticate": 'Basic realm="Attendance System"'},
            )
        return view_func(*args, **kwargs)
    return wrapped


# ==============================================================================
# 3. INITIALIZE INSIGHTFACE (ArcFace + SCRFD)
# ==============================================================================
logger.info("Loading ArcFace model (InsightFace)...")
# CUDAExecutionProvider only works with NVIDIA GPUs. On machines with an
# AMD or Intel GPU (e.g. Radeon), it will always silently fall back to
# CPU - there is no fix for that other than switching backends. DirectML
# (via the onnxruntime-directml package) gives GPU acceleration on
# Windows for AMD/Intel/NVIDIA GPUs alike through DirectX 12, so it's
# used here instead. If you do have an NVIDIA GPU with proper CUDA/cuDNN
# installed, DmlExecutionProvider still works fine, just swap the list
# below to prefer ["CUDAExecutionProvider", "CPUExecutionProvider"].
FACE_MODEL_PACK = os.environ.get("FACE_MODEL_PACK", "buffalo_l")
# buffalo_l is the most accurate but heaviest InsightFace model pack. On a
# CPU-bound machine (no working GPU execution provider), buffalo_s trades
# some accuracy for meaningfully faster detection - worth trying via
# FACE_MODEL_PACK=buffalo_s in .env if buffalo_l is too slow to keep the
# feed responsive.
DML_DEVICE_ID = int(os.environ.get("DML_DEVICE_ID", "1"))
face_engine = FaceAnalysis(
    name=FACE_MODEL_PACK,
    providers=[("DmlExecutionProvider", {"device_id": DML_DEVICE_ID}), "CPUExecutionProvider"]
)
face_engine.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE), det_thresh=DET_THRESH)

# DmlExecutionProvider silently falls back to CPUExecutionProvider if
# onnxruntime-directml isn't installed (or plain onnxruntime got
# reinstalled over it as a dependency of another package) - no error is
# raised, it just runs much slower. Log the actual active providers on
# startup so a "CPU only" fallback shows up here instead of being invisible.
try:
    active_providers = face_engine.models["detection"].session.get_providers()
    logger.info(f"Active ONNX Runtime providers: {active_providers}")
    if "DmlExecutionProvider" not in active_providers:
        logger.warning(
            "DmlExecutionProvider is not active - face detection is running "
            "on CPU. This is the most common cause of a laggy/delayed video "
            "feed. Check that onnxruntime-directml (not 'onnxruntime' or "
            "'onnxruntime-gpu') is installed, and that your GPU driver is "
            "up to date."
        )
except Exception:
    logger.exception("Could not determine active ONNX Runtime providers.")


# ==============================================================================
# 4. HELPER: CONTRAST ENHANCEMENT FOR OVERHEAD LIGHTING
# ==============================================================================
CLAHE_BRIGHTNESS_THRESHOLD = float(os.environ.get("CLAHE_BRIGHTNESS_THRESHOLD", "115"))


def enhance_backlit_face(img_bgr):
    """CLAHE histogram equalization to boost detail on dark/underexposed frames.

    Skips the (relatively expensive) LAB conversion + CLAHE pass when the
    frame is already reasonably well lit, since running it unconditionally
    on every live frame adds avoidable latency for no benefit on frames
    that don't need it. This function is used for both enrollment photos
    and live queries, so the same brightness rule applies to both -
    embeddings stay comparable either way."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if float(np.mean(l)) >= CLAHE_BRIGHTNESS_THRESHOLD:
        return img_bgr

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ==============================================================================
# 5. ENROLLMENT: PARSE `captures/<ID>/*.jpg` DYNAMICALLY
# ==============================================================================
def load_database_from_id_folders(captures_root):
    """Scans each ID folder inside 'captures' and builds ArcFace embeddings."""
    db = {}
    if not os.path.exists(captures_root):
        logger.error(f"Captures path '{captures_root}' does not exist!")
        return db

    person_ids = [d for d in os.listdir(captures_root) if os.path.isdir(os.path.join(captures_root, d))]
    logger.info(f"Found {len(person_ids)} ID folders inside 'captures'. Building embeddings...")

    for person_id in person_ids:
        folder_path = os.path.join(captures_root, person_id)
        img_paths = (
            glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]"))
            + glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]"))
            + glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
        )
        if not img_paths:
            logger.warning(f"ID '{person_id}': no images found in folder.")
            continue

        db[person_id] = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            enhanced = enhance_backlit_face(img)
            faces = face_engine.get(enhanced)
            if len(faces) == 0:
                faces = face_engine.get(img)  # Retry on raw frame if CLAHE fails

            if len(faces) > 0:
                db[person_id].append(faces[0].normed_embedding)
            else:
                logger.warning(f"ID '{person_id}': could not detect face in {os.path.basename(img_path)}")

        logger.info(f"Registered ID '{person_id}' with {len(db[person_id])}/{len(img_paths)} pose vectors.")

    return db


def build_embedding_matrix(db):
    """Flattens {person_id: [embeddings]} into a single (N, 512) matrix plus a
    parallel labels list, so matching is one matrix multiply instead of a
    nested Python loop. This is what keeps recognition 'immediate' as the
    enrolled headcount grows."""
    labels, vectors = [], []
    for person_id, emb_list in db.items():
        for emb in emb_list:
            labels.append(person_id)
            vectors.append(emb)
    if not vectors:
        return np.zeros((0, 512), dtype=np.float32), []
    return np.vstack(vectors).astype(np.float32), labels


def match_against_db(query_embedding):
    """Vectorized nearest-neighbor cosine-similarity search. Returns
    (person_id_or_'Not Enrolled', best_similarity) â€” equivalent semantics to
    'best match overall, accepted only if it clears the threshold'."""
    if emb_matrix.shape[0] == 0:
        return "Not Enrolled", -1.0
    sims = emb_matrix @ query_embedding
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= SIMILARITY_THRESHOLD:
        return emb_labels[best_idx], best_sim
    return "Not Enrolled", best_sim


face_db = load_database_from_id_folders(CAPTURES_DIR)
emb_matrix, emb_labels = build_embedding_matrix(face_db)
reload_lock = threading.Lock()  # guards face_db / emb_matrix / emb_labels swaps triggered by enrollment


# ==============================================================================
# 5b. ENROLLMENT HELPERS (quality + pose gating for guided capture)
# ==============================================================================
def decode_data_uri_image(data_uri):
    """Decodes a 'data:image/jpeg;base64,...' string (as sent by <canvas>.
    toDataURL() in the browser) into a BGR numpy frame."""
    if "," in data_uri:
        _, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def compute_blur_score(img_bgr):
    """Variance of the Laplacian - a standard, cheap sharpness proxy. Low
    variance means few sharp edges, i.e. a blurry/motion-smeared frame."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_head_pose(face):
    """Derives yaw/pitch ratios from the 5-point landmarks InsightFace
    already returns with every detection (left eye, right eye, nose, left
    mouth corner, right mouth corner) - no extra model or computation needed.

    yaw_ratio: how far the nose sits from the eye-midpoint, as a fraction of
    eye separation. ~0 = facing the camera; large |value| = turned to a side.

    pitch_ratio: where the nose sits vertically between the eye line and the
    mouth line, as a fraction. ~0.5 = centered/front-on; smaller = chin
    down / looking up; larger = chin up / looking down.
    """
    kps = face.kps
    eye_l, eye_r, nose, mouth_l, mouth_r = kps
    eye_center = (eye_l + eye_r) / 2.0
    mouth_center = (mouth_l + mouth_r) / 2.0
    eye_dist = float(np.linalg.norm(eye_r - eye_l)) or 1.0
    vert_span = float(mouth_center[1] - eye_center[1]) or 1.0

    yaw_ratio = float(nose[0] - eye_center[0]) / eye_dist
    pitch_ratio = float(nose[1] - eye_center[1]) / vert_span
    return yaw_ratio, pitch_ratio


def classify_pose(yaw_ratio, pitch_ratio):
    """Buckets a yaw/pitch reading into one of ENROLL_POSES, or None if it
    doesn't clearly match any of them (e.g. mid-turn). Front requires BOTH
    yaw and pitch to be centered; left/right/up/down each require the
    corresponding axis to clearly exceed its threshold."""
    if abs(yaw_ratio) <= ENROLL_YAW_FRONT_MAX and ENROLL_PITCH_FRONT_MIN <= pitch_ratio <= ENROLL_PITCH_FRONT_MAX:
        return "front"
    if yaw_ratio <= -ENROLL_YAW_SIDE_MIN:
        return "left"
    if yaw_ratio >= ENROLL_YAW_SIDE_MIN:
        return "right"
    if pitch_ratio <= ENROLL_PITCH_UP_MAX:
        return "up"
    if pitch_ratio >= ENROLL_PITCH_DOWN_MIN:
        return "down"
    return None


def sanitize_person_id(raw):
    safe = "".join(c for c in (raw or "").strip() if c.isalnum() or c in ("-", "_"))
    return safe[:64]


def count_existing_captures(person_id, pose):
    folder = os.path.join(CAPTURES_DIR, person_id)
    return len(glob.glob(os.path.join(folder, f"{pose}_*.jpg")))


def build_embeddings_for_person(person_id):
    """Re-runs ArcFace over every saved image for one person and returns the
    list of usable embeddings. Same logic as load_database_from_id_folders,
    factored out so a single enrollment doesn't require rescanning everyone
    else's folders too."""
    folder = os.path.join(CAPTURES_DIR, person_id)
    img_paths = sorted(
        glob.glob(os.path.join(folder, "*.[jJ][pP][gG]"))
        + glob.glob(os.path.join(folder, "*.[jJ][pP][eE][gG]"))
        + glob.glob(os.path.join(folder, "*.[pP][nN][gG]"))
    )
    embeddings = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        enhanced = enhance_backlit_face(img)
        faces = face_engine.get(enhanced)
        if len(faces) == 0:
            faces = face_engine.get(img)
        if len(faces) > 0:
            embeddings.append(faces[0].normed_embedding)
    return embeddings


enroll_lock = threading.Lock()  # serializes capture validation + file writes per request


# ==============================================================================
# 6. PERSISTENT ATTENDANCE STORAGE (SQLite â€” once-per-day-per-person)
# ==============================================================================
db_lock = threading.Lock()
db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        attendance_date TEXT NOT NULL,
        first_seen_time TEXT NOT NULL,
        similarity REAL NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(person_id, attendance_date)
    )
    """
)
db_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS movement_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        direction TEXT NOT NULL,
        event_date TEXT NOT NULL,
        event_time TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
)
db_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS employees (
        person_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        designation TEXT,
        phone TEXT,
        email TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """
)
db_conn.commit()


def log_movement(person_id, direction):
    """Records one entry/exit event. person_id may be 'Not Enrolled' - we
    still log the movement (useful for footfall counts) even when identity
    isn't known, it just won't be attributable to a specific person."""
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    now_iso = datetime.now().isoformat(timespec="seconds")
    with db_lock:
        db_conn.execute(
            "INSERT INTO movement_log (person_id, direction, event_date, event_time, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (person_id, direction, today, now_time, now_iso),
        )
        db_conn.commit()


def get_movements_for_date(date_str):
    with db_lock:
        rows = db_conn.execute(
            """
            SELECT m.person_id, m.direction, m.event_time, e.name, e.department
            FROM movement_log m
            LEFT JOIN employees e ON e.person_id = m.person_id
            WHERE m.event_date = ?
            ORDER BY m.event_time ASC
            """,
            (date_str,),
        ).fetchall()
    return [
        {"id": r[0], "direction": r[1], "time": r[2], "name": r[3] or r[0], "department": r[4] or ""}
        for r in rows
    ]


def mark_attendance(person_id, similarity):
    """Inserts a first-sighting-of-the-day record. Returns True if this call
    created a new record (i.e. this is the first time today this person was
    seen), False if they were already marked present today."""
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    now_iso = datetime.now().isoformat(timespec="seconds")
    with db_lock:
        cur = db_conn.execute(
            "INSERT OR IGNORE INTO attendance "
            "(person_id, attendance_date, first_seen_time, similarity, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (person_id, today, now_time, float(similarity), now_iso),
        )
        db_conn.commit()
        return cur.rowcount == 1


def get_attendance_for_date(date_str):
    with db_lock:
        rows = db_conn.execute(
            """
            SELECT a.person_id, a.first_seen_time, a.similarity, e.name, e.department, e.designation
            FROM attendance a
            LEFT JOIN employees e ON e.person_id = a.person_id
            WHERE a.attendance_date = ?
            ORDER BY a.first_seen_time ASC
            """,
            (date_str,),
        ).fetchall()
    return [
        {
            "id": r[0], "time": r[1], "similarity": round(r[2], 3),
            "name": r[3] or r[0], "department": r[4] or "", "designation": r[5] or "",
        }
        for r in rows
    ]


def save_attendance_snapshot(person_id, frame_bgr):
    """Saves the exact frame a person was first marked present on, as an
    audit trail. Cheap and best-effort - a failure here should never block
    attendance being marked, so it's wrapped defensively."""
    try:
        day_dir = os.path.join(SNAPSHOTS_DIR, datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        safe_id = "".join(c for c in str(person_id) if c.isalnum() or c in ("-", "_")) or "unknown"
        fname = f"{safe_id}_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(day_dir, fname), frame_bgr)
    except Exception:
        logger.exception(f"Failed to save attendance snapshot for '{person_id}'")


# --- Employee details (name/department/designation shown alongside a match) -
# Kept separate from face_db/emb_matrix (which are purely about recognition)
# so enrollment can save "who this ID belongs to" independently of "what
# their face looks like". Cached in memory since annotate_frame() looks this
# up on every processed frame and a SQLite round trip per frame would add
# needless latency to something that changes rarely.
employee_lock = threading.Lock()


def upsert_employee(person_id, name, department="", designation="", phone="", email=""):
    now_iso = datetime.now().isoformat(timespec="seconds")
    with db_lock:
        db_conn.execute(
            """
            INSERT INTO employees (person_id, name, department, designation, phone, email, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                name=excluded.name, department=excluded.department, designation=excluded.designation,
                phone=excluded.phone, email=excluded.email, updated_at=excluded.updated_at
            """,
            (person_id, name, department, designation, phone, email, now_iso, now_iso),
        )
        db_conn.commit()

    record = {"name": name, "department": department, "designation": designation, "phone": phone, "email": email}
    with employee_lock:
        employee_cache[person_id] = record
    return record


def get_employee(person_id):
    with employee_lock:
        return employee_cache.get(person_id)


def load_employee_cache():
    with db_lock:
        rows = db_conn.execute(
            "SELECT person_id, name, department, designation, phone, email FROM employees"
        ).fetchall()
    return {
        r[0]: {"name": r[1], "department": r[2] or "", "designation": r[3] or "", "phone": r[4] or "", "email": r[5] or ""}
        for r in rows
    }


employee_cache = load_employee_cache()


def get_all_employee_details():

    with employee_lock:
        emp_snapshot = dict(employee_cache)

    with db_lock:
        rows = db_conn.execute("SELECT person_id, created_at FROM employees").fetchall()
    created_map = {r[0]: r[1] for r in rows}

    all_ids = set(emp_snapshot.keys()) | set(face_db.keys())

    details = []
    for person_id in sorted(all_ids):
        emp = emp_snapshot.get(person_id, {})
        pose_counts = {pose: count_existing_captures(person_id, pose) for pose in ENROLL_POSES}
        details.append({
            "person_id": person_id,
            "name": emp.get("name") or person_id,
            "department": emp.get("department", ""),
            "designation": emp.get("designation", ""),
            "phone": emp.get("phone", ""),
            "email": emp.get("email", ""),
            "created_at": created_map.get(person_id, ""),
            "pose_counts": pose_counts,
            "total_captured": sum(pose_counts.values()),
            "embeddings_count": len(face_db.get(person_id, [])),
            "thumbnail": get_thumbnail_data_uri(person_id),
        })
    return details


# ==============================================================================
# 7. SHARED STATE (updated by the capture thread, read by Flask routes)
# ==============================================================================
state_lock = threading.Lock()
latest_jpeg = None        # most recent annotated frame, JPEG-encoded bytes
match_history = []        # list of dicts shown in the sidebar (most recent first)
last_seen_at = {}          # dedupe_key -> unix timestamp, for cooldown


def get_thumbnail_data_uri(person_id):
    """Returns a base64 data URI for a person's reference photo (for the sidebar)."""
    folder = os.path.join(CAPTURES_DIR, person_id)
    preferred = [os.path.join(folder, f"center.{ext}") for ext in ("jpg", "jpeg", "png")]
    candidates = [p for p in preferred if os.path.exists(p)] or glob.glob(os.path.join(folder, "*"))
    if not candidates:
        return None
    with open(candidates[0], "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def encode_crop_data_uri(face_crop_bgr):
    """Encodes a cropped face image (numpy array) into a JPEG data URI, used
    as the sidebar thumbnail for faces that were detected but not enrolled."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None
    ok, buffer = cv2.imencode(".jpg", face_crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def record_match(person_id, similarity, status, face_crop_bgr=None, dedupe_key=None):
    """Appends a recognition event to the live sidebar feed, respecting a
    per-identity cooldown so a flickering track doesn't spam the feed.

    status is one of "present" (first sighting today â€” attendance just
    marked), "already_marked" (recognized again, already marked earlier
    today), "unknown" (face detected but not in the DB), "entry" or "exit"
    (a completed movement/direction event).

    dedupe_key controls the cooldown bucket. Known people are deduped by
    their person_id. Unknown faces are deduped by a caller-supplied unique
    key (e.g. the track id) instead of the literal string "Not Enrolled" â€”
    otherwise two *different* unrecognized people appearing within the
    cooldown window would collide on the same key and the second one would
    silently be dropped from the feed.
    """
    dedupe_key = dedupe_key or person_id
    now = time.time()
    if now - last_seen_at.get(dedupe_key, 0) < MATCH_COOLDOWN_SECONDS:
        return
    last_seen_at[dedupe_key] = now

    is_enrolled = person_id != "Not Enrolled"
    thumb = get_thumbnail_data_uri(person_id) if is_enrolled else encode_crop_data_uri(face_crop_bgr)
    emp = get_employee(person_id) if is_enrolled else None

    entry = {
        "id": person_id,
        "name": (emp["name"] if emp else None) or person_id,
        "department": emp["department"] if emp else "",
        "designation": emp["designation"] if emp else "",
        "similarity": round(float(similarity), 3),
        "time": datetime.now().strftime("%H:%M:%S"),
        "thumb": thumb,
        "status": status,
    }

    with state_lock:
        match_history.insert(0, entry)
        del match_history[MATCH_HISTORY_SIZE:]


# ==============================================================================
# 8. LIGHTWEIGHT FACE TRACKING
# ==============================================================================
# Detection still runs on every frame (so the box follows the person smoothly
# as they walk), but once a face has been matched against the database, its
# identity is "locked" onto a track. As long as that same face keeps showing
# up frame-to-frame (matched by bounding-box overlap), we just reuse the
# locked label instead of re-running the database matching / re-triggering a
# history entry. The track expires (and a fresh recognition happens) only
# after the face has been missing for TRACK_TIMEOUT_SECONDS - i.e. the person
# actually left the frame.
active_tracks = {}     # track_id -> {bbox, label, similarity, last_seen, ...}
next_track_id = 0


def compute_iou(box_a, box_b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / float(area_a + area_b - inter_area)


def compute_bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_bbox_centroid_y(box):
    x1, y1, x2, y2 = box
    return (y1 + y2) / 2.0


def compute_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_bbox_diag(box):
    """Diagonal length of a box, used to scale how far a centroid is
    'allowed' to move between frames relative to that box's own size -
    a small face near the door can move several times its own width
    between two processed frames without it meaning a different person,
    while the same absolute pixel movement would be huge for a box that's
    already large/close to the camera. Floored at 1.0 to avoid div-by-zero
    on a degenerate (zero-area) box."""
    x1, y1, x2, y2 = box
    return max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


def determine_direction(size_history, position_history):
    if len(position_history) < DIRECTION_MIN_SAMPLES or len(size_history) < DIRECTION_MIN_SAMPLES:
        return None

    n = min(DIRECTION_EDGE_SAMPLES, len(position_history) // 2) or 1

    start_y = sum(y for _, y in position_history[:n]) / n
    end_y = sum(y for _, y in position_history[-n:]) / n

    # PRIMARY SIGNAL: did the person's tracked position actually cross
    # DOOR_LINE_Y? This works regardless of the angle someone walks at -
    # unlike the size-based check below, it doesn't require the face to
    # visibly grow/shrink, so it also catches people who approach at an
    # angle or walk more laterally than head-on toward the camera.
    if start_y < DOOR_LINE_Y <= end_y:
        return "ENTERING"
    if start_y > DOOR_LINE_Y >= end_y:
        return "EXITING"

    # FALLBACK: the track never crossed the line at all - e.g. the person
    # was first detected already past it (track started mid-walk after a
    # brief re-detection gap), or the camera framing doesn't put them on
    # both sides during their visible time in frame. In that case, fall
    # back to the old size+position trend as a secondary heuristic so
    # these cases aren't silently dropped outright. Both signals (moving
    # further into frame AND growing bigger, or the reverse) must still
    # agree here, since without an actual line crossing this is a weaker
    # signal and needs the extra confirmation to avoid false positives.
    start_area = sum(a for _, a in size_history[:n]) / n
    end_area = sum(a for _, a in size_history[-n:]) / n
    if start_area <= 0:
        return None

    y_delta = end_y - start_y
    area_fractional_change = (end_area - start_area) / start_area

    if y_delta > 0 and area_fractional_change >= DIRECTION_TREND_THRESHOLD:
        return "ENTERING"
    if y_delta < 0 and area_fractional_change <= -DIRECTION_TREND_THRESHOLD:
        return "EXITING"
    return None


DIRECTION_TREND_THRESHOLD = float(os.environ.get("DIRECTION_TREND_THRESHOLD", "0.15"))

# Real-time crossing detection (see _update_track_side_and_maybe_log below).
# A small dead-zone straddling DOOR_LINE_Y so a person hovering right at the
# line doesn't get classified as flip-flopping sides from ordinary per-frame
# jitter - they must move clearly past the line (by this many pixels) before
# a side commits, in either direction.
DIRECTION_LINE_HYSTERESIS = int(os.environ.get("DIRECTION_LINE_HYSTERESIS", "15"))


def classify_side(y):
    """Buckets a (smoothed) vertical position into 'above' (door side) or
    'below' (interior/desk side) of DOOR_LINE_Y, or None if it's still
    inside the hysteresis dead-zone and not confidently on either side yet."""
    if y < DOOR_LINE_Y - DIRECTION_LINE_HYSTERESIS:
        return "above"
    if y > DOOR_LINE_Y + DIRECTION_LINE_HYSTERESIS:
        return "below"
    return None


def _update_track_side_and_maybe_log(track, tid):
    """Runs every frame a track is updated (not just when it expires) and
    fires the entry/exit event the instant a person's smoothed position
    actually crosses DOOR_LINE_Y. This replaces waiting for the track to
    time out (TRACK_TIMEOUT_SECONDS, 4s) before deciding a direction, which
    was adding a several-second delay - often much longer if the person
    lingered anywhere in frame afterward, since every additional sighting
    reset that timeout clock. Smoothing over the last few samples (instead
    of the single newest one) keeps a single noisy detection from firing a
    false crossing."""
    if track.get("movement_fired"):
        return
    position_history = track["position_history"]
    if not position_history:
        return

    n = min(DIRECTION_EDGE_SAMPLES, len(position_history))
    smoothed_y = sum(y for _, y in position_history[-n:]) / n
    side = classify_side(smoothed_y)
    if side is None:
        return  # still in the dead-zone, not confident enough to commit yet

    known_side = track.get("known_side")
    if known_side is None:
        track["known_side"] = side
        return

    if side != known_side:
        direction = "ENTERING" if side == "below" else "EXITING"
        track["movement_fired"] = True
        log_movement(track["label"], direction)
        record_match(
            track["label"], track["similarity"],
            "entry" if direction == "ENTERING" else "exit",
            dedupe_key=f"movement-{tid}",
        )


# ==============================================================================
# 9. DETECTION + RECOGNITION (detection runs every frame, matching is locked
#    per-track so the same person isn't re-recognized on every frame)
# ==============================================================================
def annotate_frame(frame):
    global next_track_id

    # Detection/embedding runs on a CLAHE-enhanced copy so live queries are
    # computed the same way as the enrollment reference photos were (see
    # load_database_from_id_folders). Drawing still happens on the original
    # `frame` further down, so the displayed video quality is untouched.
    detected_faces = face_engine.get(enhance_backlit_face(frame))
    now = time.time()

    # Draw the calibrated door line for visual reference / calibration.
    cv2.line(frame, (0, DOOR_LINE_Y), (frame.shape[1], DOOR_LINE_Y), (255, 180, 0), 1)

    detections = []
    for face in detected_faces:
        bbox = face.bbox.astype(int)
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
        detections.append({
            "box": (x1, y1, x2, y2),
            "embedding": face.normed_embedding,
        })

    # ---- GLOBAL, FRAME-WIDE track assignment -------------------------------
    # Score every (detection, active track) pair ONCE, then resolve all of
    # them together, strongest match first - instead of the old approach of
    # walking detected_faces one at a time and letting each one immediately
    # claim whichever track looked good enough. That per-detection ordering
    # is what let two people who cross paths (or simply sit/stand near each
    # other) accidentally SWAP identities: if the detector happened to
    # return the "wrong" face first, it could claim a track that was
    # obviously a much better match for someone else, locking that person
    # onto the wrong enrolled identity for the rest of their time in frame.
    # Scoring and assigning the whole frame together means the single best
    # pairing always wins first, so two nearby tracks can no longer steal
    # each other's identity purely because of processing order.
    candidate_pairs = []  # (score, detection_idx, track_id)
    for i, det in enumerate(detections):
        det_centroid = compute_centroid(det["box"])
        for tid, track in active_tracks.items():
            iou = compute_iou(det["box"], track["bbox"])
            track_centroid = compute_centroid(track["bbox"])
            dist = ((det_centroid[0] - track_centroid[0]) ** 2 +
                    (det_centroid[1] - track_centroid[1]) ** 2) ** 0.5
            centroid_ratio = dist / compute_bbox_diag(track["bbox"])
            embed_sim = float(np.dot(det["embedding"], track["anchor_embedding"]))

            valid = (
                iou >= TRACK_IOU_THRESHOLD
                or centroid_ratio <= TRACK_CENTROID_DIST_RATIO
                or embed_sim >= TRACK_EMBEDDING_REASSOC_THRESHOLD
            )
            if not valid:
                continue

            # Weighted so a strong IOU or embedding match dominates a weak
            # geometric one - this is only used to ORDER candidate pairs for
            # greedy assignment below, not as a hard pass/fail gate (that's
            # what `valid` above is for).
            score = iou * 3.0 + embed_sim * 1.5 + max(0.0, 1.0 - centroid_ratio)
            candidate_pairs.append((score, i, tid))

    candidate_pairs.sort(key=lambda p: p[0], reverse=True)

    assigned_track_for_detection = {}
    used_track_ids = set()
    used_detection_idxs = set()
    for score, i, tid in candidate_pairs:
        if i in used_detection_idxs or tid in used_track_ids:
            continue
        assigned_track_for_detection[i] = tid
        used_detection_idxs.add(i)
        used_track_ids.add(tid)
    # -------------------------------------------------------------------------

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        current_box = det["box"]
        query_embedding = det["embedding"]
        best_track_id = assigned_track_for_detection.get(i)

        if best_track_id is not None:
            # Same face as a previous frame -> reuse the locked identity,
            # skip database matching and skip re-triggering a history entry.
            track = active_tracks[best_track_id]
            track["bbox"] = current_box
            track["last_seen"] = now
            track["anchor_embedding"] = query_embedding  # keep the reference fresh
            track["size_history"].append((now, compute_bbox_area(current_box)))
            track["position_history"].append((now, compute_bbox_centroid_y(current_box)))
            del track["size_history"][:-DIRECTION_HISTORY_SIZE]
            del track["position_history"][:-DIRECTION_HISTORY_SIZE]
            matched_id = track["label"]
            highest_sim = track["similarity"]
            _update_track_side_and_maybe_log(track, best_track_id)

            # If this track has never matched anyone (e.g. the very first
            # frame it appeared on was a bad angle / motion blur), don't
            # leave it stuck as "Not Enrolled" for its entire time in frame
            # -> periodically retry the DB match with each new frame's fresh
            # embedding, so a better-angled later frame still gets matched.
            if matched_id == "Not Enrolled" and now - track.get("last_match_attempt", 0) > REMATCH_INTERVAL_SECONDS:
                track["last_match_attempt"] = now
                retry_id, retry_sim = match_against_db(query_embedding)
                if retry_id != "Not Enrolled":
                    matched_id, highest_sim = retry_id, retry_sim
                    track["label"], track["similarity"] = matched_id, highest_sim
                    is_new_today = mark_attendance(matched_id, highest_sim)
                    status = "present" if is_new_today else "already_marked"
                    record_match(matched_id, highest_sim, status, dedupe_key=matched_id)
                    if is_new_today:
                        save_attendance_snapshot(matched_id, frame)
        else:
            # Genuinely new face (either just walked in, or the previous track
            # timed out) -> run the full database match once, immediately.
            matched_id, highest_sim = match_against_db(query_embedding)

            best_track_id = next_track_id
            next_track_id += 1

            if matched_id != "Not Enrolled":
                is_new_today = mark_attendance(matched_id, highest_sim)
                status = "present" if is_new_today else "already_marked"
                record_match(matched_id, highest_sim, status, dedupe_key=matched_id)
                if is_new_today:
                    save_attendance_snapshot(matched_id, frame)
            else:
                # Crop the face BEFORE drawing the box/label, so the sidebar
                # thumbnail for not-enrolled faces stays clean (no overlay).
                face_crop = frame[y1:y2, x1:x2].copy()
                record_match(
                    matched_id, highest_sim, "unknown", face_crop,
                    dedupe_key=f"unknown-{best_track_id}",
                )

            active_tracks[best_track_id] = {
                "bbox": current_box,
                "label": matched_id,
                "similarity": highest_sim,
                "last_seen": now,
                "anchor_embedding": query_embedding,
                "size_history": [(now, compute_bbox_area(current_box))],
                "position_history": [(now, compute_bbox_centroid_y(current_box))],
            }
            _update_track_side_and_maybe_log(active_tracks[best_track_id], best_track_id)

        color = (0, 255, 0) if matched_id != "Not Enrolled" else (0, 0, 255)
        emp = get_employee(matched_id) if matched_id != "Not Enrolled" else None
        display_name = emp["name"] if emp and emp.get("name") else matched_id
        label = f"{display_name} ({highest_sim:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Drop tracks that haven't been seen in a while -> person has left the
    # frame, so next time a face appears there it will be recognized fresh.
    # Direction is now normally decided in real time as it happens (see
    # _update_track_side_and_maybe_log, called every frame above) rather
    # than here - waiting until the track times out to decide added a
    # multi-second delay between the actual crossing and the event showing
    # up. This end-of-track pass is now only a FALLBACK, for the minority
    # of tracks that ended without ever confidently firing in real time
    # (e.g. they hovered inside the DOOR_LINE_Y hysteresis band the whole
    # time, or a couple of frames got skipped right as they crossed).
    for tid in list(active_tracks.keys()):
        track = active_tracks[tid]
        if now - track["last_seen"] > TRACK_TIMEOUT_SECONDS:
            if track.get("movement_fired"):
                del active_tracks[tid]
                continue

            direction = determine_direction(track["size_history"], track["position_history"])
            if direction is not None:
                log_movement(track["label"], direction)
                record_match(
                    track["label"], track["similarity"],
                    "entry" if direction == "ENTERING" else "exit",
                    dedupe_key=f"movement-{tid}",
                )
            else:
                # No direction was determined - log why, so misses show up
                # in the log instead of just silently never appearing in
                # the movement table. samples < DIRECTION_MIN_SAMPLES means
                # the track was too short-lived (fast walker / brief
                # detection). Otherwise it means the track never crossed
                # DOOR_LINE_Y and the size+position trend wasn't clear
                # enough either - often a lateral path, or DOOR_LINE_Y not
                # yet calibrated to this camera's geometry.
                num_samples = len(track["position_history"])
                if num_samples < DIRECTION_MIN_SAMPLES:
                    logger.info(
                        f"Track for '{track['label']}' ended with only "
                        f"{num_samples} samples (< DIRECTION_MIN_SAMPLES="
                        f"{DIRECTION_MIN_SAMPLES}) - too brief to call a "
                        f"direction."
                    )
                else:
                    logger.info(
                        f"Track for '{track['label']}' ended with "
                        f"{num_samples} samples but no clear direction - "
                        f"didn't cross DOOR_LINE_Y={DOOR_LINE_Y} and the "
                        f"size/position trend wasn't decisive."
                    )
            del active_tracks[tid]

    return frame


# ==============================================================================
# 10. TWO DECOUPLED THREADS: grab frames as fast as possible, process separately
# ==============================================================================
# Why two threads: if reading the camera and running detection/recognition
# happen in one serial loop, the camera's internal buffer backs up while
# detection is busy, and the stream falls further and further behind.
# Splitting them means the grabber never waits on detection â€” it always
# holds only the single newest frame â€” while the processor always works on
# the freshest frame available. No frame is skipped from *recognition*: the
# processor still runs full detection + full ArcFace matching, at full
# det_size, on every frame it processes. This removes the lag without
# touching accuracy at all.
latest_raw_frame = None      # newest frame straight from the camera, unprocessed
raw_frame_lock = threading.Lock()
capture_running = True


def run_forever(name, target):
    """Runs `target()` in a loop, catching and logging any exception instead
    of letting it silently kill the thread. Without this, one bad frame or a
    transient OpenCV error would freeze the video feed forever with no error
    visible anywhere."""
    while capture_running:
        try:
            target()
        except Exception:
            logger.exception(f"[{name}] crashed, restarting in 2s")
            time.sleep(2)


def grab_loop():
    """Continuously reads from the camera/RTSP source, always keeping only
    the newest frame. Reconnects automatically (with exponential backoff) if
    the stream drops, instead of freezing forever."""
    global latest_raw_frame

    source = build_video_source()
    reconnect_delay = RECONNECT_MIN_DELAY

    while capture_running:
        logger.info(f"Connecting to video source: {source if USE_WEBCAM else '<rtsp stream>'}")
        # Explicitly request the FFMPEG backend for RTSP so the
        # OPENCV_FFMPEG_CAPTURE_OPTIONS low-latency flags set above are
        # actually honored - letting OpenCV auto-pick a backend can silently
        # ignore them on some Windows builds. Webcams don't go through
        # FFMPEG, so leave those on the default backend.
        cap = cv2.VideoCapture(source) if USE_WEBCAM else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep the OS/driver buffer as small as possible too

        if not cap.isOpened():
            logger.error(f"Failed to open video source, retrying in {reconnect_delay}s")
            cap.release()
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RECONNECT_MAX_DELAY)
            continue

        logger.info("Video source connected.")
        reconnect_delay = RECONNECT_MIN_DELAY
        consecutive_failures = 0

        while capture_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > MAX_CONSECUTIVE_READ_FAILURES:
                    logger.warning("Too many failed reads in a row, reconnecting...")
                    break
                time.sleep(0.02)
                continue

            consecutive_failures = 0
            frame = maybe_downscale(frame)
            with raw_frame_lock:
                latest_raw_frame = frame

        cap.release()
        if capture_running:
            time.sleep(reconnect_delay)


def process_loop():
    """Continuously runs full detection + recognition on whatever the newest frame is."""
    global latest_jpeg

    last_processed_id = None  # avoid re-processing the exact same frame object twice
    last_process_time = 0.0

    while capture_running:
        if PROCESS_MIN_INTERVAL > 0:
            elapsed = time.time() - last_process_time
            remaining = PROCESS_MIN_INTERVAL - elapsed
            if remaining > 0:
                time.sleep(remaining)

        with raw_frame_lock:
            frame = latest_raw_frame

        if frame is None:
            time.sleep(0.02)
            continue

        frame_id = id(frame)
        if frame_id == last_processed_id:
            time.sleep(0.005)
            continue
        last_processed_id = frame_id

        annotated = annotate_frame(frame.copy())
        last_process_time = time.time()

        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with state_lock:
                latest_jpeg = buffer.tobytes()


def mjpeg_generator():
    while True:
        with state_lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


# ==============================================================================
# 11. ROUTES
# ==============================================================================
@flask_app.route("/")
@require_auth
def index():
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = len(get_attendance_for_date(today))
    return render_template(
        "index.html",
        threshold=SIMILARITY_THRESHOLD,
        enrolled_count=len(face_db),
        enrolled_ids=sorted(face_db.keys()),
        source_label="Webcam" if USE_WEBCAM else f"{RTSP_HOST}:{RTSP_PORT}",
        today=today,
        today_count=today_count,
    )


@flask_app.route("/video_feed")
@require_auth
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.route("/api/matches")
@require_auth
def api_matches():
    with state_lock:
        return jsonify(list(match_history))


@flask_app.route("/api/attendance")
@require_auth
def api_attendance():
    date_str = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    return jsonify({"date": date_str, "records": get_attendance_for_date(date_str)})


@flask_app.route("/api/movements")
@require_auth
def api_movements():
    date_str = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    return jsonify({"date": date_str, "records": get_movements_for_date(date_str)})


@flask_app.route("/admin")
@require_auth
def admin_page():
    return render_template("admin.html", captures_per_pose=ENROLL_CAPTURES_PER_POSE)


@flask_app.route("/api/employees")
@require_auth
def api_employees():
    return jsonify(employees=get_all_employee_details())


@flask_app.route("/enroll")
@require_auth
def enroll_page():
    return render_template(
        "enroll.html",
        poses=ENROLL_POSES,
        captures_per_pose=ENROLL_CAPTURES_PER_POSE,
    )


@flask_app.route("/api/enroll/details", methods=["POST"])
@require_auth
def api_enroll_details():
    """Saves the human-readable identity (name/department/designation) tied
    to a person_id. Called once at the start of enrollment, before any face
    capture, so the ID is never just a bare folder name - everywhere a match
    is shown (bounding box, sidebar, attendance/movement tables) can show
    the person's actual name."""
    payload = request.get_json(force=True, silent=True) or {}
    person_id = sanitize_person_id(payload.get("person_id"))
    name = (payload.get("name") or "").strip()
    department = (payload.get("department") or "").strip()
    designation = (payload.get("designation") or "").strip()
    phone = (payload.get("phone") or "").strip()
    email = (payload.get("email") or "").strip()

    if not person_id:
        return jsonify(ok=False, reason="Missing/invalid person ID."), 400
    if not name:
        return jsonify(ok=False, reason="Name is required."), 400

    record = upsert_employee(person_id, name, department, designation, phone, email)
    return jsonify(ok=True, person_id=person_id, **record)


@flask_app.route("/api/enroll/status")
@require_auth
def api_enroll_status():
    """Lets the enrollment page resume where it left off if the browser was
    refreshed or the phone lost the connection mid-session."""
    person_id = sanitize_person_id(request.args.get("person_id", ""))
    if not person_id:
        return jsonify(ok=False, reason="Missing person ID."), 400
    counts = {pose: count_existing_captures(person_id, pose) for pose in ENROLL_POSES}
    return jsonify(ok=True, person_id=person_id, counts=counts, per_pose_target=ENROLL_CAPTURES_PER_POSE)


@flask_app.route("/api/enroll/capture", methods=["POST"])
@require_auth
def api_enroll_capture():
    payload = request.get_json(force=True, silent=True) or {}
    person_id = sanitize_person_id(payload.get("person_id"))
    pose = payload.get("pose")
    image_data = payload.get("image")

    if not person_id:
        return jsonify(ok=False, reason="Missing/invalid person ID."), 400
    if pose not in ENROLL_POSES:
        return jsonify(ok=False, reason="Invalid pose."), 400
    if not image_data:
        return jsonify(ok=False, reason="No image received."), 400

    try:
        frame = decode_data_uri_image(image_data)
    except Exception:
        frame = None
    if frame is None:
        return jsonify(ok=False, reason="Could not decode image."), 400

    with enroll_lock:
        existing = count_existing_captures(person_id, pose)
        if existing >= ENROLL_CAPTURES_PER_POSE:
            return jsonify(ok=True, already_complete=True, count=existing, total=ENROLL_CAPTURES_PER_POSE)

        cooldown_key = f"enroll:{person_id}:{pose}"
        now = time.time()
        if now - last_seen_at.get(cooldown_key, 0) < ENROLL_CAPTURE_COOLDOWN:
            return jsonify(ok=False, reason="cooldown", count=existing, total=ENROLL_CAPTURES_PER_POSE)

        blur_score = compute_blur_score(frame)
        if blur_score < ENROLL_BLUR_THRESHOLD:
            return jsonify(
                ok=False, reason=f"Too blurry (score {blur_score:.0f}) - hold steady.",
                count=existing, total=ENROLL_CAPTURES_PER_POSE,
            )

        faces = face_engine.get(enhance_backlit_face(frame))
        if len(faces) == 0:
            return jsonify(ok=False, reason="No face detected.", count=existing, total=ENROLL_CAPTURES_PER_POSE)
        if len(faces) > 1:
            return jsonify(
                ok=False, reason="Multiple faces in frame - only the person enrolling should be visible.",
                count=existing, total=ENROLL_CAPTURES_PER_POSE,
            )

        face = faces[0]
        if float(face.det_score) < ENROLL_MIN_DET_SCORE:
            return jsonify(ok=False, reason="Face not clear enough.", count=existing, total=ENROLL_CAPTURES_PER_POSE)

        bbox = face.bbox
        face_w, face_h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
        if face_w < ENROLL_MIN_FACE_WIDTH or face_h < ENROLL_MIN_FACE_WIDTH:
            return jsonify(ok=False, reason="Move closer to the camera.", count=existing, total=ENROLL_CAPTURES_PER_POSE)

        yaw_ratio, pitch_ratio = estimate_head_pose(face)
        detected_pose = classify_pose(yaw_ratio, pitch_ratio)
        if detected_pose != pose:
            return jsonify(
                ok=False,
                reason=f"Pose mismatch - detected '{detected_pose or 'unclear'}', need '{pose}'.",
                yaw=round(yaw_ratio, 3), pitch=round(pitch_ratio, 3),
                count=existing, total=ENROLL_CAPTURES_PER_POSE,
            )

        folder = os.path.join(CAPTURES_DIR, person_id)
        os.makedirs(folder, exist_ok=True)
        new_count = existing + 1
        cv2.imwrite(os.path.join(folder, f"{pose}_{new_count:02d}.jpg"), frame)
        last_seen_at[cooldown_key] = now

        return jsonify(
            ok=True, count=new_count, total=ENROLL_CAPTURES_PER_POSE,
            pose_complete=(new_count >= ENROLL_CAPTURES_PER_POSE),
            yaw=round(yaw_ratio, 3), pitch=round(pitch_ratio, 3),
        )


@flask_app.route("/api/enroll/reset", methods=["POST"])
@require_auth
def api_enroll_reset():
    """Deletes captured images for one pose (or all poses) so the person can
    retake them - e.g. if lighting was bad for the whole 'left' set."""
    payload = request.get_json(force=True, silent=True) or {}
    person_id = sanitize_person_id(payload.get("person_id"))
    pose = payload.get("pose")
    if not person_id:
        return jsonify(ok=False, reason="Missing person ID."), 400

    folder = os.path.join(CAPTURES_DIR, person_id)
    if not os.path.isdir(folder):
        return jsonify(ok=True, cleared=0)

    poses_to_clear = [pose] if pose in ENROLL_POSES else ENROLL_POSES
    cleared = 0
    with enroll_lock:
        for p in poses_to_clear:
            for f in glob.glob(os.path.join(folder, f"{p}_*.jpg")):
                os.remove(f)
                cleared += 1
    return jsonify(ok=True, cleared=cleared)


@flask_app.route("/captures/<person_id>/<filename>")
@require_auth
def serve_capture_image(person_id, filename):
    """Serves one saved enrollment photo. Requires auth (same as the rest of
    the dashboard, since these are identifiable photos of employees) and
    guards against path traversal: both person_id and filename are
    re-sanitized/re-derived from the raw path components rather than trusted
    as-is, so a request like '/captures/../../secrets/x.jpg' can't escape
    CAPTURES_DIR."""
    safe_id = sanitize_person_id(person_id)
    safe_filename = os.path.basename(filename)
    if not safe_id or safe_id != person_id or not safe_filename or safe_filename != filename:
        return "Not found", 404

    folder = os.path.join(CAPTURES_DIR, safe_id)
    file_path = os.path.join(folder, safe_filename)
    if not os.path.isfile(file_path):
        return "Not found", 404

    return send_from_directory(folder, safe_filename)


@flask_app.route("/api/enroll/images")
@require_auth
def api_enroll_images():
    """Lists every captured photo for one person, grouped in pose order, so
    the enrollment-complete page can render a full gallery of exactly what
    was just captured (and later, the admin/dashboard could reuse this too)."""
    person_id = sanitize_person_id(request.args.get("person_id", ""))
    if not person_id:
        return jsonify(ok=False, reason="Missing person ID."), 400

    folder = os.path.join(CAPTURES_DIR, person_id)
    if not os.path.isdir(folder):
        return jsonify(ok=True, person_id=person_id, images=[])

    images = []
    for pose in ENROLL_POSES:
        for path in sorted(glob.glob(os.path.join(folder, f"{pose}_*.jpg"))):
            fname = os.path.basename(path)
            images.append({
                "pose": pose,
                "filename": fname,
                "url": f"/captures/{person_id}/{fname}",
            })

    return jsonify(ok=True, person_id=person_id, images=images, count=len(images))


@flask_app.route("/api/enroll/finish", methods=["POST"])
@require_auth
def api_enroll_finish():
    """Builds embeddings from every image captured for this person and
    merges them into the live matching database (emb_matrix/emb_labels)
    WITHOUT requiring an app restart, so a person enrolled just now can be
    recognized on the entrance camera immediately afterward."""
    payload = request.get_json(force=True, silent=True) or {}
    person_id = sanitize_person_id(payload.get("person_id"))
    if not person_id:
        return jsonify(ok=False, reason="Missing person ID."), 400

    counts = {pose: count_existing_captures(person_id, pose) for pose in ENROLL_POSES}
    total_captured = sum(counts.values())

    embeddings = build_embeddings_for_person(person_id)

    global face_db, emb_matrix, emb_labels
    with reload_lock:
        face_db[person_id] = embeddings
        new_matrix, new_labels = build_embedding_matrix(face_db)
        emb_matrix, emb_labels = new_matrix, new_labels

    logger.info(
        f"Enrolled '{person_id}': {len(embeddings)}/{total_captured} usable "
        f"embeddings built from {total_captured} captured images."
    )
    return jsonify(
        ok=True, person_id=person_id, counts=counts,
        total_captured=total_captured, embeddings_built=len(embeddings),
    )


# ==============================================================================
# 12. STARTUP / SHUTDOWN
# ==============================================================================
def validate_config():
    problems = []
    if not BASIC_AUTH_USER or not BASIC_AUTH_PASS:
        problems.append(
            "BASIC_AUTH_USER / BASIC_AUTH_PASS are not set. Refusing to start "
            "with an unauthenticated dashboard that shows employee identities. "
            "Set them in your .env file (see .env.example)."
        )
    if not USE_WEBCAM:
        try:
            build_video_source()
        except RuntimeError as e:
            problems.append(str(e))
    if not face_db:
        logger.warning("Face database is empty. Check your captures/ folder structure â€” "
                        "no one will be recognized until it's populated.")
    return problems


def shutdown(*_args):
    global capture_running
    logger.info("Shutting down...")
    capture_running = False
    time.sleep(0.3)
    try:
        db_conn.close()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    config_problems = validate_config()
    if config_problems:
        for p in config_problems:
            logger.error(p)
        sys.exit(1)

    threading.Thread(target=run_forever, args=("grab_loop", grab_loop), daemon=True).start()
    threading.Thread(target=run_forever, args=("process_loop", process_loop), daemon=True).start()

    port = int(os.environ.get("PORT", "8080"))

    try:
        from waitress import serve
        logger.info(f"Starting production server (waitress) on 0.0.0.0:{port}")
        serve(flask_app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        logger.warning(
            "waitress is not installed â€” falling back to Flask's development "
            "server, which is NOT suitable for production. Run "
            "'pip install waitress' to use the production server."
        )
        flask_app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
