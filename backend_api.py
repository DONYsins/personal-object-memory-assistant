# ============================
# backend_api.py
# ============================
import numpy as np
import torch, os, time, sqlite3, json, faiss, bcrypt, shutil, cv2, base64
import uuid  # used for generating session ids
import clip
from typing import Optional, List, Dict, Any
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from ultralytics import YOLO
from datetime import datetime, timedelta

# Import logging helpers (creates logs/SESSION_ID/ on import)
from logger import log_info, log_error, log_debug, SESSION_ID, SESSION_LOG_DIR

# Import all constants from centralized config file
# WHY: Single source of truth for PERSONAL_CLASSES, LANDMARK_CLASSES, thresholds, etc.
# Easier to manage and update across the system
from constants import (
    PERSONAL_CLASSES,
    LANDMARK_CLASSES,
    SIM_THRESHOLD,
    STORE_INTERVAL_SECONDS,
    DISAPPEAR_TICKS,
    BASE_DIR,
    YOLO_MODEL_PATH,
    MAX_OBJECTS_PER_USER,
    RETENTION_ACTIVE_HOURS,
    CLIP_MODEL,
    EMBEDDING_DIM
)

# ============================================================
# CONFIG (EDIT HERE)
# ============================================================

# Derived paths (using BASE_DIR from constants)
USERS_DIR = os.path.join(BASE_DIR, "users")  # Per-user FAISS indices and images
MAIN_DB = os.path.join(BASE_DIR, "main.sqlite")  # Central SQLite database
LOGS_DIR = SESSION_LOG_DIR  # Session-grouped logging directory

# ephemeral state for live sessions (TRACK, ENROLL_*)
# keyed by session_id; will hold objects state, inferred environment, etc.
SESSION_STATE: Dict[str, Any] = {}  # Live session data (TRACK/ENROLL states)

# ============================================================
# APP + MODELS
# ============================================================

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available, else CPU
log_info(f"[BACKEND] Using device: {DEVICE}")

yolo_model = YOLO(YOLO_MODEL_PATH)  # Custom trained model for 12 classes

clip_model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)  # For visual re-identification

ACTIVE_SECONDS = 0.0  # Track cumulative runtime for auto-cleanup
_last_active_tick = time.time()  # Last frame processing timestamp

# ============================================================
# DB INIT
# ============================================================

def db_conn():
    return sqlite3.connect(MAIN_DB, check_same_thread=False)

def init_db():
    # Ensure base directory exists before creating database
    os.makedirs(BASE_DIR, exist_ok=True)
    
    conn = db_conn()
    c = conn.cursor()

    # Users
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Personal objects (user-defined labels, multiple per type)
    c.execute("""
    CREATE TABLE IF NOT EXISTS personal_objects (
        user_object_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        generic_type TEXT NOT NULL,
        user_label TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    # unique label per user to prevent duplicates (user_id+user_label composite)
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_label ON personal_objects(user_id, user_label)")

    # Environments
    c.execute("""
    CREATE TABLE IF NOT EXISTS environments (
        environment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        environment_label TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    # make environment_label unique per user
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_envlabel ON environments(user_id, environment_label)")

    # Environment landmarks (what model detected + user custom label)
    c.execute("""
    CREATE TABLE IF NOT EXISTS environment_landmarks (
        env_landmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
        environment_id INTEGER NOT NULL,
        landmark_class TEXT NOT NULL,
        user_label TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Events (what was seen, where, when)
    # location_text stores the full human-readable description, e.g. "on white_chair (Bedroom)"
    c.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        user_object_id INTEGER,
        location_text TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        image_path TEXT NOT NULL
    )
    """)

    # Sessions table for live enrollment/tracking
    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        mode TEXT NOT NULL,
        user_object_id INTEGER,
        environment_id INTEGER,
        landmark_id INTEGER,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_error TEXT
    )
    """)

    # Migrate sessions table to include landmark_id if needed
    def _ensure_sessions_columns():
        c2 = conn.cursor()
        c2.execute("PRAGMA table_info(sessions)")
        cols = [r[1] for r in c2.fetchall()]
        if "landmark_id" not in cols:
            c2.execute("ALTER TABLE sessions ADD COLUMN landmark_id INTEGER")
        conn.commit()

    _ensure_sessions_columns()

    conn.commit()
    conn.close()

init_db()

# ============================================================
# FILE/FOLDER HELPERS
# ============================================================

def user_dir(user_id: int) -> str:
    """Create and return user's data directory structure."""
    p = os.path.join(USERS_DIR, str(user_id))
    os.makedirs(p, exist_ok=True)
    os.makedirs(os.path.join(p, "faiss", "object"), exist_ok=True)  # Object FAISS indices
    os.makedirs(os.path.join(p, "faiss", "environment"), exist_ok=True)  # Landmark FAISS indices
    os.makedirs(os.path.join(p, "images"), exist_ok=True)  # Detection event images
    return p

def faiss_path_for_object(user_id: int, user_object_id: int) -> str:
    """Path for object FAISS indices: {user_id}/faiss/object/{object_id}.index"""
    return os.path.join(user_dir(user_id), "faiss", "object", f"{user_object_id}.index")

def faiss_path_for_landmark(user_id: int, environment_id: int, landmark_id: int) -> str:
    """Path for landmark FAISS indices: {user_id}/faiss/environment/{env_id}/{landmark_id}.index"""
    env_dir = os.path.join(user_dir(user_id), "faiss", "environment", str(environment_id))
    os.makedirs(env_dir, exist_ok=True)
    return os.path.join(env_dir, f"{landmark_id}.index")

def ensure_faiss_index(path: str):
    """Create empty FAISS index if doesn't exist (inner product = cosine similarity)."""
    if os.path.exists(path):
        return
    base = faiss.IndexFlatIP(512)  # Inner product on L2-normalized 512-dim embeddings
    index = faiss.IndexIDMap2(base)  # Allows custom IDs for embeddings
    faiss.write_index(index, path)

# ============================================================
# VISION HELPERS
# ============================================================

def get_embedding_from_bbox(frame_bgr: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    """Extract 512-dim L2-normalized CLIP embedding from bounding box crop."""
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]  # Extract region of interest
    if crop.size == 0:
        return None

    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))  # BGR→RGB for CLIP
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)  # CLIP preprocessing pipeline

    with torch.no_grad():
        emb = clip_model.encode_image(img_t).float()  # Extract visual features
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2-normalize for cosine similarity

    return emb.cpu().numpy().astype("float32")  # shape (1,512)

def bbox_iou(a, b) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # Calculate intersection rectangle
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:  # No overlap
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)  # IoU formula

def bbox_center(b):
    """Return (cx, cy) center coordinates of bounding box."""
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def nearest_landmark(obj_bbox, landmarks: List[tuple]) -> Optional[str]:
    """Find closest landmark to object by Euclidean distance between centers."""
    ox, oy = bbox_center(obj_bbox)
    best = None
    best_d = 1e18  # Initialize with large value
    for lname, lbbox in landmarks:
        lx, ly = bbox_center(lbbox)
        d = (ox - lx) ** 2 + (oy - ly) ** 2  # Squared distance (no sqrt needed for comparison)
        if d < best_d:
            best_d = d
            best = lname
    return best

def infer_location(obj_bbox, landmarks: List[tuple]) -> Optional[str]:
    """Determine spatial relationship: 'on' if overlapping, 'near' if close proximity."""
    if not landmarks:  # No landmarks detected
        return None
    
    # Strategy 1: Check for physical overlap (object resting ON landmark)
    best_iou = 0.0
    best_name = None
    for lname, lbbox in landmarks:
        i = bbox_iou(obj_bbox, lbbox)
        log_debug(f"[LOCATION_IoU] vs '{lname}': iou={i:.3f} (threshold=0.05)")
        if i > best_iou:
            best_iou = i
            best_name = lname

    if best_name and best_iou > 0.25:  # Overlap threshold for 'on' relationship
        log_debug(f"[LOCATION] Overlap match -> 'on {best_name}' (iou={best_iou:.3f})")
        return f"on {best_name}"

    # Strategy 2: Find closest landmark by distance (object NEAR landmark)
    near = nearest_landmark(obj_bbox, landmarks)
    if near:
        log_debug(f"[LOCATION] Centroid match -> 'near {near}' (best_iou={best_iou:.3f} was below 0.05)")
        return f"near {near}"

    return None

# ============================================================
# TRACKING HELPERS (refactored for clarity)
# ============================================================

def detect_objects_in_frame(frame: np.ndarray) -> tuple:
    """
    Run YOLO inference and categorize detections into personal objects and landmarks.
    
    Returns: (personal_objects, landmarks) as lists of (class_label, bbox) tuples
    """
    try:
        r0 = yolo_model(frame, verbose=False)[0]  # YOLO detection
        boxes = r0.boxes
    except Exception as e:
        log_error(f"YOLO inference failed: {e}")
        raise

    personal = []  # Personal items (Watch, Wallet, etc.)
    landmarks = []  # Environment markers (Chair, Bed, etc.)
    for b in boxes:
        cls_id = int(b.cls[0])
        label = yolo_model.names[cls_id]  # YOLO class name
        bbox = tuple(map(int, b.xyxy[0].tolist()))  # (x1, y1, x2, y2)
        
        if label in PERSONAL_CLASSES:
            personal.append((label, bbox))
        if label in LANDMARK_CLASSES:
            landmarks.append((label, bbox))

    return personal, landmarks


def extract_embedding_and_match(
    frame: np.ndarray,
    bbox: tuple,
    user_id: int,
    candidates: List[tuple]
) -> tuple:
    """
    Extract CLIP embedding from bounding box and match against FAISS indices.
    
    WHY THIS FUNCTION:
      Core of multi-instance tracking. When we detect a "Watch", we need to figure out
      which watch it is (blue_watch? silver_watch? black_watch?). This function:
      1. Extracts a CLIP embedding (neural network features) from the detected region
      2. Compares against all candidate labels' FAISS indices
      3. Returns the best match
    
    ARGS:
        frame: BGR frame from camera
        bbox: (x1, y1, x2, y2) bounding box of detected object
        user_id: database user ID (to find FAISS indices)
        candidates: list of (user_object_id, user_label) - all labels of this object type
    
    RETURNS: (embedding, best_user_object_id, best_user_label, best_score)
        - embedding: CLIP feature vector for this detection
        - best_user_object_id: database ID of best-matching label (or None if no match)
        - best_user_label: user's custom label (e.g. "blue_watch")
        - best_score: similarity score [0, 1] from FAISS
    
    EXAMPLE:
        User has: blue_watch (obj_id=1), silver_watch (obj_id=2), black_watch (obj_id=3)
        Detected a watch in frame. Call:
          extract_embedding_and_match(frame, bbox, user_id=5, 
                                      candidates=[(1,"blue_watch"), (2,"silver_watch"), (3,"black_watch")])
        Returns: (embedding_vector, 1, "blue_watch", 0.89) - 89% match to blue_watch
    """
    # Step 1: Extract CLIP embedding from this detection's crop
    emb = get_embedding_from_bbox(frame, bbox)
    if emb is None:
        return None, None, None, -1.0

    best_obj_id = None
    best_obj_label = None
    best_score = -1.0

    # Step 2: Query each candidate's FAISS index to find best match
    for oid, lbl in candidates:
        idx_path = faiss_path_for_object(user_id, oid)
        if not os.path.exists(idx_path):
            continue
        
        try:
            index = faiss.read_index(idx_path)
            # Search for 1 nearest neighbor
            # FAISS returns: distances = similarity scores, indices = neighbor IDs
            # distances[0,0] = how similar this embedding is to the stored embeddings
            # (higher = more similar, since we use inner product distance)
            distances, indices = index.search(emb, 1)
            score = float(distances[0, 0]) if len(distances) > 0 else -1.0
            
            # DEBUGGING: Log similarity scores for each candidate
            log_debug(f"[FAISS_SCORE] '{lbl}' similarity={score:.3f} (threshold={SIM_THRESHOLD})")
            
            # If score < SIM_THRESHOLD, reject it
            if score > best_score and score >= SIM_THRESHOLD:
                best_score = score
                best_obj_id = oid
                best_obj_label = lbl
        except Exception as e:
            log_debug(f"[FAISS_SCORE] Error querying FAISS for object {oid}: {e}")

    return emb, best_obj_id, best_obj_label, best_score


def store_object_event(
    conn: sqlite3.Connection,
    user_id: int,
    user_object_id: int,
    location_text: str,
    timestamp: datetime,
    image: np.ndarray,
    object_type: str,
    object_label: str
) -> bool:
    """
    Store a detection event: save image and create database record.
    
    WHY THIS FUNCTION:
      When we detect a personal object (blue_watch found at 2:15 PM on table), we need to:
      1. Save the image to disk (for later review/visualization)
      2. Create a database record (for querying/searching later)
      3. Link it to the right user, object label, and location
    
    location_text already contains the full human-readable description including
    landmark context and environment name, e.g. "on white_chair (Bedroom)".
    No separate environment/landmark columns are needed.
    
    ARGS:
        conn: database connection (already open)
        user_id: database user ID
        user_object_id: which labeled object was detected (from personal_objects table)
        location_text: human-readable location ("on white_chair (Bedroom)", etc)
        timestamp: datetime of detection
        image: BGR frame from camera
        object_type: generic type ("Watch", "Wallet", "Sunglasses", etc) - used for folder organization
        object_label: user's custom label for this object ("blue_watch", "my_wallet")
    
    RETURNS: True if successful, False otherwise
    
    FILE ORGANIZATION:
      Saves to: reid_store/users/{user_id}/images/{object_type}/{object_label}_{TIMESTAMP}.jpg
      Example: reid_store/users/5/images/Watch/blue_watch_20240115_143022_123456.jpg
    """
    try:
        # Create folder: reid_store/users/{user_id}/images/{object_type}/
        user_folder = user_dir(user_id)
        obj_type_folder = os.path.join(user_folder, "images", object_type.replace(" ", "_"))
        os.makedirs(obj_type_folder, exist_ok=True)

        # Save image with timestamp for uniqueness and chronological ordering
        # Format: "{object_label}_{YYYYMMDD_HHMMSS_MICROSECONDS}.jpg"
        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(obj_type_folder, f"{object_label}_{ts}.jpg")
        cv2.imwrite(image_path, image)

        # Record event in database so we can query: "When/where did I last see blue_watch?"
        # location_text already contains the full context (landmark + environment), e.g. "on white_chair (Bedroom)"
        c = conn.cursor()
        c.execute("""
            INSERT INTO events(user_id, user_object_id, location_text, timestamp, image_path)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, user_object_id, location_text, timestamp.isoformat(), image_path))
        conn.commit()
        return True
    except Exception as e:
        log_error(f"Failed to store event: {e}")
        return False


def find_environment_from_landmarks(
    detected_landmarks: List[tuple],
    user_id: int
) -> tuple:
    """
    If ANY detected landmark class matches a registered landmark in one of the user's
    environments, return that environment.
    
    Example: user has Bedroom with Chair + Bed defined.
    Frame detects Chair → returns (env_id_of_bedroom, "Bedroom")
    """
    if not detected_landmarks:
        return None, None
    
    # Extract just the YOLO classes detected
    detected_classes = {class_label for class_label, bbox in detected_landmarks}
    log_debug(f"[ENV_MATCH] Detected landmark YOLO classes: {sorted(detected_classes)}")
    
    conn = db_conn()
    c = conn.cursor()
    
    # Get all user's environments
    c.execute("SELECT environment_id, environment_label FROM environments WHERE user_id=?", (user_id,))
    environments = c.fetchall()
    
    # For each environment, check if ANY registered landmark class matches detections
    for env_id, env_label in environments:
        c.execute("""
            SELECT landmark_class FROM environment_landmarks 
            WHERE environment_id = ?
        """, (env_id,))
        
        env_landmark_classes = {row[0] for row in c.fetchall()}  # Set of registered classes
        intersection = detected_classes & env_landmark_classes  # Class overlap
        log_debug(f"[ENV_MATCH] Env '{env_label}' registered: {sorted(env_landmark_classes)} | intersection: {sorted(intersection)}")
        
        # If ANY class matches, this is the environment
        if intersection:
            conn.close()
            log_debug(f"[ENV_MATCH] Matched env '{env_label}' (env_id={env_id})")
            return env_id, env_label
    
    log_debug(f"[ENV_MATCH] No environment matched for detected classes: {sorted(detected_classes)}")
    conn.close()
    return None, None


def get_location_text_for_event(
    object_bbox: tuple,
    detected_landmarks: List[tuple],
    environment_id: Optional[int],
    user_id: int
) -> str:
    """
    Compute human-readable location text for a tracking event.
    
    WHY: When storing "watch was detected", we want to say:
         "on my bed (Bedroom)" not "near Bed"
    
    This uses:
      1. Spatial proximity (is watch on/near a landmark?)
      2. User's custom landmark names from THIS environment
      3. Environment context
    
    RETURNS: "on white_chair (Bedroom)" or "near bed" or "Unknown"
    """
    # Step 1: Find closest landmark spatially
    closest_landmark_class = infer_location(object_bbox, detected_landmarks)
    
    if not closest_landmark_class:
        # No landmarks detected this frame
        log_debug(f"[LOCATION_TEXT] infer_location returned None -> storing 'Unknown'")
        return "Unknown"
    
    # Step 2: Get user's custom labels for landmarks in THIS environment
    if not environment_id:
        # No environment known, just use YOLO class name
        log_debug(f"[LOCATION_TEXT] No environment matched -> using raw YOLO result: '{closest_landmark_class}'")
        return closest_landmark_class
    
    conn = db_conn()
    c = conn.cursor()
    c.execute("""
        SELECT landmark_class, user_label 
        FROM environment_landmarks 
        WHERE environment_id = ?
    """, (environment_id,))
    
    # Map YOLO class → user's custom label (e.g., "Chair" → "white_chair")
    landmark_mapping = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    log_debug(f"[LOCATION_TEXT] landmark_mapping for env_id={environment_id}: {landmark_mapping}")
    
    # Step 3: Replace YOLO class with user's custom label
    for yolo_class, custom_label in landmark_mapping.items():
        if yolo_class in closest_landmark_class:
            closest_landmark_class = closest_landmark_class.replace(yolo_class, custom_label)
            log_debug(f"[LOCATION_TEXT] Replaced YOLO class '{yolo_class}' -> custom label '{custom_label}'")
            break
    
    # Step 4: Add environment context
    if environment_id:
        conn = db_conn()
        c = conn.cursor()
        c.execute("SELECT environment_label FROM environments WHERE environment_id = ?", (environment_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            env_label = row[0]
            result = f"{closest_landmark_class} ({env_label})"
            log_debug(f"[LOCATION_TEXT] Final location_text: '{result}'")
            return result
    
    return closest_landmark_class


def tick_active_runtime():
    """Update cumulative active runtime (used for auto-cleanup threshold)."""
    global ACTIVE_SECONDS, _last_active_tick
    now = time.time()
    dt = max(0.0, now - _last_active_tick)  # Time since last frame
    _last_active_tick = now
    ACTIVE_SECONDS += dt  # Accumulate total active time

def cleanup_if_needed(conn: sqlite3.Connection, user_id: int):
    # active runtime based cleanup
    active_hours = ACTIVE_SECONDS / 3600.0
    if active_hours < RETENTION_ACTIVE_HOURS:
        return

    # We only cleanup when the app has been actively running long enough,
    # but we delete based on actual timestamps within the DB (so it’s real data management).
    cutoff = datetime.now() - timedelta(hours=RETENTION_ACTIVE_HOURS)

    c = conn.cursor()
    c.execute("""
        SELECT event_id, image_path, timestamp FROM events
        WHERE user_id = ?
    """, (user_id,))
    rows = c.fetchall()

    to_delete = []
    for event_id, image_path, ts in rows:
        try:
            dt = datetime.fromisoformat(ts)
        except:
            continue
        if dt < cutoff:
            to_delete.append((event_id, image_path))

    if not to_delete:
        return

    log_info(f"[CLEANUP] Deleting {len(to_delete)} old events for user {user_id}")

    for event_id, image_path in to_delete:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            log_error(f"Failed to remove image {image_path}: {e}")

        c.execute("DELETE FROM events WHERE event_id = ?", (event_id,))

    conn.commit()

# ============================================================
# AUTH
# ============================================================

@app.post("/register_user")
def register_user(username: str = Form(...), password: str = Form(...)):
    # Log registration attempt (avoid logging plaintext password)
    log_info(f"Register attempt: username='{username}'")
    
    try:
        conn = db_conn()
        c = conn.cursor()
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        c.execute(
            "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, pw_hash, datetime.now().isoformat())
        )
        conn.commit()
        log_info(f"[BACKEND] Registered user: '{username}'")
        return {"status": "success"}
    except Exception as e:
        log_error(f"[BACKEND] Register failed for '{username}': {e}")
        return {"status": "error", "message": "Username already exists"}
    finally:
        try: conn.close()
        except: pass

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    # Log login attempt (avoid logging plaintext password)
    log_info(f"[BACKEND] Login attempt for username='{username}'")
    
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT user_id, password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()

    if not row:
        log_info(f"[BACKEND] Login failed - user not found: '{username}'")
        return {"status": "error", "message": "Invalid credentials"}

    user_id, pw_hash = row
    if bcrypt.checkpw(password.encode("utf-8"), pw_hash.encode("utf-8")):
        user_dir(user_id)  # ensure folders exist
        log_info(f"[BACKEND] Login successful for username='{username}', user_id={user_id}")
        return {"status": "success", "user_id": user_id}

    log_info(f"[BACKEND] Login failed - incorrect password for '{username}'")
    return {"status": "error", "message": "Invalid credentials"}

@app.get("/get_user_info")
def get_user_info(user_id: int):
    """Get user information (username, created_at, etc)."""
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT username, created_at FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return {"status": "error", "message": "User not found"}
    
    username, created_at = row
    return {"status": "success", "username": username, "created_at": created_at, "user_id": user_id}

# ============================================================
# USER DASHBOARD DATA
# ============================================================

@app.get("/dashboard")
def dashboard(user_id: int):
    conn = db_conn()
    c = conn.cursor()

    c.execute("SELECT user_object_id, generic_type, user_label FROM personal_objects WHERE user_id=?",
              (user_id,))
    objs = [{"user_object_id": r[0], "generic_type": r[1], "user_label": r[2]} for r in c.fetchall()]

    c.execute("SELECT environment_id, environment_label FROM environments WHERE user_id=?",
              (user_id,))
    envs = [{"environment_id": r[0], "environment_label": r[1]} for r in c.fetchall()]

    conn.close()
    return {"objects": objs, "environments": envs}

# ============================================================
# PERSONAL OBJECT CRUD
# ============================================================

@app.post("/add_personal_object")
def add_personal_object(
    user_id: int = Form(...),
    generic_type: str = Form(...),
    user_label: str = Form(...)
):
    if generic_type not in PERSONAL_CLASSES:
        return {"status": "error", "message": "Invalid object type"}

    if not user_label.strip():
        return {"status": "error", "message": "Label required"}

    try:
        conn = db_conn()
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM personal_objects WHERE user_id=?", (user_id,))
        cnt = int(c.fetchone()[0])
        if cnt >= MAX_OBJECTS_PER_USER:
            conn.close()
            return {"status": "error", "message": "Max objects reached"}

        c.execute("""
            INSERT INTO personal_objects(user_id, generic_type, user_label, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, generic_type, user_label.strip(), datetime.now().isoformat()))
        conn.commit()
        user_object_id = c.lastrowid  # Get auto-generated ID

        # create FAISS for this object
        idx_path = faiss_path_for_object(user_id, user_object_id)
        ensure_faiss_index(idx_path)

        conn.close()
        log_info("[BACKEND] Created personal object: label='%s', user=%d", user_label, user_id)
        return {"status": "success", "user_object_id": user_object_id}
    except sqlite3.IntegrityError:
        try: conn.close()
        except: pass
        log_error("[BACKEND] Duplicate label '%s' for user %d", user_label, user_id)
        return {"status": "error", "message": "A label with that name already exists"}
    except Exception as e:
        try: conn.close()
        except: pass
        log_error("[BACKEND] add_personal_object failed: %s", str(e))
        return {"status": "error", "message": str(e)}

# ============================================================
# ENVIRONMENT CRUD
# ============================================================

@app.post("/add_environment")
def add_environment(user_id: int = Form(...), environment_label: str = Form(...)):
    if not environment_label.strip():
        return {"status": "error", "message": "Environment label required"}

    conn = db_conn()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO environments(user_id, environment_label, created_at)
            VALUES (?, ?, ?)
        """, (user_id, environment_label.strip(), datetime.now().isoformat()))
        conn.commit()
        env_id = c.lastrowid
        conn.close()
        return {"status": "success", "environment_id": env_id}
    except sqlite3.IntegrityError as ie:
        # duplicate environment label for this user
        conn.rollback()
        conn.close()
        log_error("[BACKEND] add_environment duplicate label: %s", ie)
        return {"status": "error", "message": "Environment label already exists"}
    except Exception as e:
        conn.rollback()
        conn.close()
        log_error("[BACKEND] add_environment exception: %s", e)
        return {"status": "error", "message": "Internal error"}

# ----------------------------
# Deletion endpoints
# ----------------------------
@app.post("/delete_personal_object")
def delete_personal_object(user_id: int = Form(...), user_object_id: int = Form(...)):
    # remove object and its events + faiss index
    conn = db_conn()
    c = conn.cursor()
    # verify ownership
    c.execute("SELECT user_object_id FROM personal_objects WHERE user_id=? AND user_object_id=?",
              (user_id, user_object_id))
    if not c.fetchone():
        conn.close()
        return {"status": "error", "message": "object not found"}
    # delete events and associated image files
    rows = c.execute("SELECT image_path FROM events WHERE user_object_id=?", (user_object_id,)).fetchall()
    for (path,) in rows:
        try:
            if os.path.exists(path):
                os.remove(path)  # Delete image from disk
        except:
            pass  # Ignore file errors
    c.execute("DELETE FROM events WHERE user_object_id=?", (user_object_id,))
    # delete object record
    c.execute("DELETE FROM personal_objects WHERE user_object_id=?", (user_object_id,))
    conn.commit()
    conn.close()
    # remove faiss index file
    idx_path = os.path.join(user_dir(user_id), "faiss", f"{user_object_id}.index")
    try:
        if os.path.exists(idx_path):
            os.remove(idx_path)  # Delete FAISS index from disk
    except:
        pass
    return {"status": "success"}

@app.post("/delete_environment")
def delete_environment(user_id: int = Form(...), environment_id: int = Form(...)):
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT environment_id FROM environments WHERE user_id=? AND environment_id=?", (user_id, environment_id))
    if not c.fetchone():
        conn.close()
        return {"status": "error", "message": "environment not found"}
    c.execute("DELETE FROM environment_landmarks WHERE environment_id=?", (environment_id,))
    c.execute("DELETE FROM environments WHERE environment_id=?", (environment_id,))
    conn.commit()
    conn.close()
    # delete any stored scan images (if still present)
    try:
        env_imgs_dir = os.path.join(user_dir(user_id), f"env_landmarks_{environment_id}")
        if os.path.exists(env_imgs_dir):
            shutil.rmtree(env_imgs_dir)
    except:
        pass
    return {"status": "success"}

@app.get("/get_environment_landmarks")
def get_environment_landmarks(user_id: int, environment_id: int):
    """Return list of saved landmarks (class + user label) for an environment."""
    try:
        conn = db_conn()
        c = conn.cursor()
        c.execute("SELECT landmark_class,user_label,env_landmark_id FROM environment_landmarks WHERE environment_id=?", (environment_id,))
        rows = c.fetchall()
        conn.close()
        items = []
        for lc, ul, landmark_id in rows:
            items.append({"landmark_class": lc, "user_label": ul, "environment_landmark_id": landmark_id})
        return {"status": "ok", "landmarks": items}
    except Exception as e:
        log_error(f"[BACKEND] [GET_ENV_LANDMARKS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_environment_landmark")
def add_environment_landmark(
    user_id: int = Form(...),
    environment_id: int = Form(...),
    landmark_class: str = Form(...),
    user_label: str = Form(...)
):
    """
    Add a single landmark to an environment.
    
    WHY THIS ENDPOINT:
      When user defines "I have a 'master_bed' (Bed class)" in my Bedroom,
      they use this endpoint to add that specific landmark.
    
    ARGS:
      user_id: owner of the environment
      environment_id: which environment to add landmark to
      landmark_class: YOLO class (e.g., "Bed", "Chair") - must be from LANDMARK_CLASSES
      user_label: user's custom name (e.g., "master_bed", "desk_chair")
    """
    try:
        conn = db_conn()
        c = conn.cursor()
        
        # Verify environment belongs to this user
        c.execute("SELECT environment_id FROM environments WHERE user_id=? AND environment_id=?", (user_id, environment_id))
        if not c.fetchone():
            conn.close()
            return {"status": "error", "message": "environment not found"}
        
        user_label_clean = user_label.strip()
        
        # Prevent duplicate user_label in this environment
        c.execute(
            "SELECT env_landmark_id FROM environment_landmarks WHERE environment_id=? AND user_label=?",
            (environment_id, user_label_clean)
        )
        if c.fetchone():
            conn.close()
            return {"status": "error", "message": f"Label '{user_label_clean}' already exists in this environment"}
        
        # Insert new landmark (allows multiple labels per landmark_class)
        c.execute(
            """
            INSERT INTO environment_landmarks(environment_id, landmark_class, user_label, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (environment_id, landmark_class, user_label_clean, datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": f"Added '{user_label_clean}' as {landmark_class}"}
    except Exception as e:
        log_error(f"[BACKEND] [ADD_ENV_LANDMARK] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_environment_landmark")
def delete_environment_landmark(user_id: int = Form(...), environment_id: int = Form(...), user_label: str = Form(...)):
    conn = db_conn()
    c = conn.cursor()
    # ensure env belongs to user
    c.execute("SELECT environment_id FROM environments WHERE user_id=? AND environment_id=?", (user_id, environment_id))
    if not c.fetchone():
        conn.close()
        return {"status": "error", "message": "environment not found"}
    c.execute("DELETE FROM environment_landmarks WHERE environment_id=? AND user_label=?", (environment_id, user_label))
    conn.commit()
    conn.close()
    return {"status": "success"}

# ============================================================
# SESSION START/STOP (TRACK / ENROLL_OBJECT / ENROLL_ENV)
# ============================================================

@app.post("/start_session")
def start_session(
    user_id: int = Form(...),
    mode: str = Form(...),
    user_object_id: Optional[int] = Form(None),
    environment_id: Optional[int] = Form(None),
    landmark_id: Optional[int] = Form(None)
):
    # wrap entire handler so unanticipated errors give JSON response
    try:
        log_debug("start_session called user_id=%s mode=%s user_object_id=%s environment_id=%s landmark_id=%s",
                  user_id, mode, user_object_id, environment_id, landmark_id)

        mode = mode.strip().upper()
        if mode not in ["TRACK", "ENROLL_OBJECT", "ENROLL_LANDMARK", "TEST_ENVIRONMENT"]:
            return {"status": "error", "message": "Invalid mode"}  # Must be one of the 4 supported modes

        if mode == "ENROLL_OBJECT" and not user_object_id:
            return {"status": "error", "message": "user_object_id required"}

        if mode == "ENROLL_LANDMARK" and (not environment_id or not landmark_id):
            return {"status": "error", "message": "environment_id and landmark_id required"}

        # allow tracking inside a specific environment (optional)
        if mode == "TRACK" and environment_id is not None:
            # verify environment belongs to user
            conn = db_conn()
            c = conn.cursor()
            c.execute("SELECT environment_id FROM environments WHERE environment_id=? AND user_id=?", (environment_id, user_id))
            if not c.fetchone():
                conn.close()
                return {"status": "error", "message": "environment not found for user"}
            conn.close()

        session_id = str(uuid.uuid4())  # Generate unique session identifier
        now = datetime.now().isoformat()

        conn = db_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO sessions(session_id, user_id, mode, user_object_id, environment_id, landmark_id, status, created_at, updated_at, last_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, user_id, mode, user_object_id, environment_id, landmark_id, "RUNNING", now, now, None))
        conn.commit()
        conn.close()

        # if we're starting a TRACK session, set up ephemeral state
        if mode == "TRACK":
            SESSION_STATE[session_id] = {
                "user_id": user_id,
                "environment_id": environment_id,
                "objects": {},    # key: user_label → per-object tracking state
                "landmarks": {},  # key: env_landmark_id → rate-limit state
            }
        
        # if we're starting ENROLL_LANDMARK, store landmark context
        if mode == "ENROLL_LANDMARK":
            SESSION_STATE[session_id] = {
                "user_id": user_id,
                "environment_id": environment_id,
                "landmark_id": landmark_id
            }
        
        log_info(f"[SESSION] Started {mode} session={session_id} user={user_id}")
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_error(f"[SESSION] start_session exception: {e}\n{tb}")
        # ensure we always return JSON so client doesn't crash
        return {"status": "error", "message": f"Internal server error: {e}"}

@app.post("/stop_session")
def stop_session(session_id: str = Form(...)):
    try:
        conn = db_conn()
        c = conn.cursor()
        c.execute("UPDATE sessions SET status=?, updated_at=? WHERE session_id=?",
                  ("STOPPED", datetime.now().isoformat(), session_id))
        conn.commit()
        conn.close()

        # drop ephemeral state if we created one
        if session_id in SESSION_STATE:
            del SESSION_STATE[session_id]

        log_info(f"[SESSION] Stopped session={session_id}")
        return {"status": "success"}
    except Exception as e:
        log_debug("Error stopping session %s", session_id)
        # ensure we always return JSON so clients don't crash
        return {"status": "error", "message": str(e)}

@app.get("/session_status")
def session_status(session_id: str):
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT status, last_error, mode FROM sessions WHERE session_id=?", (session_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"status": "error", "message": "session not found"}

    return {"status": "success", "session_status": row[0], "last_error": row[1], "mode": row[2]}

# ============================================================
# FRAME HANDLING FOR SESSIONS
# ============================================================

@app.post("/session_frame")
async def session_frame(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    # This endpoint is called by ingest_ipcam.py, with a session_id.
    # We look up session mode and route logic accordingly.

    tick_active_runtime()

    conn = db_conn()
    c = conn.cursor()

    c.execute("""
        SELECT user_id, mode, user_object_id, environment_id, status
        FROM sessions WHERE session_id=?
    """, (session_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return {"status": "error", "message": "Invalid session_id"}

    user_id, mode, user_object_id, environment_id, status = row
    if status != "RUNNING":
        conn.close()
        return {"status": "stopped"}

    # decode image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)  # Convert bytes to numpy array
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode JPEG to BGR frame
    if frame is None:
        log_error(f"[FRAME] Failed to decode frame (bytes: {len(contents)})")
        conn.close()
        return {"status": "error", "message": "Bad frame"}

    # YOLO inference
    try:
        r0 = yolo_model(frame, verbose=False)[0]
        boxes = r0.boxes
    except Exception as e:
        log_error(f"[YOLO] Inference failed: {e}")
        c.execute("UPDATE sessions SET last_error=?, updated_at=? WHERE session_id=?",
                  (str(e), datetime.now().isoformat(), session_id))
        conn.commit()
        conn.close()
        return {"status": "error", "message": "YOLO failed"}

    # Parse detections
    personal = []
    landmarks = []
    detected_classes = []  # For debugging
    for b in boxes:
        cls_id = int(b.cls[0])
        label = yolo_model.names[cls_id]
        bbox = tuple(map(int, b.xyxy[0].tolist()))
        detected_classes.append(label)
        if label in PERSONAL_CLASSES:
            personal.append((label, bbox))
        if label in LANDMARK_CLASSES:
            landmarks.append((label, bbox))
    
    # Only log when something is actually detected — skip silent empty frames
    if detected_classes:
        personal_names = [cls for cls, _ in personal]
        landmark_names = [cls for cls, _ in landmarks]
        log_debug(f"[DETECTION] mode={mode} | personal={personal_names} | landmarks={landmark_names}")

    # Route by mode
    if mode == "ENROLL_OBJECT":
        # We only store embeddings for the selected object type.
        # We match by object generic_type in DB.
        c.execute("SELECT generic_type, user_label FROM personal_objects WHERE user_object_id=? AND user_id=?",
                  (user_object_id, user_id))
        obj_row = c.fetchone()
        if not obj_row:
            conn.close()
            return {"status": "error", "message": "Object not found"}

        wanted_type, user_label = obj_row

        # find detections of wanted type
        candidates = [(lab, bb) for (lab, bb) in personal if lab == wanted_type]
        if not candidates:  # Target object not visible in frame
            conn.close()
            return {"status": "ok", "note": "no target object in frame"}

        # take highest area bbox as best candidate
        def area(bb): return max(1, (bb[2]-bb[0])*(bb[3]-bb[1]))
        best_bbox = max(
            (bb for _, bb in candidates),
            key=area  # Pick largest detection to avoid occlusions
        )

        emb = get_embedding_from_bbox(frame, best_bbox)
        if emb is None:
            conn.close()
            return {"status": "ok", "note": "empty crop"}

        idx_path = faiss_path_for_object(user_id, user_object_id)
        ensure_faiss_index(idx_path)
        index = faiss.read_index(idx_path)

        # we store with a simple incremental id (faiss needs ids)
        new_id = int(time.time() * 1000)  # millisecond id
        index.add_with_ids(emb, np.array([new_id], dtype=np.int64))
        faiss.write_index(index, idx_path)

        # ============================================================
        # DEBUGGING: Save enrollment frames for visual verification
        # Saves 1 frame every 2 seconds so user can confirm the object was detected.
        # ============================================================
        try:
            debug_dir = os.path.join(user_dir(user_id), "enroll_debug_images", wanted_type)
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save 1 frame every 2 seconds
            current_slot = int(time.time()) // 2
            debug_file = os.path.join(debug_dir, f"{user_label}_{current_slot}.jpg")
            
            if not os.path.exists(debug_file):
                debug_frame = frame.copy()
                x1, y1, x2, y2 = best_bbox
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(debug_frame, f"{user_label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.imwrite(debug_file, debug_frame)
                log_debug(f"[ENROLL_OBJ_DEBUG] Saved: {debug_file}")
        except Exception as e:
            log_debug(f"[ENROLL_OBJ_DEBUG] Failed to save debug image: {e}")
        # ============================================================
        # END DEBUGGING BLOCK
        # ============================================================

        conn.close()
        return {"status": "ok", "note": f"enrolled embedding for {user_label}"}

    if mode == "ENROLL_LANDMARK":
        # Similar to ENROLL_OBJECT, but for environment landmarks
        
        # Get landmark_id from session state (set during start_session)
        session_data = SESSION_STATE.get(session_id, {})
        landmark_id = session_data.get("landmark_id")
        environment_id = session_data.get("environment_id")
        
        if not landmark_id or not environment_id:
            conn.close()
            return {"status": "error", "message": "landmark_id or environment_id not in session"}
        
        # Get landmark info from DB
        c.execute("""
            SELECT user_label, landmark_class FROM environment_landmarks 
            WHERE env_landmark_id=? AND environment_id=?
        """, (landmark_id, environment_id))
        lm_row = c.fetchone()
        if not lm_row:
            conn.close()
            return {"status": "error", "message": "Landmark not found"}
        
        user_label, landmark_class = lm_row
        
        # Find detections of this landmark class
        candidates = [(lab, bb) for (lab, bb) in landmarks if lab == landmark_class]
        if not candidates:
            conn.close()
            return {"status": "ok", "note": f"no {landmark_class} detected in frame"}
        
        # Take highest area bbox as best candidate
        def area(bb): return max(1, (bb[2]-bb[0])*(bb[3]-bb[1]))
        # best_bbox = sorted([bb for _, bb in candidates], key=area, reverse=True)[0]
        best_bbox = max(
            (bb for _, bb in candidates),
            key=area
        )
        
        # Extract embedding
        emb = get_embedding_from_bbox(frame, best_bbox)
        if emb is None:
            conn.close()
            return {"status": "ok", "note": "empty crop"}
        
        # Load or create FAISS index for this landmark
        idx_path = faiss_path_for_landmark(user_id, environment_id, landmark_id)
        ensure_faiss_index(idx_path)
        index = faiss.read_index(idx_path)
        
        # Store embedding with millisecond timestamp as ID
        new_id = int(time.time() * 1000)
        index.add_with_ids(emb, np.array([new_id], dtype=np.int64))
        faiss.write_index(index, idx_path)
        
        # Save debug frame
        try:
            debug_dir = os.path.join(user_dir(user_id), "landmark_debug_images", f"env_{environment_id}")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save 1 frame every 2 seconds
            current_slot = int(time.time()) // 2
            debug_file = os.path.join(debug_dir, f"{user_label}_{current_slot}.jpg")
            
            if not os.path.exists(debug_file):
                debug_frame = frame.copy()
                x1, y1, x2, y2 = best_bbox
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(debug_frame, f"{user_label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                cv2.imwrite(debug_file, debug_frame)
                log_debug(f"[ENROLL_LM_DEBUG] Saved: {debug_file}")
        except Exception as e:
            log_debug(f"[ENROLL_LM_DEBUG] Debug save failed: {e}")
        
        conn.close()
        return {"status": "ok", "note": f"enrolled embedding for {user_label}"}

    # Four-step process each frame:
    # 1. Detect objects in frame (YOLO)
    # 2. Extract embedding from each detection
    # 3. Match embedding to user's FAISS indices (find which labeled object it is)
    # 4. Store event if enough time passed or object disappeared
    if mode == "TRACK":
        session_data = SESSION_STATE.get(session_id, {})
        
        # Step 1: Detect all personal objects and landmarks in this frame
        personal, landmarks = detect_objects_in_frame(frame)

        # ---------- SIMPLE Environment Inference ----------        
        environment_id, env_label = find_environment_from_landmarks(landmarks, user_id)
        
        # ---------- Label based Object Tracking ----------
        object_state = session_data.setdefault("objects", {})
        
        # Load user's objects: (user_object_id, generic_type, user_label)
        c.execute("SELECT user_object_id, generic_type, user_label FROM personal_objects WHERE user_id=?",
                  (user_id,))
        user_objects = c.fetchall()
        user_folder = user_dir(user_id)

        # Ensure state entries exist for each user-labeled object (key by label)
        for object_id, generic_type, user_label in user_objects:
            if user_label not in object_state:
                object_state[user_label] = {
                    "user_object_id": object_id,
                    "generic_type": generic_type,
                    "last_stored": None,  # Timestamp of last stored event
                    "last_seen": None,    # Last frame we saw this object
                    "last_frame": None,   # Raw image from last detection
                    "last_bbox": None,    # Bounding box from last detection
                    "last_embedding": None,  # Embedding vector from last detection
                    "missing_ticks": 0    # Consecutive frames without detection
                }

        stored_count = 0  # Number of events stored this frame
        now_dt = datetime.now()
        detected_labels_this_frame = set()  # Track which labels were detected THIS frame

        # Step 2 & 3: Update state for currently visible personal objects
        # For each detected object, match it to the best user-defined label via FAISS
        for detected_class, detected_bbox in personal:
            # Example: For detected_class="Watch", get all Watch labels: ["blue_watch", "silver_watch"]
            candidate_labels = [(oid, lbl) for (oid, gtype, lbl) in user_objects if gtype == detected_class]
            if not candidate_labels:
                continue

            # Extract embedding from this detection and match to best user defined label
            embedding, best_object_id, best_label, confidence = extract_embedding_and_match(
                frame, detected_bbox, user_id, candidate_labels
            )
            
            # DEBUGGING: Log all candidates evaluated and their scores
            candidate_info = f"Candidates evaluated: {[lbl for _, lbl in candidate_labels]}"
            if best_label is None:
                log_debug(f"[TRACK_MATCH] {detected_class} detection → NO MATCH (confidence={confidence:.3f}) | {candidate_info} | threshold={SIM_THRESHOLD}")
            else:
                log_debug(f"[TRACK_MATCH] {detected_class} detection → MATCHED to '{best_label}' (confidence={confidence:.3f}) | {candidate_info} | threshold={SIM_THRESHOLD}")
            
            if embedding is None or best_label is None:
                continue

            # Record that this label was detected this frame (for disappearance detection)
            detected_labels_this_frame.add(best_label)

            # Update tracking state for this specific labeled object
            object_info = object_state[best_label]
            object_info["last_seen"] = now_dt
            object_info["last_frame"] = frame.copy()
            object_info["last_bbox"] = detected_bbox
            object_info["last_embedding"] = embedding
            object_info["missing_ticks"] = 0

            # Step 4: Decide whether to store an event (periodic sampling)
            #   (a) First time seeing this object (last_stored is None), OR
            #   (b) Enough time has passed since last store (STORE_INTERVAL_SECONDS)
            should_store = False
            if object_info["last_stored"] is None:
                should_store = True  # First detection of this object
            else:
                elapsed_seconds = (now_dt - object_info["last_stored"]).total_seconds()
                if elapsed_seconds >= STORE_INTERVAL_SECONDS:
                    should_store = True  # Enough time has passed

            if should_store and best_object_id is not None:
                # Compute human-readable location (e.g., "on white_chair (Bedroom)")
                location_text = get_location_text_for_event(
                    detected_bbox, landmarks, environment_id, user_id
                )
                
                # Store event to database
                store_object_event(
                    conn=conn,
                    user_id=user_id,
                    user_object_id=best_object_id,
                    location_text=location_text,
                    timestamp=now_dt,
                    image=object_info["last_frame"],
                    object_type=detected_class,
                    object_label=best_label
                )
                # DEBUGGING: Log when we store an event
                log_debug(f"[STORE_EVENT] Stored '{best_label}' at '{location_text}' with confidence={confidence:.3f}")
                object_info["last_stored"] = now_dt
                stored_count += 1

        # Handle object disappearance/not detected case
        for obj_label, obj_info in list(object_state.items()):
            # Skip if this labeled object was detected in THIS frame
            if obj_label in detected_labels_this_frame:
                continue

            # Object was never seen
            if obj_info["last_seen"] is None:
                continue
            
            # Increment counter for frames without detection
            obj_info["missing_ticks"] += 1
            
            # After DISAPPEAR_TICKS consecutive frames without seeing object,
            # register a final disappearance event at last known location
            if obj_info["missing_ticks"] >= DISAPPEAR_TICKS:
                if obj_info["last_frame"] is not None and obj_info["last_bbox"] is not None:
                    final_time = obj_info["last_seen"]
                    
                    # Only store disappearance if we haven't already stored this timepoint
                    if obj_info["last_stored"] is None or final_time > obj_info["last_stored"]:
                        user_object_id = obj_info.get("user_object_id")
                        generic_type = obj_info.get("generic_type")
                        
                        location_text = get_location_text_for_event(
                            obj_info["last_bbox"], landmarks, environment_id, user_id
                        )
                        
                        store_object_event(
                            conn=conn,
                            user_id=user_id,
                            user_object_id=user_object_id,
                            location_text=location_text,
                            timestamp=final_time,
                            image=obj_info["last_frame"],
                            object_type=generic_type,
                            object_label=obj_label
                        )
                        stored_count += 1

                # Reset state for this labeled object
                obj_info["missing_ticks"] = 0
                obj_info["last_seen"] = None
                obj_info["last_frame"] = None
                obj_info["last_bbox"] = None
                obj_info["last_embedding"] = None

        cleanup_if_needed(conn, user_id)
        conn.close()
        return {"status": "ok", "stored": stored_count}

    # TEST_ENVIRONMENT mode: Live video with YOLO detections + landmark overlays
    # Process:
    # 1. Detect all objects and landmarks (YOLO)
    # 2. For each landmark, query its FAISS index to get the matching landmark_id
    # 3. Draw bounding boxes + labels on frame
    # 4. Return annotated frame
    if mode == "TEST_ENVIRONMENT":
        # Get detected objects/landmarks
        personal, landmarks = detect_objects_in_frame(frame)
        
        # Load environment landmarks 
        c.execute("""
            SELECT env_landmark_id, landmark_class, user_label
            FROM environment_landmarks
            WHERE environment_id=?
        """, (environment_id,))
        env_landmarks = c.fetchall()
        
        # Load user's personal objects 
        c.execute("""
            SELECT user_object_id, generic_type, user_label
            FROM personal_objects
            WHERE user_id=?
        """, (user_id,))
        user_objects = c.fetchall()
        
        # Draw detections on frame
        annotated_frame = frame.copy()
        
        def draw_label(img, text, x1, y1, color):
            """Draw text with a dark background rectangle for readability."""
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            # Background box
            cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(img, text, (x1 + 2, y1 - baseline - 2), font, scale, color, thickness, cv2.LINE_AA)
        
        # Draw landmarks (cyan boxes) - attempt FAISS matching across all same-class labels
        for landmark_class, bbox in landmarks:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Cyan, thick
            
            # Get all registered landmarks of this class in current environment and compare
            lm_candidates = [
                (lm_id, lm_label)
                for lm_id, lm_cls, lm_label in env_landmarks
                if lm_cls == landmark_class
            ]

            if not lm_candidates:
                label_text = f"[LM] {landmark_class}"
            else:
                label_text = f"[LM] {landmark_class}?"
                lm_embedding = get_embedding_from_bbox(frame, bbox)
                best_lm_label = None
                best_lm_score = -1.0

                if lm_embedding is not None:
                    for lm_id, lm_label in lm_candidates:
                        lm_faiss_path = faiss_path_for_landmark(user_id, environment_id, lm_id)
                        if not os.path.exists(lm_faiss_path):
                            continue
                        try:
                            lm_index = faiss.read_index(lm_faiss_path)
                            if lm_index.ntotal == 0:
                                continue
                            distances, _ = lm_index.search(lm_embedding, 1)
                            score = float(distances[0, 0])
                            log_debug(f"[TEST_ENV_LM] class={landmark_class} candidate='{lm_label}' score={score:.3f}")
                            if score > best_lm_score and score >= SIM_THRESHOLD:
                                best_lm_score = score
                                best_lm_label = lm_label
                        except Exception as e:
                            log_debug(f"[TEST_ENV_LM] FAISS error for lm_id={lm_id}: {e}")

                if best_lm_label is not None:
                    label_text = f"[LM] {best_lm_label}"
            
            draw_label(annotated_frame, label_text, x1, y1, (0, 255, 255))
        
        # Draw personal objects (green boxes) - attempt FAISS matching across all same-class labels
        for obj_class, bbox in personal:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green, thick
            
            # Compare this detection with every user label of the same generic class
            obj_candidates = [
                (obj_id, obj_label)
                for obj_id, obj_type, obj_label in user_objects
                if obj_type == obj_class
            ]

            if not obj_candidates:
                draw_label(annotated_frame, obj_class, x1, y2 + 18, (0, 255, 0))
                continue

            display_label = f"{obj_class}?"
            try:
                obj_embedding = get_embedding_from_bbox(frame, bbox)
                if obj_embedding is not None:
                    best_obj_label = None
                    best_obj_score = -1.0

                    for obj_id, obj_label in obj_candidates:
                        object_faiss_path = faiss_path_for_object(user_id, obj_id)
                        if not os.path.exists(object_faiss_path):
                            continue

                        index = faiss.read_index(object_faiss_path)
                        if index.ntotal == 0:
                            continue

                        distances, _ = index.search(obj_embedding, 1)
                        score = float(distances[0, 0])
                        log_debug(f"[TEST_ENV_OBJ] class={obj_class} candidate='{obj_label}' score={score:.3f}")

                        if score > best_obj_score and score >= SIM_THRESHOLD:
                            best_obj_score = score
                            best_obj_label = obj_label

                    if best_obj_label is not None:
                        display_label = best_obj_label
            except Exception as e:
                log_debug(f"[TEST_ENV] FAISS matching failed: {e}")
                display_label = f"{obj_class}?"
            
            draw_label(annotated_frame, display_label, x1, y2 + 18, (0, 255, 0))
        
        # Debug: save annotated frame every 2 seconds so user can verify detections
        try:
            debug_dir = os.path.join(user_dir(user_id), "test_env_debug_images")
            os.makedirs(debug_dir, exist_ok=True)
            current_slot = int(time.time()) // 2
            debug_file = os.path.join(debug_dir, f"frame_{current_slot}.jpg")
            if not os.path.exists(debug_file):
                cv2.imwrite(debug_file, annotated_frame)
                log_debug(f"[TEST_ENV_DEBUG] Saved: {debug_file} | personal={len(personal)}, landmarks={len(landmarks)}")
        except Exception as e:
            log_debug(f"[TEST_ENV_DEBUG] Save failed: {e}")
        
        # Encode annotated frame to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        conn.close()
        return {
            "status": "ok",
            "frame": base64.b64encode(frame_bytes).decode('utf-8'),
            "detections": {
                "personal_count": len(personal),
                "landmark_count": len(landmarks)
            }
        }

    conn.close()
    return {"status": "ok"}

# ============================================================
# QUERY + MANUAL CLEANUP
# ============================================================

@app.post("/query")
def query(
    user_id: int = Form(...),
    user_label: str = Form(...),
    k: int = Form(10)
):
    user_label = user_label.strip()
    if not user_label:
        return {"status": "error", "message": "Label required", "results": []}

    conn = db_conn()
    c = conn.cursor()

    c.execute("""
        SELECT user_object_id FROM personal_objects
        WHERE user_id=? AND user_label=?
    """, (user_id, user_label))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"status": "ok", "results": []}

    obj_id = row[0]
    c.execute("""
        SELECT event_id, location_text, timestamp, image_path
        FROM events
        WHERE user_id=? AND user_object_id=?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, obj_id, k))
    rows = c.fetchall()
    conn.close()

    results = []
    for row in rows:
        event_id, loc, ts, path = row[0], row[1], row[2], row[3]
        results.append({
            "event_id": event_id,
            "location_text": loc,
            "timestamp": ts,
            "image_path": path
        })

    return {"status": "ok", "results": results}

@app.post("/manual_cleanup")
def manual_cleanup(
    user_id: int = Form(...),
    older_than_minutes: int = Form(...)
):
    cutoff = datetime.now() - timedelta(minutes=int(older_than_minutes))

    conn = db_conn()
    c = conn.cursor()
    c.execute("""
        SELECT event_id, image_path, timestamp
        FROM events
        WHERE user_id=?
    """, (user_id,))
    rows = c.fetchall()

    deleted = 0
    for event_id, image_path, ts in rows:
        try:
            dt = datetime.fromisoformat(ts)
        except:
            continue
        if dt < cutoff:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass
            c.execute("DELETE FROM events WHERE event_id=?", (event_id,))
            deleted += 1

    conn.commit()
    conn.close()
    return {"status": "ok", "deleted": deleted}
