import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # note: avoids OpenMP crash on some Windows setups

import logging
import sqlite3
from datetime import datetime, timedelta

import cv2
import faiss
import numpy as np
import torch
import clip
from PIL import Image
from ultralytics import YOLO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Create logs directory
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", f"Log_Backend_{ts}")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "backend_cmd.log")

# ---------------------------
# Logging (important)
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
)
log = logging.getLogger("backend")

# ---------------------------
# Config (edit here only)
# ---------------------------
YOLO_WEIGHTS = "best.pt"

DATA_DIR = "reid_store"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "events.sqlite")
FAISS_PATH = os.path.join(DATA_DIR, "index.faiss")

# These are the personal objects I care about
PERSONAL_OBJECTS = {"MyWatch", "MyWallet", "MyBikeKeys"}

# These are detected but I do NOT want to use them as landmarks
EXCLUDED_LANDMARKS = {"MyLeftHand", "MyRightHand"}

# Store rules
STORE_INTERVAL_SECONDS = 30        # if object stays visible, store again only after this
DISAPPEAR_TICKS = 2                # number of ingest calls without seeing object to treat as disappeared
RETENTION_HOURS = 2                # delete events older than this (only when we store something)

# CLIP config
EMBED_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Startup: create folders + db + load models
# ---------------------------
os.makedirs(IMAGES_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object TEXT NOT NULL,
    time_iso TEXT NOT NULL,
    image_path TEXT NOT NULL,
    location TEXT,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
)
""")
conn.commit()

# FAISS index: cosine similarity (because CLIP vectors are normalized)
if os.path.exists(FAISS_PATH):
    index = faiss.read_index(FAISS_PATH)
else:
    base = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIDMap2(base)

log.info("Loading YOLO weights: %s", YOLO_WEIGHTS)
yolo = YOLO(YOLO_WEIGHTS)

log.info("Loading CLIP on %s", DEVICE)
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# ---------------------------
# Runtime state (not saved in DB)
# NOTE: DB stays across restarts, but this state resets.
# ---------------------------
state = {
    obj: {
        "last_stored": None,   # datetime
        "last_seen": None,     # datetime
        "last_frame": None,    # full frame snapshot
        "last_bbox": None,     # bbox
        "last_emb": None,      # embedding
        "missing_ticks": 0,    # how many ingest cycles missing
    }
    for obj in PERSONAL_OBJECTS
}

# ---------------------------
# Helpers
# ---------------------------
def get_embedding(frame_bgr, bbox):
    # note to me: crop the object and convert it to a 512-number vector (CLIP embedding)
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = clip_model.encode_image(img_t).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize so dot-product==cosine

    return emb.squeeze(0).cpu().numpy().astype("float32")  # shape (512,)

def save_event(obj, frame, bbox, emb_vec, location, event_time_dt=None):
    # note to me: store full frame for context so I can see "where" later
    now_dt = event_time_dt or datetime.now()
    now_iso = now_dt.isoformat(timespec="seconds")

    # important: store ABSOLUTE paths so UI can always find the file
    fname = f"frame_{obj}_{now_dt.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    fpath = os.path.abspath(os.path.join(IMAGES_DIR, fname))

    cv2.imwrite(fpath, frame)

    x1, y1, x2, y2 = bbox
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events(object, time_iso, image_path, location, x1, y1, x2, y2) VALUES (?,?,?,?,?,?,?,?)",
        (obj, now_iso, fpath, location, x1, y1, x2, y2)
    )
    conn.commit()
    event_id = cur.lastrowid

    index.add_with_ids(emb_vec.reshape(1, -1), np.array([event_id], dtype=np.int64))

    log.info("[STORE] obj=%s id=%s time=%s location=%s path=%s", obj, event_id, now_iso, location, fpath)

    cleanup_retention_if_needed()
    save_faiss()

    return event_id

def save_faiss():
    faiss.write_index(index, FAISS_PATH)

def cleanup_retention_if_needed():
    # note to me: only delete old stuff when we store a new event (as requested)
    cutoff = datetime.now() - timedelta(hours=RETENTION_HOURS)
    cutoff_iso = cutoff.isoformat(timespec="seconds")

    rows = conn.execute(
        "SELECT id, image_path FROM events WHERE datetime(time_iso) < datetime(?)",
        (cutoff_iso,)
    ).fetchall()

    if not rows:
        return

    ids_to_delete = []
    for event_id, image_path in rows:
        ids_to_delete.append(int(event_id))
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                log.warning("Could not delete image %s (%s)", image_path, e)

    # delete from DB
    q = ",".join(["?"] * len(ids_to_delete))
    conn.execute(f"DELETE FROM events WHERE id IN ({q})", ids_to_delete)
    conn.commit()

    # delete from FAISS
    index.remove_ids(np.array(ids_to_delete, dtype=np.int64))

    log.info("[CLEANUP] deleted %d old events (>%d hours)", len(ids_to_delete), RETENTION_HOURS)

def infer_location_from_landmarks(item_bbox, landmark_list):
    # note to me: "location" = nearest/overlap with a detected furniture-like object
    # landmark_list: list of {"label": str, "bbox": (x1,y1,x2,y2)}
    if not landmark_list:
        return "unknown"

    x1, y1, x2, y2 = item_bbox
    icx = (x1 + x2) / 2
    icy = (y1 + y2) / 2

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
        area_b = max(1, (bx2 - bx1)) * max(1, (by2 - by1))
        return inter / float(area_a + area_b - inter)

    best = None
    best_score = -1e18
    best_relation = "near"

    for lm in landmark_list:
        lb = lm["bbox"]
        overlap = iou(item_bbox, lb)

        lx1, ly1, lx2, ly2 = lb
        lcx = (lx1 + lx2) / 2
        lcy = (ly1 + ly2) / 2
        dist = ((icx - lcx) ** 2 + (icy - lcy) ** 2) ** 0.5

        score = overlap * 10000.0 - dist
        if score > best_score:
            best_score = score
            best = lm["label"]
            best_relation = "on" if overlap >= 0.10 else "near"

    return f"{best_relation} {best}" if best else "unknown"

def fetch_last_k(object_name, k):
    rows = conn.execute(
        """
        SELECT id, object, time_iso, image_path, location, x1, y1, x2, y2
        FROM events
        WHERE object = ?
        ORDER BY datetime(time_iso) DESC
        LIMIT ?
        """,
        (object_name, k)
    ).fetchall()

    results = []
    for r in rows:
        _id, obj, time_iso, image_path, location, x1, y1, x2, y2 = r
        results.append({
            "id": _id,
            "object": obj,
            "time_iso": time_iso,
            "image_path": image_path,
            "location": location,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
    return results

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title="Item Memory Assistant (Minimal)")

@app.get("/health")
def health():
    return {"ok": True, "faiss_total": int(index.ntotal)}

@app.post("/ingest_frame")
async def ingest_frame(file: UploadFile = File(...)):
    img_bytes = await file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    now_dt = datetime.now()

    # YOLO detect
    results = yolo(frame, verbose=False)
    boxes = results[0].boxes if results else []

    # gather landmarks (everything except personal objects and excluded objects)
    landmarks = []
    # gather personal detections (keep only 1 bbox per personal object per frame)
    personal_seen = {}

    for b in boxes:
        cls_id = int(b.cls[0])
        label = yolo.names[cls_id]
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        bbox = (x1, y1, x2, y2)

        if label in PERSONAL_OBJECTS:
            personal_seen[label] = bbox
        else:
            if label in EXCLUDED_LANDMARKS:
                continue
            landmarks.append({"label": label, "bbox": bbox})

    stored_count = 0

    # (1) update state for seen personal objects
    for obj, bbox in personal_seen.items():
        emb = get_embedding(frame, bbox)
        if emb is None:
            continue

        st = state[obj]
        st["last_seen"] = now_dt
        st["last_frame"] = frame.copy()
        st["last_bbox"] = bbox
        st["last_emb"] = emb
        st["missing_ticks"] = 0

        # location based on landmarks
        location = infer_location_from_landmarks(bbox, landmarks)

        # store first time
        if st["last_stored"] is None:
            save_event(obj, st["last_frame"], bbox, emb, location, event_time_dt=now_dt)
            st["last_stored"] = now_dt
            stored_count += 1
        else:
            # store only after 30 seconds if continuous
            elapsed = (now_dt - st["last_stored"]).total_seconds()
            if elapsed >= STORE_INTERVAL_SECONDS:
                save_event(obj, st["last_frame"], bbox, emb, location, event_time_dt=now_dt)
                st["last_stored"] = now_dt
                stored_count += 1

    # (2) handle disappear logic for objects not seen now
    for obj in PERSONAL_OBJECTS:
        if obj in personal_seen:
            continue

        st = state[obj]
        if st["last_seen"] is None:
            continue

        st["missing_ticks"] += 3  # added 3 to make it more likely to trigger disappearance (since we might have lag/skips)

        if st["missing_ticks"] >= DISAPPEAR_TICKS:
            # store the last moment we saw it (even if < 30 seconds)
            final_time = st["last_seen"]
            if st["last_frame"] is not None and st["last_bbox"] is not None and st["last_emb"] is not None:
                if st["last_stored"] is None or final_time > st["last_stored"]:
                    location = infer_location_from_landmarks(st["last_bbox"], landmarks)
                    save_event(obj, st["last_frame"], st["last_bbox"], st["last_emb"], location, event_time_dt=final_time)
                    st["last_stored"] = final_time
                    stored_count += 1

            # reset after disappearance handled
            st["missing_ticks"] = 0
            st["last_seen"] = None
            st["last_frame"] = None
            st["last_bbox"] = None
            st["last_emb"] = None

    return {"stored_events": stored_count, "seen_personal": list(personal_seen.keys())}

@app.post("/query")
async def query(payload: dict):
    object_name = (payload.get("object_name") or "").strip()
    k = int(payload.get("k") or 3)
    k = max(1, min(10, k))  # clamp 1..10

    if not object_name:
        return JSONResponse(status_code=400, content={"error": "object_name is required"})

    results = fetch_last_k(object_name, k)
    return {"object_name": object_name, "results": results}
