# Session Frame TRACK Mode — Deep Dive

**Purpose:** This document explains how the `session_frame` endpoint processes each camera frame in TRACK mode, including state management, object detection, landmark identification, location inference, and storage decisions.

---

## Table of Contents
1. [Overview](#overview)
2. [Part A: Loop-by-Loop Code Walkthrough](#part-a-loop-by-loop-code-walkthrough)
3. [Part B: Visual Flow Diagrams](#part-b-visual-flow-diagrams)

---

## Overview

**What happens each frame in TRACK mode:**
1. **Frame arrives** → YOLO detects objects & landmarks
2. **For each personal object detection** → Extract CLIP embedding → Match against FAISS → Update tracking state
3. **Check storage rule** → Store event if 15 seconds passed OR object disappeared
4. **For each tracked object NOT detected** → Increment missing_ticks → Store disappearance event if threshold reached

**Key concepts:**
- **Session state**: Ephemeral dict `SESSION_STATE[session_id]` holds per-object tracking data (last_seen, last_frame, missing_ticks)
- **15-second rule**: Store event only if `STORE_INTERVAL_SECONDS` (15s) passed since last storage
- **Disappearance tracking**: If object not detected for `DISAPPEAR_TICKS` (20 frames), store final event at last known location
- **Location inference**: Use IoU overlap (for "on") or Euclidean distance (for "near") to determine object-landmark relationship

---

## Part A: Loop-by-Loop Code Walkthrough

### 1. Frame Arrival & Session Setup

**File:** `backend_api.py`  
**Function:** `@app.post("/session_frame")`  
**Lines:** ~1049-1078

```python
# Step 1: Endpoint entry point
@app.post("/session_frame")
async def session_frame(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    tick_active_runtime()  # Update cumulative runtime for auto-cleanup
    
    conn = db_conn()
    c = conn.cursor()
    
    # Step 2: Load session metadata from database
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
```

**What happens:**
- `tick_active_runtime()` updates `ACTIVE_SECONDS` global variable (used for auto-cleanup after 4 hours)
- Database query retrieves session info (user_id, mode, status)
- If session stopped (e.g., camera timeout), immediately return without processing

---

### 2. Frame Decoding

**Lines:** ~1074-1080

```python
# Step 3: Decode JPEG bytes to BGR numpy array
contents = await file.read()                      # Raw JPEG bytes from ingest_ipcam.py
np_arr = np.frombuffer(contents, np.uint8)        # Convert to 1-D uint8 array
frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    # Decode to BGR image (H, W, 3)

if frame is None:
    log_error(f"[FRAME] Failed to decode frame (bytes: {len(contents)})")
    conn.close()
    return {"status": "error", "message": "Bad frame"}
```

**What happens:**
- Receives JPEG-compressed frame from `ingest_ipcam.py` (typically ~20-50 KB)
- Decodes to uncompressed BGR numpy array (e.g., 480×640×3 = 921,600 bytes)
- Error handling for corrupted frames

---

### 3. YOLO Inference

**Lines:** ~1083-1094

```python
# Step 4: Run YOLO object detection
try:
    r0 = yolo_model(frame, verbose=False)[0]  # Run inference (GPU or CPU)
    boxes = r0.boxes                           # Extract detected bounding boxes
except Exception as e:
    log_error(f"[YOLO] Inference failed: {e}")
    c.execute("UPDATE sessions SET last_error=?, updated_at=? WHERE session_id=?",
              (str(e), datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()
    return {"status": "error", "message": "YOLO failed"}
```

**What happens:**
- YOLO model runs on frame (takes ~50-200ms depending on hardware)
- Returns bounding boxes with class IDs
- On error: update session table with error message and stop processing

---

### 4. Parse Detections

**Lines:** ~1095-1115

```python
# Step 5: Categorize detections into personal objects vs landmarks
personal = []       # Will hold: [(class_label, bbox), ...]
landmarks = []      # Will hold: [(class_label, bbox), ...]
detected_classes = []  # For debug logging

for b in boxes:
    cls_id = int(b.cls[0])                        # Class index (0-11)
    label = yolo_model.names[cls_id]              # Class name: "Watch", "Chair", etc.
    bbox = tuple(map(int, b.xyxy[0].tolist()))    # (x1, y1, x2, y2)
    
    detected_classes.append(label)
    
    if label in PERSONAL_CLASSES:   # ["Watch", "Wallet", "Bike Key", "Car Key"]
        personal.append((label, bbox))
    if label in LANDMARK_CLASSES:   # ["Bed", "Chair", "Table", ...]
        landmarks.append((label, bbox))

# Log only when something detected (avoid spam from empty frames)
if detected_classes:
    personal_names = [cls for cls, _ in personal]
    landmark_names = [cls for cls, _ in landmarks]
    log_debug(f"[DETECTION] mode={mode} | personal={personal_names} | landmarks={landmark_names}")
```

**What happens:**
- Each detection is classified as personal object (tracked item) or landmark (location context)
- Bounding box stored as `(x1, y1, x2, y2)` pixel coordinates
- Example output: `personal=[("Watch", (100,50,150,100))]`, `landmarks=[("Chair", (200,150,350,400))]`

---

### 5. Environment Inference from Landmarks

**Lines:** ~1269-1271

```python
# Step 6: Determine which environment we're in (if any)
# Uses ANY-intersection strategy: if detected landmark classes overlap with
# registered environment's landmarks, that's the environment
environment_id, env_label = find_environment_from_landmarks(landmarks, user_id)
```

**Function:** `find_environment_from_landmarks()` — Lines ~453-494

```python
def find_environment_from_landmarks(
    detected_landmarks: List[tuple],
    user_id: int
) -> tuple:
    if not detected_landmarks:
        return None, None
    
    # Extract just YOLO classes (ignore bboxes for matching)
    detected_classes = {class_label for class_label, bbox in detected_landmarks}
    # Example: {"Chair", "Bed"}
    
    conn = db_conn()
    c = conn.cursor()
    
    # Get all user's environments
    c.execute("SELECT environment_id, environment_label FROM environments WHERE user_id=?", (user_id,))
    environments = c.fetchall()
    # Example: [(1, "Bedroom"), (2, "Living Room")]
    
    # For each environment, check if ANY registered landmark class matches detections
    for env_id, env_label in environments:
        c.execute("""
            SELECT landmark_class FROM environment_landmarks 
            WHERE environment_id = ?
        """, (env_id,))
        
        env_landmark_classes = {row[0] for row in c.fetchall()}
        # Example for Bedroom: {"Chair", "Bed", "Night Table"}
        
        intersection = detected_classes & env_landmark_classes
        # Example: {"Chair", "Bed"} & {"Chair", "Bed", "Night Table"} = {"Chair", "Bed"}
        
        # If ANY class matches, this is the environment
        if intersection:
            conn.close()
            return env_id, env_label  # e.g., (1, "Bedroom")
    
    conn.close()
    return None, None  # No environment matched
```

**What happens:**
- Compares detected landmark YOLO classes against user's registered environments
- First environment with ANY matching class wins (no scoring/confidence)
- If "Bedroom" has Chair+Bed+Night Table registered, and frame detects Chair → "Bedroom" is inferred
- If no match → `environment_id = None`, location text will be raw YOLO class name (e.g., "on Chair" instead of "on study_chair (Bedroom)")

---

### 6. Object Tracking State Initialization

**Lines:** ~1273-1295

```python
# Step 7: Get or create session state (ephemeral dict, not in database)
session_data = SESSION_STATE.get(session_id, {})
object_state = session_data.setdefault("objects", {})
# object_state = {
#     "blue_watch": {
#         "user_object_id": 2,
#         "generic_type": "Watch",
#         "last_stored": datetime(...),
#         "last_seen": datetime(...),
#         "last_frame": np.ndarray(...),
#         "last_bbox": (100, 50, 150, 100),
#         "last_embedding": np.ndarray(...),
#         "missing_ticks": 0
#     },
#     "black_wallet": { ... }
# }

# Load user's registered objects from database
c.execute("SELECT user_object_id, generic_type, user_label FROM personal_objects WHERE user_id=?", (user_id,))
user_objects = c.fetchall()
# Example: [(1, "Watch", "blue_watch"), (2, "Watch", "silver_watch"), (3, "Wallet", "black_wallet")]

# Ensure state entry exists for each user-labeled object (key by label)
for object_id, generic_type, user_label in user_objects:
    if user_label not in object_state:
        object_state[user_label] = {
            "user_object_id": object_id,
            "generic_type": generic_type,
            "last_stored": None,      # When we last stored an event for this object
            "last_seen": None,        # When we last detected this object
            "last_frame": None,       # Frame image when last detected
            "last_bbox": None,        # Bounding box when last detected
            "last_embedding": None,   # CLIP embedding when last detected
            "missing_ticks": 0        # Consecutive frames without detection
        }

stored_count = 0                          # How many events stored this frame
now_dt = datetime.now()                   # Current timestamp
detected_labels_this_frame = set()        # Which labels were matched this frame
```

**What happens:**
- `SESSION_STATE` is a global dict holding ephemeral tracking data (not persisted to database)
- For each registered object label (e.g., "blue_watch"), create state entry if doesn't exist
- State persists across frames until session stops
- `missing_ticks` counts consecutive frames without detection (used for disappearance logic)

---

### 7. Main Detection Loop — Match & Update State

**Lines:** ~1297-1332

```python
# Step 8 & 9: For each detected personal object, match to best user label via FAISS
for detected_class, detected_bbox in personal:
    # Example: detected_class="Watch", detected_bbox=(100, 50, 150, 100)
    
    # Get all user labels of this object type
    candidate_labels = [(oid, lbl) for (oid, gtype, lbl) in user_objects if gtype == detected_class]
    # Example for detected_class="Watch": [(1, "blue_watch"), (2, "silver_watch")]
    
    if not candidate_labels:
        continue  # User has no registered labels for this object type
    
    # Extract CLIP embedding from this detection's crop and match against FAISS
    embedding, best_object_id, best_label, confidence = extract_embedding_and_match(
        frame, detected_bbox, user_id, candidate_labels
    )
    # Returns: (512-dim vector, 2, "silver_watch", 0.87)
    # Meaning: 87% similarity to "silver_watch" FAISS index
    
    # Log matching results for debugging
    candidate_info = f"Candidates evaluated: {[lbl for _, lbl in candidate_labels]}"
    if best_label is None:
        log_debug(f"[TRACK_MATCH] {detected_class} detection → NO MATCH (confidence={confidence:.3f}) | {candidate_info} | threshold={SIM_THRESHOLD}")
    else:
        log_debug(f"[TRACK_MATCH] {detected_class} detection → MATCHED to '{best_label}' (confidence={confidence:.3f}) | {candidate_info} | threshold={SIM_THRESHOLD}")
    
    if embedding is None or best_label is None:
        continue  # CLIP extraction failed OR similarity below threshold (0.65)
    
    # Record that this label was detected this frame (for disappearance detection)
    detected_labels_this_frame.add(best_label)
    # Example: detected_labels_this_frame = {"silver_watch"}
    
    # Update tracking state for this specific labeled object
    object_info = object_state[best_label]
    object_info["last_seen"] = now_dt                  # Timestamp of this frame
    object_info["last_frame"] = frame.copy()           # Store frame image (for later storage)
    object_info["last_bbox"] = detected_bbox           # Bounding box
    object_info["last_embedding"] = embedding          # CLIP embedding (not currently used)
    object_info["missing_ticks"] = 0                   # Reset disappearance counter
```

**What happens:**
- For each YOLO detection (e.g., "Watch"), extract CLIP embedding from bounding box crop
- Compare embedding against ALL user labels of that class (e.g., "blue_watch", "silver_watch")
- Pick label with highest FAISS similarity score (if above threshold 0.65)
- Update state dict with latest detection info
- Frame image is stored in RAM (not disk yet) for potential later storage

**Critical:** Frame is NOT saved to disk yet — only stored in session state.

---

### 8. FAISS Matching Deep Dive

**Function:** `extract_embedding_and_match()` — Lines ~312-382

```python
def extract_embedding_and_match(
    frame: np.ndarray,
    bbox: tuple,
    user_id: int,
    candidates: List[tuple]  # [(user_object_id, user_label), ...]
) -> tuple:
    # Step 1: Extract CLIP embedding from bounding box crop
    emb = get_embedding_from_bbox(frame, bbox)
    # Returns: (1, 512) numpy array, L2-normalized
    
    if emb is None:
        return None, None, None, -1.0
    
    best_obj_id = None
    best_obj_label = None
    best_score = -1.0
    
    # Step 2: Query each candidate's FAISS index to find best match
    for oid, lbl in candidates:
        # Example: oid=1, lbl="blue_watch"
        
        idx_path = faiss_path_for_object(user_id, oid)
        # Example: "reid_store/users/5/faiss/object/1.index"
        
        if not os.path.exists(idx_path):
            continue
        
        try:
            index = faiss.read_index(idx_path)
            # FAISS index contains all stored embeddings for this object label
            # Each index.ntotal = number of embeddings stored during enrollment
            
            # Search for 1 nearest neighbor
            distances, indices = index.search(emb, 1)
            # distances[0,0] = similarity score (inner product, higher = more similar)
            # indices[0,0] = ID of nearest stored embedding (not used here)
            
            score = float(distances[0, 0]) if len(distances) > 0 else -1.0
            
            log_debug(f"[FAISS_SCORE] '{lbl}' similarity={score:.3f} (threshold={SIM_THRESHOLD})")
            
            # If score below threshold (0.65), reject it
            if score > best_score and score >= SIM_THRESHOLD:
                best_score = score
                best_obj_id = oid
                best_obj_label = lbl
        except Exception as e:
            log_debug(f"[FAISS_SCORE] Error querying FAISS for object {oid}: {e}")
    
    return emb, best_obj_id, best_obj_label, best_score
```

**How FAISS matching works:**
1. During enrollment, user points camera at "blue_watch" for ~10 seconds → system stores 50-100 CLIP embeddings in `1.index`
2. During tracking, each Watch detection extracts 1 embedding → searches `1.index` for nearest match
3. FAISS returns similarity score (inner product of L2-normalized vectors = cosine similarity)
4. If score ≥ 0.65, it's a match. If multiple matches, pick highest score.

**Example scenario:**
- User has "blue_watch" (enrolled 80 embeddings) and "silver_watch" (enrolled 60 embeddings)
- Frame detects Watch at (100,50,150,100)
- Extract embedding → search `blue_watch` index → best score = 0.89
- Extract same embedding → search `silver_watch` index → best score = 0.52
- Result: "blue_watch" wins (0.89 > 0.65 threshold, 0.89 > 0.52)

---

### 9. Storage Decision — 15-Second Rule

**Lines:** ~1334-1366

```python
# Step 10: Decide whether to store an event (periodic sampling)
should_store = False

if object_info["last_stored"] is None:
    should_store = True  # First detection of this object (never stored before)
else:
    elapsed_seconds = (now_dt - object_info["last_stored"]).total_seconds()
    if elapsed_seconds >= STORE_INTERVAL_SECONDS:  # STORE_INTERVAL_SECONDS = 15
        should_store = True  # Enough time passed since last storage

if should_store and best_object_id is not None:
    # Compute human-readable location (e.g., "on white_chair (Bedroom)")
    location_text = get_location_text_for_event(
        detected_bbox, landmarks, environment_id, user_id
    )
    
    # Store event to database + save image to disk
    store_object_event(
        conn=conn,
        user_id=user_id,
        user_object_id=best_object_id,
        location_text=location_text,
        timestamp=now_dt,
        image=object_info["last_frame"],  # Frame stored in RAM from step 7
        object_type=detected_class,
        object_label=best_label
    )
    
    log_debug(f"[STORE_EVENT] Stored '{best_label}' at '{location_text}' with confidence={confidence:.3f}")
    
    object_info["last_stored"] = now_dt  # Update last storage timestamp
    stored_count += 1
```

**What happens:**
- **First detection:** Always store (last_stored = None)
- **Subsequent detections:** Store only if 15+ seconds passed since last storage
- **Example timeline:**
  - 14:00:00 — Detect "blue_watch" → STORE (first time)
  - 14:00:05 — Detect "blue_watch" → SKIP (only 5 seconds passed)
  - 14:00:10 — Detect "blue_watch" → SKIP (only 10 seconds passed)
  - 14:00:15 — Detect "blue_watch" → STORE (15 seconds passed)
  - 14:00:20 — Detect "blue_watch" → SKIP (only 5 seconds since last storage)

**Why 15 seconds?** Prevents database spam. If watch detected continuously for 1 hour, without this rule we'd store ~3600 events. With 15s rule, we store only ~240 events.

---

### 10. Location Text Computation

**Function:** `get_location_text_for_event()` — Lines ~499-565

```python
def get_location_text_for_event(
    object_bbox: tuple,
    detected_landmarks: List[tuple],
    environment_id: Optional[int],
    user_id: int
) -> str:
    # Step 1: Find closest landmark spatially (uses IoU + Euclidean distance)
    closest_landmark_class = infer_location(object_bbox, detected_landmarks)
    # Example: "on Chair" or "near Bed"
    
    if not closest_landmark_class:
        return "Unknown"  # No landmarks detected this frame
    
    # Step 2: Get user's custom labels for landmarks in THIS environment
    if not environment_id:
        return closest_landmark_class  # No environment matched, use raw YOLO class
    
    conn = db_conn()
    c = conn.cursor()
    c.execute("""
        SELECT landmark_class, user_label 
        FROM environment_landmarks 
        WHERE environment_id = ?
    """, (environment_id,))
    
    # Map YOLO class → user's custom label (e.g., "Chair" → "white_chair")
    landmark_mapping = {row[0]: row[1] for row in c.fetchall()}
    # Example: {"Chair": "study_chair", "Bed": "main_bed"}
    conn.close()
    
    # Step 3: Replace YOLO class with user's custom label
    for yolo_class, custom_label in landmark_mapping.items():
        if yolo_class in closest_landmark_class:
            closest_landmark_class = closest_landmark_class.replace(yolo_class, custom_label)
            # Example: "on Chair" → "on study_chair"
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
            # Example: "on study_chair (Bedroom)"
            return result
    
    return closest_landmark_class
```

**Function:** `infer_location()` — Lines ~253-279

```python
def infer_location(obj_bbox, landmarks: List[tuple]) -> Optional[str]:
    if not landmarks:
        return None
    
    # Strategy 1: Check for physical overlap (object resting ON landmark)
    best_iou = 0.0
    best_name = None
    
    for lname, lbbox in landmarks:
        i = bbox_iou(obj_bbox, lbbox)  # Intersection over Union
        # Example: obj_bbox overlaps 30% with chair_bbox → iou=0.30
        
        if i > best_iou:
            best_iou = i
            best_name = lname
    
    if best_name and best_iou > 0.25:  # Overlap threshold for 'on' relationship
        return f"on {best_name}"  # Example: "on Chair"
    
    # Strategy 2: Find closest landmark by distance (object NEAR landmark)
    near = nearest_landmark(obj_bbox, landmarks)
    # Computes Euclidean distance between centers
    
    if near:
        return f"near {near}"  # Example: "near Bed"
    
    return None
```

**How "on" vs "near" is decided:**

| Scenario | IoU | Distance | Result |
|---|---|---|---|
| Watch resting on chair | 0.35 | N/A | `"on Chair"` (IoU > 0.25) |
| Watch on floor near bed | 0.02 | 150px | `"near Bed"` (IoU < 0.25, closest landmark) |
| Watch in hand (no landmarks) | 0.0 | N/A | `"Unknown"` |

**Full location text examples:**

| Detected landmarks | Environment matched | User labels | Final location_text |
|---|---|---|---|
| Chair (IoU=0.30) | Bedroom | {"Chair": "study_chair"} | `"on study_chair (Bedroom)"` |
| Bed (distance=100px) | Bedroom | {"Bed": "main_bed"} | `"near main_bed (Bedroom)"` |
| Chair (IoU=0.28) | None | N/A | `"on Chair"` |
| None | N/A | N/A | `"Unknown"` |

---

### 11. Image Storage

**Function:** `store_object_event()` — Lines ~386-449

```python
def store_object_event(
    conn: sqlite3.Connection,
    user_id: int,
    user_object_id: int,
    location_text: str,
    timestamp: datetime,
    image: np.ndarray,  # BGR frame stored in RAM from step 7
    object_type: str,
    object_label: str
) -> bool:
    try:
        # Create folder: reid_store/users/{user_id}/images/{object_type}/
        user_folder = user_dir(user_id)
        obj_type_folder = os.path.join(user_folder, "images", object_type.replace(" ", "_"))
        os.makedirs(obj_type_folder, exist_ok=True)
        
        # Save image with timestamp for uniqueness and chronological ordering
        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")  # 20260302_143522_123456
        image_path = os.path.join(obj_type_folder, f"{object_label}_{ts}.jpg")
        cv2.imwrite(image_path, image)  # Write frame to disk
        # Example: "reid_store/users/5/images/Watch/blue_watch_20260302_143522_123456.jpg"
        
        # Record event in database
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
```

**What happens:**
- Frame stored in RAM (from step 7) is now written to disk as JPEG
- Filename includes object label + timestamp for uniqueness
- Database record links: user → object label → location → timestamp → image path
- Typical file size: ~30-80 KB per event

**Example folder structure:**
```
reid_store/
  users/
    5/
      images/
        Watch/
          blue_watch_20260302_143522_123456.jpg
          blue_watch_20260302_143537_987654.jpg
        Wallet/
          black_wallet_20260302_144102_456789.jpg
```

---

### 12. Disappearance Detection Loop

**Lines:** ~1368-1416

```python
# Step 11: Handle object disappearance (not detected this frame)
for obj_label, obj_info in list(object_state.items()):
    # Example: obj_label = "blue_watch"
    
    # Skip if this labeled object WAS detected this frame
    if obj_label in detected_labels_this_frame:
        continue
    
    # Skip if object never seen (state entry created but never detected)
    if obj_info["last_seen"] is None:
        continue
    
    # Increment counter for frames without detection
    obj_info["missing_ticks"] += 1
    # Example: missing_ticks was 5 → now 6
    
    # After DISAPPEAR_TICKS (20) consecutive frames without seeing object,
    # register a final disappearance event at last known location
    if obj_info["missing_ticks"] >= DISAPPEAR_TICKS:
        if obj_info["last_frame"] is not None and obj_info["last_bbox"] is not None:
            final_time = obj_info["last_seen"]
            # Timestamp of LAST time we saw it (not current time)
            
            # Only store disappearance if we haven't already stored this timepoint
            if obj_info["last_stored"] is None or final_time > obj_info["last_stored"]:
                user_object_id = obj_info.get("user_object_id")
                generic_type = obj_info.get("generic_type")
                
                # Compute location at LAST seen position
                location_text = get_location_text_for_event(
                    obj_info["last_bbox"], landmarks, environment_id, user_id
                )
                
                # Store event with LAST seen frame and timestamp
                store_object_event(
                    conn=conn,
                    user_id=user_id,
                    user_object_id=user_object_id,
                    location_text=location_text,
                    timestamp=final_time,  # NOT current time
                    image=obj_info["last_frame"],  # Frame from when last seen
                    object_type=generic_type,
                    object_label=obj_label
                )
                stored_count += 1
        
        # Reset state for this labeled object (prepare for re-detection)
        obj_info["missing_ticks"] = 0
        obj_info["last_seen"] = None
        obj_info["last_frame"] = None
        obj_info["last_bbox"] = None
        obj_info["last_embedding"] = None
```

**What happens:**
- For each tracked object NOT detected this frame, increment `missing_ticks`
- If `missing_ticks` reaches 20 (at 10 FPS = 2 seconds of absence):
  - Store event using LAST seen frame/timestamp/location
  - Reset tracking state
- This creates a "final sighting" event before object leaves camera view

**Example timeline:**
```
Frame 100: Detect "blue_watch" at (100,50,150,100) → Update state, missing_ticks=0
Frame 101: Detect "blue_watch" at (105,52,155,102) → Update state, missing_ticks=0
Frame 102: NOT detected → missing_ticks=1
Frame 103: NOT detected → missing_ticks=2
...
Frame 121: NOT detected → missing_ticks=20 → STORE disappearance event (using Frame 101's data)
Frame 122: NOT detected → state reset, missing_ticks=0
```

**Why store disappearance?** Captures "last known location" even if 15-second rule not met. Example: user picks up watch at 14:00:05 (only 5 seconds since last storage) and walks away. Without disappearance logic, we'd never store this sighting. With it, we store at 14:00:07 (2 seconds later when missing_ticks=20).

---

### 13. Frame Processing Complete

**Lines:** ~1418-1420

```python
# Step 12: Auto-cleanup old events if threshold reached
cleanup_if_needed(conn, user_id)
conn.close()
return {"status": "ok", "stored": stored_count}
```

**What happens:**
- Check if cumulative active runtime > 4 hours → delete events older than 4 hours
- Close database connection
- Return success status with count of events stored this frame

---

## Part B: Visual Flow Diagrams

### Diagram 1: High-Level Frame Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Camera Frame Arrives (JPEG bytes from ingest_ipcam.py)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Decode JPEG → BGR numpy array (H×W×3)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. YOLO Inference → List of bounding boxes + class IDs         │
│     Example: [(Watch, (100,50,150,100)), (Chair, (200,150,...))]│
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Categorize Detections                                       │
│     personal = [(Watch, bbox)]                                  │
│     landmarks = [(Chair, bbox), (Bed, bbox)]                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Environment Inference (from landmarks)                      │
│     detected_classes = {Chair, Bed}                             │
│     → Check user's registered environments                      │
│     → "Bedroom" has {Chair, Bed, Night Table}                   │
│     → Intersection {Chair, Bed} → Match!                        │
│     environment_id = 1, env_label = "Bedroom"                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. For Each Personal Object Detection:                         │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ a. Extract CLIP embedding from bbox crop                ││
│     │    embedding = 512-dim float vector                     ││
│     └─────────────────┬───────────────────────────────────────┘│
│                       │                                          │
│                       ▼                                          │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ b. Match against ALL candidate FAISS indices            ││
│     │    Candidates for "Watch": [blue_watch, silver_watch]   ││
│     │    → Query blue_watch.index → score = 0.89              ││
│     │    → Query silver_watch.index → score = 0.52            ││
│     │    → Best match: blue_watch (0.89 > 0.65 threshold)     ││
│     └─────────────────┬───────────────────────────────────────┘│
│                       │                                          │
│                       ▼                                          │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ c. Update tracking state                                ││
│     │    object_state["blue_watch"] = {                       ││
│     │      last_seen: now                                     ││
│     │      last_frame: frame.copy()  ← STORED IN RAM          ││
│     │      last_bbox: (100,50,150,100)                        ││
│     │      missing_ticks: 0                                   ││
│     │    }                                                     ││
│     └─────────────────┬───────────────────────────────────────┘│
│                       │                                          │
│                       ▼                                          │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ d. Check storage rule (15-second rule)                  ││
│     │    IF first detection OR 15+ seconds passed:            ││
│     │      → Compute location_text                            ││
│     │      → Save frame to disk as JPEG                       ││
│     │      → Insert row into events table                     ││
│     │    ELSE:                                                 ││
│     │      → Skip storage (keep in RAM only)                  ││
│     └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. For Each Tracked Object NOT Detected This Frame:            │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ a. Increment missing_ticks counter                      ││
│     │    missing_ticks: 5 → 6                                 ││
│     └─────────────────┬───────────────────────────────────────┘│
│                       │                                          │
│                       ▼                                          │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ b. If missing_ticks >= 20:                              ││
│     │    → Store disappearance event (using LAST seen data)   ││
│     │    → Reset state (missing_ticks=0, last_seen=None)      ││
│     └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. Cleanup old events if 4 hours of active runtime passed      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. Return {"status": "ok", "stored": N}                        │
└─────────────────────────────────────────────────────────────────┘
```

---

### Diagram 2: Session State Evolution (3 Frames Example)

**Scenario:** User has "blue_watch" and "silver_watch" registered. Camera detects a watch.

```
═══════════════════════════════════════════════════════════════════════════════
FRAME 100 (t=0s)
═══════════════════════════════════════════════════════════════════════════════

YOLO Detection:
  [Watch at (100, 50, 150, 100)]

CLIP Matching:
  → Extract embedding from (100,50,150,100)
  → Query blue_watch FAISS: score = 0.89 ✓
  → Query silver_watch FAISS: score = 0.51 ✗
  → Winner: "blue_watch"

Session State BEFORE:
  object_state = {
    "blue_watch": {
      last_stored: None,
      last_seen: None,
      last_frame: None,
      last_bbox: None,
      missing_ticks: 0
    },
    "silver_watch": { ... }
  }

Session State AFTER:
  object_state = {
    "blue_watch": {
      last_stored: 2026-03-02 14:00:00,  ← STORED (first detection)
      last_seen: 2026-03-02 14:00:00,
      last_frame: <frame_100_copy>,
      last_bbox: (100, 50, 150, 100),
      missing_ticks: 0
    },
    "silver_watch": { (unchanged) }
  }

Storage Decision:
  ✓ STORE EVENT (first detection)
    - location_text: "on study_chair (Bedroom)"
    - image_path: "reid_store/users/5/images/Watch/blue_watch_20260302_140000_123456.jpg"
    - timestamp: 2026-03-02 14:00:00

───────────────────────────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════════════════════════
FRAME 105 (t=0.5s, 5 frames later at 10 FPS)
═══════════════════════════════════════════════════════════════════════════════

YOLO Detection:
  [Watch at (105, 52, 155, 102)]  ← Slightly moved

CLIP Matching:
  → Extract embedding from (105,52,155,102)
  → Query blue_watch FAISS: score = 0.91 ✓
  → Query silver_watch FAISS: score = 0.49 ✗
  → Winner: "blue_watch"

Session State BEFORE:
  object_state = {
    "blue_watch": {
      last_stored: 2026-03-02 14:00:00,  ← 0.5 seconds ago
      last_seen: 2026-03-02 14:00:00,
      last_frame: <frame_100_copy>,
      last_bbox: (100, 50, 150, 100),
      missing_ticks: 0
    }
  }

Session State AFTER:
  object_state = {
    "blue_watch": {
      last_stored: 2026-03-02 14:00:00,  ← NOT updated (15s rule)
      last_seen: 2026-03-02 14:00:00.5,  ← Updated
      last_frame: <frame_105_copy>,      ← Updated (overwrites old frame)
      last_bbox: (105, 52, 155, 102),    ← Updated
      missing_ticks: 0
    }
  }

Storage Decision:
  ✗ SKIP STORAGE
    - Reason: Only 0.5 seconds passed since last_stored
    - Threshold: 15 seconds (STORE_INTERVAL_SECONDS)
    - Frame stored in RAM but not saved to disk

───────────────────────────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════════════════════════
FRAME 255 (t=15.5s, 150 frames later)
═══════════════════════════════════════════════════════════════════════════════

YOLO Detection:
  [Watch at (120, 60, 170, 110)]  ← Moved again

CLIP Matching:
  → Extract embedding from (120,60,170,110)
  → Query blue_watch FAISS: score = 0.88 ✓
  → Query silver_watch FAISS: score = 0.50 ✗
  → Winner: "blue_watch"

Session State BEFORE:
  object_state = {
    "blue_watch": {
      last_stored: 2026-03-02 14:00:00,  ← 15.5 seconds ago
      last_seen: 2026-03-02 14:00:15.5,  (from Frame 254)
      last_frame: <frame_254_copy>,
      last_bbox: (119, 59, 169, 109),
      missing_ticks: 0
    }
  }

Session State AFTER:
  object_state = {
    "blue_watch": {
      last_stored: 2026-03-02 14:00:15.5,  ← UPDATED (stored again)
      last_seen: 2026-03-02 14:00:15.5,
      last_frame: <frame_255_copy>,
      last_bbox: (120, 60, 170, 110),
      missing_ticks: 0
    }
  }

Storage Decision:
  ✓ STORE EVENT (15+ seconds passed)
    - location_text: "on study_chair (Bedroom)"
    - image_path: "reid_store/users/5/images/Watch/blue_watch_20260302_140015_654321.jpg"
    - timestamp: 2026-03-02 14:00:15.5
    - Elapsed since last storage: 15.5 seconds

═══════════════════════════════════════════════════════════════════════════════
```

---

### Diagram 3: Disappearance Tracking (missing_ticks Logic)

**Scenario:** Watch detected continuously, then user picks it up and walks away.

```
═══════════════════════════════════════════════════════════════════════════════
Timeline (10 FPS = 0.1s per frame)
═══════════════════════════════════════════════════════════════════════════════

Frame 100 (t=0.0s):   Detect blue_watch → Update state, missing_ticks=0
Frame 101 (t=0.1s):   Detect blue_watch → Update state, missing_ticks=0
Frame 102 (t=0.2s):   Detect blue_watch → Update state, missing_ticks=0
Frame 103 (t=0.3s):   Detect blue_watch → Update state, missing_ticks=0
Frame 104 (t=0.4s):   Detect blue_watch → Update state, missing_ticks=0
Frame 105 (t=0.5s):   Detect blue_watch → Update state, missing_ticks=0
                      → STORE EVENT (first detection)
                      → last_stored = t=0.5s

Frame 106 (t=0.6s):   Detect blue_watch → Update state, missing_ticks=0
Frame 107 (t=0.7s):   Detect blue_watch → Update state, missing_ticks=0
Frame 108 (t=0.8s):   Detect blue_watch → Update state, missing_ticks=0
                      last_seen = t=0.8s  ← LAST DETECTION
                      last_frame = <frame_108_copy>
                      last_bbox = (110, 55, 160, 105)

─── USER PICKS UP WATCH AND WALKS AWAY ───────────────────────────────────────

Frame 109 (t=0.9s):   NOT detected → missing_ticks=1
Frame 110 (t=1.0s):   NOT detected → missing_ticks=2
Frame 111 (t=1.1s):   NOT detected → missing_ticks=3
Frame 112 (t=1.2s):   NOT detected → missing_ticks=4
Frame 113 (t=1.3s):   NOT detected → missing_ticks=5
...
Frame 128 (t=2.8s):   NOT detected → missing_ticks=20 ← THRESHOLD REACHED
                      → STORE DISAPPEARANCE EVENT
                         - timestamp: t=0.8s (last_seen, NOT current time)
                         - image: <frame_108_copy> (last seen frame)
                         - location_text: "on study_chair (Bedroom)" (computed from frame 108 bbox + landmarks)
                         - Reason: Only 0.3 seconds since last storage (t=0.5s → t=0.8s)
                                   but object disappeared, so store final sighting
                      → Reset state: missing_ticks=0, last_seen=None, last_frame=None

Frame 129 (t=2.9s):   NOT detected → missing_ticks=0 (already reset)
Frame 130 (t=3.0s):   NOT detected → missing_ticks=0 (no increment after reset)

═══════════════════════════════════════════════════════════════════════════════

Events Table Result:
┌──────────┬────────────────┬──────────────────────────────┬────────────────┐
│ event_id │ user_object_id │        location_text         │   timestamp    │
├──────────┼────────────────┼──────────────────────────────┼────────────────┤
│   42     │       2        │ on study_chair (Bedroom)     │  t=0.5s        │  ← First detection
│   43     │       2        │ on study_chair (Bedroom)     │  t=0.8s        │  ← Disappearance event
└──────────┴────────────────┴──────────────────────────────┴────────────────┘

Total duration: 0.3 seconds between events (15-second rule bypassed by disappearance)
```

---

### Diagram 4: Multi-Instance Tracking (Same Class, Different Labels)

**Scenario:** User has "blue_watch" and "silver_watch". Camera sees both simultaneously.

```
═══════════════════════════════════════════════════════════════════════════════
FRAME 100
═══════════════════════════════════════════════════════════════════════════════

YOLO Detections:
  [Watch at (100, 50, 150, 100)]   ← Detection #1
  [Watch at (300, 80, 350, 130)]   ← Detection #2

User's Registered Objects:
  1. blue_watch (user_object_id=1)
  2. silver_watch (user_object_id=2)

─────────────────────────────────────────────────────────────────────────────
Processing Detection #1: Watch at (100, 50, 150, 100)
─────────────────────────────────────────────────────────────────────────────

Step 1: Extract CLIP embedding
  embedding_1 = [0.12, -0.34, 0.56, ...] (512 dims)

Step 2: Match against all Watch candidates
  Candidates: [(1, "blue_watch"), (2, "silver_watch")]
  
  Query blue_watch FAISS (user_object_id=1):
    → Nearest neighbor distance: 0.89
    → Above threshold (0.65) ✓
  
  Query silver_watch FAISS (user_object_id=2):
    → Nearest neighbor distance: 0.51
    → Below threshold (0.65) ✗
  
  Winner: blue_watch (score=0.89)

Step 3: Update session state
  object_state["blue_watch"] = {
    last_seen: now,
    last_frame: <frame_copy>,
    last_bbox: (100, 50, 150, 100),
    missing_ticks: 0
  }

─────────────────────────────────────────────────────────────────────────────
Processing Detection #2: Watch at (300, 80, 350, 130)
─────────────────────────────────────────────────────────────────────────────

Step 1: Extract CLIP embedding
  embedding_2 = [0.45, -0.12, 0.78, ...] (512 dims)

Step 2: Match against all Watch candidates
  Candidates: [(1, "blue_watch"), (2, "silver_watch")]
  
  Query blue_watch FAISS (user_object_id=1):
    → Nearest neighbor distance: 0.48
    → Below threshold (0.65) ✗
  
  Query silver_watch FAISS (user_object_id=2):
    → Nearest neighbor distance: 0.92
    → Above threshold (0.65) ✓
  
  Winner: silver_watch (score=0.92)

Step 3: Update session state
  object_state["silver_watch"] = {
    last_seen: now,
    last_frame: <frame_copy>,
    last_bbox: (300, 80, 350, 130),
    missing_ticks: 0
  }

─────────────────────────────────────────────────────────────────────────────

Final Session State:
  object_state = {
    "blue_watch": {
      last_seen: 2026-03-02 14:00:00,
      last_bbox: (100, 50, 150, 100),
      missing_ticks: 0
    },
    "silver_watch": {
      last_seen: 2026-03-02 14:00:00,
      last_bbox: (300, 80, 350, 130),
      missing_ticks: 0
    }
  }

detected_labels_this_frame = {"blue_watch", "silver_watch"}

Result:
  ✓ Both watches tracked independently
  ✓ Each matched to correct label via FAISS similarity
  ✓ Both state entries updated
```

---

### Diagram 5: Location Text Computation (on/near Logic)

```
═══════════════════════════════════════════════════════════════════════════════
SCENARIO 1: Watch OVERLAPPING Chair (on logic)
═══════════════════════════════════════════════════════════════════════════════

Frame:
  ┌────────────────────────────────────┐
  │                                    │
  │      ┌──────────┐                 │
  │      │  Watch   │  ← (100,50,150,100)
  │      │  bbox    │                 │
  │  ┌───┴──────────┴────┐            │
  │  │                    │            │
  │  │   Chair bbox       │            │
  │  │   (90,70,200,180)  │            │
  │  │                    │            │
  │  └────────────────────┘            │
  │                                    │
  └────────────────────────────────────┘

Step 1: Compute IoU (Intersection over Union)
  watch_bbox = (100, 50, 150, 100)
  chair_bbox = (90, 70, 200, 180)
  
  Intersection rectangle:
    x1 = max(100, 90) = 100
    y1 = max(50, 70) = 70
    x2 = min(150, 200) = 150
    y2 = min(100, 180) = 100
    
  Intersection area = (150-100) * (100-70) = 50 * 30 = 1500 px²
  Watch area = (150-100) * (100-50) = 50 * 50 = 2500 px²
  Chair area = (200-90) * (180-70) = 110 * 110 = 12100 px²
  
  IoU = 1500 / (2500 + 12100 - 1500) = 1500 / 13100 = 0.115

Step 2: Check IoU threshold
  IoU = 0.115 < 0.25 threshold → NOT "on"
  
  (In this example, IoU too low despite visual overlap due to chair being much larger)

Step 3: Fallback to distance logic
  watch_center = ((100+150)/2, (50+100)/2) = (125, 75)
  chair_center = ((90+200)/2, (70+180)/2) = (145, 125)
  distance² = (145-125)² + (125-75)² = 20² + 50² = 400 + 2500 = 2900
  
  Closest landmark: Chair

Result:
  infer_location() returns "near Chair"

Step 4: Map to user label
  environment_id = 1 (Bedroom)
  landmark_mapping = {"Chair": "study_chair"}
  "near Chair" → "near study_chair"

Step 5: Add environment context
  Final: "near study_chair (Bedroom)"

═══════════════════════════════════════════════════════════════════════════════
SCENARIO 2: Watch RESTING ON Table (on logic)
═══════════════════════════════════════════════════════════════════════════════

Frame:
  ┌────────────────────────────────────┐
  │                                    │
  │  ┌────────────────────────────┐   │
  │  │                            │   │
  │  │   Table bbox               │   │
  │  │   (50,80,350,120)          │   │
  │  │                            │   │
  │  │     ┌───────┐              │   │
  │  │     │ Watch │              │   │
  │  │     │ bbox  │              │   │
  │  │     │(100,85,│              │   │
  │  │     │140,115)│              │   │
  │  │     └───────┘              │   │
  │  │                            │   │
  │  └────────────────────────────┘   │
  │                                    │
  └────────────────────────────────────┘

Step 1: Compute IoU
  watch_bbox = (100, 85, 140, 115)
  table_bbox = (50, 80, 350, 120)
  
  Intersection rectangle:
    x1 = max(100, 50) = 100
    y1 = max(85, 80) = 85
    x2 = min(140, 350) = 140
    y2 = min(115, 120) = 115
    
  Intersection area = (140-100) * (115-85) = 40 * 30 = 1200 px²
  Watch area = (140-100) * (115-85) = 40 * 30 = 1200 px²
  Table area = (350-50) * (120-80) = 300 * 40 = 12000 px²
  
  IoU = 1200 / (1200 + 12000 - 1200) = 1200 / 12000 = 0.10

Step 2: Check IoU threshold
  IoU = 0.10 < 0.25 threshold → NOT "on"
  
  (Watch fully contained within table, but IoU low because table much larger)

Adjusted Scenario (watch takes more table area):
  watch_bbox = (100, 85, 250, 115)  ← Wider watch
  table_bbox = (50, 80, 350, 120)
  
  Intersection area = (250-100) * (115-85) = 150 * 30 = 4500 px²
  Watch area = 150 * 30 = 4500 px²
  Table area = 300 * 40 = 12000 px²
  
  IoU = 4500 / (4500 + 12000 - 4500) = 4500 / 12000 = 0.375

Step 2: Check IoU threshold
  IoU = 0.375 > 0.25 threshold ✓ → "on"

Result:
  infer_location() returns "on Table"

Step 4: Map to user label
  environment_id = 1 (Bedroom)
  landmark_mapping = {"Table": "night_table"}
  "on Table" → "on night_table"

Step 5: Add environment context
  Final: "on night_table (Bedroom)"

═══════════════════════════════════════════════════════════════════════════════
SCENARIO 3: No Environment Matched
═══════════════════════════════════════════════════════════════════════════════

Detected landmarks: [("Chair", bbox)]
User's environments: [("Bedroom", {Bed, Night Table}), ("Living Room", {Sofa, TV})]
Intersection: {Chair} ∩ {Bed, Night Table} = ∅ (empty)
              {Chair} ∩ {Sofa, TV} = ∅ (empty)

Result: environment_id = None

Location text computation:
  infer_location() → "on Chair" (raw YOLO class)
  environment_id = None → skip mapping
  Final: "on Chair" (no custom label, no environment context)

═══════════════════════════════════════════════════════════════════════════════
```

---

## Summary

**Key takeaways:**

1. **State is ephemeral (RAM only):** `SESSION_STATE` dict holds tracking info until session stops. Not persisted to database.

2. **Frame storage is deferred:** Frame copied to RAM immediately, but only written to disk when storage rule met (15s OR disappearance).

3. **15-second rule prevents spam:** Without it, 1 hour of continuous detection = 36,000 events. With it, ~240 events.

4. **Disappearance tracking captures exit:** If watch detected 5 times in 0.5 seconds then user walks away, disappearance logic stores final sighting even if 15s rule not met.

5. **FAISS enables multi-instance tracking:** System distinguishes "blue_watch" from "silver_watch" even though YOLO only says "Watch". CLIP embeddings + FAISS matching provides identity.

6. **Location inference is two-stage:** IoU overlap (for "on") → Euclidean distance (for "near"). Then map YOLO class → user label → add environment context.

7. **Environment matching is simple:** ANY detected landmark class matching registered environment → that's the environment. No scoring/confidence.

---

**This document explains every step of the TRACK mode frame processing pipeline, from camera bytes to database storage.**
