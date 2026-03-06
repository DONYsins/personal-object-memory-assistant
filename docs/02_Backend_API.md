# Backend API Reference — `backend_api.py`

---

## Endpoint Summary Table

| Method | Path | Purpose |
|---|---|---|
| POST | `/register_user` | Create a new user account |
| POST | `/login` | Authenticate user, get user_id |
| GET | `/get_user_info` | Get username and created_at for a user_id |
| GET | `/dashboard` | Get user's objects and environments |
| POST | `/add_personal_object` | Register a new labeled personal object |
| POST | `/delete_personal_object` | Delete object, its events, and FAISS index |
| POST | `/add_environment` | Create a new environment (room) |
| POST | `/delete_environment` | Delete environment and its landmarks |
| GET | `/get_environment_landmarks` | List all landmarks defined for an environment |
| POST | `/add_environment_landmark` | Add a landmark definition to an environment |
| POST | `/delete_environment_landmark` | Remove a landmark definition |
| POST | `/start_session` | Begin TRACK, ENROLL_OBJECT, ENROLL_LANDMARK, or TEST_ENVIRONMENT session |
| POST | `/stop_session` | Stop a running session |
| GET | `/session_status` | Poll session status (RUNNING / STOPPED) |
| POST | `/session_frame` | Receive a camera frame — core frame processing endpoint |
| POST | `/query` | Retrieve stored events for a given object label |
| POST | `/manual_cleanup` | Delete events older than N minutes |

---

## Auth

### `POST /register_user`
**Purpose:** Create a new user with a bcrypt-hashed password.

**Request (form data):**
```
username=alice
password=mysecret
```

**Response:**
```json
{"status": "success"}
{"status": "error", "message": "Username already exists"}
```

**DB writes:** INSERT into `users`
**Files created:** None
**Error cases:** Duplicate username → IntegrityError caught, returns error JSON

---

### `POST /login`
**Purpose:** Validate credentials, return `user_id`.

**Request (form data):**
```
username=alice
password=mysecret
```

**Response:**
```json
{"status": "success", "user_id": 5}
{"status": "error", "message": "Invalid credentials"}
```

**DB reads:** `users` (password_hash lookup)
**Files created:** `reid_store/users/{user_id}/` folder tree created via `user_dir()` on success
**Error cases:** User not found, bcrypt mismatch

---

### `GET /get_user_info`
**Purpose:** Get display name for the logged-in user.

**Request (query params):**
```
?user_id=5
```

**Response:**
```json
{"status": "success", "username": "alice", "created_at": "2026-03-01T10:00:00", "user_id": 5}
{"status": "error", "message": "User not found"}
```

**DB reads:** `users`

---

## Dashboard

### `GET /dashboard`
**Purpose:** One-call fetch of all objects and environments for the dashboard.

**Request:**
```
?user_id=5
```

**Response:**
```json
{
  "objects": [
    {"user_object_id": 2, "generic_type": "Watch", "user_label": "black_watch"}
  ],
  "environments": [
    {"environment_id": 1, "environment_label": "Bedroom"}
  ]
}
```

**DB reads:** `personal_objects`, `environments`

---

## Personal Objects

### `POST /add_personal_object`
**Purpose:** Register a new labeled personal object under a user.

**Request (form data):**
```
user_id=5
generic_type=Watch
user_label=black_watch
```

**Response:**
```json
{"status": "success", "user_object_id": 2}
{"status": "error", "message": "Max objects reached"}
{"status": "error", "message": "A label with that name already exists"}
{"status": "error", "message": "Invalid object type"}
```

**DB writes:** INSERT into `personal_objects`
**Files created:** Empty FAISS index at `reid_store/users/5/faiss/object/2.index` (via `ensure_faiss_index`)
**Limits:** `MAX_OBJECTS_PER_USER = 10` (`constants.py`)
**Error cases:** `generic_type` not in `PERSONAL_CLASSES`, blank label, duplicate `(user_id, user_label)`

---

### `POST /delete_personal_object`
**Purpose:** Delete object, all its events (+ image files), and its FAISS index.

**Request (form data):**
```
user_id=5
user_object_id=2
```

**Response:**
```json
{"status": "success"}
{"status": "error", "message": "object not found"}
```

**DB writes:** DELETE from `events` (for this object), DELETE from `personal_objects`
**Files deleted:** All `image_path` files from events; FAISS index file

---

## Environments

### `POST /add_environment`
**Purpose:** Create a named environment (room context).

**Request (form data):**
```
user_id=5
environment_label=Bedroom
```

**Response:**
```json
{"status": "success", "environment_id": 1}
{"status": "error", "message": "Environment label already exists"}
```

**DB writes:** INSERT into `environments`
**Error cases:** Duplicate label per user

---

### `POST /delete_environment`
**Purpose:** Delete an environment and all its landmark definitions.

**Request (form data):**
```
user_id=5
environment_id=1
```

**Response:**
```json
{"status": "success"}
{"status": "error", "message": "environment not found"}
```

**DB writes:** DELETE from `environment_landmarks`, DELETE from `environments`
**Files deleted:** `env_landmarks_{environment_id}/` folder if it exists

---

### `GET /get_environment_landmarks`
**Purpose:** List all landmark definitions for one environment.

**Request:**
```
?user_id=5&environment_id=1
```

**Response:**
```json
{
  "status": "ok",
  "landmarks": [
    {"landmark_class": "Chair", "user_label": "study_chair", "environment_landmark_id": 3},
    {"landmark_class": "Bed",   "user_label": "main_bed",    "environment_landmark_id": 4}
  ]
}
```

**DB reads:** `environment_landmarks`

---

### `POST /add_environment_landmark`
**Purpose:** Add a landmark definition (YOLO class + custom label) to an environment.

**Request (form data):**
```
user_id=5
environment_id=1
landmark_class=Chair
user_label=study_chair
```

**Response:**
```json
{"status": "success", "message": "Added 'study_chair' as Chair"}
{"status": "error", "message": "Label 'study_chair' already exists in this environment"}
```

**DB writes:** INSERT into `environment_landmarks`
**Note:** Multiple landmarks of the same `landmark_class` are allowed (e.g. two chairs), as long as `user_label` differs.

---

### `POST /delete_environment_landmark`
**Purpose:** Remove a single landmark definition by label.

**Request (form data):**
```
user_id=5
environment_id=1
user_label=study_chair
```

**Response:**
```json
{"status": "success"}
{"status": "error", "message": "environment not found"}
```

**DB writes:** DELETE from `environment_landmarks` WHERE `environment_id` + `user_label`

---

## Sessions

### `POST /start_session`
**Purpose:** Begin a camera session in one of four modes.

**Modes:**
| Mode | Extra Required Params | What Happens |
|---|---|---|
| `TRACK` | none (optional: `environment_id`) | Sets up `SESSION_STATE[session_id]` with object tracking state |
| `ENROLL_OBJECT` | `user_object_id` | Frames will add CLIP embeddings to that object's FAISS index |
| `ENROLL_LANDMARK` | `environment_id`, `landmark_id` | Frames will add CLIP embeddings to that landmark's FAISS index |
| `TEST_ENVIRONMENT` | `environment_id` | Frames annotated with bounding boxes, returned as base64 JPEG |

**Request (form data):**
```
user_id=5
mode=TRACK
```
or for enrollment:
```
user_id=5
mode=ENROLL_OBJECT
user_object_id=2
```

**Response:**
```json
{"status": "success", "session_id": "abc-def-123-456"}
{"status": "error", "message": "Invalid mode"}
{"status": "error", "message": "user_object_id required"}
```

**DB writes:** INSERT into `sessions` (status=`RUNNING`)
**Memory:** Initialises `SESSION_STATE[session_id]` dict for TRACK and ENROLL_LANDMARK modes

---

### `POST /stop_session`
**Purpose:** Mark session as STOPPED and clear in-memory state.

**Request (form data):**
```
session_id=abc-def-123-456
```

**Response:**
```json
{"status": "success"}
{"status": "error", "message": "..."}
```

**DB writes:** UPDATE `sessions` SET `status=STOPPED`
**Memory:** `del SESSION_STATE[session_id]`

---

### `GET /session_status`
**Purpose:** Poll whether a session is still running (used by UI to detect camera disconnection).

**Request:**
```
?session_id=abc-def-123-456
```

**Response:**
```json
{"status": "success", "session_status": "RUNNING", "last_error": null, "mode": "TRACK"}
{"status": "success", "session_status": "STOPPED", "last_error": null, "mode": "TRACK"}
```

**DB reads:** `sessions`

---

## Core Frame Processing

### `POST /session_frame`
**Purpose:** Main processing endpoint. Receives one JPEG frame from `ingest_ipcam.py` and acts based on session mode.

**Request (multipart form):**
```
session_id=abc-def-123-456
file=<JPEG bytes>
```

**Common response:**
```json
{"status": "ok"}
{"status": "stopped"}
{"status": "error", "message": "Invalid session_id"}
{"status": "error", "message": "Bad frame"}
```

**Processing by mode:**

#### ENROLL_OBJECT
1. Look up `generic_type` and `user_label` for `user_object_id` in `personal_objects`
2. Find all YOLO detections of that class
3. Pick the largest bounding box
4. `get_embedding_from_bbox()` → 512-dim CLIP vector
5. `faiss.add_with_ids(embedding, timestamp_ms_id)` → write to `faiss/object/{user_object_id}.index`
6. Save debug JPEG to `enroll_debug_images/{generic_type}/{user_label}_{slot}.jpg` (1 per 2 sec)

**Response:**
```json
{"status": "ok", "note": "enrolled embedding for black_watch"}
{"status": "ok", "note": "no Watch detected in frame"}
```

**DB reads:** `personal_objects`
**Files written:** FAISS index, debug JPEG

---

#### ENROLL_LANDMARK
1. Look up `landmark_class` and `user_label` from `environment_landmarks` (using `landmark_id` from `SESSION_STATE`)
2. Find all YOLO detections of that class
3. Pick the largest bbox
4. Extract CLIP embedding
5. Append to `faiss/environment/{environment_id}/{landmark_id}.index`
6. Save debug JPEG to `landmark_debug_images/env_{environment_id}/{user_label}_{slot}.jpg`

**Response:**
```json
{"status": "ok", "note": "enrolled embedding for study_chair"}
{"status": "ok", "note": "no Chair detected in frame"}
```

**DB reads:** `environment_landmarks`
**Files written:** FAISS index, debug JPEG

---

#### TRACK
Full pipeline per frame:

1. `detect_objects_in_frame(frame)` → `personal[]`, `landmarks[]`
2. `find_environment_from_landmarks(landmarks, user_id)` → `environment_id`, `env_label`
3. For each detected personal object:
   - `extract_embedding_and_match(frame, bbox, user_id, candidates)` → `best_label`, `confidence`
   - If `confidence >= SIM_THRESHOLD (0.65)` and time since last store `>= STORE_INTERVAL_SECONDS (15)`:
     - `get_location_text_for_event()` → e.g. `"on study_chair (Bedroom)"`
     - `store_object_event()` → save JPEG + INSERT into `events`
4. For objects not seen this frame: `missing_ticks += 1`; at `DISAPPEAR_TICKS (20)` store final disappearance event

**Response:**
```json
{"status": "ok", "stored": 1}
{"status": "ok", "stored": 0}
```

**DB reads:** `personal_objects`, `environments`, `environment_landmarks`
**DB writes:** INSERT into `events`
**Files written:** JPEG to `images/{generic_type}/{label}_{timestamp}.jpg`
**Debug logs written:** `app_debug.log` — only when detections found (`[DETECTION]`, `[TRACK_MATCH]`, `[ENV_MATCH]`, `[LOCATION_IoU]`, `[LOCATION_TEXT]`, `[STORE_EVENT]`)

---

#### TEST_ENVIRONMENT
1. Detect all objects and landmarks
2. For each landmark: draw cyan bounding box, label with user custom name
3. For each personal object: attempt FAISS match, draw green bounding box with matched label
4. Encode annotated frame as base64 JPEG

**Response:**
```json
{
  "status": "ok",
  "frame": "<base64-encoded JPEG>",
  "detections": {"personal_count": 1, "landmark_count": 2}
}
```

**Files written:** Annotated debug frames to `test_env_debug_images/frame_{slot}.jpg`

---

## Query & Cleanup

### `POST /query`
**Purpose:** Return the N most recent events for a given object label.

**Request (form data):**
```
user_id=5
user_label=black_watch
k=3
```

**Response:**
```json
{
  "status": "ok",
  "results": [
    {
      "event_id": 42,
      "location_text": "on study_chair (Bedroom)",
      "timestamp": "2026-03-02T14:35:22.123456",
      "image_path": "reid_store/users/5/images/Watch/black_watch_20260302_143522_123456.jpg"
    }
  ]
}
```

**DB reads:** `personal_objects` (to resolve `user_label` → `user_object_id`), `events`
**Note:** Label matching is exact string match; NLP parsing happens in `ui_components/query.py` before calling this endpoint.

---

### `POST /manual_cleanup`
**Purpose:** Delete events (and their image files) older than a given number of minutes.

**Request (form data):**
```
user_id=5
older_than_minutes=240
```

**Response:**
```json
{"status": "ok", "deleted": 7}
```

**DB reads + writes:** SELECT then DELETE from `events`
**Files deleted:** Image files referenced by deleted event rows
**Note:** Auto-cleanup also runs inside TRACK frames based on `RETENTION_ACTIVE_HOURS` (`cleanup_if_needed()`, `backend_api.py` line ~575)
