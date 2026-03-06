# Copilot Instructions — Vision Memory Assistant

## Project Overview
A personal item re-identification system: a YOLOv8 custom model detects personal objects (Watch, Wallet, etc.) and environmental landmarks (Chair, Bed, etc.) from an IP camera stream. CLIP embeddings + FAISS indices identify *which* specific item was seen and *where*, storing timestamped events in SQLite for later "where did I last see my keys?" queries.

## Architecture
```
ui_streamlit.py          ← Streamlit frontend (routing only)
ui_components/           ← Modular UI pages: dashboard, tracking, query, environment, login
backend_api.py           ← FastAPI backend (http://127.0.0.1:8000), all ML logic lives here
ingest_ipcam.py          ← Subprocess launched by UI; reads IP camera frames → POST /session_frame
constants.py             ← Single source of truth for all config (thresholds, classes, paths)
logger.py                ← Session-grouped logging to logs/{SESSION_ID}/app.log
reid_store/              ← All persistent data
  main.sqlite            ← Users, objects, environments, events, sessions
  users/{id}/faiss/object/{obj_id}.index          ← CLIP embeddings per labeled object
  users/{id}/faiss/environment/{env_id}/{lm_id}.index  ← CLIP embeddings per landmark
  users/{id}/images/{ObjectType}/{label}_{ts}.jpg ← Saved detection frames
```

## Developer Workflows

**Run the full application** (starts backend + UI in separate terminals):
```bat
run_application.bat
```
Or manually:
```
conda activate itemmem
uvicorn backend_api:app --reload       # terminal 1
streamlit run ui_streamlit.py          # terminal 2
```

**Conda environment**: `itemmem` (Python 3.10). Install deps via `pip install -r requirements.txt`.

**Reset logs session**: Delete `logs/.current_session` — next run creates a new `logs/{SESSION_ID}/` folder.

## Key Patterns & Conventions

### Centralized Config — always edit `constants.py`
`PERSONAL_CLASSES`, `LANDMARK_CLASSES`, `SIM_THRESHOLD` (0.65), `STORE_INTERVAL_SECONDS` (15), `DISAPPEAR_TICKS` (20), `BACKEND_FPS` (10), and all paths flow from here. Both backend and UI import from `constants.py`. Never hardcode these values elsewhere.

### FAISS Index Layout
All indices use `IndexFlatIP` (inner product = cosine similarity on L2-normalized 512-dim CLIP embeddings) wrapped in `IndexIDMap2`. Use `ensure_faiss_index(path)` before reading. Paths are always constructed via `faiss_path_for_object()` / `faiss_path_for_landmark()`.

### Session State Machine
`SESSION_STATE` dict in `backend_api.py` holds ephemeral live-session data keyed by `session_id` (UUID). Session modes: `TRACK`, `ENROLL_OBJECT`, `ENROLL_LANDMARK`, `TEST_ENVIRONMENT`. The `sessions` SQLite table persists status across Streamlit reruns.

### UI → Backend Communication
All UI API calls go through `utils.api_post(path, data)` and `utils.api_get(path, params)` in `ui_components/utils.py`. Backend is hardcoded at `BACKEND = "http://127.0.0.1:8000"`. Form data (not JSON) is used for POST requests.

### Camera Ingestion is a Subprocess
`ingest_ipcam.py` is launched by `tracking.launch_ingest()` via `subprocess.Popen`. It streams frames to `/session_frame`. The subprocess handle is stored in `st.session_state.ingest_proc` and must be terminated on stop/logout via `stop_ingest_if_running()`.

### Location Text Format
Events store a single `location_text` string like `"on white_chair (Bedroom)"`. This is assembled in `get_location_text_for_event()`: spatial inference (IoU/nearest centroid) → user's custom landmark label → environment name appended in parentheses.

### Object vs Landmark Distinction
- **Personal objects** (`PERSONAL_CLASSES`): tracked items with per-object FAISS indices; multi-instance distinguished by CLIP matching against enrolled embeddings.
- **Landmarks** (`LANDMARK_CLASSES`): furniture/fixtures providing spatial context only; matched via separate per-environment FAISS indices.

### Landmark Enrollment Flow
1. User creates an environment (`/add_environment`), then adds landmark definitions via `/add_environment_landmark` — each pairs a YOLO class (e.g. `"Chair"`) with a custom label (e.g. `"white_chair"`), yielding an `env_landmark_id`.
2. User starts an `ENROLL_LANDMARK` session passing `environment_id` + `landmark_id` (`env_landmark_id`).
3. Camera streams frames → for each frame, YOLO picks the largest bbox of the target class, extracts a CLIP embedding, appends to `reid_store/users/{id}/faiss/environment/{env_id}/{landmark_id}.index`.
4. User physically points the camera at one landmark at a time (multiple angles/lighting), then stops and repeats for the next. These embeddings disambiguate multiple landmarks of the same YOLO class in a room (e.g. two chairs).

### Query NLP Parsing (Two-Layer, UI-Side)
`/query` backend is a pure label lookup — it takes `user_label` exactly and returns events. All parsing is in `ui_components/query.py`:
1. Fetch the user's registered object labels, iterate them, and do a **case-insensitive substring check** against the free-text query (e.g. label `"black_wallet"` matches `"where is my black wallet"`).
2. If no registered label matches, fall back to hardcoded keyword checks (`"watch"`, `"wallet"`, `"key"`).

Voice queries are transcribed via `faster_whisper` to plain text, then fed through the same two-layer logic — no separate NLP model involved.

### `best.pt` — Custom YOLO Model
`best.pt` is trained on exactly the 12 classes in `constants.py`: `PERSONAL_CLASSES` (Watch, Wallet, Bike Key, Car Key) and `LANDMARK_CLASSES` (Bed, Computer Table, Cupboard, Dressing Table, Night Table, Laptop, Chair, Table). There are no retraining scripts in the repo. If you add/remove classes in `constants.py`, the YOLO training data labels must be kept in sync. `yolov8n.pt` is present but unused at runtime.

## Critical Files
- [constants.py](../constants.py) — change any threshold/class/path here first
- [backend_api.py](../backend_api.py) — `extract_embedding_and_match()`, `store_object_event()`, `find_environment_from_landmarks()` are the core tracking functions
- [ui_components/utils.py](../ui_components/utils.py) — `api_post` / `api_get` wrappers used by all UI components
- [ingest_ipcam.py](../ingest_ipcam.py) — camera loop; posts base64-encoded JPEG frames

## Gotchas
- `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` is set in `ui_streamlit.py` to prevent OpenMP conflicts from PyTorch + OpenCV.
- The `sessions` table has a `landmark_id` column added via an inline migration (`_ensure_sessions_columns`) — always use this pattern for schema changes.
- Voice query uses `faster_whisper` (base model, CPU, int8); the Whisper model is cached with `@st.cache_resource`.
- `find_environment_from_landmarks()` uses a simple ANY-intersection strategy — if detected YOLO landmark classes overlap with any registered environment's landmark classes, that environment is returned. No scoring or hysteresis.
