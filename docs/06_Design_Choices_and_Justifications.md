# Design Choices and Justifications

---

## 1. Major Libraries

### FastAPI (`backend_api.py`)

**Why used:**
- Async support via `async def` — critical for `session_frame` which needs to `await file.read()` without blocking other requests
- Automatic request validation via Python type hints (Form fields, optional params)
- Auto-generated API docs at `/docs` (Swagger UI) — useful during development
- Lightweight: starts in <1 second

**Alternative considered:** Flask
- Flask is synchronous by default; handling concurrent frame POSTs would require `gevent` or threading workarounds
- FastAPI's native `async` is cleaner for the frame-streaming use case
- Rejected: Flask

---

### Streamlit (`ui_streamlit.py`, `ui_components/`)

**Why used:**
- Pure Python UI — no HTML/CSS/JavaScript required, suitable for a research/demo project
- Built-in state management via `st.session_state`
- Trivial to add sliders, images, audio components
- Hot-reload on file save during development

**Alternative considered:** React / Next.js frontend
- Would require a separate Node.js project, API integration boilerplate, auth tokens
- Overkill for a solo final-year project
- Rejected: React

**Limitation acknowledged:** Streamlit reruns the entire script on every interaction.
This is why the camera loop must be a subprocess, not inline Python — see Architecture doc.

---

### OpenCV (`cv2`) — `backend_api.py`, `ingest_ipcam.py`

**Why used:**
- `VideoCapture` natively handles MJPEG streams from IP Webcam app on Android
- `imread` / `imencode` for fast JPEG decode/encode
- Drawing primitives (`rectangle`, `putText`) for TEST_ENVIRONMENT annotated frames
- Battle-tested, cross-platform

**Alternative considered:** `imageio` or `PIL` for image I/O
- Neither has native `VideoCapture` for streaming
- PIL was still used for the CLIP preprocessing step (CLIP's `preprocess` expects a PIL Image)
- Rejected as a VideoCapture replacement

**Note:** `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` is set in `ui_streamlit.py` line 6
because PyTorch and OpenCV both ship OpenMP, causing a "duplicate library" crash on Windows when
both are imported in the same process.

---

### Ultralytics YOLOv8 (`best.pt`) — `backend_api.py`

**Why used:**
- State-of-the-art real-time object detection
- Python API: `YOLO("best.pt")(frame)` returns structured `boxes` object
- Custom training workflow well-documented (Ultralytics docs)
- `best.pt` is the custom-trained model covering all 12 project-specific classes

**Alternative considered:** Detectron2 (Facebook)
- Heavier library, slower inference, more complex setup
- Less straightforward custom training pipeline
- Rejected

**Alternative considered:** Generic COCO-pretrained model
- COCO has no "Watch", "Wallet", "Bike Key", "Car Key" classes
- Would require re-identification to distinguish object types — defeats the purpose
- Rejected

**Why a custom model is necessary:**
The project tracks personal items (Watch, Wallet, keys) that do not appear in standard COCO-80 or
ImageNet class lists. A custom model was trained specifically for these 12 classes.

---

### OpenAI CLIP (`clip`) — `backend_api.py`

**Why used:**
- Pre-trained on 400M image–text pairs: learns powerful visual features without task-specific training
- Produces 512-dim embeddings that capture appearance (colour, shape, texture) well enough to
  distinguish "blue watch" from "silver watch" from the same YOLO class
- No additional training required — enrollment = just collecting embeddings, not gradient updates
- Inference is fast on GPU (~5ms); acceptable on CPU (~100–300ms) for 5–10 FPS use case

**Why not train a classifier instead:**
Training a per-object classifier would require labelled images of each specific item, a training
pipeline, and retraining whenever the user adds a new item. With CLIP + FAISS, the user simply
shows the object to the camera for 30 seconds — no training, no compute cost.

**Alternative considered:** ResNet / EfficientNet feature extractor (ImageNet-pretrained)
- Would work but CLIP's contrastive training produces more semantically meaningful embeddings
- CLIP representations cluster by visual appearance more tightly than classification-head features
- Rejected

**Model choice — ViT-B/32 vs ViT-L/14:**
- `ViT-B/32` (chosen): 512-dim, ~150MB, fast
- `ViT-L/14`: 768-dim, ~900MB, 3–5× slower, more accurate
- Configurable via `CLIP_MODEL` in `constants.py` — swap without code changes

---

### FAISS (`faiss-cpu`) — `backend_api.py`

**Why used:**
- Industry-standard library for approximate (and exact) nearest-neighbour search
- `IndexFlatIP` = exact inner product search: correct result guaranteed, no approximation error
- `IndexIDMap2` wrapper: lets us associate custom integer IDs with vectors (used for timestamp IDs)
- Read/write to disk: `faiss.write_index / faiss.read_index` — no separate vector DB needed
- Scales to millions of vectors if needed; overkill for this project but future-proof

**Alternative considered:** sklearn `NearestNeighbors`
- No GPU support, no persistent index files, slower for large N
- Would work for this project's scale (< 1000 vectors per object) but less maintainable
- Rejected

**Alternative considered:** Chroma / Weaviate (vector databases)
- Full server deployments — too heavy for a local demo project
- Rejected

**Why IndexFlatIP (not IndexFlatL2)?**
L2 distance on normalised vectors is equivalent to cosine distance, but inner product (IP) directly
gives cosine similarity as a score in `[−1, 1]` which is easier to threshold against `SIM_THRESHOLD`.

---

### SQLite (`sqlite3`) — `backend_api.py`

**Why used:**
- Zero-configuration: single file (`reid_store/main.sqlite`), no server to install
- Perfect for single-user or small-scale multi-user local deployment
- Built into Python standard library
- SQL queries for event retrieval are natural and readable

**Alternative considered:** PostgreSQL / MySQL
- Require a running DB server, user management, connection strings
- Unnecessary for a local demo running on one machine
- Rejected

**Alternative considered:** JSON files
- No query capabilities, no transactions, slow for event retrieval
- Rejected

---

### `speech_recognition` + Google Web Speech API — `ui_components/query.py`

**Why used:**
- `SpeechRecognition` library wraps the Google Web Speech API in 5 lines of code
- No local model to download, no GPU needed
- Transcription quality is high for short queries

**Alternative considered:** `faster-whisper` (originally used)
- Runs locally (no internet needed)
- Required `sounddevice` + `scipy` for mic capture — complex dependency chain
- `faster-whisper` model download (~150MB) adds setup friction
- Replaced with `SpeechRecognition` for simplicity

**Limitation:** Requires internet connection. Shown to user via spinner text "Requires internet".

---

### `bcrypt` — `backend_api.py`

**Why used:**
- Industry-standard password hashing with built-in salt
- Python: `bcrypt.hashpw(password.encode(), bcrypt.gensalt())`
- Even if the SQLite file is accessed directly, passwords cannot be recovered

**Alternative considered:** `hashlib.sha256`
- No built-in salting — vulnerable to rainbow table attacks
- Rejected

---

## 2. Architectural Design Decisions

### Three-Process Architecture (UI + Backend + Ingest subprocess)

**Why not a monolith?**
- Streamlit reruns the entire script on every button click. A camera capture loop inside Streamlit
  would be killed and restarted on every interaction.
- YOLO + CLIP load time is 5–10 seconds. Loading them inside Streamlit's process would cause
  that delay on every page rerun.
- Separating into `backend_api.py` (persistent ML process) + `ui_streamlit.py` (stateless UI) +
  `ingest_ipcam.py` (camera loop subprocess) gives each process a single clear responsibility.

See [01_Architecture_Overview.md](01_Architecture_Overview.md) §4 for the full explanation.

---

### Session-Based Frame Processing

**Why sessions instead of a simpler design?**
- Sessions allow the backend to know *context* for each frame (mode, user_id, which object is
  being enrolled) without the UI having to send all that data with every frame
- Multiple users can have simultaneous active sessions without conflict
- Sessions persist in SQLite — if the backend restarts mid-session, the UI can detect the
  session is gone via `GET /session_status` and show an appropriate message

See [01_Architecture_Overview.md](01_Architecture_Overview.md) §5 for a beginner-friendly explanation.

---

### Embeddings for Re-identification (not a second detector)

**The problem:** YOLO says "I see a Watch." But the user has three watches. Which one?

**Option A — Train a per-object classifier**
- Requires labelled images of each watch, a training pipeline, and retraining when user adds a new watch
- Not scalable for a personal assistant

**Option B — Use CLIP embeddings + FAISS (chosen)**
- Enrollment = collect CLIP embeddings during a 30-second camera session. No training.
- Matching = cosine similarity search in real-time. Works for any new object the user adds.
- The system generalises to new objects without any code changes.

---

### `location_text` as a Single String Column

**Why not store `environment_id` and `landmark_id` as separate FK columns in `events`?**
- The location at the time of detection is a snapshot — if the user later renames the environment
  or deletes a landmark, a FK would either break or point to stale data
- A denormalised string (`"on study_chair (Bedroom)"`) is self-contained and always correct
- Simpler query: no joins needed to display results

---

### Centralized `constants.py`

All thresholds, class lists, paths, and UI options are in one file.
- Both `backend_api.py` and all `ui_components/*.py` import from it
- No magic numbers scattered across files
- To adjust `SIM_THRESHOLD` from 0.65 to 0.70: one line change in one file
- To add a new object class: add to `PERSONAL_CLASSES` and retrain YOLO — nothing else to change

---

### Per-User Isolated Storage

Each user gets their own folder: `reid_store/users/{user_id}/`
- FAISS indices for user A are completely isolated from user B
- Image files never mix between users
- Deleting a user's objects only affects that user's data
- SQLite foreign keys ensure database-level isolation (all tables include `user_id`)
