# Quick Start Guide — Vision Memory Assistant

---

## 1. Prerequisites

| Requirement | Value |
|---|---|
| OS | Windows (tested) |
| Python | 3.10 (via Conda) |
| Conda environment name | `itemmem` |
| YOLO model | `best.pt` (must be in project root) |
| IP Camera app | "IP Webcam" on Android, or any MJPEG stream |

---

## 2. Environment Setup (first time only)

```bat
conda create -n itemmem python=3.10 -y
conda activate itemmem
pip install -r requirements.txt
```

> `requirements.txt` includes: `ultralytics`, `opencv-python`, `torch`, `faiss-cpu`,
> `fastapi`, `uvicorn`, `streamlit`, `clip` (from GitHub), `SpeechRecognition`, `pyaudio`, `bcrypt`, etc.

---

## 3. Running the Application

### Option A — One-click batch file (recommended)
```bat
run_application.bat
```
This activates `itemmem`, opens **BACKEND** in one terminal (`uvicorn backend_api:app --reload`),
waits 12 seconds, then opens **UI** in a second terminal (`streamlit run ui_streamlit.py`).

### Option B — Manual (two terminals)
```bat
# Terminal 1
conda activate itemmem
uvicorn backend_api:app --reload

# Terminal 2 (after backend is ready)
conda activate itemmem
streamlit run ui_streamlit.py
```

**Default ports:**
- Backend API: `http://127.0.0.1:8000`
- Streamlit UI: `http://localhost:8501`

The camera ingest process (`ingest_ipcam.py`) is **not** started manually — the UI launches it as
a subprocess via `tracking.launch_ingest()` (`ui_components/tracking.py`, line 20) whenever you
press **Start Tracking** or **Start Enroll**.

---

## 4. Happy-Path Demo (end to end)

### Step 1 — Register & Login
1. Open `http://localhost:8501`
2. Click **Register** tab → enter username + password → click **Register**
3. Switch to **Login** tab → enter same credentials → click **Login**
4. You land on the **Dashboard**

### Step 2 — Add a Personal Object
1. Scroll to **Personal Objects** → **Add Personal Object**
2. Choose type: e.g. `Watch`
3. Enter label: e.g. `black_watch`
4. Click **Create Object** → success toast shows

### Step 3 — Enroll the Object
1. Enter your IP camera URL (e.g. `http://192.168.x.x:8080/video`)
2. Scroll to **Enroll Object (Live)** → select `black_watch`
3. Click **Start Enroll Object** — a camera preview window opens on screen
4. Hold the watch in front of the camera for **30–60 seconds** (multiple angles, lighting)
5. Click **Stop Enroll Object**

### Step 4 — Create an Environment and Add Landmarks
1. Scroll to **Add Environment** → enter `Bedroom` → click **Create Environment**
2. Scroll to **Landmarks** → select `Bedroom` environment
3. Choose YOLO class: e.g. `Chair`, enter label: `study_chair` → click **Add Landmark**
4. Repeat for `Bed` → label `main_bed`

### Step 5 — Enroll a Landmark (optional but improves accuracy)
1. Scroll to **Enroll Landmark (Live)** → select `Bedroom`, then `study_chair`
2. Click **Start Enroll Landmark**, point camera at the chair for ~30 seconds
3. Click **Stop Enroll Landmark**

### Step 6 — Start Tracking
1. Scroll to **Tracking** → click **Start Tracking**
2. Walk around with the camera. When a watch is detected the app matches it to `black_watch`
3. Events are stored every **15 seconds** while the object is visible (configurable: `constants.py → STORE_INTERVAL_SECONDS`)

### Step 7 — Query
1. In the **Query (Last Seen)** box, type: `Where did I last see my black watch?`
2. Click **Search**
3. A list of sighting timestamps, locations, and images appears

---

## 5. Common Runtime Errors and Fixes

| Error | Likely Cause | Fix |
|---|---|---|
| `Camera failed to start after retries` | IP camera URL wrong or app not running | Open IP Webcam app on phone, check URL, confirm same WiFi |
| `uvicorn: error: … port already in use` | Port 8000 taken | Run `netstat -ano \| findstr :8000`, kill the PID, or change port with `--port 8001` and update `BACKEND` in `ui_components/utils.py` |
| `streamlit: port 8501 in use` | Previous Streamlit still running | Kill it, or run `streamlit run ui_streamlit.py --server.port 8502` |
| `Error loading best.pt: file not found` | Model not in project root | Copy `best.pt` to project root (same folder as `backend_api.py`) |
| `clip load failed / OSError` | CLIP not installed from GitHub source | Run `pip install git+https://github.com/openai/CLIP.git` |
| `KMP_DUPLICATE_LIB_OK` OpenMP crash | PyTorch + OpenCV sharing OpenMP | Already handled: `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` is set in `ui_streamlit.py` line 6 |
| `No speech detected` on voice input | Mic not ready, quiet room | Speak within 5 seconds of spinner appearing; check mic access |
| `Voice input requires internet` | Google Web Speech API unreachable | Requires internet connection; use text input instead |
| `pyaudio install fails on Windows` | Binary dependency missing | Install from wheel: `pip install pipwin && pipwin install pyaudio` |
| `FAISS: empty index` during tracking | Object not enrolled yet | Run Enroll Object before tracking |
| All events show `location_text = Unknown` | No landmarks detected in frames | Ensure landmarks (Chair, Bed, etc.) are visible in camera frame and enrolled; check `app_debug.log` for `[ENV_MATCH]` lines |

---

## 6. Resetting Logs / Starting a Fresh Session

The log folder is `logs/<SESSION_ID>/`. A new session folder is created when
the `logs/.current_session` marker file does not exist.

```bat
del logs\.current_session
```

Next run creates `logs/<new_timestamp>/app.log` and `app_debug.log`.

---

## 7. Key Config Values (all in `constants.py`)

| Constant | Default | What it controls |
|---|---|---|
| `SIM_THRESHOLD` | `0.65` | Min cosine similarity to accept a FAISS match |
| `STORE_INTERVAL_SECONDS` | `15` | Min gap between stored events for same object |
| `DISAPPEAR_TICKS` | `20` | Frames without detection before "disappeared" event |
| `BACKEND_FPS` | `10` | Target frames/sec sent from ingest to backend |
| `MAX_OBJECTS_PER_USER` | `10` | Cap on registered personal objects per user |
| `CLIP_MODEL` | `"ViT-B/32"` | Which CLIP variant to load |
