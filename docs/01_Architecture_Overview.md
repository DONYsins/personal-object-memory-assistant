# Architecture Overview — Vision Memory Assistant

---

## 1. High-Level Description

The system has **three processes** running simultaneously, plus persistent storage:

| Process | File | Role |
|---|---|---|
| **Backend API** | `backend_api.py` | All ML logic, DB access, FAISS operations. Exposes REST API on port 8000. |
| **Streamlit UI** | `ui_streamlit.py` + `ui_components/` | Web-based frontend. Routes pages, shows controls, calls backend via HTTP. |
| **Camera Ingest** | `ingest_ipcam.py` | Subprocess launched by UI. Reads IP camera frames and POSTs them to `/session_frame`. |

Storage layers:
- **SQLite** (`reid_store/main.sqlite`) — users, objects, environments, events, sessions
- **FAISS indices** (`reid_store/users/{id}/faiss/`) — CLIP embedding vectors
- **Image files** (`reid_store/users/{id}/images/`) — saved detection frames
- **Log files** (`logs/{SESSION_ID}/app.log` + `app_debug.log`) — runtime and debug logs

---

## 2. Component Diagram

```mermaid
graph TB
    subgraph Browser["Browser (localhost:8501)"]
        UI["Streamlit UI<br>ui_streamlit.py<br>ui_components/*.py"]
    end

    subgraph Backend["Backend Process (localhost:8000)"]
        API["FastAPI<br>backend_api.py"]
        YOLO["YOLOv8 (best.pt)<br>Object Detection"]
        CLIP["CLIP ViT-B/32<br>Embedding Extraction"]
        FAISS_LIB["FAISS IndexFlatIP<br>Similarity Search"]
    end

    subgraph Ingest["Ingest Subprocess"]
        CAM["ingest_ipcam.py<br>(subprocess.Popen)"]
    end

    subgraph Storage["Persistent Storage (reid_store/)"]
        DB[("SQLite<br>main.sqlite")]
        FAISS_FILES["FAISS .index files<br>users/{id}/faiss/"]
        IMAGES["Image files<br>users/{id}/images/"]
    end

    subgraph Camera["Physical Camera"]
        IPCAM["IP Webcam<br>(phone app)"]
    end

    subgraph Logs["Logs (logs/{SESSION_ID}/)"]
        APPLOG["app.log (INFO+ERROR)"]
        DEBUGLOG["app_debug.log (DEBUG)"]
    end

    UI -- "HTTP POST/GET<br>requests library" --> API
    CAM -- "launch subprocess<br>subprocess.Popen" --> Ingest
    IPCAM -- "MJPEG stream<br>OpenCV VideoCapture" --> CAM
    CAM -- "POST /session_frame<br>multipart JPEG" --> API
    API --> YOLO
    API --> CLIP
    API --> FAISS_LIB
    API <--> DB
    FAISS_LIB <--> FAISS_FILES
    API --> IMAGES
    API --> Logs
```

---

## 3. Sequence Diagrams

### 3a. Login / Register Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI<br>(login.py)
    participant API as FastAPI Backend

    User->>UI: Enter username + password
    UI->>API: POST /login {username, password}
    API->>API: bcrypt.checkpw(password, stored_hash)
    alt Credentials valid
        API-->>UI: {status: success, user_id: 5}
        UI->>UI: st.session_state.user_id = 5
        UI-->>User: Redirect to Dashboard
    else Invalid
        API-->>UI: {status: error, message: Invalid credentials}
        UI-->>User: Show error message
    end
```

### 3b. Enroll Object Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Dashboard<br>(dashboard.py)
    participant API as FastAPI Backend
    participant Ingest as ingest_ipcam.py<br>(subprocess)
    participant Camera as IP Camera

    User->>UI: Click "Start Enroll Object"<br>(object: black_watch)
    UI->>API: POST /start_session {mode: ENROLL_OBJECT, user_object_id: 2}
    API-->>UI: {session_id: "abc-123"}
    UI->>Ingest: subprocess.Popen(ingest_ipcam.py --session_id abc-123)

    loop Every SKIP_INTERVAL frames
        Camera->>Ingest: MJPEG frame
        Ingest->>API: POST /session_frame {session_id, file: frame.jpg}
        API->>API: YOLO detect → find "Watch" bbox
        API->>API: CLIP encode bbox crop → 512-dim vector
        API->>API: faiss.add_with_ids(embedding, timestamp_id)
        API->>API: Save debug image to enroll_debug_images/
        API-->>Ingest: {status: ok, note: enrolled embedding}
    end

    User->>UI: Click "Stop Enroll Object"
    UI->>API: POST /stop_session {session_id: abc-123}
    UI->>Ingest: proc.terminate()
```

### 3c. Enroll Environment (Landmark) Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Dashboard<br>(dashboard.py)
    participant API as FastAPI Backend
    participant Ingest as ingest_ipcam.py

    User->>UI: Create Environment "Bedroom"
    UI->>API: POST /add_environment {environment_label: Bedroom}
    API-->>UI: {environment_id: 1}

    User->>UI: Add Landmark: Chair → "study_chair"
    UI->>API: POST /add_environment_landmark<br>{environment_id:1, landmark_class:Chair, user_label:study_chair}
    API-->>UI: {env_landmark_id: 3}

    User->>UI: Click "Start Enroll Landmark"<br>(study_chair in Bedroom)
    UI->>API: POST /start_session<br>{mode: ENROLL_LANDMARK, environment_id:1, landmark_id:3}
    API-->>UI: {session_id: "def-456"}
    UI->>Ingest: subprocess.Popen(--session_id def-456)

    loop Every SKIP_INTERVAL frames
        Ingest->>API: POST /session_frame
        API->>API: YOLO detect → find "Chair" bbox (largest)
        API->>API: CLIP encode bbox crop
        API->>API: Append to FAISS index:<br>users/1/faiss/environment/1/3.index
        API-->>Ingest: {status: ok}
    end

    User->>UI: Click "Stop Enroll Landmark"
    UI->>API: POST /stop_session
    UI->>Ingest: proc.terminate()
```

### 3d. Tracking + Event Storage Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Dashboard<br>(tracking.py)
    participant API as FastAPI Backend<br>(TRACK mode)
    participant Ingest as ingest_ipcam.py

    User->>UI: Click "Start Tracking"
    UI->>API: POST /start_session {mode: TRACK, user_id: 5}
    API-->>UI: {session_id: "ghi-789"}
    API->>API: SESSION_STATE[ghi-789] = {objects:{}, ...}
    UI->>Ingest: subprocess.Popen(--session_id ghi-789)

    loop Every SKIP_INTERVAL frames
        Ingest->>API: POST /session_frame {session_id, file}
        API->>API: YOLO → personal[] + landmarks[]
        Note over API: Only logs if detections found
        API->>API: find_environment_from_landmarks()<br>→ match YOLO classes to registered env
        loop Each detected personal object
            API->>API: extract_embedding_and_match()<br>→ CLIP embed + FAISS search all candidates
            API->>API: Best label scored ≥ SIM_THRESHOLD (0.65)?
            alt Match found
                API->>API: Update object_state[label].last_seen
                API->>API: Elapsed ≥ 15s since last store?
                alt Time to store
                    API->>API: get_location_text_for_event()<br>→ IoU/centroid + user label + env name
                    API->>API: store_object_event()<br>→ save JPEG + INSERT into events
                end
            end
        end
        loop Each labeled object NOT seen this frame
            API->>API: missing_ticks += 1
            API->>API: missing_ticks ≥ 20?<br>→ store disappearance event, reset state
        end
        API-->>Ingest: {status: ok, stored: N}
    end

    User->>UI: Click "Stop Tracking"
    UI->>API: POST /stop_session
    UI->>Ingest: proc.terminate()
```

### 3e. Query + Result Rendering Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as query.py
    participant API as FastAPI Backend

    User->>UI: Type "where is my black watch?" → Click Search
    UI->>UI: Fetch all user labels from /dashboard
    UI->>UI: Substring match "black_watch" in query text
    UI->>API: POST /query {user_id:5, user_label:black_watch, k:3}
    API->>API: SELECT events WHERE user_object_id=2 ORDER BY timestamp DESC LIMIT 3
    API-->>UI: [{event_id, location_text, timestamp, image_path}, ...]

    UI->>UI: st.radio → user picks sighting
    UI-->>User: Show location_text, timestamp, image (st.image)
```

---

## 4. Why Three Separate Processes?

Two main reasons:

**1. Streamlit reruns everything on each interaction.**
Every button click causes Streamlit to re-execute `ui_streamlit.py` top to bottom. If the
camera capture loop ran inside Streamlit, it would be killed and restarted on every click.
By putting it in a subprocess (`ingest_ipcam.py`), the camera loop runs independently and
continuously, unaffected by UI reruns.

**2. Heavy ML models can't live in a browser-facing process.**
YOLO and CLIP take several seconds to load. They are loaded **once** when `backend_api.py`
starts (`backend_api.py` lines 56–61) and stay in memory. If they loaded in the UI, every
page navigation would re-load them. FastAPI keeps them alive and serves any number of requests.

**How they communicate:**
- UI → Backend: synchronous HTTP (POST/GET via `requests` library, `ui_components/utils.py`)
- Ingest → Backend: HTTP multipart POST for each frame (`/session_frame`)
- UI → Ingest: OS process control (`subprocess.Popen`, `proc.terminate()`)

---

## 5. Additional Info — What Is a Session and Why?

### The Problem It Solves

When you click **Start Tracking**, the camera starts sending frames. But Streamlit is a web app —
it doesn't keep a "connection" open between you and the server. Every button click is a separate
HTTP request. So how does the backend know "this frame belongs to *this* user in *this* mode"?
That's exactly what a **session** solves.

### What a Session Is

A session is a **unique ID** (a UUID like `"abc-def-123-456"`) that acts like a
"ticket" tying everything together. When you click Start Tracking:

```
UI sends: POST /start_session {user_id: 5, mode: TRACK}
Backend creates: session_id = "abc-def-123-456"
                  → Saved in SQLite sessions table (status=RUNNING)
                  → Saved in SESSION_STATE dict (live tracking data in memory)
Backend returns: {session_id: "abc-def-123-456"}
UI stores it:    st.session_state.session_id = "abc-def-123-456"
```

The ingest subprocess is then launched with `--session_id abc-def-123-456`. Every frame it sends
carries that ID:
```
POST /session_frame {session_id: "abc-def-123-456", file: frame.jpg}
```

The backend looks up the session, finds `user_id=5, mode=TRACK, status=RUNNING`, and processes
the frame accordingly.

### What Happens Without Sessions

Without sessions you'd need to somehow keep track of "who is tracking right now" and "what mode
are they in" using some other mechanism — like a single global dict keyed by user_id. But then:
- Two users can't track simultaneously (they'd overwrite each other's state)
- You can't distinguish "this user is enrolling object #2" from "this user is tracking"
- If the backend restarts, you lose all context; with sessions in SQLite, you can query the
  DB to find all sessions that were `RUNNING` and clean them up

### Why There Are Two Stores (SQLite + SESSION_STATE dict)

| Store | What | Why |
|---|---|---|
| `sessions` table (SQLite) | session_id, user_id, mode, status, timestamps | Persists across backend restarts. UI can always check session status with `GET /session_status`. |
| `SESSION_STATE` dict (in-memory) | Live tracking data: per-object last_seen, last_frame, last_bbox, missing_ticks | Changes every single frame — too fast for DB writes. Cleared when session stops. |

### Session Lifecycle

```
/start_session → status = RUNNING (DB) + live dict created (memory)
                           ↓
            frames arrive → SESSION_STATE updated per frame
                           ↓
/stop_session  → status = STOPPED (DB) + live dict deleted (memory)
```

If the camera disconnects unexpectedly (`ingest_ipcam.py` line 63–69), the ingest script calls
`/stop_session` automatically before exiting, so the session is cleanly closed.
