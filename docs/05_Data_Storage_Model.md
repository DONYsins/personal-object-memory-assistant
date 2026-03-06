# Data Storage Model — `backend_api.py`, `logger.py`

---

## 1. Database Schema (`reid_store/main.sqlite`)

Initialised by `init_db()` in `backend_api.py` (line ~71). Uses SQLite via `sqlite3` standard library.

### Table: `users`

```sql
CREATE TABLE users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,         -- bcrypt hash, never plaintext
    created_at    TEXT NOT NULL          -- ISO 8601 timestamp
)
```

| Column | Example |
|---|---|
| `user_id` | `5` |
| `username` | `"alice"` |
| `password_hash` | `"$2b$12$..."` |
| `created_at` | `"2026-03-01T10:00:00.123456"` |

---

### Table: `personal_objects`

```sql
CREATE TABLE personal_objects (
    user_object_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    generic_type   TEXT NOT NULL,   -- must be in PERSONAL_CLASSES
    user_label     TEXT NOT NULL,   -- user's custom name
    created_at     TEXT NOT NULL
)
-- Unique index prevents duplicate label per user
CREATE UNIQUE INDEX idx_user_label ON personal_objects(user_id, user_label)
```

| Column | Example |
|---|---|
| `user_object_id` | `2` |
| `user_id` | `5` |
| `generic_type` | `"Watch"` |
| `user_label` | `"black_watch"` |

**Relationship:** One user → many objects. Each object has exactly one FAISS `.index` file.

---

### Table: `environments`

```sql
CREATE TABLE environments (
    environment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL,
    environment_label TEXT NOT NULL,
    created_at        TEXT NOT NULL
)
CREATE UNIQUE INDEX idx_user_envlabel ON environments(user_id, environment_label)
```

| Column | Example |
|---|---|
| `environment_id` | `1` |
| `user_id` | `5` |
| `environment_label` | `"Bedroom"` |

---

### Table: `environment_landmarks`

```sql
CREATE TABLE environment_landmarks (
    env_landmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    environment_id  INTEGER NOT NULL,    -- FK → environments
    landmark_class  TEXT NOT NULL,       -- YOLO class name e.g. "Chair"
    user_label      TEXT,                -- custom label e.g. "study_chair"
    created_at      TEXT NOT NULL
)
```

| Column | Example |
|---|---|
| `env_landmark_id` | `3` |
| `environment_id` | `1` |
| `landmark_class` | `"Chair"` |
| `user_label` | `"study_chair"` |

**Note:** Multiple rows per `(environment_id, landmark_class)` are allowed, as long as
`user_label` differs. This supports two chairs in one room: `"study_chair"` and `"reading_chair"`.

---

### Table: `events`

```sql
CREATE TABLE events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    user_object_id INTEGER,              -- FK → personal_objects
    location_text  TEXT NOT NULL,        -- full human-readable location
    timestamp      TEXT NOT NULL,        -- ISO 8601
    image_path     TEXT NOT NULL         -- absolute path to saved JPEG
)
```

| Column | Example |
|---|---|
| `event_id` | `42` |
| `user_id` | `5` |
| `user_object_id` | `2` |
| `location_text` | `"on study_chair (Bedroom)"` |
| `timestamp` | `"2026-03-02T14:35:22.123456"` |
| `image_path` | `"reid_store/users/5/images/Watch/black_watch_20260302_143522_123456.jpg"` |

**`location_text` format patterns:**
| Pattern | Meaning |
|---|---|
| `"on study_chair (Bedroom)"` | Object overlapping chair; environment identified |
| `"near main_bed (Bedroom)"` | Closest landmark; environment identified |
| `"on Chair"` | No environment registered; YOLO class used directly |
| `"Unknown"` | No landmarks detected in that frame |

---

### Table: `sessions`

```sql
CREATE TABLE sessions (
    session_id     TEXT PRIMARY KEY,     -- UUID e.g. "abc-def-123"
    user_id        INTEGER NOT NULL,
    mode           TEXT NOT NULL,        -- TRACK / ENROLL_OBJECT / ENROLL_LANDMARK / TEST_ENVIRONMENT
    user_object_id INTEGER,              -- set for ENROLL_OBJECT
    environment_id INTEGER,              -- set for ENROLL_LANDMARK / TEST_ENVIRONMENT
    landmark_id    INTEGER,              -- set for ENROLL_LANDMARK (env_landmark_id)
    status         TEXT NOT NULL,        -- RUNNING / STOPPED
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    last_error     TEXT                  -- last error message if any
)
```

**Schema migration:** `_ensure_sessions_columns()` (inside `init_db()`) checks for missing
`landmark_id` column and adds it via `ALTER TABLE` if absent — allows upgrading existing databases.

---

### Entity Relationship Summary

```
users
  └── personal_objects  (user_id FK)
        └── events       (user_object_id FK)
  └── environments       (user_id FK)
        └── environment_landmarks  (environment_id FK)
  └── sessions           (user_id FK)
```

---

## 2. File Storage Directories

All files live under `reid_store/` (configurable via `BASE_DIR` in `constants.py`).

```
reid_store/
├── main.sqlite                            ← single database file
└── users/
    └── {user_id}/                         ← created by user_dir() on first login
        ├── faiss/
        │   ├── object/
        │   │   └── {user_object_id}.index  ← CLIP embeddings for one personal object
        │   └── environment/
        │       └── {environment_id}/
        │           └── {env_landmark_id}.index  ← CLIP embeddings for one landmark
        ├── images/
        │   └── {generic_type}/             ← e.g. "Watch/", "Wallet/"
        │       └── {label}_{YYYYMMDD_HHMMSS_µs}.jpg  ← saved detection frames
        ├── enroll_debug_images/
        │   └── {generic_type}/
        │       └── {label}_{slot}.jpg      ← debug: 1 frame per 2 seconds during enrollment
        ├── landmark_debug_images/
        │   └── env_{environment_id}/
        │       └── {label}_{slot}.jpg      ← debug: 1 frame per 2 seconds during landmark enroll
        └── test_env_debug_images/
            └── frame_{slot}.jpg            ← debug: annotated frames during TEST_ENVIRONMENT
```

### FAISS Index Files

| Path | Content | Created by |
|---|---|---|
| `faiss/object/{user_object_id}.index` | Embeddings for one personal object | `ensure_faiss_index()` on `add_personal_object` |
| `faiss/environment/{env_id}/{lm_id}.index` | Embeddings for one landmark | `ensure_faiss_index()` on first ENROLL_LANDMARK frame |

Index format: `IndexIDMap2(IndexFlatIP(512))`. IDs are millisecond timestamps.

### Image Files

Saved by `store_object_event()` (`backend_api.py` line ~423):
```python
ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")
image_path = f"reid_store/users/5/images/Watch/black_watch_20260302_143522_123456.jpg"
cv2.imwrite(image_path, image)
```

Microseconds in filename ensure uniqueness even at 10 FPS.

---

## 3. Logging Strategy

### Log Location

```
logs/
├── .current_session     ← marker file: contains current SESSION_ID
└── {SESSION_ID}/        ← e.g. "20260302_213541/"
    ├── app.log          ← INFO + ERROR messages (startup, login, events stored, errors)
    └── app_debug.log    ← DEBUG messages (detection traces, FAISS scores, location logic)
```

### Log Files Explained

| File | Level | Content | When written |
|---|---|---|---|
| `app.log` | INFO + ERROR | Login/register, session start/stop, enrollment notes, errors | Always (every run) |
| `app_debug.log` | DEBUG | Frame detections, FAISS scores per candidate, IoU values, location text steps | Only when YOLO detects something (not on empty frames) |

### Session ID Lifecycle (`logger.py` lines 37–58)

1. On import, `get_session_id()` checks if `logs/.current_session` exists
2. If yes: reads and reuses the stored ID → **same folder for backend + UI in same terminal**
3. If no: generates `datetime.now().strftime("%Y%m%d_%H%M%S")`, writes to marker
4. To force a new session: `del logs\.current_session`

Both `backend_api.py` and `ui_streamlit.py` import `logger` at startup, so both write to
the same `logs/{SESSION_ID}/app.log` file.

### Log Format

```
2026-03-02 14:35:22,123 INFO: Login successful for username='alice', user_id=5
2026-03-02 14:35:30,456 INFO: [SESSION] Started TRACK session=abc-123 user=5
```

```
2026-03-02 14:35:31,001 DEBUG: [DETECTION] mode=TRACK | personal=['Watch'] | landmarks=['Chair']
2026-03-02 14:35:31,002 DEBUG: [FAISS_SCORE] 'black_watch' similarity=0.871 (threshold=0.65)
2026-03-02 14:35:31,003 DEBUG: [TRACK_MATCH] Watch detection → MATCHED to 'black_watch' (confidence=0.871)
2026-03-02 14:35:31,004 DEBUG: [ENV_MATCH] Detected landmark YOLO classes: ['Chair']
2026-03-02 14:35:31,005 DEBUG: [ENV_MATCH] Env 'Bedroom' registered: ['Bed', 'Chair'] | intersection: ['Chair']
2026-03-02 14:35:31,006 DEBUG: [LOCATION_IoU] vs 'Chair': iou=0.08 (threshold=0.05)
2026-03-02 14:35:31,007 DEBUG: [LOCATION_TEXT] Final location_text: 'on study_chair (Bedroom)'
2026-03-02 14:35:31,008 DEBUG: [STORE_EVENT] Stored 'black_watch' at 'on study_chair (Bedroom)' with confidence=0.871
```

Both files use `RotatingFileHandler`: max 5 MB (app.log) / 10 MB (app_debug.log), 3 backup files.

---

## 4. Data Retention and Cleanup

### Automatic Cleanup (`cleanup_if_needed()`, `backend_api.py` ~line 575)

- Triggered inside every TRACK frame
- Condition: `ACTIVE_SECONDS / 3600 >= RETENTION_ACTIVE_HOURS` (default: 4 hours of active runtime)
- What gets deleted: events (and their image files) with `timestamp < now - 4 hours`
- **Not calendar time** — only counts time while the app is actively running (counting via `tick_active_runtime()`)

### Manual Cleanup (`POST /manual_cleanup`)

- User-triggered from dashboard "Data Cleanup" section
- Options: 1h, 2h, 4h, 12h, 1 day, 2 days
- Deletes events by calendar timestamp regardless of active runtime
- Deletes the corresponding image file from disk before removing the DB row

### What is Never Automatically Deleted

| Data | Survives cleanup? | Manual removal? |
|---|---|---|
| FAISS index files | ✅ Yes | Only when object or environment is deleted |
| User accounts | ✅ Yes | Not exposed via UI |
| Environment / landmark definitions | ✅ Yes | Delete Environment button |
| Debug images (`enroll_debug_images/`, etc.) | ✅ Yes | Manual file deletion only |
| Log files | ✅ Yes (rotate at 5/10MB) | Manual deletion |
