# Vision Pipeline — `backend_api.py`

---

## 1. Frame Representation

Frames enter the backend as raw JPEG bytes (from `ingest_ipcam.py`), decoded into a NumPy array:

```python
# backend_api.py  /session_frame  ~line 1072
contents = await file.read()                         # bytes
np_arr   = np.frombuffer(contents, np.uint8)         # 1-D uint8 array
frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    # shape: (H, W, 3) uint8, BGR
```

**Key point:** OpenCV stores images in **BGR** channel order (Blue, Green, Red).
CLIP expects **RGB**. The conversion happens inside `get_embedding_from_bbox()`:

```python
# backend_api.py  get_embedding_from_bbox()  ~line 200
img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
```

Typical frame size at default IP Webcam resolution: `(480, 640, 3)` → 921,600 bytes uncompressed.
As JPEG on the wire it's ~20–50 KB depending on content.

---

## 2. YOLO Inference

**Model:** `best.pt` — a custom-trained YOLOv8 model.  
**Loaded once at startup:** `backend_api.py` line 57 — `yolo_model = YOLO(YOLO_MODEL_PATH)`  
**Device:** CUDA if available, else CPU (`backend_api.py` line 55)

### Classes

| Category | Class names |
|---|---|
| **Personal objects** (`PERSONAL_CLASSES`) | `Watch`, `Wallet`, `Bike Key`, `Car Key` |
| **Landmarks** (`LANDMARK_CLASSES`) | `Bed`, `Computer Table`, `Cupboard`, `Dressing Table`, `Night Table`, `Laptop`, `Chair`, `Table` |

### Inference call

```python
# detect_objects_in_frame()  ~line 280
r0 = yolo_model(frame, verbose=False)[0]
boxes = r0.boxes
```

`verbose=False` suppresses YOLO's per-frame console output.

### Parsing detections

```python
for b in boxes:
    cls_id = int(b.cls[0])          # integer class index
    label  = yolo_model.names[cls_id]  # e.g. "Watch"
    bbox   = tuple(map(int, b.xyxy[0].tolist()))  # (x1, y1, x2, y2) pixel coords
```

**Bounding box format:** `(x1, y1, x2, y2)` — top-left and bottom-right corners in pixel coordinates.

**Example:**
```
bbox = (120, 80, 200, 160)   →   width=80px, height=80px, top-left=(120,80)
```

Detections are split into two lists based on which class list they belong to:
```python
if label in PERSONAL_CLASSES:  personal.append((label, bbox))
if label in LANDMARK_CLASSES:  landmarks.append((label, bbox))
```

Note: a detection can only be personal OR landmark, never both, since the class lists are disjoint.

---

## 3. CLIP Embedding Extraction

**Model:** `ViT-B/32` (Vision Transformer, 32×32 patch size)  
**Loaded once:** `backend_api.py` line 61 — `clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)`  
**Output dimension:** 512 floats

### Function: `get_embedding_from_bbox()` (line ~196)

```python
def get_embedding_from_bbox(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]          # crop to bounding box
    img  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img_t = preprocess(img).unsqueeze(0).to(DEVICE)  # resize + normalize + add batch dim
    with torch.no_grad():
        emb = clip_model.encode_image(img_t).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)   # L2 normalization
    return emb.cpu().numpy().astype("float32")        # shape: (1, 512)
```

### What the vector looks like

- Shape: `(1, 512)` — one row, 512 columns
- Values: floats in approximately `[-0.15, +0.15]` after L2 normalization
- After normalization: `||emb|| = 1.0` exactly (unit vector)

**Why normalize?**  
FAISS uses `IndexFlatIP` (inner product). For unit vectors, inner product equals cosine similarity.
Normalizing makes the similarity score directly interpretable as cosine similarity in range `[−1, 1]`,
where `1.0 = identical`, `0.0 = unrelated`, `−1.0 = opposite`.

### Example embedding snippet
```
[ 0.043, -0.112,  0.088,  0.071, -0.055, ..., 0.012 ]  ← 512 values
```

---

## 4. FAISS Similarity Search

### Index type

```python
# ensure_faiss_index()  backend_api.py  ~line 193
base  = faiss.IndexFlatIP(512)      # Inner Product (dot product) on 512-dim vectors
index = faiss.IndexIDMap2(base)     # Wrapper that lets us store and retrieve by custom int ID
```

**IndexFlatIP** performs exhaustive (brute force) search — compares query against every stored vector.
This is fine for small indices (tens to hundreds of embeddings per object).

### How embeddings are stored (enrollment)

```python
new_id = int(time.time() * 1000)  # millisecond timestamp as ID
index.add_with_ids(emb, np.array([new_id], dtype=np.int64))
faiss.write_index(index, idx_path)  # persist to .index file
```

IDs are millisecond timestamps — guaranteed to be unique and chronologically ordered.

### How matching works during TRACK (`extract_embedding_and_match()` ~line 320)

For a detected "Watch", all registered Watch labels are candidates:
```python
candidates = [(2, "black_watch"), (3, "silver_watch")]
```

For each candidate:
```python
index = faiss.read_index(idx_path)          # load from disk each time
distances, indices = index.search(emb, 1)   # find 1 nearest neighbour
score = float(distances[0, 0])              # cosine similarity score
```

The candidate with the **highest score that also exceeds `SIM_THRESHOLD` (0.65)** wins.

**Threshold logic:**
```python
if score > best_score and score >= SIM_THRESHOLD:
    best_score = score
    best_obj_id = oid
    best_obj_label = lbl
```

If no candidate exceeds 0.65, `best_label = None` and the detection is **discarded** (not stored,
not tracked). This prevents a watch belonging to someone else from being matched as "black_watch".

### Scoring example

| Candidate | FAISS score | Accepted? |
|---|---|---|
| `black_watch` | 0.87 | ✅ Best match |
| `silver_watch` | 0.61 | ❌ Below threshold |
| `black_watch` (empty index) | N/A | ❌ File doesn't exist |

**Debug log output (app_debug.log):**
```
2026-03-02 14:35:22 DEBUG: [FAISS_SCORE] 'black_watch' similarity=0.871 (threshold=0.65)
2026-03-02 14:35:22 DEBUG: [FAISS_SCORE] 'silver_watch' similarity=0.612 (threshold=0.65)
2026-03-02 14:35:22 DEBUG: [TRACK_MATCH] Watch detection → MATCHED to 'black_watch' (confidence=0.871)
```

---

## 5. Location Inference

When a personal object is matched and about to be stored, the system computes a human-readable
location using the landmark detections in the same frame.

### Step 1: IoU-based overlap (`infer_location()` ~line 248)

For each detected landmark, compute Intersection over Union with the object's bounding box:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

If the best IoU > 0.05 (5% overlap), the object is considered **on** that landmark:
```
"on study_chair"
```

**Debug log:**
```
[LOCATION_IoU] vs 'Chair': iou=0.12 (threshold=0.05)
[LOCATION] Overlap match -> 'on Chair' (iou=0.12)
```

### Step 2: Centroid distance fallback

If no IoU > 0.05, find the landmark whose **centre point** is nearest to the object's centre:

```python
ox, oy = bbox_center(obj_bbox)
lx, ly = bbox_center(lbbox)
d = (ox - lx)**2 + (oy - ly)**2   # Euclidean² (no sqrt needed for comparison)
```

Result: `"near study_chair"`

### Step 3: No landmarks

If `landmarks = []`, `infer_location()` returns `None` → stored as `"Unknown"`.

### Step 4: User label and environment substitution (`get_location_text_for_event()` ~line 490)

1. Map YOLO class name to user's custom label using `environment_landmarks` table:
   `"Chair"` → `"study_chair"`
2. Look up environment name: `env_id=1` → `"Bedroom"`
3. Final result: `"on study_chair (Bedroom)"`

**Debug log trace:**
```
[ENV_MATCH] Detected landmark YOLO classes: ['Chair']
[ENV_MATCH] Env 'Bedroom' registered: ['Bed', 'Chair'] | intersection: ['Chair']
[ENV_MATCH] Matched env 'Bedroom' (env_id=1)
[LOCATION_IoU] vs 'Chair': iou=0.08 (threshold=0.05)
[LOCATION] Overlap match -> 'on Chair' (iou=0.08)
[LOCATION_TEXT] landmark_mapping for env_id=1: {'Chair': 'study_chair', 'Bed': 'main_bed'}
[LOCATION_TEXT] Replaced YOLO class 'Chair' -> custom label 'study_chair'
[LOCATION_TEXT] Final location_text: 'on study_chair (Bedroom)'
```

### Common reasons `location_text = "Unknown"`

| Cause | Debug log indicator |
|---|---|
| No landmarks detected by YOLO | `[DETECTION] mode=TRACK \| personal=['Watch'] \| landmarks=[]` |
| Landmark detected but no environment registered | `[ENV_MATCH] No environment matched` |
| No YOLO class registered as landmark in any environment | Same as above |

---

## 6. Object Disappearance Detection

Implemented in TRACK mode, `backend_api.py` lines ~1380–1420:

```python
obj_info["missing_ticks"] += 1
if obj_info["missing_ticks"] >= DISAPPEAR_TICKS:   # default: 20 frames
    # store a final event at last known location
    store_object_event(...)
    # reset all state for this label
    obj_info["missing_ticks"] = 0
    obj_info["last_seen"] = None
    ...
```

**Why 20 ticks?** At `BACKEND_FPS=10`, 20 frames ≈ 2 seconds of absence. This tolerates brief
occlusion (hand passing in front of camera, YOLO momentarily missing a detection) without
triggering false disappearance events.

---

## 7. Performance Considerations

| Consideration | Details |
|---|---|
| **Frame sampling** | `ingest_ipcam.py` computes `SKIP_INTERVAL = round(camera_fps / target_fps)`. At camera 30fps + target 5fps → send every 6th frame. Backend processes at ~10 FPS configured in `BACKEND_FPS`. |
| **FAISS reads from disk per frame** | Every TRACK frame reads each candidate's `.index` file from disk. For small indices (<1000 embeddings) this is fast. For large indices, performance degrades — but typical enrollment is 30–60 seconds × 5 FPS = 150–300 vectors. |
| **CLIP inference** | Most expensive step. On CPU, ~100–300ms per embedding. This is the main bottleneck. GPU (`DEVICE = "cuda"`) reduces this to ~5–10ms. |
| **YOLO inference** | YOLOv8n backbone: ~30–80ms on CPU, ~5ms on GPU. |
| **JPEG encode/decode overhead** | `cv2.imencode / cv2.imdecode` ~1–3ms for 640×480. |
| **Lag mitigations** | Low target FPS (5–10), `verbose=False` on YOLO, `torch.no_grad()` for CLIP. |
