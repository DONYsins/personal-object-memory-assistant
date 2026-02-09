import time
import logging
import requests
import cv2
from datetime import datetime
import os

# Create logs directory
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", f"Log_Ingest_{ts}")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "ingest_cmd.log")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")])
log = logging.getLogger("ingest")

# NOTE: change this to your phone IP
IPCAM_URL = "http://192.168.8.183:8080/video"

API_INGEST = "http://127.0.0.1:8000/ingest_frame"

# note to me: lower this if lagging
FPS_SAMPLE = 5
CAMERA_FPS = 30
SKIP_INTERVAL = max(1, CAMERA_FPS // FPS_SAMPLE)

cap = cv2.VideoCapture(IPCAM_URL)
frame_count = 0
failure_count = 0
last_warning_time = 0

log.info("Starting ingest. Press Q to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        failure_count += 1
        current_time = time.time()
        if current_time - last_warning_time > 30:
            log.warning("No frame read from camera.")
            last_warning_time = current_time
        if failure_count > 10:
            log.info("Attempting to reconnect to camera...")
            cap.release()
            cap = cv2.VideoCapture(IPCAM_URL)
            failure_count = 0
        time.sleep(1)
        continue
    else:
        failure_count = 0

    frame_count += 1
    if frame_count % SKIP_INTERVAL != 0:
        cv2.imshow("Ingest Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ok:
        try:
            files = {"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")}
            r = requests.post(API_INGEST, files=files, timeout=5)
            if r.status_code != 200:
                log.warning("Backend error %s: %s", r.status_code, r.text)
            else:
                data = r.json()
                log.info("Stored=%s | Seen=%s", data.get("stored_events"), data.get("seen_personal"))
        except Exception as e:
            log.error("Ingest error: %s", e)

    cv2.imshow("Ingest Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
log.info("Ingest stopped.")
