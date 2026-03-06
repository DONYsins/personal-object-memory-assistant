# ============================
# ingest_ipcam.py
# ============================

import os
import sys
import time
import argparse
import base64
import logging
from datetime import datetime
from logger import log_info, log_error, log_debug, SESSION_ID, SESSION_LOG_DIR

import cv2
import numpy as np
import requests

def try_open_camera(url: str, retries: int, log):
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(url)
        ok, _ = cap.read()
        if ok:
            log_info(f"[INGEST] Camera connected on attempt {attempt}")
            return cap
        cap.release()
        log_error(f"[INGEST] Camera not ready (attempt {attempt}/{retries}). Retrying...")
        time.sleep(1.0)
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session_id", required=True)
    p.add_argument("--ip_cam_url", required=True)
    p.add_argument("--backend", default="http://127.0.0.1:8000")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--retries", type=int, default=5)
    args = p.parse_args()

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Logging
    LOGS_DIR = SESSION_LOG_DIR
    LOG_PATH = os.path.join(LOGS_DIR, f"Ingest - run_{run_stamp}.log")

    endpoint = f"{args.backend}/session_frame"

    cap = try_open_camera(args.ip_cam_url, args.retries, LOG_PATH)
    if cap is None:
        log_error("[INGEST] Camera failed to start after retries. Exiting.")
        print("ERROR: Camera failed to start. Check IP Webcam.")
        sys.exit(2)

    # determine skip interval based on camera FPS to keep preview smooth
    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    try:
        camera_fps = float(camera_fps)
    except:
        camera_fps = 30.0

    # how many frames to skip between sends to approximately achieve args.fps
    SKIP_INTERVAL = max(1, int(round(camera_fps / args.fps)))
    frame_count = 0

    log_info(f"[INGEST] Streaming session_id={args.session_id} target_fps={args.fps} camera_fps={camera_fps:.1f} skip={SKIP_INTERVAL}")
    print("Streaming started. Press Q on the preview window to stop.")

    last_frame_time = time.time()
    frame_read_failures = 0
    MAX_CONSECUTIVE_FAILURES = 20  # If fail 10 times in a row, camera is disconnected
    
    while True:
        ok, frame = cap.read()
        if not ok:
            frame_read_failures += 1
            log_error(f"[INGEST] Frame read failed ({frame_read_failures}/{MAX_CONSECUTIVE_FAILURES}). Will retry read...")
            
            # If camera is disconnected for too long, auto-stop session
            if frame_read_failures >= MAX_CONSECUTIVE_FAILURES:
                log_error("[TIMEOUT] Camera disconnected (too many read failures). Auto-stopping session.")
                try:
                    requests.post(f"{args.backend}/stop_session", data={"session_id": args.session_id}, timeout=5)
                except:
                    pass
                break
            
            time.sleep(0.2)
            continue
        
        # Reset failure counter on successful read
        frame_read_failures = 0
        last_frame_time = time.time()

        frame_count += 1
        cv2.imshow("IP Camera Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_count % SKIP_INTERVAL != 0:
            # show every frame; only send limited frames
            continue

        ok2, buf = cv2.imencode(".jpg", frame)
        if not ok2:
            continue

        display_frame = frame  # Default: show raw frame
        try:
            r = requests.post(
                endpoint,
                data={"session_id": args.session_id},
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                timeout=5
            )
            if r.status_code != 200:
                log_error(f"[INGEST] Backend status={r.status_code} body={r.text[:200]}")
            else:
                # If backend returned an annotated frame (TEST_ENVIRONMENT mode), display it
                resp = r.json()
                if resp.get("frame"):
                    jpg_bytes = base64.b64decode(resp["frame"])
                    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    annotated = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if annotated is not None:
                        display_frame = annotated
        except Exception as e:
            log_error(f"[INGEST] POST failed: {e}")

        cv2.imshow("IP Camera Preview", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_info("[INGEST] Ingest stopped.")

if __name__ == "__main__":
    main()