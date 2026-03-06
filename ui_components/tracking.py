"""Tracking related UI controls and helpers."""
import subprocess
import streamlit as st
from logger import log_info, log_error

from ui_components import utils


def stop_ingest_if_running():
    p = st.session_state.get("ingest_proc")
    if p is not None:
        try:
            p.terminate()
        except:
            pass
        st.session_state.ingest_proc = None


def launch_ingest(session_id, ip_cam_url, fps=5):
    """Launch IP camera ingestion subprocess in background."""
    # start the external ingest process as background
    cmd = [
        "python", "ingest_ipcam.py",
        "--session_id", session_id,
        "--ip_cam_url", ip_cam_url,
        "--backend", utils.BACKEND,
        "--fps", str(fps),
        "--retries", "5"
    ]
    log_info("[UI] Launching ingest: " + " ".join(cmd))
    return subprocess.Popen(cmd)  # Subprocess streams frames to backend


def tracking_controls(uid, ip_cam_url, fps):
    """Render tracking buttons on the dashboard's right panel."""
    st.markdown("### Tracking")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Tracking"):
            log_info("[UI] Start Tracking button clicked")
            # optionally let user pick environment later (handled at dashboard level)
            payload = {"user_id": uid, "mode": "TRACK"}
            # include chosen environment if stored in session_state
            env_sel = st.session_state.get("selected_environment")
            if env_sel:
                payload["environment_id"] = env_sel["environment_id"]
            res = utils.api_post("/start_session", payload)
            if res.get("status") == "success":
                stop_ingest_if_running()
                st.session_state.session_id = res["session_id"]
                st.session_state.mode = "TRACK"
                st.session_state.ingest_proc = launch_ingest(res["session_id"], ip_cam_url, fps=fps)
                st.success("Tracking started. Check the video window on your camera display.")
                log_info(f"Tracking started: session_id={res['session_id']}")
            else:
                msg = res.get("message", "Failed to start tracking")
                st.error(msg)
                log_error(f"Failed to start tracking: {msg}")

    with col2:
        if st.button("Stop Tracking"):
            log_info("[UI] Stop Tracking button clicked")
            sid = st.session_state.session_id
            if sid:
                utils.api_post("/stop_session", {"session_id": sid})
            stop_ingest_if_running()
            st.session_state.session_id = None
            st.session_state.mode = None
            st.success("Stopped.")
            log_info("Tracking stopped")    
    
    # Check if session auto-stopped (e.g., camera disconnected)
    if st.session_state.get("session_id"):
        utils.auto_stop_if_session_ended()