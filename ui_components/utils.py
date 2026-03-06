"""Utility functions shared across UI components."""
import os
from datetime import datetime

import streamlit as st
import requests

from logger import log_info, log_error
# Import constants (centralized configuration)
from constants import IP_CAM_URL_DEFAULT as DEFAULT_IP_CAM_URL

# configuration constants
BACKEND = "http://127.0.0.1:8000"

# Use IP camera URL from constants (can be overridden in environment)
IP_CAM_URL_DEFAULT = os.getenv("IP_CAM_URL", DEFAULT_IP_CAM_URL)

# API helpers

def api_post(path, data):
    """POST to backend and return JSON response (with timeout)."""
    try:
        resp = requests.post(f"{BACKEND}{path}", data=data, timeout=15)
        # raise an exception for 4xx/5xx so we can log status and body
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as he:
            log_error("[API] API POST returned error status %s for %s: %s", resp.status_code, path, resp.text[:200])
            # attempt to parse any JSON error body before giving up
            try:
                return resp.json()
            except Exception:
                return {"status": "error", "message": f"Server returned status {resp.status_code}"}
        # when backend crashes or returns HTML the .json() call may fail
        return resp.json()
    except requests.exceptions.JSONDecodeError as jde:
        # backend returned non-json (html, empty, etc.)
        body = resp.text if 'resp' in locals() else ''
        log_error("[API] Non-JSON response from backend POST %s status=%s: %s", path, getattr(resp, 'status_code', None), body[:200])
        return {"status": "error", "message": "Invalid response from server"}
    except requests.exceptions.RequestException as e:
        # network error or timeout (includes HTTPError since subclass)
        log_error("[API] API POST request failed: %s %s", path, e)
        return {"status": "error", "message": f"Network error: {e}"}
    except ValueError:
        # catch any other JSON decode problems
        body = resp.text if 'resp' in locals() else ''
        log_error("[API] Non-JSON response from backend POST %s: %s", path, body[:200])
        return {"status": "error", "message": "Invalid response from server"}


def api_get(path, params):
    """GET from backend and return JSON response."""
    try:
        resp = requests.get(f"{BACKEND}{path}", params=params, timeout=15)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as he:
            log_error("[API] API GET returned error status %s for %s: %s", resp.status_code, path, resp.text[:200])
            try:
                return resp.json()
            except Exception:
                return {"status": "error", "message": f"Server returned status {resp.status_code}"}
        return resp.json()
    except requests.exceptions.JSONDecodeError as jde:
        body = resp.text if 'resp' in locals() else ''
        log_error("[API] Non-JSON response from backend GET %s status=%s: %s", path, getattr(resp, 'status_code', None), body[:200])
        return {"status": "error", "message": "Invalid response from server"}
    except requests.exceptions.RequestException as e:
        log_error("[API] API GET request failed: %s %s", path, e)
        return {"status": "error", "message": f"Network error: {e}"}
    except ValueError:
        body = resp.text if 'resp' in locals() else ''
        log_error("[API] Non-JSON response from backend GET %s: %s", path, body[:200])
        return {"status": "error", "message": "Invalid response from server"}

# formatting helpers

def pretty_time(iso_str: str) -> str:
    """Convert ISO timestamp to human-readable form, local timezone."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d %b %Y • %I:%M %p")
    except Exception:
        return iso_str


def ordinal(i: int) -> str:
    names = [
        "Last seen",
        "Second most recent",
        "Third most recent",
        "Fourth most recent",
        "Fifth most recent",
        "Sixth most recent",
        "Seventh most recent",
        "Eighth most recent",
        "Ninth most recent",
        "Tenth most recent",
    ]
    return names[i] if i < len(names) else f"Result #{i+1}"

def auto_stop_if_session_ended():
    """If session stopped (camera timeout), reset UI and rerun."""
    sid = st.session_state.get("session_id")
    if sid and api_get("/session_status", {"session_id": sid}).get("session_status") == "STOPPED":
        st.session_state.session_id = None
        st.session_state.mode = None
        from ui_components.tracking import stop_ingest_if_running
        stop_ingest_if_running()
        st.warning("✓ Session auto-stopped")
        # Removed st.rerun() - keep buttons responsive