import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import logging
from datetime import datetime

import requests
import streamlit as st

import psutil

from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# ---------------------------
# Config
# ---------------------------
API_QUERY = "http://127.0.0.1:8000/query"

SAMPLE_RATE = 16000
RECORD_SECONDS = 4
AUDIO_PATH = "query.wav"

OBJECTS = ["MyWatch", "MyWallet", "MyBikeKeys"]

# Create logs directory
if 'ui_log_ts' not in st.session_state:
    st.session_state.ui_log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
ts = st.session_state.ui_log_ts
log_dir = os.path.join("logs", f"Logs_UI_{ts}")
os.makedirs(log_dir, exist_ok=True)
LOG_FILE = os.path.join(log_dir, "ui_cmd.log")

# ---------------------------
# Logging (file only)
# ---------------------------
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")]
)
log = logging.getLogger("ui")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource
def load_whisper():
    # note to me: load once so it doesn’t reload every click
    return WhisperModel("base", device="cpu", compute_type="int8")

def kill_application():
    """Kill backend, ingest, and UI processes."""
    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline:
                cmd_str = ' '.join(cmdline)
                if 'uvicorn' in cmd_str and 'backend_api:app' in cmd_str:
                    proc.kill()
                    killed.append("Backend (uvicorn)")
                elif 'python' in cmd_str and 'ingest_ipcam.py' in cmd_str:
                    proc.kill()
                    killed.append("Ingest")
                elif 'streamlit' in cmd_str and 'ui_streamlit.py' in cmd_str:
                    proc.kill()
                    killed.append("UI (streamlit)")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed

def record_audio():
    log.info("Recording audio for %ss", RECORD_SECONDS)
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    write(AUDIO_PATH, SAMPLE_RATE, audio)
    log.info("Saved audio to %s", AUDIO_PATH)

def transcribe_audio():
    whisper = load_whisper()
    segments, info = whisper.transcribe(AUDIO_PATH, beam_size=5)
    text = "".join(seg.text for seg in segments).strip()
    log.info("Whisper detected language=%s", getattr(info, "language", None))
    log.info("Transcription='%s'", text)
    return text

def parse_object(text: str):
    t = text.lower()
    if "watch" in t:
        return "MyWatch"
    if "wallet" in t:
        return "MyWallet"
    if "key" in t or "keys" in t:
        return "MyBikeKeys"
    for obj in OBJECTS:
        if obj.lower() in t:
            return obj
    return None

def parse_k(text: str, default_k=3):
    # note to me: support "last 5" or just "5" in the question
    t = text.lower()
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        return max(1, min(10, int(m.group(1))))
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for w, n in words.items():
        if re.search(rf"\b{w}\b", t):
            return n
    return default_k

def pretty_time(iso_str: str):
    dt = datetime.fromisoformat(iso_str)
    return dt.strftime("%d %b %Y • %I:%M %p")

def ordinal(i: int):
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

def call_query(obj: str, k: int):
    payload = {"object_name": obj, "k": k}
    log.info("POST %s payload=%s", API_QUERY, payload)
    r = requests.post(API_QUERY, json=payload, timeout=10)
    log.info("Response status=%s", r.status_code)
    if r.status_code != 200:
        log.error("Error body=%s", r.text)
        return None, f"Backend error: {r.status_code}"
    return r.json(), None

def show_results_from_state():
    data = st.session_state.get("last_data")
    obj = st.session_state.get("last_obj")
    if not data or not obj:
        return

    results = data.get("results", [])
    if not results:
        st.warning("No saved sightings found.")
        return

    st.subheader(f"Results for: {obj}")

    # radio causes rerun -> we keep selection via key
    idx = st.radio(
        "Select sighting",
        options=list(range(len(results))),
        format_func=lambda i: f"{ordinal(i)} — {pretty_time(results[i]['time_iso'])}",
        key="sighting_radio"
    )

    r = results[idx]

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**{ordinal(idx)}**")
        st.caption(f"Location: {r.get('location','unknown')}")
    with right:
        st.markdown(f"**{pretty_time(r['time_iso'])}**")

    path = (r.get("image_path") or "").strip()
    if not path:
        st.error("No image path returned by backend.")
        log.error("No image_path in result: %s", r)
        return

    if not os.path.isabs(path):
        path = os.path.abspath(path)

    log.info("Displaying image path: %s", path)
    if os.path.exists(path):
        # Streamlit changed: width='stretch' replaces use_container_width=True
        st.image(path, width="stretch")
    else:
        st.error("Image file not found on disk.")
        st.code(path)
        log.error("Image missing on disk: %s", path)

# ---------------------------
# UI
# ---------------------------

# # Close Application button
# col1, col2, col3 = st.columns([6, 2, 2])

# with col3:
#     if st.button("Close App", help="Kill all running processes (backend, ingest, UI)"):
#         killed = kill_application()
#         if killed:
#             st.success(f"Closed: {', '.join(killed)}")
#         else:
#             st.info("No processes found to close.")

st.set_page_config(page_title="Item Memory Assistant", layout="centered")
st.title("Item Memory Assistant")


st.write("Examples: **Where did I last see my watch?** • **Show last 5 wallet sightings**")

query_text = st.text_input("Type your question", value="Where did I last see my watch?")

c1, c2 = st.columns(2)

if c1.button("Search (Text)"):
    if not query_text.strip():
        st.warning("Type a question first.")
    else:
        obj = parse_object(query_text)
        k = parse_k(query_text, default_k=3)

        if not obj:
            st.error("Mention: watch / wallet / keys.")
        else:
            data, err = call_query(obj, k)
            if err:
                st.error(err)
            else:
                # note to me: save in session_state so radio clicks still show results
                st.session_state["last_data"] = data
                st.session_state["last_obj"] = obj

if c2.button("🎤 Speak"):
    st.info("Listening... Please speak now.")
    try:
        record_audio()
        text = transcribe_audio()
        st.write("You said:", text)

        if not text.strip():
            st.error("I couldn’t hear anything clearly. Try again closer to mic.")
            log.warning("Empty transcription")
        else:
            obj = parse_object(text)
            k = parse_k(text, default_k=3)

            if not obj:
                st.error("Mention: watch / wallet / keys.")
            else:
                data, err = call_query(obj, k)
                if err:
                    st.error(err)
                else:
                    st.session_state["last_data"] = data
                    st.session_state["last_obj"] = obj

    except Exception as e:
        st.error(f"Voice error: {e}")
        log.exception("Voice error")

# Always render last results if we have them
show_results_from_state()
