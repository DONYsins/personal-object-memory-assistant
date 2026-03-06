"""
Session-based logging wrapper for the project.

This module creates a single logs folder and uses a SESSION_ID to track
which terminal session is running. When the app starts (backend or UI),
we check if a session marker exists. If not, we create a new session folder.
This way, logs from backend and UI in the same terminal are grouped together,
and only when you kill the terminal and restart do you get a new session folder.

Each session has its own subfolder: logs/<SESSION_ID>/

Key insight for beginners:
- When you run `streamlit run ui_streamlit.py` in terminal A, a SESSION_ID is created
- When you rerun the UI (click button that triggers rerun), it uses SAME SESSION_ID
- Logs go to logs/SESSION_ID/app.log
- When you kill terminal A and start a new terminal, a NEW SESSION_ID is created
- This way all logs from one terminal session are in one folder
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# =====================================
# SESSION ID MANAGEMENT
# This tracks which terminal session we're in
# =====================================

# Check if a session marker file exists
# If it does, use that session ID; if not, create a new one
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

SESSION_MARKER_FILE = os.path.join(LOG_DIR, ".current_session")

def get_session_id():
    """
    Get or create the current session ID.
    When the app starts, if a session marker exists, use that ID.
    Otherwise create a new one with a timestamp.
    """
    if os.path.exists(SESSION_MARKER_FILE):
        try:
            with open(SESSION_MARKER_FILE, "r") as f:
                return f.read().strip()
        except:
            pass
    
    # Create new session ID based on current timestamp
    # Format: 20260226_143025
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        with open(SESSION_MARKER_FILE, "w") as f:
            f.write(session_id)
    except:
        pass
    
    return session_id

# Get session ID and create session-specific log folder
SESSION_ID = get_session_id()
SESSION_LOG_DIR = os.path.join(LOG_DIR, SESSION_ID)
os.makedirs(SESSION_LOG_DIR, exist_ok=True)

# =====================================
# MAIN LOGGER — INFO + ERROR → app.log
# =====================================

LOG_PATH = os.path.join(SESSION_LOG_DIR, "app.log")

logger = logging.getLogger("vision_memory")
logger.setLevel(logging.INFO)  # Only INFO and ERROR go here

if not logger.handlers:
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =====================================
# DEBUG LOGGER — DEBUG only → app_debug.log
# A separate file so debug noise doesn't pollute the main log.
# Only written when detections actually occur (controlled in backend_api.py).
# =====================================

DEBUG_LOG_PATH = os.path.join(SESSION_LOG_DIR, "app_debug.log")

debug_logger = logging.getLogger("vision_memory_debug")
debug_logger.setLevel(logging.DEBUG)
debug_logger.propagate = False  # Don't bubble up to root logger

if not debug_logger.handlers:
    debug_handler = RotatingFileHandler(DEBUG_LOG_PATH, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
    debug_formatter = logging.Formatter("%(asctime)s DEBUG: %(message)s")
    debug_handler.setFormatter(debug_formatter)
    debug_logger.addHandler(debug_handler)


def log_info(message: str, *args) -> None:
    """Log an informational message to app.log."""
    if args:
        logger.info(message % args)
    else:
        logger.info(message)


def log_error(message: str, *args) -> None:
    """Log an error message to app.log."""
    if args:
        logger.error(message % args)
    else:
        logger.error(message)


def log_debug(message: str, *args) -> None:
    """Log a debug message to app_debug.log (separate file, only when detections occur)."""
    if args:
        debug_logger.debug(message % args)
    else:
        debug_logger.debug(message)
