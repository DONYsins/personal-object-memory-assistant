"""
Central configuration and constants for the re-identification system.

This file serves as the single source of truth for all configuration values.
Benefits:
  - Easy to manage: change thresholds, classes, limits in one place
  - Suitable for loading from external files (e.g., YAML, JSON, environment variables)
  - Beginner-friendly: clear variable names and comments explaining "why"
  - Maintainable: other modules import from here instead of duplicating values
"""

# ============================================================
# PERSONAL OBJECT TYPES
# ============================================================

PERSONAL_CLASSES = ["Watch", "Wallet", "Bike Key", "Car Key"]

# ============================================================
# ENVIRONMENTAL LANDMARKS
# ============================================================

LANDMARK_CLASSES = [
    "Bed", "Computer Table", "Cupboard", "Dressing Table", "Night Table",
    "Chair", "Table"
]

# ============================================================
# TRACKING PARAMETERS
# ============================================================

# Similarity Threshold for FAISS matching
SIM_THRESHOLD = 0.65

# How many seconds must pass before storing another event for the same object
STORE_INTERVAL_SECONDS = 15

# Tolerate brief detection glitches (occlusion, YOLO miss)
DISAPPEAR_TICKS = 20

# ============================================================
# DATABASE & STORAGE
# ============================================================

# Base directory for all data storage
BASE_DIR = "reid_store"

# YOLO model file path
YOLO_MODEL_PATH = "best.pt"

# Max personal objects per user
MAX_OBJECTS_PER_USER = 10

# ============================================================
# SESSION & RETENTION
# ============================================================

# delete events older than 4 hours of accumulated active time
RETENTION_ACTIVE_HOURS = 4

# Fixed backend FPS
BACKEND_FPS = 10

# ============================================================
# UI CONSTANTS
# ============================================================

# Default IP camera URL shown in UI
IP_CAM_URL_DEFAULT = "http://192.168.1.100:8080/video"

# Data cleanup options (in minutes)
CLEANUP_OPTIONS_HOURS = {
    "1 Hour": 60,
    "2 Hours": 120,
    "4 Hours": 240,
    "12 Hours": 720,
    "1 Day": 1440,
    "2 Days": 2880
}

# ============================================================
# CLIP EMBEDDING MODEL
# ============================================================

# CLIP model version for embedding extraction
# WHY: Smaller model (ViT-B/32) balances speed vs accuracy
# Options: "ViT-B/32" (fast), "ViT-L/14" (slower but more accurate)
CLIP_MODEL = "ViT-B/16"

# Embedding dimension (CLIP used 512)
EMBEDDING_DIM = 512 
