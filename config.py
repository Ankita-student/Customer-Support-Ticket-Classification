"""Central configuration: all paths and constants live here.

Paths are built from this file's location, so every script works no matter
which folder you run it from.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

RAW_CSV = DATA_DIR / "tickets.csv"
CLEAN_CSV = DATA_DIR / "tickets_clean.csv"
MODEL_PATH = MODEL_DIR / "ticket_classifier.joblib"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20            # 80% train / 20% test
LOW_CONFIDENCE = 0.45       # below this, the prediction is flagged for human review


def ensure_dirs():
    for d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR, SCREENSHOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
