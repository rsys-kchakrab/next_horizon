
# FILE: util_models.py
from __future__ import annotations
from pathlib import Path

def save_model(obj, path: Path):
    try:
        import joblib
        joblib.dump(obj, path)
    except Exception:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(obj, f)

def load_model(path: Path):
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
