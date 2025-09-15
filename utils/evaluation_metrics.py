
# FILE: evaluation_metrics.py
# Purpose: Lightweight event logger for offline/online metrics.
from __future__ import annotations
import csv, os, time, hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any

LOG_PATH = os.environ.get("NH_EVAL_LOG", "/mnt/data/nexthorizon_eval_log.csv")

@dataclass
class EvalEvent:
    timestamp: float
    user_id: str
    step: str
    metric: str
    value: float
    payload_hash: str

def _hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()[:16]

def log_event(user_id: str, step: str, metric: str, value: float, payload: Dict[str, Any] | None = None) -> None:
    payload = payload or {}
    ev = EvalEvent(time.time(), user_id, step, metric, value, _hash_payload(payload))
    header = [f.name for f in EvalEvent.__dataclass_fields__.values()]
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(asdict(ev))
