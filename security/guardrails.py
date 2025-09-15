
# FILE: guardrails.py
# Purpose: Simple guardrails + tracing (placeholder for now).
from __future__ import annotations
from typing import Dict, Any, Callable
import functools, time

def approval_gate(plan: Dict[str, Any]) -> bool:
    risks = plan.get("risks", [])
    return True if len(risks) <= 3 else False

def check_prompt_safety(text: str) -> bool:
    lower = text.lower()
    return not any(term in lower for term in ["password", "ssn"])

def trace(step: str):
    def _wrap(fn: Callable):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            start = time.time()
            try:
                out = fn(*args, **kwargs)
                return out
            finally:
                dur = (time.time() - start) * 1000.0
                print(f"[TRACE] step={step} fn={fn.__name__} ms={dur:.1f}")
        return inner
    return _wrap
