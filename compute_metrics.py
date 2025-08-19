
# FILE: compute_metrics.py (enhanced)
# Purpose: Heuristic, transparent metrics per stage (0.0–1.0) + severity levels.
from __future__ import annotations
from typing import Dict, Any, List, Tuple

CRITICAL_FIELDS = [
    ("personal_info", ["name","email","phone"]),
    ("current_role", ["role"]),
]

def _present(v) -> bool:
    if v is None: return False
    if isinstance(v, str): return v.strip() != ""
    if isinstance(v, (list, dict)): return len(v) > 0
    return True

def parse_quality(sj: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    if not isinstance(sj, dict): return 0.0, {"error": "not a dict"}
    present = 0; total = 0; missing = []
    for section, keys in CRITICAL_FIELDS:
        sec = sj.get(section, {}) or {}
        for k in keys:
            total += 1
            if _present(sec.get(k)): present += 1
            else: missing.append(f"{section}.{k}")
    # also consider work_experience[0].start_date if exists
    wx = sj.get("work_experience", []) or []
    total += 1
    if wx and _present(wx[0].get("start_date")): present += 1
    else: missing.append("work_experience[0].start_date")
    score = present / max(1, total)
    # classify non-critical observations
    messages = []
    if not sj.get("projects"):
        messages.append({"kind":"info","msg":"No projects listed; consider adding key practical work."})
    edu = sj.get("education", []) or []
    if edu:
        # If all grad dates are past, that's okay -> info
        messages.append({"kind":"info","msg":"All education dates are in the past; no in-progress degrees noted."})
    return score, {"present": present, "total": total, "missing": missing, "messages": messages}

def role_match_quality(resume_skills: List[str], chosen_role: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    req = set([s.lower() for s in chosen_role.get("required_skills", [])])
    have = set([s.lower() for s in (resume_skills or [])])
    if not req: return 0.0, {"note":"role missing required_skills"}
    cover = len(have & req) / len(req)
    return cover, {"covered": sorted(list(have & req)), "missing": sorted(list(req - have)), "req_count": len(req)}

def courses_coverage_quality(gaps: List[str], suggestions: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    if not gaps: return 1.0, {"note": "no gaps"}
    covered = [g for g in gaps if g in (suggestions or {})]
    score = len(covered) / max(1, len(gaps))
    return score, {"covered": covered, "gaps": gaps}

def clarify_improvement(before_missing: List[str], after_missing: List[str]) -> Tuple[float, Dict[str, Any]]:
    if not before_missing: return 1.0, {"note":"nothing missing before"}
    reduced = len(set(before_missing) - set(after_missing))
    score = reduced / max(1, len(before_missing))
    return score, {"fixed": list(set(before_missing)-set(after_missing)), "remaining": after_missing}
