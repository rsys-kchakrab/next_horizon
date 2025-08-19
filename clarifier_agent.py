
# FILE: clarifier_agent.py
# Purpose: Ask targeted questions when information confidence is low; update session state.
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Set

REQUIRED_SECTIONS = ["personal_info", "current_role", "work_experience", "education", "technical_skills"]

@dataclass
class Question:
    id: str
    text: str
    options: List[str] | None = None
    required: bool = True

class ClarifierAgent:
    def __init__(self, min_required_skill_coverage: float = 0.5):
        self.min_required_skill_coverage = min_required_skill_coverage

    def _collect_resume_skills(self, resume_json: Dict[str, Any]) -> Set[str]:
        skills: Set[str] = set()
        for k in ("technical_skills", "soft_skills"):
            for s in resume_json.get(k, []) or []:
                skills.add(str(s).lower().strip())
        for w in resume_json.get("work_experience", []) or []:
            resp = (w.get("responsibilities") or "").lower()
            for token in ["python","java","c++","mlops","kubernetes","pytorch","tensorflow","onnx","llms","vector databases","spark","airflow"]:
                if token in resp:
                    skills.add(token)
        return skills

    def _skill_coverage(self, have: Set[str], required_skills: List[str]) -> float:
        need = set([s.lower().strip() for s in required_skills if s])
        if not need:
            return 1.0
        return len(have & need) / max(1, len(need))

    def generate_questions(self, resume_json: Dict[str, Any], required_skills: List[str]) -> List[Question]:
        qs: List[Question] = []
        # 1) Missing critical sections/fields
        for f in REQUIRED_SECTIONS:
            if f not in resume_json or resume_json.get(f) in (None, "", [], {}):
                qs.append(Question(id=f, text=f"Could you provide more details for '{f.replace('_',' ')}'?"))
        # 2) Personal info sanity
        pi = resume_json.get("personal_info", {}) or {}
        for key in ("name", "email"):
            if not pi.get(key):
                qs.append(Question(id=f"personal_info.{key}", text=f"What's your {key}?"))
        # 3) Low coverage for selected role
        have = self._collect_resume_skills(resume_json)
        coverage = self._skill_coverage(have, required_skills)
        if coverage < self.min_required_skill_coverage and required_skills:
            missing = sorted(set([s.lower() for s in required_skills]) - set(have))
            if missing:
                qs.append(Question(
                    id="confirm_missing_required_skills",
                    text=f"You're missing some required skills for this role: {', '.join(missing[:12])}. Which do you already have but didn't list?",
                    options=missing, required=False
                ))
        return qs

    def incorporate_answers(self, answers: Dict[str, Any], resume_json: Dict[str, Any]) -> Dict[str, Any]:
        # Answers can be flat (e.g., "technical_skills") or dotted ("personal_info.email")
        for k, v in (answers or {}).items():
            if "." in k:
                head, tail = k.split(".", 1)
                sub = resume_json.setdefault(head, {})
                if isinstance(sub, dict):
                    sub[tail] = v
                    resume_json[head] = sub
            else:
                # Merge lists for skills
                if k in ("technical_skills","soft_skills"):
                    cur = set([str(x).strip() for x in (resume_json.get(k) or [])])
                    if isinstance(v, list):
                        cur |= set([str(x).strip() for x in v])
                    else:
                        cur.add(str(v).strip())
                    resume_json[k] = sorted([x for x in cur if x])
                else:
                    resume_json[k] = v
        return resume_json
