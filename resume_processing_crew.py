#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crew pipeline for resume extraction + structured parsing + validation + profiling,
PLUS a GuidanceEngine to recommend roles, compute skill gaps, and suggest courses.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Set

from crewai import Agent, Task, Crew, Process

# -----------------------------
# LLM helper (tries multiple backends)
# -----------------------------

def _get_llm(model: str, temperature: float = 0.2):
    """Return an LLM object compatible with CrewAI Agents.
    Tries crewai.llm, then langchain_openai, then legacy langchain.llms.OpenAI.
    """
    try:
        from crewai.llm import LLM  # type: ignore
        return LLM(model=model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        pass
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(model=model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        pass
    try:
        from langchain.llms import OpenAI  # type: ignore
        return OpenAI(model_name=model, temperature=temperature)
    except Exception:
        pass
    raise RuntimeError("Could not initialize an OpenAI-compatible LLM. Install crewai>=0.30 or langchain-openai.")

# -----------------------------
# Utilities
# -----------------------------

def _to_text(x: Any) -> str:
    if x is None:
        return ""
    for attr in ("raw", "result", "output", "content", "text"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            return v if isinstance(v, str) else str(v)
    return str(x)


def extract_text_from_file(file_path: str) -> str:
    from pathlib import Path
    p = Path(file_path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(p))
            pages = []
            for pg in reader.pages:
                try:
                    pages.append(pg.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n".join(pages)
        except Exception as e:
            return f"[ERROR] PDF parse failed: {e}"
    if suffix == ".docx":
        try:
            import docx as docx_lib
            d = docx_lib.Document(str(p))
            return "\n".join([para.text for para in d.paragraphs])
        except Exception:
            try:
                import docx2txt
                return docx2txt.process(str(p)) or ""
            except Exception as e:
                return f"[ERROR] DOCX parse failed: {e}"
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return p.read_text(encoding="latin-1", errors="ignore")
        except Exception as e:
            return f"[ERROR] Text read failed: {e}"

# -----------------------------
# Parsing helpers (JSON-first, fallback to headers + regex rescue)
# -----------------------------

def parse_structured_info(structured_info: Any) -> Dict[str, Any]:
    import json as _json

    def _coerce_text(x: Any) -> str:
        return _to_text(x).strip()

    text = _coerce_text(structured_info)

    fields = {
        'personal_info':'',
        'professional_summary':'',
        'current_role':'',
        'work_experience':'',
        'education':'',
        'technical_skills':'',
        'soft_skills':'',
        'certifications':'',
        'key_achievements':'',
        'projects':'',
        'career_level':'',
        'industry_focus':'',
    }

    def _extract_json_block(s: str):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        candidate = m.group(0) if m else s
        try:
            return _json.loads(candidate)
        except Exception:
            return None

    js = _extract_json_block(text)
    if isinstance(js, dict):
        return js

    # header-based fallback
    headers = {
        'personal_info': ['PERSONAL INFORMATION','PERSONAL INFO','CONTACT','CONTACT DETAILS','CONTACT INFORMATION'],
        'professional_summary': ['PROFESSIONAL SUMMARY','SUMMARY','OBJECTIVE','SUMMARY OF QUALIFICATIONS'],
        'current_role': ['CURRENT ROLE','CURRENT POSITION','PRESENT ROLE','PRESENT POSITION'],
        'work_experience': ['WORK EXPERIENCE','EXPERIENCE','EMPLOYMENT','EMPLOYMENT HISTORY','PROFESSIONAL EXPERIENCE'],
        'education': ['EDUCATION','ACADEMIC QUALIFICATIONS','ACADEMICS'],
        'technical_skills': ['TECHNICAL SKILLS','TECHNICAL','PROGRAMMING','SKILLS'],
        'soft_skills': ['SOFT SKILLS','INTERPERSONAL','BEHAVIORAL'],
        'certifications': ['CERTIFICATIONS','CERTIFICATES','LICENSES'],
        'key_achievements': ['ACHIEVEMENTS','AWARDS','ACCOMPLISHMENTS','KEY ACHIEVEMENTS'],
        'projects': ['PROJECTS','PROJECT'],
    }

    out = fields.copy(); current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        up = line.upper().rstrip(':')
        matched = False
        for key, variants in headers.items():
            if any(v in up for v in variants):
                current = key
                matched = True
                break
        if matched:
            continue
        if current and not line.endswith(':'):
            out[current] += (('\n' if out[current] else '') + line)

    if not out['personal_info']:
        email = re.search(r'[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}', text)
        phone = re.search(r'(?:\+?\d[\s-]?)?(?:\(?\d{3,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}', text)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        name_line = lines[0] if lines else ''
        bits = []
        if name_line: bits.append(f"Name: {name_line}")
        if email: bits.append(f"Email: {email.group(0)}")
        if phone: bits.append(f"Phone: {phone.group(0)}")
        out['personal_info'] = "\n".join(bits)

    return out


def normalize_structured_json(struct_like: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce various parsed shapes into the canonical schema used by the UI.
    Also performs small normalizations like mapping 'Till date' → 'Present'.
    """
    def _list(x):
        return x if isinstance(x, list) else ([] if x in (None, "") else [x])

    def _norm_date(s: str) -> str:
        s = (s or "").strip()
        s_up = s.upper()
        if s_up in {"TILL DATE", "TILLDATE", "PRESENT", "CURRENT", "NOW", "TODAY"}:
            return "Present"
        return s

    out = {
        "personal_info": {"name": "", "email": "", "phone": "", "location": ""},
        "professional_summary": "",
        "current_role": {"role": "", "company": ""},
        "work_experience": [],
        "education": [],
        "technical_skills": [],
        "soft_skills": [],
        "certifications": [],
        "projects": [],
        "career_level": struct_like.get("career_level", ""),
        "industry_focus": struct_like.get("industry_focus", ""),
    }

    pi = struct_like.get("personal_info")
    if isinstance(pi, dict):
        out["personal_info"].update({
            "name": pi.get("name", ""),
            "email": pi.get("email", ""),
            "phone": pi.get("phone", ""),
            "location": pi.get("location", ""),
        })
    elif isinstance(pi, str):
        name_m = re.search(r"(?i)name\s*:\s*(.+)", pi)
        email_m = re.search(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}", pi)
        phone_m = re.search(r"(?:\+?\d[\s-]?)?(?:\(?\d{3,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}", pi)
        out["personal_info"]["name"] = (name_m.group(1).strip() if name_m else "")
        out["personal_info"]["email"] = (email_m.group(0) if email_m else "")
        out["personal_info"]["phone"] = (phone_m.group(0) if phone_m else "")

    out["professional_summary"] = struct_like.get("professional_summary", "") or struct_like.get("summary", "")

    cr = struct_like.get("current_role")
    if isinstance(cr, dict):
        out["current_role"]["role"] = cr.get("role", "")
        out["current_role"]["company"] = cr.get("company", "")

    wx = struct_like.get("work_experience")
    if isinstance(wx, list):
        for w in wx:
            out["work_experience"].append({
                "title": (w.get("title") if isinstance(w, dict) else ""),
                "company": (w.get("company") if isinstance(w, dict) else ""),
                "start_date": _norm_date((w.get("start_date") if isinstance(w, dict) else "")),
                "end_date": _norm_date((w.get("end_date") if isinstance(w, dict) else "")),
                "responsibilities": (w.get("responsibilities") if isinstance(w, dict) else ""),
            })

    ed = struct_like.get("education")
    if isinstance(ed, list):
        for e in ed:
            out["education"].append({
                "degree": (e.get("degree") if isinstance(e, dict) else ""),
                "institution": (e.get("institution") if isinstance(e, dict) else ""),
                "graduation_date": _norm_date((e.get("graduation_date") if isinstance(e, dict) else "")),
            })

    out["technical_skills"] = [s for s in _list(struct_like.get("technical_skills")) if s]
    out["soft_skills"] = [s for s in _list(struct_like.get("soft_skills")) if s]
    out["certifications"] = [s for s in _list(struct_like.get("certifications")) if s]
    out["projects"] = [s for s in _list(struct_like.get("projects")) if s]

    return out

# -----------------------------
# Crew for extraction/validation/profile
# -----------------------------

class ResumeProcessingCrew:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature
        self.llm = _get_llm(model=self.model, temperature=self.temperature)

        self.file_processor_agent = Agent(
            role="ResumeTextCleaner",
            goal="Clean and normalize raw resume text while preserving semantic content.",
            backstory=(
                "You remove noise (headers, footers, page numbers), deduplicate lines, and produce clean text."
            ),
            allow_delegation=False,
            llm=self.llm,
        )

        self.resume_parser_agent = Agent(
            role="ResumeParser",
            goal="Extract structured information from resumes in a strict JSON schema.",
            backstory=("You always return valid JSON strictly matching the schema provided."),
            allow_delegation=False,
            llm=self.llm,
        )

        self.validator_agent = Agent(
            role="ResumeValidator",
            goal="Validate the structured JSON and identify missing or inconsistent fields.",
            backstory=("You check completeness and coherence and write a short plain report."),
            allow_delegation=False,
            llm=self.llm,
        )

        self.profile_agent = Agent(
            role="CareerProfiler",
            goal="Create a concise career profile summary with role, experience, and top skills.",
            backstory=("You summarize the candidate in 5–7 crisp bullets."),
            allow_delegation=False,
            llm=self.llm,
        )

    def _create_tasks(self, raw_text: str):
        # Task 1
        t1 = Task(
            description=(
                "You are given RAW_RESUME_TEXT between triple backticks.\n"
                "- Remove headers/footers/page numbers/duplicate lines\n"
                "- Normalize bullets to '-'\n"
                "- Preserve headings\n"
                "Return ONLY the cleaned text.\n\n"
                f"```\n{raw_text}\n```"
            ),
            expected_output="Cleaned resume text only",
            agent=self.file_processor_agent,
        )

        # Task 2
        schema = {
            "personal_info": {"name": "", "email": "", "phone": "", "location": ""},
            "professional_summary": "",
            "current_role": {"role": "", "company": ""},
            "work_experience": [
                {"title": "", "company": "", "start_date": "", "end_date": "", "responsibilities": ""}
            ],
            "education": [
                {"degree": "", "institution": "", "graduation_date": ""}
            ],
            "technical_skills": [""],
            "soft_skills": [""],
            "certifications": [""],
            "key_achievements": [""],
            "projects": [""],
            "career_level": "",
            "industry_focus": "",
        }
        t2 = Task(
            description=(
                "Analyze the CLEANED text and extract structured professional information.\n"
                "Return ONLY valid JSON matching EXACTLY this schema (keys & nesting):\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                "Use empty strings/lists if unknown. No commentary outside JSON."
            ),
            expected_output="Valid JSON strictly matching the schema.",
            agent=self.resume_parser_agent,
            context=[t1],
        )

        # Task 3
        t3 = Task(
            description=(
                "Validate the extracted JSON for completeness/coherence.\n"
                "- Flag missing critical fields (name/email/phone/recent role)\n"
                "- Accept future graduation dates as 'in-progress'\n"
                "- Map labels like 'Till date' to 'Present' rather than error\n"
                "Return a short plain report: 'OK' or 'Issues Found' with bullets."
            ),
            expected_output="Short validation report.",
            agent=self.validator_agent,
            context=[t2],
        )

        # Task 4
        t4 = Task(
            description=(
                "Create a concise career profile in 5–7 bullets: current role/company, years (if inferable),"
                " top technical skills, domains/industries, short value proposition."
            ),
            expected_output="5–7 bullet profile.",
            agent=self.profile_agent,
            context=[t2],
        )

        return t1, t2, t3, t4

    def process_resume(self, file_path: str) -> Dict[str, Any]:
        raw_text = extract_text_from_file(file_path)
        if raw_text.startswith("[ERROR]") or not raw_text.strip():
            return {"success": False, "extracted_text": raw_text, "structured_info": "", "validation_report": "", "career_profile": "", "final_result": None}

        t1, t2, t3, t4 = self._create_tasks(raw_text)
        crew = Crew(agents=[self.file_processor_agent, self.resume_parser_agent, self.validator_agent, self.profile_agent], tasks=[t1, t2, t3, t4], process=Process.sequential, verbose=True)
        final = crew.kickoff()
        return {
            "success": True,
            "extracted_text": _to_text(t1.output),
            "structured_info": _to_text(t2.output),
            "validation_report": _to_text(t3.output),
            "career_profile": _to_text(t4.output),
            "final_result": final,
        }

# -----------------------------
# Guidance Engine (roles, gaps, courses)
# -----------------------------

_COURSE_CATALOG: Dict[str, List[Dict[str, str]]] = {
    # Example mappings; extend as needed.
    "python": [
        {"name": "Python for Everybody", "provider": "Coursera", "link": "https://www.coursera.org/specializations/python"},
    ],
    "machine learning": [
        {"name": "Machine Learning", "provider": "Coursera (Andrew Ng)", "link": "https://www.coursera.org/learn/machine-learning"},
        {"name": "Intro to Machine Learning", "provider": "Kaggle Learn", "link": "https://www.kaggle.com/learn/intro-to-machine-learning"},
    ],
    "deep learning": [
        {"name": "Deep Learning Specialization", "provider": "Coursera (DeepLearning.AI)", "link": "https://www.coursera.org/specializations/deep-learning"},
        {"name": "Practical Deep Learning for Coders", "provider": "fast.ai", "link": "https://course.fast.ai/"},
    ],
    "mlops": [
        {"name": "MLOps Specialization", "provider": "Coursera", "link": "https://www.coursera.org/specializations/mlops"},
        {"name": "MLOps with Azure ML", "provider": "Microsoft Learn", "link": "https://learn.microsoft.com/azure/machine-learning/"},
    ],
    "llms": [
        {"name": "LLM Engineering", "provider": "DeepLearning.AI", "link": "https://www.deeplearning.ai/short-courses/"},
    ],
    "vector databases": [
        {"name": "Intro to Vector Databases", "provider": "Pinecone", "link": "https://www.pinecone.io/learn/"},
    ],
    "telecom": [
        {"name": "5G Core Networking", "provider": "Coursera", "link": "https://www.coursera.org/learn/5g-core-networking"},
        {"name": "5G NR Radio Access", "provider": "edX", "link": "https://www.edx.org/"},
    ],
    "ran optimization": [
        {"name": "Radio Access Network Optimization", "provider": "Coursera", "link": "https://www.coursera.org/"},
    ],
}

class GuidanceEngine:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature
        self.llm = None
        try:
            self.llm = _get_llm(model=self.model, temperature=self.temperature)
        except Exception:
            self.llm = None  # allow fallback

    # ---------- Public API ----------
    def generate_guidance(self, resume_json: Dict[str, Any], aspirations: Dict[str, Any]) -> Dict[str, Any]:
        roles = self._recommend_roles(resume_json, aspirations)
        return {"recommended_roles": roles}

    def compute_gaps(self, resume_json: Dict[str, Any], role: Dict[str, Any]) -> List[str]:
        have = self._collect_resume_skills(resume_json)
        need = set([s.lower().strip() for s in role.get("required_skills", []) if s])
        return sorted(list(need - have))

    def suggest_courses(self, missing_skills: List[str]) -> Dict[str, List[Dict[str, str]]]:
        out: Dict[str, List[Dict[str, str]]] = {}
        for skill in missing_skills:
            k = skill.lower()
            out[k] = _COURSE_CATALOG.get(k, [])
        return out

    # ---------- Internals ----------
    def _llm_complete(self, prompt: str) -> str:
        """Call the LLM regardless of backend differences, return text."""
        if self.llm is None:
            raise RuntimeError("No LLM backend")
        # crewai.llm.LLM is callable; ChatOpenAI has invoke(); OpenAI may be callable
        try:
            if hasattr(self.llm, "invoke"):
                resp = self.llm.invoke(prompt)
                return _to_text(resp)
            if callable(self.llm):
                resp = self.llm(prompt)
                return _to_text(resp)
            if hasattr(self.llm, "predict"):
                resp = self.llm.predict(prompt)
                return _to_text(resp)
        except Exception as e:
            raise
        raise RuntimeError("Unsupported LLM interface")

    def _collect_resume_skills(self, resume_json: Dict[str, Any]) -> Set[str]:
        skills: Set[str] = set()
        for k in ("technical_skills", "soft_skills"):
            for s in resume_json.get(k, []) or []:
                skills.add(str(s).lower().strip())
        # Try to harvest from responsibilities text (very naive)
        for w in resume_json.get("work_experience", []) or []:
            resp = (w.get("responsibilities") or "").lower()
            for token in ["python", "c++", "java", "mlops", "kubernetes", "pytorch", "tensorflow", "onnx", "llms", "vector databases", "telecom", "ran", "o-ran", "ric", "cu", "du"]:
                if token in resp:
                    skills.add(token)
        return skills

    def _recommend_roles(self, resume_json: Dict[str, Any], aspirations: Dict[str, Any]) -> List[Dict[str, Any]]:
        # If LLM available, ask it to propose roles JSON; else use a heuristic fallback.
        if self.llm is not None:
            prompt = (
                "You are a career coach for telecom + AI professionals. Given RESUME_JSON and ASPIRATIONS, "
                "return ONLY JSON with this schema:\n\n"
                "{\n  \"recommended_roles\": [\n    {\n      \"title\": \"\",\n      \"why\": \"\",\n      \"required_skills\": [\"\"],\n      \"nice_to_have\": [\"\"\n      ]\n    }\n  ]\n}\n\n"
                "Keep titles specific (e.g., 'AI/ML Engineer (Telecom)'). "
                "Use 6–12 required skills; reflect resume strengths and aspiration focus. No prose outside JSON.\n\n"
                f"RESUME_JSON = {json.dumps(resume_json)}\n\nASPIRATIONS = {json.dumps(aspirations)}"
            )
            try:
                text = self._llm_complete(prompt)
                data = json.loads(text)
                roles = data.get("recommended_roles", [])
                # ensure structure
                norm = []
                for r in roles:
                    norm.append({
                        "title": r.get("title", ""),
                        "why": r.get("why", ""),
                        "required_skills": r.get("required_skills", []) or [],
                        "nice_to_have": r.get("nice_to_have", []) or [],
                    })
                return norm
            except Exception:
                pass

        # Fallback (heuristic): map on simple keywords
        have = self._collect_resume_skills(resume_json)
        roles: List[Dict[str, Any]] = []
        def _mk(title: str, why: str, req: List[str], nice: List[str]):
            roles.append({"title": title, "why": why, "required_skills": req, "nice_to_have": nice})

        if any(k in have for k in ["ran", "o-ran", "ric", "telecom"]):
            _mk(
                "AI/ML Engineer (Telecom)",
                "Strong telecom background with interest in AI makes this a good bridge role.",
                ["python", "machine learning", "deep learning", "pytorch", "mlops", "telecom", "vector databases", "llms"],
                ["kubernetes", "onnx", "o-ran", "ric"],
            )
            _mk(
                "RAN Optimization Lead",
                "Direct leverage of RAN expertise with analytics and automation.",
                ["telecom", "ran optimization", "python", "machine learning", "feature engineering", "time-series analysis"],
                ["spark", "airflow", "powerbi"],
            )
        _mk(
            "Solution Architect (AI/5G)",
            "Architectural background + AI adoption across RAN/CU/DU.",
            ["telecom", "5g", "python", "mlops", "cloud", "kubernetes", "llms", "vector databases"],
            ["azure", "aws", "gcp"],
        )
        _mk(
            "MLOps Engineer",
            "If you enjoy productionizing models and infra.",
            ["mlops", "python", "kubernetes", "docker", "ci/cd", "model registry", "monitoring"],
            ["azure ml", "mlflow", "kubeflow"],
        )
        return roles
