#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crew pipeline for resume extraction + structured parsing + validation + profiling.
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    try:
        from crewai.llm import LLM  # type: ignore
        return LLM(model=model, temperature=temperature, api_key=api_key)
    except Exception:
        pass
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
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
