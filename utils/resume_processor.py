# FILE: utils/resume_processor.py - Resume Processing Pipeline (AI Provider Agnostic)
from __future__ import annotations
import json
from typing import Dict, Any, List
from pathlib import Path

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file formats"""
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
    
    elif suffix == ".docx":
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
    
    # Plain text files
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return p.read_text(encoding="latin-1", errors="ignore")
        except Exception as e:
            return f"[ERROR] Text read failed: {e}"

def clean_resume_text(raw_text: str) -> str:
    """Clean and normalize raw resume text"""
    if not raw_text or raw_text.startswith("[ERROR]"):
        return raw_text
    
    lines = raw_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip headers/footers that might be page numbers or metadata
        if len(line) < 5 and line.isdigit():
            continue
        
        # Normalize bullets
        if line.startswith(('•', '●', '◦', '-', '*')):
            line = '- ' + line[1:].strip()
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_structured_resume_data(resume_text: str) -> Dict[str, Any]:
    """
    Extract structured data from resume text using AI parsing (provider-agnostic).
    
    This function delegates to AI provider-specific implementations in the ai/ module.
    Currently uses OpenAI, but can be easily swapped for other AI providers by
    modifying the import and function call below.
    """
    from ai.openai_client import openai_parse_resume
    
    try:
        structured_data = openai_parse_resume(resume_text)
        return normalize_structured_data(structured_data)
    except Exception as e:
        raise RuntimeError(f"Failed to extract structured data: {str(e)}")

def normalize_structured_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate the structured data"""
    
    def _list(x):
        return x if isinstance(x, list) else ([] if x in (None, "") else [x])
    
    def _norm_date(s: str) -> str:
        s = (s or "").strip()
        s_up = s.upper()
        if s_up in {"TILL DATE", "TILLDATE", "PRESENT", "CURRENT", "NOW", "TODAY"}:
            return "Present"
        return s
    
    normalized = {
        "professional_summary": "",
        "current_role": {"role": "", "company": ""},
        "technical_skills": [],
        "career_level": data.get("career_level", ""),
        "industry_focus": data.get("industry_focus", ""),
        "work_experience": [],
        "key_achievements": [],
        "soft_skills": [],
        "location": "",
        "projects": [],
        "education": [],
        "certifications": [],
    }
    
    # Normalize location
    normalized["location"] = data.get("location", "")
    
    # Normalize professional summary
    normalized["professional_summary"] = data.get("professional_summary", "") or data.get("summary", "")
    
    # Normalize current role
    cr = data.get("current_role", {})
    if isinstance(cr, dict):
        normalized["current_role"]["role"] = cr.get("role", "")
        normalized["current_role"]["company"] = cr.get("company", "")
    
    # Normalize work experience
    for exp in data.get("work_experience", []) or []:
        if isinstance(exp, dict):
            normalized["work_experience"].append({
                "title": exp.get("title", ""),
                "company": exp.get("company", ""),
                "start_date": _norm_date(exp.get("start_date", "")),
                "end_date": _norm_date(exp.get("end_date", "")),
                "responsibilities": exp.get("responsibilities", ""),
            })
    
    # Normalize education
    for edu in data.get("education", []) or []:
        if isinstance(edu, dict):
            normalized["education"].append({
                "degree": edu.get("degree", ""),
                "institution": edu.get("institution", ""),
                "graduation_date": _norm_date(edu.get("graduation_date", "")),
            })
    
    # Normalize skill lists
    normalized["technical_skills"] = [s for s in _list(data.get("technical_skills")) if s]
    normalized["soft_skills"] = [s for s in _list(data.get("soft_skills")) if s]
    normalized["certifications"] = [s for s in _list(data.get("certifications")) if s]
    normalized["key_achievements"] = [s for s in _list(data.get("key_achievements")) if s]
    normalized["projects"] = [s for s in _list(data.get("projects")) if s]
    
    return normalized

def validate_resume_data(structured_data: Dict[str, Any]) -> List[str]:
    """Validate resume data and return list of issues"""
    issues = []
    
    location = structured_data.get("location", "")
    if not location:
        issues.append("Missing location information")
    
    if not structured_data.get("technical_skills"):
        issues.append("No technical skills found")
    
    if not structured_data.get("work_experience"):
        issues.append("No work experience found")
    
    return issues

def create_career_profile(structured_data: Dict[str, Any]) -> str:
    """Create a concise career profile summary"""
    
    profile_points = []
    
    # Current role and company
    current_role = structured_data.get("current_role", {})
    if current_role.get("role") and current_role.get("company"):
        profile_points.append(f"Currently working as {current_role['role']} at {current_role['company']}")
    
    # Years of experience (estimated from work history)
    work_exp = structured_data.get("work_experience", [])
    if work_exp:
        profile_points.append(f"Professional with {len(work_exp)} role(s) of experience")
    
    # Top technical skills
    tech_skills = structured_data.get("technical_skills", [])[:5]  # Top 5
    if tech_skills:
        profile_points.append(f"Skilled in: {', '.join(tech_skills)}")
    
    # Industry focus
    industry = structured_data.get("industry_focus", "")
    if industry:
        profile_points.append(f"Industry focus: {industry}")
    
    # Education
    education = structured_data.get("education", [])
    if education:
        latest_edu = education[0]  # Assume first is latest
        if latest_edu.get("degree"):
            profile_points.append(f"Education: {latest_edu['degree']}")
    
    return "• " + "\n• ".join(profile_points) if profile_points else "Career profile information not available"

def create_optimized_resume_text(structured_data: Dict[str, Any]) -> str:
    """Create resume text with strategic ordering for course recommendations"""
    
    text_parts = []
    
    # 1. Professional Summary
    summary = structured_data.get("professional_summary", "").strip()
    if summary:
        text_parts.append(f"PROFESSIONAL SUMMARY: {summary}")
    
    # 2. Current Role
    current_role = structured_data.get("current_role", {})
    if current_role.get("role") or current_role.get("company"):
        role_text = f"CURRENT ROLE: {current_role.get('role', '')} at {current_role.get('company', '')}".strip()
        text_parts.append(role_text)
    
    # 3. Technical Skills
    tech_skills = structured_data.get("technical_skills", [])
    if tech_skills:
        skills_text = ", ".join(tech_skills)
        text_parts.append(f"TECHNICAL SKILLS: {skills_text}")
    
    # 4. Career Level
    career_level = structured_data.get("career_level", "").strip()
    if career_level:
        text_parts.append(f"CAREER LEVEL: {career_level}")
    
    # 5. Industry Focus
    industry_focus = structured_data.get("industry_focus", "").strip()
    if industry_focus:
        text_parts.append(f"INDUSTRY FOCUS: {industry_focus}")
    
    # 6. Work Experience (condensed)
    work_exp = structured_data.get("work_experience", [])
    if work_exp:
        exp_parts = []
        for exp in work_exp[:3]:  # Top 3 most recent
            title = exp.get("title", "")
            company = exp.get("company", "")
            if title or company:
                exp_parts.append(f"{title} at {company}".strip())
        if exp_parts:
            text_parts.append(f"WORK EXPERIENCE: {'; '.join(exp_parts)}")
    
    # 7. Key Achievements
    achievements = structured_data.get("key_achievements", [])
    if achievements:
        achievements_text = "; ".join(achievements[:3])  # Top 3 achievements
        text_parts.append(f"KEY ACHIEVEMENTS: {achievements_text}")
    
    # 8. Soft Skills
    soft_skills = structured_data.get("soft_skills", [])
    if soft_skills:
        soft_skills_text = ", ".join(soft_skills)
        text_parts.append(f"SOFT SKILLS: {soft_skills_text}")
    
    # 9. Location
    location = structured_data.get("location", "").strip()
    if location:
        text_parts.append(f"LOCATION: {location}")
    
    # 10. Projects
    projects = structured_data.get("projects", [])
    if projects:
        projects_text = "; ".join(projects[:2])  # Top 2 projects
        text_parts.append(f"PROJECTS: {projects_text}")
    
    # 11. Education (condensed)
    education = structured_data.get("education", [])
    if education:
        edu_parts = []
        for edu in education[:2]:  # Top 2 degrees
            degree = edu.get("degree", "")
            institution = edu.get("institution", "")
            if degree or institution:
                edu_parts.append(f"{degree} from {institution}".strip())
        if edu_parts:
            text_parts.append(f"EDUCATION: {'; '.join(edu_parts)}")
    
    # 12. Certifications
    certifications = structured_data.get("certifications", [])
    if certifications:
        certs_text = ", ".join(certifications)
        text_parts.append(f"CERTIFICATIONS: {certs_text}")
    
    return " | ".join(text_parts)

def prepare_resume_text(resume_doc: Dict[str, Any]) -> str:
    """Convert structured resume data to text format for ML processing"""
    parts: List[str] = []
    if not isinstance(resume_doc, dict):
        return str(resume_doc) if resume_doc else ""
    
    # Extract current role
    cr = resume_doc.get("current_role") or {}
    if isinstance(cr, dict):
        parts.append(str(cr.get("role","")))
        parts.append(str(cr.get("summary","")))
    elif isinstance(cr, str):
        parts.append(cr)
    
    # Extract skills
    skills = resume_doc.get("technical_skills") or resume_doc.get("skills") or []
    if isinstance(skills, (list, tuple)):
        parts.extend([str(s) for s in skills if s])
    elif isinstance(skills, str):
        parts.append(skills)
    
    # Extract work experience
    for w in (resume_doc.get("work_experience") or []):
        if not isinstance(w, dict): 
            parts.append(str(w))
            continue
        parts.append(str(w.get("title","")))
        parts.append(str(w.get("company","")))
        parts.append(str(w.get("summary","")))
        b = w.get("bullets") or w.get("responsibilities")
        if isinstance(b, str): 
            parts.append(b)
        elif isinstance(b, (list, tuple)): 
            parts.extend([str(x) for x in b if x])
    
    # Extract projects
    for p in (resume_doc.get("projects") or []):
        if isinstance(p, dict):
            parts.append(str(p.get("name","")))
            parts.append(str(p.get("description","")))
        else:
            parts.append(str(p))
    
    return " ".join([x for x in parts if x]).strip()

def build_resume_text() -> str:
    """Build comprehensive resume text from structured JSON data"""
    import streamlit as st
    
    # Get the updated structured JSON from resume parsing tab
    structured_json = st.session_state.get("structured_json", {})
    
    if structured_json:
        # Build comprehensive resume text from structured data
        text_parts = []
        
        # Professional summary
        if structured_json.get("professional_summary"):
            text_parts.append(f"Professional Summary: {structured_json['professional_summary']}")
        
        # Current role
        current_role = structured_json.get("current_role", {})
        if current_role.get("role") and current_role.get("company"):
            text_parts.append(f"Current Role: {current_role['role']} at {current_role['company']}")
        
        # Work experience
        work_exp = structured_json.get("work_experience", [])
        if work_exp:
            work_text = "Work Experience: "
            for exp in work_exp:
                if exp.get("title") and exp.get("company"):
                    work_text += f"{exp['title']} at {exp['company']} ({exp.get('start_date', '')} - {exp.get('end_date', '')}). "
                    if exp.get("responsibilities"):
                        work_text += f"Responsibilities: {exp['responsibilities']} "
            text_parts.append(work_text)
        
        # Education
        education = structured_json.get("education", [])
        if education:
            edu_text = "Education: "
            for edu in education:
                if edu.get("degree") and edu.get("institution"):
                    edu_text += f"{edu['degree']} in {edu.get('specialization', '')} from {edu['institution']} ({edu.get('graduation_date', '')}). "
            text_parts.append(edu_text)
        
        # Skills
        tech_skills = structured_json.get("technical_skills", [])
        if tech_skills:
            text_parts.append(f"Technical Skills: {', '.join(tech_skills)}")
        
        soft_skills = structured_json.get("soft_skills", [])
        if soft_skills:
            text_parts.append(f"Soft Skills: {', '.join(soft_skills)}")
        
        # Certifications
        certifications = structured_json.get("certifications", [])
        if certifications:
            text_parts.append(f"Certifications: {', '.join(certifications)}")
        
        # Total experience
        total_exp = structured_json.get("total_years_experience")
        if total_exp:
            text_parts.append(f"Total Years of Experience: {total_exp}")
        
        return " ".join(text_parts)
    
    # Fallback to original logic
    resume_json = st.session_state.get("resume_json") or {}
    try:
        txt = prepare_resume_text(resume_json)
    except Exception:
        txt = ""
    if not txt:
        txt = st.session_state.get("cleaned_text", "")
    return txt or ""

def process_resume(file_path: str) -> Dict[str, Any]:
    """Main function to process resume from file to structured data"""
    
    try:
        # Extract and clean text
        raw_text = extract_text_from_file(file_path)
        if raw_text.startswith("[ERROR]") or not raw_text.strip():
            return {
                "success": False,
                "extracted_text": raw_text,
                "structured_info": "",
                "validation_report": "Failed to extract text from file",
                "career_profile": "",
                "final_result": None
            }
        
        cleaned_text = clean_resume_text(raw_text)
        
        # Extract structured data
        structured_data = extract_structured_resume_data(cleaned_text)
        
        # Validate data
        validation_issues = validate_resume_data(structured_data)
        validation_report = "OK" if not validation_issues else f"Issues Found: {', '.join(validation_issues)}"
        
        # Generate career profile
        career_profile = create_career_profile(structured_data)
        
        return {
            "success": True,
            "extracted_text": cleaned_text,
            "structured_info": json.dumps(structured_data, indent=2),
            "validation_report": validation_report,
            "career_profile": career_profile,
            "final_result": structured_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "extracted_text": "",
            "structured_info": "",
            "validation_report": f"Processing failed: {str(e)}",
            "career_profile": "",
            "final_result": None
        }
