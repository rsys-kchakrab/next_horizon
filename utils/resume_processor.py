# FILE: utils/resume_processor.py
from __future__ import annotations
import os
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
    """Extract structured data from resume text using OpenAI"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    # Define the expected schema
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
    
    prompt = f"""
    Analyze the following resume text and extract structured professional information.
    Return ONLY valid JSON matching EXACTLY this schema (keys & nesting):
    
    {json.dumps(schema, indent=2)}
    
    Use empty strings/lists if information is unknown. No commentary outside JSON.
    
    Resume text:
    {resume_text}
    """
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        structured_data = json.loads(response.choices[0].message.content)
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
        "personal_info": {"name": "", "email": "", "phone": "", "location": ""},
        "professional_summary": "",
        "current_role": {"role": "", "company": ""},
        "work_experience": [],
        "education": [],
        "technical_skills": [],
        "soft_skills": [],
        "certifications": [],
        "projects": [],
        "career_level": data.get("career_level", ""),
        "industry_focus": data.get("industry_focus", ""),
    }
    
    # Normalize personal info
    pi = data.get("personal_info", {})
    if isinstance(pi, dict):
        normalized["personal_info"].update({
            "name": pi.get("name", ""),
            "email": pi.get("email", ""),
            "phone": pi.get("phone", ""),
            "location": pi.get("location", ""),
        })
    
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
    normalized["projects"] = [s for s in _list(data.get("projects")) if s]
    
    return normalized

def validate_resume_data(structured_data: Dict[str, Any]) -> List[str]:
    """Validate resume data and return list of issues"""
    issues = []
    
    personal_info = structured_data.get("personal_info", {})
    if not personal_info.get("name"):
        issues.append("Missing name")
    if not personal_info.get("email"):
        issues.append("Missing email")
    if not personal_info.get("phone"):
        issues.append("Missing phone number")
    
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
