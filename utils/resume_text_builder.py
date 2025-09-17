# FILE: utils/resume_text_builder.py - Resume Text Construction Utilities
from __future__ import annotations
from typing import Any, Dict, List
import streamlit as st

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
