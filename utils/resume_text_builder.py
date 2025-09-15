# FILE: utils/resume_text_builder.py - Resume Text Construction Utilities
from __future__ import annotations
import streamlit as st
from ml_training import prepare_resume_text

def build_resume_text() -> str:
    """Build comprehensive resume text from structured JSON data"""
    # Get the updated structured JSON from review & edit tab
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
