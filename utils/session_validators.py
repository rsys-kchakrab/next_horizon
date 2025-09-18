# FILE: utils/session_validators.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Optional

def validate_resume_parsed() -> bool:
    """Check if resume is uploaded and parsed"""
    if not st.session_state.get("structured_json"):
        st.info("Complete Tab 1: Upload a resume and complete the Review & Edit section")
        return False
    return True

def validate_role_selected() -> bool:
    """Check if resume is parsed and role is selected"""
    if not st.session_state.get("structured_json") or not st.session_state.get("chosen_role_title"):
        st.info("Complete Tabs 1-2: Upload resume, parse it, and select a role from 'Aspirations' tab.")
        return False
    return True

def validate_skill_gaps_completed() -> bool:
    """Check if resume, role, and skill gaps are completed"""
    if (not st.session_state.get("structured_json") or 
        not st.session_state.get("chosen_role_title") or 
        not st.session_state.get("skill_gaps")):
        st.info("Complete Tabs 1-3: Upload resume, parse it, select a role from 'Aspirations' tab, and find the skill gaps")
        return False
    return True

def get_jd_dataframe() -> pd.DataFrame:
    """Get JD dataframe and show warning if empty"""
    jd_df = st.session_state.get("jd_df", pd.DataFrame())
    if jd_df.empty:
        st.warning("JD database is required. Upload jd_database.csv in the sidebar.")
    return jd_df

def get_training_dataframe() -> pd.DataFrame:
    """Get training dataframe from session state"""
    return st.session_state.get("training_df", pd.DataFrame())

def get_resume_text() -> str:
    """Get optimized resume text from session state for course recommendations"""
    # First try to get structured data and create optimized text
    structured_json = st.session_state.get("structured_json", {})
    if structured_json:
        from utils.resume_processor import create_optimized_resume_text
        return create_optimized_resume_text(structured_json)
    
    # Fallback to cleaned text if structured data not available
    return st.session_state.get('cleaned_text', '')

def get_candidate_skills() -> list:
    """Get candidate's technical skills from structured resume data"""
    structured_json = st.session_state.get("structured_json", {})
    return structured_json.get("technical_skills", [])

def get_current_role() -> str:
    """Get the currently selected role"""
    return st.session_state.get("chosen_role_title", "")

def get_skill_gaps() -> list:
    """Get the identified skill gaps"""
    return st.session_state.get("skill_gaps", [])

def get_user_aspirations() -> str:
    """Get user's career aspirations text"""
    return st.session_state.get("user_aspirations", "")
