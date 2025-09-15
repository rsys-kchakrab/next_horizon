# FILE: config/session_config.py - Session State Management
from __future__ import annotations
import os
import streamlit as st
import pandas as pd

def initialize_session_state():
    """Initialize session state with default values"""
    defaults = {
        "jd_df": pd.DataFrame(),
        "structured_json": {},
        "cleaned_text": "",
        "validation_report": "",
        "chosen_role_title": "",
        "user_aspirations": "",
    }
    
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    
    # Load sample data if files don't exist and are empty
    if st.session_state.jd_df.empty:
        try:
            st.session_state.jd_df = pd.read_csv("jd_database.csv")
        except Exception:
            pass

def check_environment():
    """Check environment configuration and display status"""
    openai_api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    
    with st.sidebar.expander("Environment checks", expanded=False):
        st.write("OPENAI_API_KEY:", "✅" if openai_api_key_set else "❌")
