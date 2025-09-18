# FILE: app.py - Main Application Entry Point
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Disable telemetry
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")

# Import modules
from config.session_config import initialize_session_state, check_environment
from ui.sidebar import render_sidebar
from ui.resume_parsing import render as tab1_render
from ui.role_recommendations import render as tab2_render  
from ui.skill_gaps import render as tab3_render
from ui.course_recommendations import render as tab4_render

def main():
    """Main application entry point"""
    # Configure Streamlit
    st.set_page_config(
        page_title="NextHorizon - Your Personalized Career Guide", 
        page_icon="🧭", 
        layout="wide"
    )
    
    # Load environment
    load_dotenv(override=True)
    
    # Initialize application
    st.title("🧭 NextHorizon - Your Personalized Career Guide")
    initialize_session_state()
    
    # Environment checks
    check_environment()
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Resume Parsing",
        "🎯 Job Role Recommendation", 
        "🔍 Skill Gaps Identification",
        "📚 Course Recommendation"
    ])
    
    with tab1:
        tab1_render()
    with tab2:
        tab2_render()
    with tab3:
        tab3_render()
    with tab4:
        tab4_render()

if __name__ == "__main__":
    main()
