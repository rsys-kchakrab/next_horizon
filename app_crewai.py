
# FILE: app_crewai_modular.py
from __future__ import annotations
import os, json
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")  # disable CrewAI’s own telemetry
os.environ.setdefault("OTEL_SDK_DISABLED", "true")         # hard kill all OpenTelemetry
# optional extra belt-and-braces:
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")

# Local tabs
from tabs.tab1_extract_parse import render as tab1_render
from tabs.tab2_review_edit import render as tab2_render
from tabs.tab4_aspirations import render as tab4_render
from tabs.tab5_skill_gaps import render as tab5_render
from tabs.tab6_guidance_courses import render as tab6_render

st.set_page_config(page_title="NextHorizon - Your Personalized Career Guide", page_icon="🧭", layout="wide")
load_dotenv(override=True)
st.title("🧭 NextHorizon - Your Personalized Career Guide")

# Session defaults
for k, v in {
    "jd_df": pd.DataFrame(),
    "structured_json": {},
    "cleaned_text": "",
    "validation_report": "",
    "chosen_role_title": "",
    "user_aspirations": "",
}.items():
    st.session_state.setdefault(k, v)

# Load sample data if files don't exist and are empty
if st.session_state.jd_df.empty:
    try: st.session_state.jd_df = pd.read_csv("jd_database.csv")
    except Exception: pass

OPENAI_API_KEY_SET = bool(os.getenv("OPENAI_API_KEY"))
with st.sidebar.expander("Environment checks", expanded=False):
    st.write("OPENAI_API_KEY:", "✅" if OPENAI_API_KEY_SET else "❌")

# ---------- Sidebar: Database Management
st.sidebar.markdown("---")
st.sidebar.header("� Database Management")
with st.sidebar.expander("Upload Databases", expanded=False):
    # Job Description Database Upload
    st.markdown("**Job Description Database**")
    jd_csv = st.file_uploader("Upload JD Database (CSV)", type=["csv"], key="jd_upload",
                             help="CSV with columns: role_title, jd_text, company, source_url, etc.")
    if jd_csv:
        try:
            df = pd.read_csv(jd_csv)
            st.session_state.jd_df = df
            st.success(f"✅ JD Database loaded: {len(df)} entries")
            st.write(f"**Columns:** {', '.join(df.columns)}")
        except Exception as e:
            st.error(f"Error loading JD database: {e}")
    
    st.markdown("---")
    
    # Course Database Upload  
    st.markdown("**Course Database**")
    course_csv = st.file_uploader("Upload Course Database (CSV)", type=["csv"], key="course_upload",
                                 help="CSV with columns: course_title, course_url, skills, description, etc.")
    if course_csv:
        try:
            df = pd.read_csv(course_csv)
            st.session_state.training_df = df
            st.success(f"✅ Course Database loaded: {len(df)} entries")
            st.write(f"**Columns:** {', '.join(df.columns)}")
        except Exception as e:
            st.error(f"Error loading course database: {e}")
    
    # Show current database status
    st.markdown("---")
    st.markdown("**Current Status:**")
    jd_count = len(st.session_state.get("jd_df", pd.DataFrame()))
    course_count = len(st.session_state.get("training_df", pd.DataFrame()))
    st.write(f"• JD Database: {jd_count} entries")
    st.write(f"• Course Database: {course_count} entries")

# ---------- Tabs
T1, T2, T3, T4 = st.tabs([
    "1) Resume Parsing",
    "2) Job Role Recommendation",
    "3) Skill Gaps Identification",
    "4) Course Recommendation",
])

with T1: 
    tab1_render()
    # Show Review & Edit section only after parsing is done
    if st.session_state.get("structured_json"):
        st.markdown("---")
        tab2_render()
with T2: tab4_render()
with T3: tab5_render()
with T4: tab6_render()

st.sidebar.caption("Tip: Keep this structure; extend tabs independently without touching the main app.")
