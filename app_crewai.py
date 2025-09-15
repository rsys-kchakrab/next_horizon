
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
from tabs.tab3_dev_train import render as tab3_render
from tabs.tab4_aspirations import render as tab4_render
from tabs.tab5_skill_gaps import render as tab5_render
from tabs.tab6_guidance_courses import render as tab6_render

st.set_page_config(page_title="NextHorizon - Your Personalized Career Guide", page_icon="🧭", layout="wide")
load_dotenv(override=True)
st.title("🧭 NextHorizon - Your Personalized Career Guide")

# Session defaults
for k, v in {
    "jd_df": pd.DataFrame(),
    "train_df": pd.DataFrame(),
    "training_df": pd.DataFrame(),  # For web-scraped training content
    "structured_json": {},
    "cleaned_text": "",
    "validation_report": "",
    "chosen_role_title": "",
    "role_model": None,
    "role_model_metrics": {},
    "training_model": None,
    "training_model_metrics": {},
}.items():
    st.session_state.setdefault(k, v)

# Load sample data if files don't exist and are empty
if st.session_state.jd_df.empty:
    try: st.session_state.jd_df = pd.read_csv("jd_database.csv")
    except Exception: pass

if st.session_state.train_df.empty:
    try: st.session_state.train_df = pd.read_csv("training_content.csv")
    except Exception: pass

# Load web-scraped training dataset if available
if st.session_state.training_df.empty:
    try: 
        training_df = pd.read_csv("training_database.csv")
        st.session_state.training_df = training_df
    except Exception: 
        # Fallback to original training content
        try:
            training_df = pd.read_csv("training_content.csv")
            st.session_state.training_df = training_df
        except Exception: 
            pass
    except Exception: pass

OPENAI_API_KEY_SET = bool(os.getenv("OPENAI_API_KEY"))
SERPAPI_SET        = bool(os.getenv("SERPAPI_API_KEY"))
with st.sidebar.expander("Environment checks", expanded=False):
    st.write("OPENAI_API_KEY:", "✅" if OPENAI_API_KEY_SET else "❌")
    st.write("SERPAPI_API_KEY:", "✅" if SERPAPI_SET else "❌")

# ---------- Sidebar: Developer section
st.sidebar.markdown("---")
st.sidebar.header("🔧 Developer: Train & Manage Models")
with st.sidebar.expander("Train Models", expanded=False):
    tab3_render(models_dir=Path("./models"))

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
