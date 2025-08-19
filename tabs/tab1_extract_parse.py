
# FILE: tabs/tab1_extract_parse.py
from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
from resume_processing_crew import (
    ResumeProcessingCrew, parse_structured_info, normalize_structured_json
)
from compute_metrics import parse_quality

try:
    from backfillers import backfill_from_text
except Exception:
    def backfill_from_text(cleaned_text, structured_json): return structured_json

def render():
    st.subheader("Upload Resume")
    up = st.file_uploader("Resume (PDF / DOCX / TXT)", type=["pdf","docx","txt"], key="resume_v5_mod")
    c1, c2 = st.columns(2)
    with c1:
        run = st.button("🚀 Run Extraction & Parsing", disabled=up is None)
    with c2:
        if st.button("🧹 Clear session (profile)"):
            for k in ["structured_json","cleaned_text","validation_report","chosen_role_title"]:
                st.session_state[k] = {} if isinstance(st.session_state.get(k), dict) else ""
            st.success("Profile data cleared.")
    if run:
        tmp = Path(".streamlit_tmp"); tmp.mkdir(exist_ok=True)
        p = tmp / up.name
        with open(p, "wb") as f: f.write(up.read())
        crew = ResumeProcessingCrew()
        with st.spinner("Processing resume…"):
            out = crew.process_resume(str(p))
        st.session_state.cleaned_text = out.get("extracted_text","")
        st.session_state.validation_report = out.get("validation_report","")
        raw_struct = out.get("structured_info","")
        try:
            struct = json.loads(raw_struct)
        except Exception:
            struct = parse_structured_info(raw_struct)
        struct = normalize_structured_json(struct)
        struct = backfill_from_text(st.session_state.cleaned_text, struct)
        st.session_state.structured_json = struct

    if st.session_state.get("cleaned_text"):
        st.subheader("🧹 Cleaned Text")
        st.text_area("cleaned_text", value=st.session_state.cleaned_text, height=160)

    if st.session_state.get("validation_report"):
        st.subheader("🧪 Validation Report (from pipeline)")
        st.text(st.session_state.validation_report)

    if st.session_state.get("structured_json"):
        st.subheader("📦 Structured Information (JSON)")
        st.code(json.dumps(st.session_state.structured_json, indent=2, ensure_ascii=False), language="json")
        score, detail = parse_quality(st.session_state.structured_json)
        st.markdown(f"**Metric — Parse Quality:** `{score:.2f}`")
        if detail.get("missing"):
            st.caption("Missing: " + ", ".join(detail["missing"]))
