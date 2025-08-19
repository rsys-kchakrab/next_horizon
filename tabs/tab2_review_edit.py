
# FILE: tabs/tab2_review_edit.py
from __future__ import annotations
import json
import streamlit as st
from resume_processing_crew import normalize_structured_json
from compute_metrics import parse_quality
from evaluation_metrics import log_event

try:
    from backfillers import backfill_from_text
except Exception:
    def backfill_from_text(cleaned_text, structured_json): return structured_json

def render():
    st.subheader("Review & Edit Extracted Fields")
    sj = st.session_state.get("structured_json", {})
    if not sj:
        st.info("Run Tab 1 first.")
        return
    with st.form("edit_form_mod_v5"):
        st.markdown("### Personal Info")
        pi = sj.get("personal_info", {})
        col1, col2 = st.columns(2)
        with col1:
            name  = st.text_input("Name",  value=pi.get("name",""))
            email = st.text_input("Email", value=pi.get("email",""))
        with col2:
            phone = st.text_input("Phone", value=pi.get("phone",""))
            location = st.text_input("Location", value=pi.get("location",""))

        st.markdown("### Professional Summary")
        summary = st.text_area("Summary", value=sj.get("professional_summary",""))

        st.markdown("### Current Role")
        cr = sj.get("current_role", {})
        current_role    = st.text_input("Role",    value=cr.get("role",""))
        current_company = st.text_input("Company", value=cr.get("company",""))
        current_end     = st.text_input("Current Role End Date", value=cr.get("end_date",""))

        st.markdown("### Work Experience")
        st.caption("Line format: Title | Company | Start | End | Responsibilities")
        wx_list = sj.get("work_experience", []) or []
        wx_lines = [f"{w.get('title','')} | {w.get('company','')} | {w.get('start_date','')} | {w.get('end_date','')} | {w.get('responsibilities','')}" for w in wx_list]
        wx_text = st.text_area("Work Experience", value="\n".join(wx_lines), height=180)

        st.markdown("### Education")
        st.caption("Line format: Degree | Institution | Graduation Date")
        ed_list = sj.get("education", []) or []
        ed_lines = [f"{e.get('degree','')} | {e.get('institution','')} | {e.get('graduation_date','')}" for e in ed_list]
        education_text = st.text_area("Education", value="\n".join(ed_lines), height=120)

        st.markdown("### Skills, Certifications, Projects")
        tech = st.text_input("Technical Skills (comma-separated)", value=", ".join(sj.get("technical_skills", []) or []))
        soft = st.text_input("Soft Skills (comma-separated)", value=", ".join(sj.get("soft_skills", []) or []))
        certs = st.text_input("Certifications (comma-separated)", value=", ".join(sj.get("certifications", []) or []))
        projs = st.text_input("Projects (comma-separated)", value=", ".join(sj.get("projects", []) or []))

        submitted = st.form_submit_button("✅ Apply Edits")

    if submitted:
        new_sj = {
            "personal_info": {"name": name, "email": email, "phone": phone, "location": location},
            "professional_summary": summary,
            "current_role": {"role": current_role, "company": current_company, "end_date": current_end},
            "work_experience": [], "education": [],
            "technical_skills": [s.strip() for s in tech.split(",") if s.strip()],
            "soft_skills": [s.strip() for s in soft.split(",") if s.strip()],
            "certifications": [s.strip() for s in certs.split(",") if s.strip()],
            "projects": [s.strip() for s in projs.split(",") if s.strip()],
            "career_level": sj.get("career_level",""),
            "industry_focus": sj.get("industry_focus",""),
        }
        for ln in (wx_text or "").splitlines():
            if ln.strip():
                p = [x.strip() for x in ln.split("|")]; p += [""] * (5 - len(p))
                new_sj["work_experience"].append({
                    "title": p[0], "company": p[1],
                    "start_date": p[2], "end_date": p[3],
                    "responsibilities": p[4]
                })
        for ln in (education_text or "").splitlines():
            if ln.strip():
                p = [x.strip() for x in ln.split("|")]; p += [""] * (3 - len(p))
                new_sj["education"].append({
                    "degree": p[0], "institution": p[1], "graduation_date": p[2]
                })
        new_sj = normalize_structured_json(new_sj)
        new_sj = backfill_from_text(st.session_state.get("cleaned_text",""), new_sj)
        st.session_state.structured_json = new_sj
        log_event("anon", "edit", "applied_edits", 1.0, {"tech_count": len(new_sj.get("technical_skills", []))})
        st.success("Edits applied.")

    if st.session_state.get("structured_json"):
        st.subheader("Updated Profile JSON")
        st.code(json.dumps(st.session_state.structured_json, indent=2), language="json")
        score, detail = parse_quality(st.session_state.structured_json)
        st.markdown(f"**Metric — Parse Quality (after edits):** `{score:.2f}`")
        if detail.get("missing"): st.caption("Missing: " + ", ".join(detail["missing"]))
