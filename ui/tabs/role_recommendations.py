
# FILE: ui/tabs/role_recommendations.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from utils.ml_inference import rank_roles_for_resume, rank_jds_within_role
from ai.openai_client import openai_rank_roles, openai_rank_jds
from utils.resume_text_builder import build_resume_text

def _get_resume_text() -> str:
    """Get resume text using the utility function"""
    return build_resume_text()

def _get_jd_df() -> pd.DataFrame:
    df = st.session_state.get("jd_df")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def render():
    # Check if resume is uploaded and parsed (structured_json exists)
    if not st.session_state.get("structured_json"):
        st.info("Complete Tabs 1: Upload a resume and complete the Review & Edit section")
        return
    
    # Add aspirations input box
    st.markdown("### Career Aspirations")
    user_aspirations = st.text_area(
        "What are your career goals and aspirations?", 
        value=st.session_state.get("user_aspirations", ""),
        height=100,
        help="Describe your career goals, desired roles, industries you're interested in, or specific skills you want to develop",
        key="aspirations_input"
    )
    
    # Save aspirations to session state
    st.session_state.user_aspirations = user_aspirations
    
    st.markdown("---")
    
    resume_text = _get_resume_text()
    
    # Include user aspirations in the resume text for matching
    if user_aspirations.strip():
        resume_text += f" Career Aspirations: {user_aspirations}"
    
    jd_df = _get_jd_df()

    if not resume_text:
        st.info("Upload a resume in Tab 1 and complete the Review & Edit section in Tab 2 for better role matching.")
    if jd_df.empty:
        st.warning("JD database is empty. Upload `jd_database.csv` in the Dev tab.")
        return  # Don't show Role Suggestions if no JD database

    st.markdown("### Role Suggestions")
    mode = st.radio("Source", ["Trained Model","gpt-4o-mini"], index=0 if st.session_state.get("role_model") else 1, horizontal=True)

    if mode == "Trained Model":
        if not st.session_state.get("role_model"):
            st.error("No trained role model loaded. Go to 'Dev: Train Models' to train/load a model.")
            return
        k = st.slider("Top-K roles", 1, 10, 5, key="asp_k_mod")
        preds = rank_roles_for_resume(st.session_state.role_model, resume_text, top_k=k)
        if not preds:
            st.warning("No roles predicted. Train with more data or adjust preprocessing.")
            return

        st.markdown("#### Suggested roles (Trained Model)")
        for i, p in enumerate(preds, 1):
            st.write(f"**{i}. {p['role_title']}** — Match: **{int(round(p['score']*100,0))}%**")

        st.markdown("---")
        st.markdown("##### Show top JDs for a selected role")
        roles = [p["role_title"] for p in preds]
        sel = st.selectbox("Pick a role", roles, key="asp_sel_role_trained")
        if st.button("Show top-5 JDs (trained TF‑IDF space)"):
            items = rank_jds_within_role(st.session_state.role_model, resume_text, jd_df, sel, top_n=5)
            if items:
                for it in items:
                    st.write(f"- {it['title']} @ {it['company']} — Match: **{it['match_percent']}%**")
                    if it.get("link"):
                        st.markdown(f"  → [Job posting link]({it['link']})")
            else:
                st.info("No JDs found for that role in your JD database.")

        # Save the selected role to session state for skill gaps analysis
        if sel and st.button("Save Selected Role for Skill Analysis", key="save_role_trained_for_skills"):
            st.session_state.chosen_role_title = sel
            # Clear previous skill gaps to ensure fresh analysis for new role
            if "skill_gaps" in st.session_state:
                del st.session_state.skill_gaps
            if "matched_skills" in st.session_state:
                del st.session_state.matched_skills
            st.success(f"Selected role '{sel}' saved for skill gap analysis.")

    else:
        k = st.slider("Top-K roles", 1, 10, 5, key="asp_k_web")
        if jd_df.empty:
            st.error("JD database required to summarize roles. Upload jd_database.csv in the Developer section (sidebar).")
            return
        
        # Validate JD database structure
        required_cols = ['role_title', 'jd_text']
        missing_cols = [col for col in required_cols if col not in jd_df.columns]
        if missing_cols:
            st.error(f"JD database is missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(jd_df.columns)}")
            st.info("💡 Please upload a proper JD database with 'role_title' and 'jd_text' columns.")
            return
            
        grp = jd_df.groupby("role_title")["jd_text"].apply(lambda s: " ".join(s.astype(str).fillna("").tolist()[:20]))
        snippets = [{"title": r, "link": "", "snippet": txt, "source": "jd_db"} for r, txt in grp.items() if r and txt]
        ranked_roles = openai_rank_roles(resume_text, snippets, top_k=k)

        st.markdown("#### Suggested roles")
        for i, p in enumerate(ranked_roles, 1):
            st.write(f"**{i}. {p['role_title']}** — Match: **{int(round(p['score']*100,0))}%**")

        st.markdown("---")
        st.markdown("##### Show top JDs for a selected role")
        roles = [p["role_title"] for p in ranked_roles]
        sel = st.selectbox("Pick a role", roles, key="asp_sel_role_openai")
        if st.button("Show top-5 JDs"):
            rows = jd_df[jd_df["role_title"]==sel]
            jd_rows = rows[["role_title","company","source_title","source_url","jd_text"]].to_dict(orient="records")
            items = openai_rank_jds(resume_text, jd_rows, top_k=5)
            if items:
                for it in items:
                    st.write(f"- {it['title']} @ {it['company']} — Match: **{it['match_percent']}%**")
                    if it.get("link"):
                        st.markdown(f"  → [Job posting link]({it['link']})")
            else:
                st.info("No JDs found for that role in your JD database.")

        # Save the selected role to session state for skill gaps analysis
        if sel and st.button("Save Selected Role for Skill Analysis", key="save_role_for_skills"):
            st.session_state.chosen_role_title = sel
            # Clear previous skill gaps to ensure fresh analysis for new role
            if "skill_gaps" in st.session_state:
                del st.session_state.skill_gaps
            if "matched_skills" in st.session_state:
                del st.session_state.matched_skills
            st.success(f"Selected role '{sel}' saved for skill gap analysis.")
