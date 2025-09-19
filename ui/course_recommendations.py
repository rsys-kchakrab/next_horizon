
# FILE: ui/course_recommendations.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from utils.skill_analysis import get_required_skills_for_role, calculate_skill_gaps
from utils.session_helpers import (
    validate_skill_gaps_completed, 
    get_jd_dataframe, 
    get_training_dataframe,
    get_resume_text,
    get_candidate_skills,
    get_current_role,
    get_skill_gaps
)
from ai.openai_client import openai_rank_courses

def render():
    # Check if we have necessary data
    if not validate_skill_gaps_completed():
        return
    
    jd_df = get_jd_dataframe()
    if jd_df.empty:
        return
    
    role_title = get_current_role()
    candidate_skills = get_candidate_skills()
    
    # Use stored skill gaps from skill gaps tab (if available) or calculate fresh
    gaps = get_skill_gaps()
    if not gaps:
        # Fallback: calculate gaps if not available in session
        required_skills = get_required_skills_for_role(role_title, jd_df)
        gaps, _ = calculate_skill_gaps(candidate_skills, required_skills)
        st.info("⚠️ Calculating fresh skill gaps. Visit 'Skill Gap Analysis' tab first for better results.")
    
    if not gaps:
        st.success("🎉 No major skill gaps detected for the selected role!")
        if st.session_state.get("skill_gaps") is not None:
            st.info("Great! Your clarification answers helped fill in the missing skills. You seem well-prepared for this role!")
        else:
            st.info("You seem to have most of the skills required for this role. Consider looking for more advanced or specialized skills for career growth.")
        return
    
    st.write(f"**Finding courses for role:** {role_title}")
    
    # Show information about skill gaps source
    if st.session_state.get("skill_gaps") is not None:
        st.write(f"**Skill gaps identified:** {', '.join(gaps[:10])} ✨")  # Show first 10 gaps
        st.caption("✨ These gaps include any clarification answers you provided in the Skill Gap Analysis tab")
    else:
        st.write(f"**Skill gaps identified:** {', '.join(gaps[:10])}")  # Show first 10 gaps
        st.caption("💡 Tip: Complete the Skill Gap Analysis tab first for more accurate recommendations")
    
    # Check for training dataset
    training_df = get_training_dataframe()
    has_training_dataset = not training_df.empty
    
    # Use Vector Search for course recommendations
    if has_training_dataset:
        if st.button("🚀 Find Courses", key="btn_openai_training"):
            with st.spinner("Using vector search to find the best courses from training database..."):
                resume_text = get_resume_text()
                
                # Filter courses relevant to skill gaps
                relevant_courses = training_df.copy()
                for gap in gaps:
                    gap_lower = str(gap).lower().strip()
                    if gap_lower:
                        mask = (
                            training_df['skill'].str.lower().str.contains(gap_lower, na=False) |
                            training_df['title'].str.lower().str.contains(gap_lower, na=False) |
                            training_df['description'].str.lower().str.contains(gap_lower, na=False)
                        )
                        if mask.any():
                            gap_courses = training_df[mask].copy()
                            # Convert DataFrame to list of dictionaries for openai_rank_courses
                            gap_courses_list = gap_courses.to_dict('records')
                            # Use consolidated function for each skill gap
                            recs = openai_rank_courses([gap], resume_text, gap_courses_list, top_k=5)
                            
                            if recs:
                                st.markdown(f"### {gap} ({len(recs)} courses)")
                                
                                for course in recs:
                                    title = course.get('title', 'Unknown Course')
                                    provider = course.get('provider', 'Unknown')
                                    link = course.get('link', '')
                                    hours = course.get('hours')
                                    price = course.get('price', 'unknown')
                                    rating = course.get('rating')
                                    match_percent = course.get('match_percent', 0)
                                    
                                    # Create course display
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        if link:
                                            st.markdown(f"🤖 [{title}]({link}) - **{provider}**")
                                        else:
                                            st.markdown(f"🤖 {title} - **{provider}**")
                                    
                                    with col2:
                                        if hours:
                                            st.caption(f"⏱️ {hours}h")
                                        if price and price != 'unknown':
                                            st.caption(f"💰 {price}")
                                    
                                    with col3:
                                        if rating:
                                            st.caption(f"⭐ {rating}")
                                        st.caption(f"🧠 {match_percent}% Vector Match")
                                    
                                    st.divider()
    else:
        st.warning("⚠️ **No training database available.**")
        st.info("Please upload training_database.csv in the left panel (sidebar) to use vector search recommendations.")
