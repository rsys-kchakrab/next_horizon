
# FILE: ui/tabs/skill_gaps.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from agents.clarifier_agent import ClarifierAgent
from utils.compute_metrics import parse_quality, clarify_improvement
from utils.skill_extraction import (
    extract_skills_from_jd_text, 
    extract_skills_from_aspirations, 
    get_required_skills_for_role, 
    calculate_skill_gaps
)
from utils.session_validators import (
    validate_role_selected,
    get_jd_dataframe,
    get_candidate_skills,
    get_current_role,
    get_user_aspirations
)

def render():
    # Check if we have necessary data
    if not validate_role_selected():
        return
    
    jd_df = get_jd_dataframe()
    if jd_df.empty:
        return
    
    role_title = get_current_role()
    st.write(f"**Analyzing skill gaps for role:** {role_title}")
    
    # Get required skills for the selected role from JD database
    required_skills = get_required_skills_for_role(role_title, jd_df)
    
    if not required_skills:
        st.warning(f"No job descriptions found for role '{role_title}' in the database.")
        return
    
    # Get candidate's current skills
    candidate_skills = get_candidate_skills()
    
    # Also get skills from aspirations
    user_aspirations = get_user_aspirations()
    if user_aspirations:
        aspirations_skills = extract_skills_from_aspirations(user_aspirations)
        candidate_skills.extend(aspirations_skills)
        candidate_skills = list(set(candidate_skills))  # Remove duplicates
    
    # Calculate skill gaps using the shared utility function
    gaps, matched_skills = calculate_skill_gaps(candidate_skills, required_skills)
    
    # Store skill gaps in session state for use in course recommendations
    st.session_state.skill_gaps = gaps
    st.session_state.matched_skills = matched_skills
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Skills You Have")
        if matched_skills:
            for skill in matched_skills[:10]:  # Show top 10
                st.write(f"• {skill}")
            if len(matched_skills) > 10:
                st.caption(f"... and {len(matched_skills) - 10} more")
        else:
            st.write("None detected from the job requirements")
    
    with col2:
        st.markdown("### ❌ Skill Gaps")
        if gaps:
            for skill in gaps[:15]:  # Show top 15 gaps
                st.write(f"• {skill}")
            if len(gaps) > 15:
                st.caption(f"... and {len(gaps) - 15} more")
        else:
            st.success("No major skill gaps detected!")
    
    # Clarification questions (if applicable)
    if gaps:
        st.markdown("---")
        st.markdown("### 🤔 Clarification Questions")
        
        clarifier = ClarifierAgent(min_required_skill_coverage=0.5)
        
        # Generate questions based on the skill gaps
        questions = clarifier.generate_questions(st.session_state.structured_json, gaps[:10])  # Limit to top 10 gaps
        
        if questions:
            with st.form("clarify_skill_gaps"):
                answers = {}
                for i, q in enumerate(questions):
                    if getattr(q, "options", None):
                        val = st.multiselect(q.text, q.options, default=[])
                    else:
                        val = st.text_input(q.text, value="")
                    answers[q.id] = val
                
                submitted = st.form_submit_button("✅ Apply Answers")
                
            if submitted:
                before_missing = parse_quality(st.session_state.structured_json)[1].get("missing", [])
                new_json = clarifier.incorporate_answers({k:v for k,v in answers.items() if v}, st.session_state.structured_json)
                st.session_state.structured_json = new_json
                after_missing = parse_quality(new_json)[1].get("missing", [])
                sc_improve, det = clarify_improvement(before_missing, after_missing)
                st.success("Applied clarification answers successfully!")
                if det.get("fixed"): 
                    st.caption("Fixed: " + ", ".join(det["fixed"]))
                
                # Recalculate skill gaps with updated information
                updated_candidate_skills = new_json.get("technical_skills", [])
                
                # Also include aspirations skills in updated calculation
                if user_aspirations:
                    updated_aspirations_skills = extract_skills_from_aspirations(user_aspirations)
                    updated_candidate_skills.extend(updated_aspirations_skills)
                    updated_candidate_skills = list(set(updated_candidate_skills))  # Remove duplicates
                
                # Recalculate gaps using shared utility
                updated_gaps, updated_matched_skills = calculate_skill_gaps(updated_candidate_skills, required_skills)
                
                st.info(f"🔄 Updated technical skills: {', '.join(updated_candidate_skills)}")
                
                # Update session state with new gaps
                st.session_state.skill_gaps = updated_gaps
                st.session_state.matched_skills = updated_matched_skills
                
                if len(updated_gaps) < len(gaps):
                    st.success(f"🎉 Great! Reduced skill gaps from {len(gaps)} to {len(updated_gaps)}")
                    if updated_matched_skills:
                        st.info(f"✅ Now matched: {', '.join(updated_matched_skills[-3:])}")  # Show last 3 matched
                else:
                    st.info("Skills updated but gaps remain the same. Make sure you selected the skills you actually have.")
                
                # Refresh the page to show updated analysis
                st.rerun()
        else:
            st.info("No clarification questions needed at this time.")
