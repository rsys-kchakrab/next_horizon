
# FILE: ui/tabs/skill_gaps.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import re
from collections import Counter
from agents.clarifier_agent import ClarifierAgent
from utils.compute_metrics import parse_quality, clarify_improvement

def extract_skills_from_jd_text(jd_text: str) -> list:
    """Extract skills from job description text using basic NLP patterns"""
    if not jd_text or pd.isna(jd_text):
        return []
    
    # Convert to lowercase for matching
    text = str(jd_text).lower()
    
    # Common skill patterns and keywords
    skill_patterns = [
        # Programming languages
        r'\b(?:python|java|javascript|c\+\+|c#|go|rust|swift|kotlin|php|ruby|scala|r\b)\b',
        # Web technologies
        r'\b(?:html|css|react|angular|vue|node\.?js|express|django|flask|spring|laravel)\b',
        # Databases
        r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|sqlite)\b',
        # Cloud platforms
        r'\b(?:aws|azure|gcp|google cloud|docker|kubernetes|terraform|ansible)\b',
        # Data science
        r'\b(?:pandas|numpy|scikit-learn|tensorflow|pytorch|jupyter|tableau|power bi)\b',
        # DevOps/Tools
        r'\b(?:git|jenkins|gitlab|github|jira|confluence|linux|windows|mac)\b',
        # Frameworks
        r'\b(?:\.net|spring boot|rails|next\.js|nuxt\.js|fastapi)\b',
        # Other technical skills
        r'\b(?:api|rest|graphql|microservices|agile|scrum|kanban|ci/cd|machine learning|ai|blockchain)\b'
    ]
    
    skills = []
    for pattern in skill_patterns:
        matches = re.findall(pattern, text)
        skills.extend(matches)
    
    # Also look for explicit skill mentions
    skill_keywords = [
        'experience with', 'knowledge of', 'proficient in', 'familiar with',
        'skills in', 'expertise in', 'background in', 'experience in'
    ]
    
    for keyword in skill_keywords:
        if keyword in text:
            # Extract text after the keyword (up to next sentence or comma)
            pattern = f'{keyword}([^.;]+)'
            matches = re.findall(pattern, text)
            for match in matches:
                # Split by common delimiters and clean up
                potential_skills = re.split(r'[,;/&\n]', match.strip())
                for skill in potential_skills[:3]:  # Limit to first 3 to avoid noise
                    skill = skill.strip().lower()
                    if len(skill) > 2 and len(skill) < 30:  # Reasonable skill name length
                        skills.append(skill)
    
    # Remove duplicates and clean up
    unique_skills = list(set([skill.strip() for skill in skills if skill.strip()]))
    return unique_skills

def get_required_skills_for_role(role_title: str, jd_df: pd.DataFrame) -> list:
    """Extract required skills for a specific role from job descriptions"""
    if jd_df.empty or not role_title:
        return []
    
    # Filter JDs for the selected role
    role_jds = jd_df[jd_df['role_title'].str.lower() == role_title.lower()]
    
    if role_jds.empty:
        return []
    
    # Extract skills from all JDs for this role
    all_skills = []
    for _, row in role_jds.iterrows():
        jd_text = row.get('jd_text', '')
        skills = extract_skills_from_jd_text(jd_text)
        all_skills.extend(skills)
    
    # Count skill frequency and return most common ones
    skill_counts = Counter(all_skills)
    # Return skills that appear in at least 20% of JDs for this role or top 20 skills
    min_count = max(1, len(role_jds) * 0.2)
    common_skills = [skill for skill, count in skill_counts.items() if count >= min_count]
    
    # If too few skills, take top 20 most frequent
    if len(common_skills) < 10:
        common_skills = [skill for skill, count in skill_counts.most_common(20)]
    
    return sorted(common_skills)

def render():
    # Check if we have necessary data
    if not st.session_state.get("structured_json") or not st.session_state.get("chosen_role_title"):
        st.info("Complete Tabs 1-3: Upload resume, parse it, and select a role from 'Aspirations' tab.")
        return
    
    jd_df = st.session_state.get("jd_df", pd.DataFrame())
    if jd_df.empty:
        st.warning("JD database is required. Upload jd_database.csv in the sidebar.")
        return
    
    role_title = st.session_state.chosen_role_title
    st.write(f"**Analyzing skill gaps for role:** {role_title}")
    
    # Get required skills for the selected role from JD database
    required_skills = get_required_skills_for_role(role_title, jd_df)
    
    if not required_skills:
        st.warning(f"No job descriptions found for role '{role_title}' in the database.")
        return
    
    # Get candidate's current skills
    candidate_skills = st.session_state.structured_json.get("technical_skills", [])
    candidate_skills_lower = [skill.lower().strip() for skill in candidate_skills]
    
    # Calculate skill gaps
    gaps = []
    matched_skills = []
    
    for req_skill in required_skills:
        req_skill_lower = req_skill.lower().strip()
        # Check for exact match or partial match
        is_matched = any(
            req_skill_lower in candidate_skill or candidate_skill in req_skill_lower
            for candidate_skill in candidate_skills_lower
        )
        
        if is_matched:
            matched_skills.append(req_skill)
        else:
            gaps.append(req_skill)
    
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
                st.success(f"Applied. **Clarify Improvement:** {sc_improve:.2f}")
                if det.get("fixed"): 
                    st.caption("Fixed: " + ", ".join(det["fixed"]))
                
                # Recalculate skill gaps with updated information
                updated_candidate_skills = new_json.get("technical_skills", [])
                updated_gaps = []
                updated_matched_skills = []
                updated_candidate_skills_lower = [skill.lower().strip() for skill in updated_candidate_skills]
                
                st.info(f"🔄 Updated technical skills: {', '.join(updated_candidate_skills)}")
                
                for req_skill in required_skills:
                    req_skill_lower = req_skill.lower().strip()
                    # Improved matching: exact match, partial match in both directions
                    is_matched = any(
                        req_skill_lower == candidate_skill or 
                        req_skill_lower in candidate_skill or 
                        candidate_skill in req_skill_lower
                        for candidate_skill in updated_candidate_skills_lower
                    )
                    
                    if is_matched:
                        updated_matched_skills.append(req_skill)
                    else:
                        updated_gaps.append(req_skill)
                
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
