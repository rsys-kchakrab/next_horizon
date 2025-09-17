
# FILE: ui/tabs/course_recommendations.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Any
from utils.ml_inference import recommend_courses_by_model, recommend_courses_from_training_dataset

def openai_rank_training_courses(gaps: List[str], resume_text: str, training_df, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Use OpenAI to rank courses from training database based on skill gaps and resume.
    """
    import os
    import numpy as np
    
    if training_df.empty or not gaps:
        return {}
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    recommendations = {}
    
    for gap in gaps:
        gap_lower = str(gap).lower().strip()
        if not gap_lower:
            continue
        
        # Filter courses relevant to this skill gap
        relevant_courses = training_df[
            training_df['skill'].str.lower().str.contains(gap_lower, na=False) |
            training_df['title'].str.lower().str.contains(gap_lower, na=False) |
            training_df['description'].str.lower().str.contains(gap_lower, na=False)
        ].copy()
        
        if relevant_courses.empty:
            continue
        
        # Prepare course descriptions for ranking
        course_descriptions = []
        for _, row in relevant_courses.iterrows():
            desc = f"{row.get('title', '')} {row.get('description', '')} {row.get('skill', '')}"
            course_descriptions.append(desc)
        
        # Create user context
        user_context = f"Resume: {resume_text[:1000]} Skill Gap: {gap}"
        
        # Use OpenAI for intelligent ranking
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model = "text-embedding-3-small"
            
            # Get embeddings
            user_emb = client.embeddings.create(model=model, input=user_context).data[0].embedding
            course_embs = client.embeddings.create(model=model, input=course_descriptions).data
            
            # Calculate similarities
            scores = []
            for course_emb in course_embs:
                similarity = np.dot(user_emb, course_emb.embedding) / (
                    np.linalg.norm(user_emb) * np.linalg.norm(course_emb.embedding) + 1e-9
                )
                scores.append(float(similarity))
                
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")
        
        # Rank and select top courses
        course_scores = list(zip(relevant_courses.iterrows(), scores))
        course_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_courses = []
        for (_, course), score in course_scores[:top_k]:
            top_courses.append({
                'title': str(course.get('title', 'Unknown Course')),
                'provider': str(course.get('provider', 'Unknown')),
                'link': str(course.get('link', '')),
                'hours': course.get('hours'),
                'price': str(course.get('price', 'unknown')),
                'rating': course.get('rating'),
                'match_percent': round(float(score * 100), 1),
                'match_type': 'ai_ranked'
            })
        
        if top_courses:
            recommendations[gap] = top_courses
    
    return recommendations

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
    
    # Remove duplicates and clean up
    unique_skills = list(set([skill.strip() for skill in skills if skill.strip()]))
    return unique_skills

def get_skill_gaps_for_role(role_title: str, candidate_skills: list, jd_df: pd.DataFrame) -> list:
    """Get skill gaps for a specific role from job descriptions"""
    if jd_df.empty or not role_title:
        return []
    
    # Filter JDs for the selected role
    role_jds = jd_df[jd_df['role_title'].str.lower() == role_title.lower()]
    
    if role_jds.empty:
        return []
    
    # Extract skills from all JDs for this role
    all_required_skills = []
    for _, row in role_jds.iterrows():
        jd_text = row.get('jd_text', '')
        skills = extract_skills_from_jd_text(jd_text)
        all_required_skills.extend(skills)
    
    # Count skill frequency and get common required skills
    skill_counts = Counter(all_required_skills)
    min_count = max(1, len(role_jds) * 0.2)
    required_skills = [skill for skill, count in skill_counts.items() if count >= min_count]
    
    # If too few skills, take top 20 most frequent
    if len(required_skills) < 10:
        required_skills = [skill for skill, count in skill_counts.most_common(20)]
    
    # Calculate gaps
    candidate_skills_lower = [skill.lower().strip() for skill in candidate_skills]
    gaps = []
    
    for req_skill in required_skills:
        req_skill_lower = req_skill.lower().strip()
        # Check for exact match or partial match
        is_matched = any(
            req_skill_lower in candidate_skill or candidate_skill in req_skill_lower
            for candidate_skill in candidate_skills_lower
        )
        
        if not is_matched:
            gaps.append(req_skill)
    
    return sorted(gaps)

def render():
    # Check if we have necessary data
    if not st.session_state.get("structured_json") or not st.session_state.get("chosen_role_title") or not st.session_state.get("skill_gaps"):
        st.info("Complete Tabs 1-3: Upload resume, parse it, select a role from 'Aspirations' tab, and find the skill gaps")
        return
    
    jd_df = st.session_state.get("jd_df", pd.DataFrame())
    if jd_df.empty:
        st.warning("JD database is required. Upload jd_database.csv in the sidebar.")
        return
    
    role_title = st.session_state.chosen_role_title
    candidate_skills = st.session_state.structured_json.get("technical_skills", [])
    
    # Use stored skill gaps from skill gaps tab (if available) or calculate fresh
    if st.session_state.get("skill_gaps") is not None:
        gaps = st.session_state.skill_gaps
    else:
        # Fallback: calculate gaps if not available in session
        gaps = get_skill_gaps_for_role(role_title, candidate_skills, jd_df)
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
    training_df = st.session_state.get("training_df", pd.DataFrame())
    has_training_dataset = not training_df.empty
    
    # Check for trained model
    has_trained_model = st.session_state.get("training_model") is not None
    
    # Define the two main options
    mode_options = ["Trained Model", "gpt-4o-mini"]
    
    # Set default mode based on available resources
    if has_trained_model:
        default_mode = 0  # Trained Model
    else:
        default_mode = 1  # gpt-4o-mini
    
    mode_c = st.radio("Source", mode_options, index=default_mode)

    if mode_c == "Trained Model":
        if has_trained_model and has_training_dataset:
            st.markdown("### 🎯 Trained Model Recommendations")
            st.caption("Uses your trained model to classify and recommend courses from the training database.")
            
            with st.spinner("Using trained model to find courses from training database..."):
                # Use the trained model to recommend from training database
                recs = recommend_courses_by_model(gaps, training_df, st.session_state.training_model,
                                                text_col="description", title_col="title", 
                                                provider_col="provider", link_col="link")
            
            if recs:
                for gap, courses in recs.items():
                    if courses:
                        st.markdown(f"### {gap} ({len(courses)} courses)")
                        
                        for course in courses:
                            title = course.get('title', 'Unknown Course')
                            provider = course.get('provider', 'Unknown')
                            link = course.get('link', '')
                            match_percent = course.get('match_percent', 0)
                            
                            # Create course display
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                if link:
                                    st.markdown(f"🎯 [{title}]({link}) - **{provider}**")
                                else:
                                    st.markdown(f"🎯 {title} - **{provider}**")
                            
                            with col2:
                                st.caption(f"🎯 {match_percent}% Model Match")
                            
                            st.divider()
            else:
                st.info("No course recommendations found in the training database for these skills.")
                
        elif has_trained_model and not has_training_dataset:
            st.warning("⚠️ **Trained model is available but no training database found.**")
            st.info("Please upload training_database.csv in the left panel (sidebar) to use trained model recommendations.")
        elif not has_trained_model and has_training_dataset:
            st.warning("⚠️ **No trained model available.**")
            st.info("Please train a model in the Developer section (left panel) first.")
        else:
            st.warning("⚠️ **Both trained model and training database are required.**")
            st.info("1. Upload training_database.csv in the left panel (sidebar)")
            st.info("2. Train a model in the Developer section (left panel)")
    
    elif mode_c == "gpt-4o-mini":
        if has_training_dataset:
            if st.button("🚀 Find Courses", key="btn_openai_training"):
                with st.spinner("Using gpt-4o-mini to find the best courses from training database..."):
                    resume_text = st.session_state.get('cleaned_text', '')
                    recs = openai_rank_training_courses(gaps, resume_text, training_df, top_k=5)
                    
                if recs:
                    for gap, courses in recs.items():
                        if courses:
                            st.markdown(f"### {gap} ({len(courses)} courses)")
                            
                            for course in courses:
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
                                    st.caption(f"🧠 {match_percent}% AI Match")
                                
                                st.divider()
                else:
                    st.info("No courses found in the training database for the identified skill gaps.")
        else:
            st.warning("⚠️ **No training database available.**")
            st.info("Please upload training_database.csv in the left panel (sidebar) to use gpt-4o-mini recommendations.")
