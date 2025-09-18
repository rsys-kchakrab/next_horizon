
# FILE: ui/course_recommendations.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from utils.skill_extraction import get_required_skills_for_role, calculate_skill_gaps
from utils.session_validators import (
    validate_skill_gaps_completed, 
    get_jd_dataframe, 
    get_training_dataframe,
    get_resume_text,
    get_candidate_skills,
    get_current_role,
    get_skill_gaps
)

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
                                    st.caption(f"🧠 {match_percent}% Vector Match")
                                
                                st.divider()
                else:
                    st.info("No courses found in the training database for the identified skill gaps.")
    else:
        st.warning("⚠️ **No training database available.**")
        st.info("Please upload training_database.csv in the left panel (sidebar) to use vector search recommendations.")
