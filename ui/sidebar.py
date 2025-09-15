# FILE: ui/sidebar.py - Sidebar UI Components
from __future__ import annotations
import streamlit as st
import pandas as pd

def render_sidebar():
    """Render the sidebar with database management"""
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Database Management")
    
    with st.sidebar.expander("Upload Databases", expanded=False):
        # Job Description Database Upload
        st.markdown("**Job Description Database**")
        jd_csv = st.file_uploader(
            "Upload JD Database (CSV)", 
            type=["csv"], 
            key="jd_upload",
            help="CSV with columns: role_title, jd_text, company, source_url, etc."
        )
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
        course_csv = st.file_uploader(
            "Upload Course Database (CSV)", 
            type=["csv"], 
            key="course_upload",
            help="CSV with columns: course_title, course_url, skills, description, etc."
        )
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
    
    st.sidebar.caption("Tip: Keep this structure; extend tabs independently without touching the main app.")
