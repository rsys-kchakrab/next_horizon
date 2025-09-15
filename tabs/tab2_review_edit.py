
# FILE: tabs/tab2_review_edit.py
from __future__ import annotations
import streamlit as st
from resume_processing_crew import normalize_structured_json
from compute_metrics import parse_quality
from evaluation_metrics import log_event

try:
    from backfillers import backfill_from_text
except Exception:
    def backfill_from_text(cleaned_text, structured_json): return structured_json

def calculate_total_experience_years(work_experience):
    """Calculate total years of experience from work experience list"""
    import re
    from datetime import datetime
    
    total_months = 0
    current_year = datetime.now().year
    
    for work in work_experience:
        start_date = work.get('start_date', '').strip()
        end_date = work.get('end_date', '').strip()
        
        if not start_date:
            continue
            
        # Handle "Present" or "Current" end dates
        if end_date.lower() in ['present', 'current', ''] or not end_date:
            end_date = str(current_year)
        
        # Extract years from dates (handle various formats)
        start_year = None
        end_year = None
        
        # Try to extract 4-digit years
        start_match = re.search(r'\b(19|20)\d{2}\b', start_date)
        end_match = re.search(r'\b(19|20)\d{2}\b', end_date)
        
        if start_match:
            start_year = int(start_match.group())
        if end_match:
            end_year = int(end_match.group())
        elif end_date.lower() in ['present', 'current']:
            end_year = current_year
        
        # Calculate duration if both years are available
        if start_year and end_year and end_year >= start_year:
            total_months += (end_year - start_year) * 12
    
    return round(total_months / 12, 1) if total_months > 0 else 0

def render():
    st.subheader("Review & Edit Extracted Fields")
    sj = st.session_state.get("structured_json", {})
    if not sj:
        st.info("Run Tab 1 first.")
        return
    
    # Initialize work experience and education lists if not present
    if "work_experience" not in sj or not isinstance(sj["work_experience"], list):
        sj["work_experience"] = []
    if "education" not in sj or not isinstance(sj["education"], list):
        sj["education"] = []
    
    # Buttons for adding/removing entries (outside form)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("➕ Add Work Experience"):
            sj["work_experience"].append({"title": "", "company": "", "start_date": "", "end_date": "", "responsibilities": ""})
            st.session_state.structured_json = sj
            st.rerun()
    with col2:
        if st.button("➕ Add Education"):
            sj["education"].append({"degree": "", "specialization": "", "institution": "", "graduation_date": ""})
            st.session_state.structured_json = sj
            st.rerun()
    with col3:
        if len(sj["work_experience"]) > 0 and st.button("🗑️ Remove Last Work Exp"):
            sj["work_experience"].pop()
            st.session_state.structured_json = sj
            st.rerun()
    with col4:
        if len(sj["education"]) > 0 and st.button("🗑️ Remove Last Education"):
            sj["education"].pop()
            st.session_state.structured_json = sj
            st.rerun()
    
    with st.form("edit_form_mod_v6"):
        # 1. Professional Summary
        st.markdown("### 1. Professional Summary")
        summary = st.text_area("Professional Summary", 
                              value=sj.get("professional_summary", ""), 
                              height=100,
                              help="Brief overview of your professional background")

        # 2. Current Role
        st.markdown("### 2. Current Role")
        cr = sj.get("current_role", {})
        col1, col2 = st.columns(2)
        with col1:
            current_designation = st.text_input("Current Designation", value=cr.get("role", ""))
        with col2:
            current_company = st.text_input("Current Company", value=cr.get("company", ""))

        # 3. Work Experience - Separate boxes for each experience
        st.markdown("### 3. Work Experience")
        wx_list = sj.get("work_experience", []) or []
        
        work_experiences = []
        for i, work in enumerate(wx_list):
            st.markdown(f"**Experience {i+1}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                designation = st.text_input(f"Designation", value=work.get("title", ""), key=f"work_title_{i}")
            with col2:
                company = st.text_input(f"Company", value=work.get("company", ""), key=f"work_company_{i}")
            with col3:
                start_date = st.text_input(f"Start Date", value=work.get("start_date", ""), key=f"work_start_{i}")
            with col4:
                end_date = st.text_input(f"End Date", value=work.get("end_date", ""), key=f"work_end_{i}")
            
            responsibilities = st.text_area(f"Responsibilities", 
                                          value=work.get("responsibilities", ""), 
                                          height=80, 
                                          key=f"work_resp_{i}")
            
            work_experiences.append({
                "title": designation,
                "company": company,
                "start_date": start_date,
                "end_date": end_date,
                "responsibilities": responsibilities
            })

        # 4. Education - Separate boxes for each degree
        st.markdown("### 4. Education")
        ed_list = sj.get("education", []) or []
        
        educations = []
        for i, edu in enumerate(ed_list):
            st.markdown(f"**Education {i+1}**")
            col1, col2 = st.columns(2)
            with col1:
                degree = st.text_input(f"Degree", value=edu.get("degree", ""), key=f"edu_degree_{i}")
                specialization = st.text_input(f"Stream/Specialization", value=edu.get("specialization", ""), key=f"edu_spec_{i}")
            with col2:
                institution = st.text_input(f"Institution", value=edu.get("institution", ""), key=f"edu_inst_{i}")
                completion_date = st.text_input(f"Completion Date", value=edu.get("graduation_date", ""), key=f"edu_date_{i}")
            
            educations.append({
                "degree": degree,
                "specialization": specialization,
                "institution": institution,
                "graduation_date": completion_date
            })

        # 5. Technical Skills
        st.markdown("### 5. Technical Skills")
        tech_skills = st.text_input("Technical Skills (comma-separated)", 
                                   value=", ".join(sj.get("technical_skills", []) or []),
                                   help="e.g., Python, Java, SQL, Machine Learning")

        # 6. Soft Skills
        st.markdown("### 6. Soft Skills")
        soft_skills = st.text_input("Soft Skills (comma-separated)", 
                                   value=", ".join(sj.get("soft_skills", []) or []),
                                   help="e.g., Leadership, Communication, Problem Solving")

        # 7. Certifications
        st.markdown("### 7. Certifications")
        certifications = st.text_input("Certifications (comma-separated)", 
                                      value=", ".join(sj.get("certifications", []) or []),
                                      help="e.g., AWS Certified, PMP, Scrum Master")

        # 8. Total Years of Experience (editable)
        st.markdown("### 8. Total Years of Experience")
        calculated_exp = calculate_total_experience_years(work_experiences)
        current_exp = sj.get("total_years_experience", calculated_exp)
        
        col1, col2 = st.columns(2)
        with col1:
            total_experience = st.number_input("Total Years of Experience", 
                                             value=float(current_exp), 
                                             min_value=0.0, 
                                             step=0.1,
                                             help="You can edit this manually or it will be calculated from work experience")
        with col2:
            st.info(f"Calculated from work history: {calculated_exp} years")

        submitted = st.form_submit_button("✅ Apply All Edits")

    if submitted:
        # Update session state with new values
        sj["professional_summary"] = summary
        sj["current_role"] = {"role": current_designation, "company": current_company}
        sj["work_experience"] = work_experiences
        sj["education"] = educations
        sj["technical_skills"] = [s.strip() for s in tech_skills.split(",") if s.strip()]
        sj["soft_skills"] = [s.strip() for s in soft_skills.split(",") if s.strip()]
        sj["certifications"] = [s.strip() for s in certifications.split(",") if s.strip()]
        sj["total_years_experience"] = total_experience
        
        sj = normalize_structured_json(sj)
        sj = backfill_from_text(st.session_state.get("cleaned_text",""), sj)
        st.session_state.structured_json = sj
        log_event("anon", "edit", "applied_edits", 1.0, {"tech_count": len(sj.get("technical_skills", []))})
        st.success("Edits applied.")
