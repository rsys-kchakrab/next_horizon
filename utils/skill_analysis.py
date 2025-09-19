# FILE: utils/skill_analysis.py - Skill Extraction and Gap Analysis
from __future__ import annotations
import re
import pandas as pd
from collections import Counter
from typing import List

def extract_skills_from_jd_text(jd_text: str) -> List[str]:
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

def extract_skills_from_aspirations(aspirations_text: str) -> List[str]:
    """Extract skills mentioned in career aspirations text"""
    if not aspirations_text or not aspirations_text.strip():
        return []
    
    text = aspirations_text.lower()
    
    # Look for patterns like "I know Python", "I have experience with", "I am familiar with"
    skill_indicators = [
        r'i know (\w+)',
        r'i have experience (?:with|in) ([\w\s]+?)(?:[,.]|$)',
        r'i am familiar with ([\w\s]+?)(?:[,.]|$)',
        r'i have worked with ([\w\s]+?)(?:[,.]|$)',
        r'i have used ([\w\s]+?)(?:[,.]|$)',
        r'i can work with ([\w\s]+?)(?:[,.]|$)',
        r'i understand ([\w\s]+?)(?:[,.]|$)',
        r'experience in ([\w\s]+?)(?:[,.]|$)',
        r'skilled in ([\w\s]+?)(?:[,.]|$)',
    ]
    
    skills = []
    for pattern in skill_indicators:
        matches = re.findall(pattern, text)
        for match in matches:
            # Clean up the matched skill
            skill = match.strip()
            if len(skill) > 2 and len(skill) < 30:
                skills.append(skill)
    
    # Also use the regular skill extraction patterns
    regular_skills = extract_skills_from_jd_text(aspirations_text)
    skills.extend(regular_skills)
    
    return list(set(skills))

def get_required_skills_for_role(role_title: str, jd_df: pd.DataFrame, min_frequency: float = 0.2) -> List[str]:
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
    # Return skills that appear in at least min_frequency of JDs for this role
    min_count = max(1, len(role_jds) * min_frequency)
    common_skills = [skill for skill, count in skill_counts.items() if count >= min_count]
    
    # If too few skills, take top 20 most frequent
    if len(common_skills) < 10:
        common_skills = [skill for skill, count in skill_counts.most_common(20)]
    
    return sorted(common_skills)

def calculate_skill_gaps(candidate_skills: List[str], required_skills: List[str]) -> tuple[List[str], List[str]]:
    """Calculate skill gaps and matches between candidate and required skills
    
    Returns:
        tuple: (gaps, matches) - lists of skills that are missing and matched
    """
    candidate_skills_lower = [skill.lower().strip() for skill in candidate_skills]
    gaps = []
    matches = []
    
    for req_skill in required_skills:
        req_skill_lower = req_skill.lower().strip()
        # Check for exact match or partial match
        is_matched = any(
            req_skill_lower in candidate_skill or candidate_skill in req_skill_lower
            for candidate_skill in candidate_skills_lower
        )
        
        if is_matched:
            matches.append(req_skill)
        else:
            gaps.append(req_skill)
    
    return gaps, matches
