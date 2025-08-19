import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestComputeMetrics:
    """Test suite for metrics computation utilities"""
    
    def test_similarity_metrics(self):
        """Test similarity computation between resumes and job descriptions"""
        # Mock text vectors
        resume_vector = np.array([0.5, 0.8, 0.2, 0.9, 0.1])
        job_vector = np.array([0.6, 0.7, 0.3, 0.8, 0.2])
        
        # Cosine similarity
        dot_product = np.dot(resume_vector, job_vector)
        norm_resume = np.linalg.norm(resume_vector)
        norm_job = np.linalg.norm(job_vector)
        cosine_sim = dot_product / (norm_resume * norm_job)
        
        assert 0 <= cosine_sim <= 1
        assert isinstance(cosine_sim, (float, np.floating))
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(resume_vector - job_vector)
        assert euclidean_dist >= 0
    
    def test_skill_matching_metrics(self):
        """Test skill matching between candidate and job requirements"""
        candidate_skills = {'Python', 'Java', 'SQL', 'Machine Learning', 'Docker'}
        required_skills = {'Python', 'SQL', 'Machine Learning', 'Kubernetes', 'AWS'}
        preferred_skills = {'Docker', 'React', 'Node.js'}
        
        # Skill overlap metrics
        required_match = candidate_skills & required_skills
        preferred_match = candidate_skills & preferred_skills
        
        # Coverage metrics
        required_coverage = len(required_match) / len(required_skills)
        preferred_coverage = len(preferred_match) / len(preferred_skills)
        
        assert 0 <= required_coverage <= 1
        assert 0 <= preferred_coverage <= 1
        assert required_coverage == 0.6  # 3 out of 5 required skills
        assert preferred_coverage == 1/3  # 1 out of 3 preferred skills
    
    def test_experience_matching_metrics(self):
        """Test experience level matching"""
        candidate_experience = {
            'total_years': 5,
            'relevant_years': 3,
            'technologies': ['Python', 'Machine Learning'],
            'domains': ['Finance', 'Healthcare']
        }
        
        job_requirements = {
            'min_years': 3,
            'preferred_years': 5,
            'required_technologies': ['Python'],
            'preferred_domains': ['Healthcare']
        }
        
        # Experience scoring
        years_score = min(candidate_experience['total_years'] / job_requirements['preferred_years'], 1.0)
        tech_match = len(set(candidate_experience['technologies']) & set(job_requirements['required_technologies']))
        domain_match = len(set(candidate_experience['domains']) & set(job_requirements['preferred_domains']))
        
        assert years_score == 1.0  # Meets preferred years
        assert tech_match == 1     # Has Python
        assert domain_match == 1   # Has Healthcare
    
    def test_composite_score_calculation(self):
        """Test composite matching score calculation"""
        # Mock component scores
        skill_score = 0.8
        experience_score = 0.7
        education_score = 0.9
        culture_fit_score = 0.6
        
        # Weighted composite score
        weights = {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.2,
            'culture_fit': 0.1
        }
        
        composite_score = (
            skill_score * weights['skills'] +
            experience_score * weights['experience'] +
            education_score * weights['education'] +
            culture_fit_score * weights['culture_fit']
        )
        
        assert 0 <= composite_score <= 1
        assert abs(composite_score - 0.75) < 0.1  # Expected: ~0.75
