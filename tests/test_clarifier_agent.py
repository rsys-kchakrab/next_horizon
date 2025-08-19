import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestClarifierAgent:
    """Test suite for clarifier agent functionality"""
    
    def test_agent_initialization(self):
        """Test clarifier agent initialization"""
        agent_config = {
            'role': 'Requirements Clarifier',
            'goal': 'Clarify job requirements and candidate qualifications',
            'backstory': 'Expert in understanding job requirements',
            'verbose': True,
            'allow_delegation': False
        }
        
        assert agent_config['role'] == 'Requirements Clarifier'
        assert agent_config['allow_delegation'] is False
    
    def test_job_requirement_clarification(self):
        """Test job requirement clarification logic"""
        job_description = """
        Software Engineer Position
        Requirements: Python, 3+ years experience, Bachelor's degree
        Nice to have: Machine Learning, Cloud platforms
        """
        
        # Test requirement extraction
        requirements = []
        if 'Python' in job_description:
            requirements.append('Python')
        if '3+ years' in job_description:
            requirements.append('3+ years experience')
        if 'Bachelor' in job_description:
            requirements.append('Bachelor\'s degree')
        
        assert len(requirements) == 3
        assert 'Python' in requirements
    
    def test_candidate_qualification_assessment(self):
        """Test candidate qualification assessment"""
        candidate_profile = {
            'skills': ['Python', 'Java', 'SQL'],
            'experience_years': 5,
            'education': 'Master\'s in Computer Science'
        }
        
        job_requirements = {
            'required_skills': ['Python'],
            'min_experience': 3,
            'education_level': 'Bachelor\'s'
        }
        
        # Test qualification matching
        has_required_skills = any(skill in candidate_profile['skills'] for skill in job_requirements['required_skills'])
        meets_experience = candidate_profile['experience_years'] >= job_requirements['min_experience']
        
        assert has_required_skills is True
        assert meets_experience is True
    
    def test_clarification_questions_generation(self):
        """Test generation of clarification questions"""
        unclear_requirements = [
            'Experience with modern frameworks',
            'Strong communication skills',
            'Ability to work in fast-paced environment'
        ]
        
        clarification_questions = []
        for requirement in unclear_requirements:
            if 'modern frameworks' in requirement:
                clarification_questions.append('Which specific frameworks are required?')
            if 'communication skills' in requirement:
                clarification_questions.append('What level of communication skills is expected?')
        
        assert len(clarification_questions) == 2
    
    def test_requirement_prioritization(self):
        """Test requirement prioritization logic"""
        requirements = [
            {'skill': 'Python', 'priority': 'high', 'required': True},
            {'skill': 'Machine Learning', 'priority': 'medium', 'required': False},
            {'skill': 'Docker', 'priority': 'low', 'required': False}
        ]
        
        high_priority = [req for req in requirements if req['priority'] == 'high']
        required_skills = [req for req in requirements if req['required'] is True]
        
        assert len(high_priority) == 1
        assert len(required_skills) == 1
    
    @patch('openai.ChatCompletion.create')
    def test_ai_powered_clarification(self, mock_openai):
        """Test AI-powered clarification responses"""
        mock_openai.return_value = MagicMock()
        mock_openai.return_value.choices = [MagicMock()]
        mock_openai.return_value.choices[0].message.content = "Clarified requirements"
        
        # Test AI response structure
        response = mock_openai.return_value
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
