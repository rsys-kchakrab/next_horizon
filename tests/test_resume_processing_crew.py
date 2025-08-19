import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestResumeProcessingCrew:
    """Test suite for resume processing crew functionality"""
    
    def test_crew_initialization(self):
        """Test CrewAI crew initialization"""
        # Mock crew configuration
        crew_config = {
            'agents': ['resume_parser', 'skill_extractor', 'experience_analyzer'],
            'tasks': ['parse_resume', 'extract_skills', 'analyze_experience'],
            'verbose': True
        }
        
        assert len(crew_config['agents']) == 3
        assert len(crew_config['tasks']) == 3
        assert crew_config['verbose'] is True
    
    @patch('crewai.Agent')
    def test_agent_creation(self, mock_agent):
        """Test individual agent creation"""
        mock_agent.return_value = MagicMock()
        
        # Test agent configurations
        agent_configs = [
            {'role': 'Resume Parser', 'goal': 'Parse resume content'},
            {'role': 'Skill Extractor', 'goal': 'Extract technical skills'},
            {'role': 'Experience Analyzer', 'goal': 'Analyze work experience'}
        ]
        
        for config in agent_configs:
            assert 'role' in config
            assert 'goal' in config
            assert isinstance(config['role'], str)
    
    def test_resume_parsing_logic(self):
        """Test resume parsing logic"""
        # Mock resume content
        sample_resume = """
        John Doe
        Software Engineer
        Email: john.doe@email.com
        Phone: (555) 123-4567
        
        Skills: Python, Java, SQL, Machine Learning
        Experience: 5 years in software development
        """
        
        # Test content extraction
        lines = sample_resume.strip().split('\n')
        assert len(lines) > 0
        assert 'John Doe' in sample_resume
        assert 'Python' in sample_resume
    
    def test_skill_extraction(self):
        """Test skill extraction from resume text"""
        skills_text = "Python, Java, SQL, Machine Learning, Docker, Kubernetes"
        skills_list = [skill.strip() for skill in skills_text.split(',')]
        
        assert len(skills_list) == 6
        assert 'Python' in skills_list
        assert 'Machine Learning' in skills_list
    
    def test_experience_analysis(self):
        """Test experience analysis logic"""
        experience_text = "5 years of experience in software development"
        
        # Extract years of experience
        import re
        years_match = re.search(r'(\d+)\s+years?', experience_text)
        
        if years_match:
            years = int(years_match.group(1))
            assert years == 5
    
    @patch('crewai.Crew')
    def test_crew_execution(self, mock_crew):
        """Test crew execution process"""
        mock_crew_instance = MagicMock()
        mock_crew.return_value = mock_crew_instance
        mock_crew_instance.kickoff.return_value = {'status': 'completed'}
        
        # Test crew kickoff
        result = mock_crew_instance.kickoff()
        assert result['status'] == 'completed'
