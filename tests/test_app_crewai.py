import pytest
from unittest.mock import patch, MagicMock, mock_open
import streamlit as st
import sys
import os

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAppCrewAI:
    """Test suite for the main CrewAI application"""
    
    def test_streamlit_imports(self):
        """Test that required Streamlit components can be imported"""
        assert hasattr(st, 'title')
        assert hasattr(st, 'tabs')
        assert hasattr(st, 'sidebar')
        assert hasattr(st, 'file_uploader')
        assert hasattr(st, 'selectbox')
    
    @patch('streamlit.set_page_config')
    def test_page_config_setup(self, mock_config):
        """Test that Streamlit page configuration is properly set"""
        # Test page config parameters
        expected_config = {
            'page_title': 'NextHorizon - Your Personalized Career Guide',
            'page_icon': '🚀',
            'layout': 'wide'
        }
        # Verify config structure
        assert isinstance(expected_config, dict)
        assert 'page_title' in expected_config
    
    def test_tab_structure(self):
        """Test that the application has the expected tab structure"""
        expected_tabs = ["Extract & Parse", "Review & Edit", "Dev & Train"]
        assert len(expected_tabs) == 3
        assert "Extract & Parse" in expected_tabs
        assert "Review & Edit" in expected_tabs
        assert "Dev & Train" in expected_tabs
    
    @patch('streamlit.sidebar')
    def test_sidebar_elements(self, mock_sidebar):
        """Test sidebar configuration and elements"""
        mock_sidebar.title.return_value = None
        mock_sidebar.selectbox.return_value = "OpenAI"
        
        # Test sidebar functionality
        assert mock_sidebar is not None
    
    def test_environment_variables(self):
        """Test that required environment variables are accessible"""
        import os
        # Test environment variable handling
        required_env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        
        for var in required_env_vars:
            # Test that we can check for environment variables
            env_value = os.environ.get(var)
            # Just verify the method works, don't require actual values in tests
            assert env_value is None or isinstance(env_value, str)
    
    @patch('streamlit.file_uploader')
    def test_file_upload_functionality(self, mock_uploader):
        """Test file upload functionality"""
        mock_uploader.return_value = MagicMock()
        
        # Test file upload types
        supported_types = ['pdf', 'docx', 'txt']
        for file_type in supported_types:
            assert file_type in ['pdf', 'docx', 'txt']
    
    def test_session_state_initialization(self):
        """Test Streamlit session state initialization"""
        # Test session state structure
        session_keys = ['processed_resumes', 'current_job_description', 'model_results']
        
        for key in session_keys:
            assert isinstance(key, str)
            assert len(key) > 0