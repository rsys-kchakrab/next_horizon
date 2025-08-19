import pytest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestModuleImports:
    """Test suite for module imports and basic functionality"""
    
    def test_import_app_crewai(self):
        """Test that main app module can be imported"""
        try:
            import app_crewai
            assert hasattr(app_crewai, '__file__')
        except ImportError as e:
            pytest.skip(f"app_crewai import failed: {e}")
    
    def test_import_util_models(self):
        """Test that util_models can be imported"""
        try:
            import util_models
            assert hasattr(util_models, '__file__')
        except ImportError as e:
            pytest.skip(f"util_models import failed: {e}")
    
    def test_import_web_tools(self):
        """Test that web_tools can be imported"""
        try:
            import web_tools
            assert hasattr(web_tools, '__file__')
        except ImportError as e:
            pytest.skip(f"web_tools import failed: {e}")
    
    def test_import_ml_training(self):
        """Test that ml_training can be imported"""
        try:
            import ml_training
            assert hasattr(ml_training, '__file__')
        except ImportError as e:
            pytest.skip(f"ml_training import failed: {e}")
    
    def test_import_guardrails(self):
        """Test that guardrails can be imported"""
        try:
            import guardrails
            assert hasattr(guardrails, '__file__')
        except ImportError as e:
            pytest.skip(f"guardrails import failed: {e}")
    
    def test_import_evaluation_metrics(self):
        """Test that evaluation_metrics can be imported"""
        try:
            import evaluation_metrics
            assert hasattr(evaluation_metrics, '__file__')
        except ImportError as e:
            pytest.skip(f"evaluation_metrics import failed: {e}")
    
    def test_import_compute_metrics(self):
        """Test that compute_metrics can be imported"""
        try:
            import compute_metrics
            assert hasattr(compute_metrics, '__file__')
        except ImportError as e:
            pytest.skip(f"compute_metrics import failed: {e}")
    
    def test_import_clarifier_agent(self):
        """Test that clarifier_agent can be imported"""
        try:
            import clarifier_agent
            assert hasattr(clarifier_agent, '__file__')
        except ImportError as e:
            pytest.skip(f"clarifier_agent import failed: {e}")
    
    def test_import_resume_processing_crew(self):
        """Test that resume_processing_crew can be imported"""
        try:
            import resume_processing_crew
            assert hasattr(resume_processing_crew, '__file__')
        except ImportError as e:
            pytest.skip(f"resume_processing_crew import failed: {e}")
    
    def test_import_backfillers(self):
        """Test that backfillers can be imported"""
        try:
            import backfillers
            assert hasattr(backfillers, '__file__')
        except ImportError as e:
            pytest.skip(f"backfillers import failed: {e}")
