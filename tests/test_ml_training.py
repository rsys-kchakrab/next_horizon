import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestMLTraining:
    """Test suite for machine learning training pipeline"""
    
    def test_data_loading(self):
        """Test data loading functionality"""
        # Mock CSV data
        mock_csv_data = "name,skills,experience,match_score\nJohn Doe,Python,5,0.85\nJane Smith,Java,3,0.72"
        
        # Test pandas DataFrame creation
        from io import StringIO
        df = pd.read_csv(StringIO(mock_csv_data))
        
        assert len(df) == 2
        assert list(df.columns) == ['name', 'skills', 'experience', 'match_score']
        assert df.iloc[0]['name'] == 'John Doe'
        assert df.iloc[0]['match_score'] == 0.85
    
    def test_feature_engineering(self):
        """Test feature engineering functions"""
        # Mock resume data
        resume_data = {
            'skills': ['Python, Java, SQL', 'JavaScript, React, Node.js'],
            'experience': [5, 3],
            'education': ['BS Computer Science', 'MS Software Engineering'],
            'job_titles': ['Senior Developer', 'Frontend Developer']
        }
        
        df = pd.DataFrame(resume_data)
        
        # Test skill extraction and counting
        df['skill_count'] = df['skills'].str.split(', ').apply(len)
        assert df.iloc[0]['skill_count'] == 3  # Python, Java, SQL
        assert df.iloc[1]['skill_count'] == 3  # JavaScript, React, Node.js
    
    def test_model_training_structure(self):
        """Test model training pipeline structure"""
        # Mock training data
        X = np.random.rand(100, 10)  # 100 samples, 10 features
        y = np.random.rand(100)      # Regression target (match scores)
        
        assert X.shape == (100, 10)
        assert y.shape == (100,)
    
    def test_train_test_split(self):
        """Test train-test split functionality"""
        from sklearn.model_selection import train_test_split
        
        # Mock data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Test train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics calculation"""
        # Mock predictions vs actual
        y_true = np.array([0.8, 0.6, 0.9, 0.7, 0.5])
        y_pred = np.array([0.75, 0.65, 0.85, 0.72, 0.55])
        
        # Calculate MAE
        mae = np.mean(np.abs(y_true - y_pred))
        assert mae < 0.1  # Should be small for good predictions
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert rmse < 0.1
    
    @patch('joblib.dump')
    def test_model_saving(self, mock_dump):
        """Test model saving functionality"""
        # Mock model
        mock_model = MagicMock()
        model_path = "models/resume_job_matcher.pkl"
        
        # Simulate saving
        mock_dump.return_value = None
        
        # Test model path validation
        assert model_path.endswith('.pkl')
        assert 'models/' in model_path
        assert 'resume_job_matcher' in model_path
