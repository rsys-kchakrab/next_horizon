import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestEvaluationMetrics:
    """Test suite for ML model evaluation metrics"""
    
    def test_regression_metrics(self):
        """Test regression evaluation metrics"""
        # Mock continuous predictions (match scores)
        y_true = np.array([0.8, 0.6, 0.9, 0.7, 0.5, 0.85, 0.75])
        y_pred = np.array([0.75, 0.65, 0.85, 0.72, 0.55, 0.80, 0.70])
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        assert mae >= 0
        assert mae < 0.2  # Should be reasonable for match scores
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert rmse >= 0
        assert rmse >= mae  # RMSE is always >= MAE
        
        # R-squared calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        assert r2 <= 1  # R-squared should be <= 1
    
    def test_classification_metrics(self):
        """Test classification evaluation metrics"""
        # Mock binary classification for high/low match
        y_true = np.array([1, 0, 1, 1, 0, 1, 0])  # 1 = good match, 0 = poor match
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1])
        
        # Calculate confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
        tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        assert 0 <= precision <= 1
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        assert 0 <= recall <= 1
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        assert 0 <= f1 <= 1
        
        # Accuracy
        accuracy = (tp + tn) / len(y_true)
        assert 0 <= accuracy <= 1
    
    def test_ranking_metrics(self):
        """Test ranking evaluation metrics for job matching"""
        # Mock ranked list of candidates with scores
        true_relevance = [1, 1, 0, 1, 0]  # Relevant candidates
        predicted_scores = [0.9, 0.8, 0.7, 0.6, 0.5]  # Predicted match scores
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevance = [true_relevance[i] for i in sorted_indices]
        
        # Calculate Precision@K for K=3
        k = 3
        precision_at_k = sum(sorted_relevance[:k]) / k
        assert 0 <= precision_at_k <= 1
        
        # Calculate NDCG@K (simplified)
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevance[:k])])
        ideal_relevance = sorted(true_relevance, reverse=True)
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k])])
        ndcg = dcg / idcg if idcg > 0 else 0
        assert 0 <= ndcg <= 1
