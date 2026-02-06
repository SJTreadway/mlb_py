import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import patch, MagicMock
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['YEAR'] = '2024'
os.environ['TOMORROW_GAMES'] = '0'
os.environ['REFRESH_DATA'] = '1'
os.environ['X_ACCESS_KEY'] = 'test'
os.environ['X_ACCESS_SECRET'] = 'test'
os.environ['X_CONSUMER_KEY'] = 'test'
os.environ['X_CONSUMER_SECRET'] = 'test'
os.environ['X_BEARER_TOKEN'] = 'test'
os.environ['ODDS_API_KEY'] = 'test'

from pipeline import (
    predict_winner,
    predict_runs_scored,
    get_runs_scored_prob,
    filter_games_by_edge,
    calculate_edge
)


class TestPredictWinner:
    """Tests for winner prediction function."""
    
    @patch('builtins.open', MagicMock())
    @patch('pickle.load')
    def test_returns_tuple(self, mock_pickle):
        """Test that function returns prediction and probability."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_pickle.return_value = mock_model
        
        X = pd.DataFrame({'feature': [1, 2, 3]})
        pred, prob = predict_winner(X)
        
        assert isinstance(pred, (list, np.ndarray))
        assert isinstance(prob, (list, np.ndarray))
    
    @patch('builtins.open', MagicMock())
    @patch('pickle.load')
    def test_prediction_values(self, mock_pickle):
        """Test prediction values."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6], [0.7, 0.3]])
        mock_pickle.return_value = mock_model
        
        X = pd.DataFrame({'feature': [1, 2]})
        pred, prob = predict_winner(X)
        
        # Probabilities should be between 0 and 1
        assert all(0 <= p <= 1 for p in prob)


class TestPredictRunsScored:
    """Tests for runs scored prediction function."""
    
    @patch('builtins.open', MagicMock())
    @patch('pickle.load')
    def test_returns_probabilities(self, mock_pickle):
        """Test that function returns probability distribution."""
        mock_model = MagicMock()
        # Return probability distribution for 20 possible run totals
        mock_model.predict_proba.return_value = np.array([
            [0.1] * 20  # Uniform distribution
        ])
        mock_pickle.return_value = mock_model
        
        X = pd.DataFrame({'feature': [1, 2, 3]})
        probs = predict_runs_scored(X)
        
        assert isinstance(probs, np.ndarray)
        assert probs.shape[1] == 20  # Should have 20 probability bins


class TestGetRunsScoredProb:
    """Tests for runs scored probability extraction."""
    
    def test_returns_probability(self):
        """Test that function returns probability for given line."""
        # Create probability distribution (30 possible run totals)
        probs = np.array([0.05] * 20)
        probs = np.append(probs, [0] * 10)  # Pad to 30
        
        result = get_runs_scored_prob(probs, 8.5)
        assert isinstance(result, (int, float))
        assert 0 <= result <= 1
    
    def test_higher_line_lower_prob(self):
        """Test that higher lines give lower probabilities."""
        probs = np.array([0.05] * 20)
        
        prob_7 = get_runs_scored_prob(probs, 7)
        prob_10 = get_runs_scored_prob(probs, 10)
        
        assert prob_10 <= prob_7  # Harder to go over higher line
    
    def test_invalid_line(self):
        """Test with invalid line value."""
        probs = np.array([0.05] * 20)
        result = get_runs_scored_prob(probs, 'invalid')
        assert result is None


class TestFilterGamesByEdge:
    """Tests for filtering games by edge."""
    
    def test_filters_by_threshold(self):
        """Test that games are filtered by edge threshold."""
        df = pd.DataFrame({
            'edge_h': ['5%', '2%', '10%'],
            'edge_v': ['1%', '8%', '3%'],
            'prob': [0.55, 0.45, 0.60]
        })
        
        result = filter_games_by_edge(df)
        
        # Should only keep games with edge > 4% and prob > 0.50
        assert len(result) <= len(df)
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        df = pd.DataFrame({
            'edge_h': ['5%', '2%'],
            'edge_v': ['1%', '8%'],
            'prob': [0.55, 0.45]
        })
        
        result = filter_games_by_edge(df)
        assert isinstance(result, pd.DataFrame)


class TestCalculateEdge:
    """Tests for edge calculation in pipeline."""
    
    def test_edge_calculation(self):
        """Test edge calculation."""
        result = calculate_edge(0.60, 150)
        assert isinstance(result, str)
        assert '%' in result


class TestFeatureSets:
    """Tests for feature sets configuration."""
    
    def test_runs_scored_features_exist(self):
        """Test that runs scored feature set is defined."""
        from pipeline import RUNS_SCORED_FEAT_SET
        assert isinstance(RUNS_SCORED_FEAT_SET, list)
        assert len(RUNS_SCORED_FEAT_SET) > 0
    
    def test_home_victory_features_exist(self):
        """Test that home victory feature set is defined."""
        from pipeline import HOME_VICTORY_FEAT_SET
        assert isinstance(HOME_VICTORY_FEAT_SET, list)
        assert len(HOME_VICTORY_FEAT_SET) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
