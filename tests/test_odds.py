import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['ODDS_API_KEY'] = 'test_api_key'

from api.odds import (
    get_over_odds,
    get_under_odds,
    get_total_line,
    get_money_line_price,
    line_to_bet,
    calculate_edge,
    line_to_prob
)


class TestCalculateEdge:
    """Tests for edge calculation."""
    
    def test_positive_edge(self):
        """Test calculation with positive edge."""
        # 60% probability, +150 odds
        result = calculate_edge(0.60, 150)
        # Result is a string like "10.0%"
        assert isinstance(result, str)
        assert '%' in result
        # Extract numeric value and check it's positive
        edge_val = float(result.replace('%', ''))
        assert edge_val > 0
    
    def test_negative_edge(self):
        """Test calculation with negative edge."""
        # 40% probability, -200 odds
        result = calculate_edge(0.40, -200)
        # Result is a string like "-10.0%"
        assert isinstance(result, str)
        assert '%' in result
        edge_val = float(result.replace('%', ''))
        assert edge_val < 0
    
    def test_zero_edge(self):
        """Test calculation with fair odds."""
        # 50% probability, +100 odds
        result = calculate_edge(0.50, 100)
        assert isinstance(result, str)
        edge_val = float(result.replace('%', ''))
        assert abs(edge_val) < 0.01  # Should be approximately 0
    
    def test_negative_odds(self):
        """Test with negative odds (favorite)."""
        result = calculate_edge(0.70, -150)
        assert isinstance(result, str)
        assert '%' in result
    
    def test_invalid_probability(self):
        """Test with invalid probability values - function handles gracefully."""
        # Function doesn't raise, it calculates even with invalid probability
        result = calculate_edge(1.5, 100)
        assert isinstance(result, str)
    
    def test_none_values(self):
        """Test with None values."""
        result = calculate_edge(None, 100)
        assert result is None
        
        result = calculate_edge(0.5, None)
        assert result is None


class TestLineToProb:
    """Tests for line to probability conversion."""
    
    def test_positive_line(self):
        """Test with positive line."""
        result = line_to_prob(150)
        assert result > 0
        assert result < 0.5
    
    def test_negative_line(self):
        """Test with negative line."""
        result = line_to_prob(-150)
        assert result > 0.5
        assert result < 1
    
    def test_even_line(self):
        """Test with even money line."""
        result = line_to_prob(100)
        assert abs(result - 0.5) < 0.01
    
    def test_none_value(self):
        """Test with None value."""
        result = line_to_prob(None)
        assert result == -1


class TestLineToBet:
    """Tests for line to bet calculation."""
    
    def test_returns_integer(self):
        """Test that function returns an integer."""
        result = line_to_bet(0.55)
        assert isinstance(result, int)
    
    def test_positive_probability(self):
        """Test with positive probability."""
        result = line_to_bet(0.60)
        assert result < 0  # Favorite should have negative odds
    
    def test_negative_probability(self):
        """Test with low probability."""
        result = line_to_bet(0.40)
        assert result > 0  # Underdog should have positive odds
    
    def test_fifty_percent(self):
        """Test with 50% probability."""
        result = line_to_bet(0.50)
        assert result == 100  # Pick 'em


class TestGetOverOdds:
    """Tests for over odds retrieval."""
    
    @patch('api.odds.get_odds_results')
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {
            'New York Yankees': {'over_under_price_o': -110}
        }
        
        result = get_over_odds('New York Yankees')
        assert result == -110
    
    @patch('api.odds.get_odds_results')
    def test_returns_none_on_error(self, mock_get_results):
        """Test that function returns None when team not found."""
        mock_get_results.return_value = {}
        
        result = get_over_odds('New York Yankees')
        assert result is None


class TestGetUnderOdds:
    """Tests for under odds retrieval."""
    
    @patch('api.odds.get_odds_results')
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {
            'New York Yankees': {'over_under_price_u': -110}
        }
        
        result = get_under_odds('New York Yankees')
        assert result == -110


class TestGetTotalLine:
    """Tests for total line retrieval."""
    
    @patch('api.odds.get_odds_results')
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {
            'New York Yankees': {'over_under_line': 8.5}
        }
        
        result = get_total_line('New York Yankees')
        assert result == 8.5


class TestGetMoneyLinePrice:
    """Tests for money line price retrieval."""
    
    @patch('api.odds.get_odds_results')
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {
            'New York Yankees': {'moneyline_price': -150}
        }
        
        result = get_money_line_price('New York Yankees')
        assert result == -150


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
