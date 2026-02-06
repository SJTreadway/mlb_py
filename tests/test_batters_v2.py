import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['YEAR'] = '2024'

from api.batters_v2 import (
    process_batting_data,
    retro_to_mlbam,
    load_batting_data,
    transform_statcast_batter,
    calculate_cumulative_stats,
    get_historical_batting_data,
    process_batter_df,
    get_position_defaults
)


class TestRetroToMlbam:
    """Tests for retro ID to MLBAM ID conversion."""
    
    def test_valid_retro_id(self):
        """Test conversion of valid retro ID."""
        # Aaron Judge retro ID
        result = retro_to_mlbam('judga001')
        assert result == 592450
    
    def test_invalid_retro_id(self):
        """Test conversion of invalid retro ID."""
        result = retro_to_mlbam('invalid')
        assert result is None
    
    def test_none_input(self):
        """Test conversion with None input."""
        result = retro_to_mlbam(None)
        assert result is None
    
    def test_empty_string(self):
        """Test conversion with empty string."""
        result = retro_to_mlbam('')
        assert result is None


class TestGetPositionDefaults:
    """Tests for position default statistics."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        defaults = get_position_defaults()
        assert isinstance(defaults, dict)
    
    def test_has_all_positions(self):
        """Test that all positions are included."""
        defaults = get_position_defaults()
        expected_positions = ['p', 'ss', 'c', '2b', '3b', '1b', 'lf', 'rf', 'cf', 'ph', 'pr', 'dh']
        for pos in expected_positions:
            assert pos in defaults
    
    def test_position_has_required_stats(self):
        """Test that each position has required stat fields."""
        defaults = get_position_defaults()
        required_stats = ['batavg', 'obp', 'slg', 'slgmod', 'obs', 'sobat']
        for pos, stats in defaults.items():
            for stat in required_stats:
                assert stat in stats


class TestCalculateCumulativeStats:
    """Tests for cumulative batting statistics calculation."""
    
    def test_basic_calculation(self):
        """Test basic cumulative stats calculation."""
        df = pd.DataFrame({
            'date': ['1-1-2024', '1-2-2024', '1-3-2024'],
            'AB': [4, 4, 4],
            'H': [1, 2, 0],
            'BB': [1, 0, 1],
            'HBP': [0, 1, 0],
            'x2B': [0, 1, 0],
            'x3B': [0, 0, 0],
            'HR': [0, 0, 0],
            'SF': [0, 0, 0]
        })
        
        result = calculate_cumulative_stats(df)
        
        # After 3 games: 3 H / 12 AB = .250 AVG
        assert abs(result['AVG'].iloc[-1] - 0.25) < 0.01
        # OBP should be around .385 (4 H + 2 BB + 1 HBP) / (12 AB + 2 BB + 1 HBP + 0 SF)
        assert result['OBP'].iloc[-1] > 0
        assert result['SLG'].iloc[-1] > 0
    
    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()
        result = calculate_cumulative_stats(df)
        assert result.empty


class TestTransformStatcastBatter:
    """Tests for transforming Statcast batter data."""
    
    @patch('api.batters_v2.statcast_batter')
    def test_empty_input(self, mock_statcast):
        """Test with empty input."""
        result = transform_statcast_batter(pd.DataFrame())
        assert result.empty
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        # Create mock statcast data
        mock_df = pd.DataFrame({
            'game_date': ['2024-06-01', '2024-06-01', '2024-06-02'],
            'game_pk': [1, 1, 2],
            'home_team': ['NYY', 'NYY', 'NYY'],
            'away_team': ['BOS', 'BOS', 'TOR'],
            'stand': ['R', 'R', 'R'],
            'events': ['single', 'strikeout', 'double'],
            'description': ['hit_into_play', 'swinging_strike', 'hit_into_play'],
            'batting_order': [1, 1, 1],
            'position': ['RF', 'RF', 'RF']
        })
        
        result = transform_statcast_batter(mock_df)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestProcessBatterDf:
    """Tests for processing individual batter dataframes."""
    
    def test_nonexistent_file(self):
        """Test with non-existent file."""
        result = process_batter_df('nonexistent_id')
        assert result is None
    
    def test_valid_file(self, tmp_path):
        """Test with valid batter data file."""
        # Create a test CSV file
        test_data = pd.DataFrame({
            'date': ['1-1-2024', '1-2-2024', '1-3-2024'],
            'dblhead_num': ['', '', ''],
            'Pos': ['rf', 'rf', 'rf'],
            'AB': [4, 4, 4],
            'H': [1, 2, 1],
            'x2B': [0, 1, 0],
            'x3B': [0, 0, 0],
            'HR': [0, 0, 0],
            'BB': [1, 0, 1],
            'HBP': [0, 0, 0],
            'SO': [1, 0, 1],
            'SB': [0, 0, 0],
            'CS': [0, 0, 0],
            'AVG': [0.250, 0.375, 0.333],
            'OBP': [0.400, 0.444, 0.400],
            'SLG': [0.250, 0.500, 0.333]
        })
        
        os.makedirs('data/bat', exist_ok=True)
        test_data.to_csv('data/bat/batting_data_test001.csv', index=False)
        
        result = process_batter_df('test001')
        
        # Cleanup
        os.remove('data/bat/batting_data_test001.csv')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'BATAVG_30' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
