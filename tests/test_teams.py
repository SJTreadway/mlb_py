import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.teams import (
    get_team_cols,
    create_team_df,
    generate_team_window_features,
    strip_suffix
)


class TestGetTeamCols:
    """Tests for team column extraction."""
    
    def test_returns_four_lists(self):
        """Test that function returns four lists."""
        df = pd.DataFrame({
            'AB_h': [1, 2],
            'H_h': [1, 1],
            'AB_v': [3, 4],
            'H_v': [2, 2]
        })
        
        home_cols, home_stripped, visit_cols, visit_stripped = get_team_cols(df)
        
        assert isinstance(home_cols, list)
        assert isinstance(home_stripped, list)
        assert isinstance(visit_cols, list)
        assert isinstance(visit_stripped, list)
    
    def test_correct_column_separation(self):
        """Test that columns are correctly separated by suffix."""
        df = pd.DataFrame({
            'AB_h': [1, 2],
            'H_h': [1, 1],
            'AB_v': [3, 4],
            'H_v': [2, 2]
        })
        
        home_cols, home_stripped, visit_cols, visit_stripped = get_team_cols(df)
        
        assert 'AB_h' in home_cols
        assert 'H_h' in home_cols
        assert 'AB_v' in visit_cols
        assert 'H_v' in visit_cols
        
        # Check stripped versions
        assert 'AB' in home_stripped
        assert 'H' in home_stripped


class TestStripSuffix:
    """Tests for suffix stripping utility."""
    
    def test_strips_suffix(self):
        """Test that suffix is stripped correctly."""
        result = strip_suffix('AB_h', '_h')
        assert result == 'AB'
    
    def test_no_suffix_match(self):
        """Test that non-matching suffix returns original."""
        result = strip_suffix('AB_v', '_h')
        assert result == 'AB_v'
    
    def test_empty_string(self):
        """Test with empty string."""
        result = strip_suffix('', '_h')
        assert result == ''


class TestCreateTeamDf:
    """Tests for team dataframe creation."""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        df = pd.DataFrame({
            'team_h': ['NYY', 'NYY', 'BOS'],
            'team_v': ['BOS', 'TOR', 'NYY'],
            'date_dblhead': [20240601, 20240602, 20240603],
            'AB_h': [34, 32, 35],
            'H_h': [9, 8, 10],
            'x2B_h': [2, 1, 3],
            'x3B_h': [0, 0, 0],
            'HR_h': [1, 2, 1],
            'BB_h': [3, 2, 4],
            'SB_h': [1, 0, 1],
            'CS_h': [0, 0, 0],
            'AB_v': [32, 30, 34],
            'H_v': [8, 7, 9],
            'x2B_v': [1, 2, 2],
            'x3B_v': [0, 0, 0],
            'HR_v': [1, 1, 1],
            'BB_v': [2, 3, 3],
            'SB_v': [0, 1, 0],
            'CS_v': [0, 0, 0]
        })
        
        result = create_team_df(df, 'NYY')
        
        assert isinstance(result, pd.DataFrame)
        assert 'AB' in result.columns
        assert 'rollsum_AB_162' in result.columns
        assert 'rollsum_BATAVG_162' in result.columns
    
    def test_rolling_window_stats(self):
        """Test that rolling window stats are calculated."""
        df = pd.DataFrame({
            'team_h': ['NYY'] * 10,
            'team_v': ['BOS'] * 10,
            'date_dblhead': list(range(20240601, 20240611)),
            'AB_h': [34] * 10,
            'H_h': [9] * 10,
            'x2B_h': [2] * 10,
            'x3B_h': [0] * 10,
            'HR_h': [1] * 10,
            'BB_h': [3] * 10,
            'SB_h': [1] * 10,
            'CS_h': [0] * 10,
            'AB_v': [32] * 10,
            'H_v': [8] * 10,
            'x2B_v': [1] * 10,
            'x3B_v': [0] * 10,
            'HR_v': [1] * 10,
            'BB_v': [2] * 10,
            'SB_v': [0] * 10,
            'CS_v': [0] * 10
        })
        
        result = create_team_df(df, 'NYY')
        
        # Check all window sizes are calculated
        for window in [162, 90, 30]:
            assert f'rollsum_AB_{window}' in result.columns
            assert f'rollsum_BATAVG_{window}' in result.columns
            assert f'rollsum_OBP_{window}' in result.columns
            assert f'rollsum_SLG_{window}' in result.columns


class TestGenerateTeamWindowFeatures:
    """Tests for team window feature generation."""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        df = pd.DataFrame({
            'team_h': ['NYY', 'NYY'],
            'team_v': ['BOS', 'TOR'],
            'date_dblhead': [20240601, 20240602],
            'AB_h': [34, 32],
            'H_h': [9, 8],
            'x2B_h': [2, 1],
            'x3B_h': [0, 0],
            'HR_h': [1, 2],
            'BB_h': [3, 2],
            'SB_h': [1, 0],
            'CS_h': [0, 0],
            'AB_v': [32, 30],
            'H_v': [8, 7],
            'x2B_v': [1, 2],
            'x3B_v': [0, 0],
            'HR_v': [1, 1],
            'BB_v': [2, 3],
            'SB_v': [0, 1],
            'CS_v': [0, 0]
        })
        
        result = generate_team_window_features(df)
        
        assert isinstance(result, pd.DataFrame)
        # Check that team stats were added
        assert 'BATAVG_162_h' in result.columns
        assert 'OBP_162_h' in result.columns
        assert 'SLG_162_h' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
