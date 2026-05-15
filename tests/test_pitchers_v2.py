import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["YEAR"] = "2024"

from api.pitchers_v2 import (
    process_pitching_data,
    load_pitching_data,
    transform_statcast_pitcher,
    calculate_cumulative_pitching_stats,
    get_historical_pitching_data,
    load_and_process_pitch_df,
    get_bullpen_data,
)


class TestCalculateCumulativePitchingStats:
    """Tests for cumulative pitching statistics calculation."""

    def test_basic_calculation(self):
        """Test basic cumulative ERA calculation."""
        df = pd.DataFrame(
            {
                "date": ["1-1-2024", "1-2-2024", "1-3-2024"],
                "ER": [2, 3, 1],
                "IP": [6.0, 7.0, 6.0],
            }
        )

        result = calculate_cumulative_pitching_stats(df)

        # After 3 games: 6 ER / 19 IP * 9 = ~2.84 ERA
        assert result["ERA"].iloc[-1] > 0
        assert "cum_ER" not in result.columns  # Should be dropped
        assert "cum_IP" not in result.columns  # Should be dropped

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()
        result = calculate_cumulative_pitching_stats(df)
        assert result.empty


class TestTransformStatcastPitcher:
    """Tests for transforming Statcast pitcher data."""

    def test_empty_input(self):
        """Test with empty input."""
        result = transform_statcast_pitcher(pd.DataFrame())
        assert result.empty

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        # Create mock statcast data
        mock_df = pd.DataFrame(
            {
                "game_date": ["2024-06-01", "2024-06-01", "2024-06-02"],
                "game_pk": [1, 1, 2],
                "home_team": ["NYY", "NYY", "NYY"],
                "away_team": ["BOS", "BOS", "TOR"],
                "pitcher": [543037, 543037, 543037],
                "inning": [1, 2, 1],
                "inning_topbot": ["Top", "Top", "Bot"],
                "events": ["single", "strikeout", "double"],
                "description": ["hit_into_play", "swinging_strike", "hit_into_play"],
                "p_throws": ["R", "R", "R"],
                "bat_score": [0, 0, 1],
                "post_bat_score": [1, 0, 2],
                "bb_type": ["fly_ball", np.nan, "line_drive"],
                "launch_speed": [95.0, np.nan, 85.0],
            }
        )

        result = transform_statcast_pitcher(mock_df)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "IP" in result.columns
        assert "H" in result.columns


class TestLoadAndProcessPitchDf:
    """Tests for loading and processing pitcher dataframes."""

    def test_nonexistent_file(self):
        """Test with non-existent file."""
        result = load_and_process_pitch_df("nonexistent_id")
        assert result.empty

    def test_valid_file(self):
        """Test with valid pitcher data file."""
        # Create a test CSV file
        test_data = pd.DataFrame(
            {
                "date": ["1-1-2024", "1-2-2024", "1-3-2024"],
                "dblhead_num": ["", "", ""],
                "IP": [6.0, 7.0, 6.0],
                "H": [5, 6, 4],
                "BFP": [25, 28, 24],
                "HR": [1, 0, 1],
                "R": [2, 3, 1],
                "ER": [2, 3, 1],
                "BB": [2, 1, 2],
                "IB": [0, 0, 0],
                "SO": [6, 7, 5],
                "SH": [0, 0, 0],
                "SF": [0, 0, 0],
                "WP": [0, 0, 0],
                "HBP": [0, 0, 0],
                "BK": [0, 0, 0],
                "x2B": [1, 2, 0],
                "x3B": [0, 0, 0],
                "GDP": [0, 1, 0],
                "ERA": [3.00, 3.21, 2.84],
                "fly_balls": [3, 4, 2],
                "batted_balls_allowed": [10, 12, 9],
            }
        )

        os.makedirs("data/pitch", exist_ok=True)
        test_data.to_csv("data/pitch/pitching_data_testp001.csv", index=False)

        result = load_and_process_pitch_df("testp001", "data/pitch/")

        # Cleanup
        os.remove("data/pitch/pitching_data_testp001.csv")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "WHIP_10" in result.columns
        assert "ERA_10" in result.columns


class TestGetBullpenData:
    """Tests for bullpen data calculation."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        # Create minimal test dataframe
        test_df = pd.DataFrame(
            {
                "team_h": ["NYY"],
                "team_v": ["BOS"],
                "date_dblhead": [20240101],
                "Strt_IP_real_h": [6.0],
                "Strt_IP_real_v": [5.0],
                "Strt_BFP_h": [25],
                "Strt_BFP_v": [22],
                "Strt_R_h": [2],
                "Strt_R_v": [3],
                "Strt_H_h": [5],
                "Strt_H_v": [6],
                "Strt_HR_h": [1],
                "Strt_HR_v": [0],
                "Strt_x2B_h": [1],
                "Strt_x2B_v": [2],
                "Strt_x3B_h": [0],
                "Strt_x3B_v": [0],
                "Strt_BB_h": [2],
                "Strt_BB_v": [1],
                "Strt_HBP_h": [0],
                "Strt_HBP_v": [0],
                "Strt_SO_h": [6],
                "Strt_SO_v": [7],
                "AB_v": [32],
                "BB_v": [3],
                "HBP_v": [0],
                "R_v": [4],
                "H_v": [8],
                "HR_v": [1],
                "x2B_v": [2],
                "x3B_v": [0],
                "SO_v": [7],
                "AB_h": [34],
                "BB_h": [2],
                "HBP_h": [1],
                "R_h": [5],
                "H_h": [9],
                "HR_h": [2],
                "x2B_h": [1],
                "x3B_h": [0],
                "SO_h": [8],
            }
        )

        result = get_bullpen_data(test_df)

        assert isinstance(result, pd.DataFrame)
        assert "Bpen_IP_h" in result.columns
        assert "Bpen_IP_v" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
