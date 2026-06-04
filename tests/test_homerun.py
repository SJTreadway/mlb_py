import datetime
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.homerun import process_homerun_data

GAME_DATE = 20260601


def _make_batter_series():
    """Return a pd.Series matching the Snowflake batter_data_dict format."""
    return pd.Series({
        "game_date": "2026-06-01",
        "stand": "R",
        "rollsum_ab_162": 50,
        **{f"{stem}_{w}": 0.05 for w in [7, 14, 30, 75, 162, 350]
           for stem in ["barrel", "hardhit", "swspot"]},
        **{f"{stem}_{w}": 90.0 for w in [7, 14, 30, 75, 162, 350]
           for stem in ["ev"]},
        **{f"{stem}_{w}": 0.45 for w in [7, 14, 30, 75, 162, 350]
           for stem in ["slg", "est_slg"]},
        **{f"{stem}_{w}": 0.35 for w in [7, 14, 30, 75, 162, 350]
           for stem in ["obp", "obs", "est_woba"]},
        **{f"hr_per_pa_{w}": 0.04 for w in [7, 14, 30, 75, 162, 350]},
        **{f"hr_per_pa_vs_r_{w}": 0.05 for w in [7, 14, 30, 75, 162, 350]},
        **{f"hr_per_pa_vs_l_{w}": 0.03 for w in [7, 14, 30, 75, 162, 350]},
        "age": 28,
    })


def _make_pitcher_series():
    """Return a pd.Series matching the Snowflake pitcher_data_dict format."""
    return pd.Series({
        "hr_per_bf_10": 0.030,
        "hr_per_bf_35": 0.025,
        "hr_per_bf_75": 0.020,
        "fb_perc_10": 0.40,
        "fb_perc_35": 0.38,
        "fb_perc_75": 0.35,
        "gs": 1,
    })


class TestProcessHomerunData:
    """Tests for home run data processing."""

    def test_returns_dataframe(self):
        """Test function returns a DataFrame."""
        df = pd.DataFrame(
            {
                "date_dblhead": [GAME_DATE],
                "starting_pitcher_id_h": [100],
                "starting_pitcher_id_v": [200],
                "team_h": ["NYY"],
                "team_v": ["BOS"],
                "sp_throws_h": ["R"],
                "sp_throws_v": ["R"],
                "batter1_id_h": [101],
                "batter1_id_v": [201],
            }
        )
        batter_data = {"101": _make_batter_series()}
        pitcher_data = {}

        result = process_homerun_data(df, batter_data, pitcher_data)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_returns_expected_columns(self):
        """Test result has expected columns."""
        df = pd.DataFrame(
            {
                "date_dblhead": [GAME_DATE],
                "starting_pitcher_id_h": [100],
                "starting_pitcher_id_v": [200],
                "team_h": ["NYY"],
                "team_v": ["BOS"],
                "sp_throws_h": ["R"],
                "sp_throws_v": ["R"],
                "batter1_id_h": [101],
                "batter1_id_v": [201],
            }
        )
        batter_data = {"101": _make_batter_series()}
        pitcher_data = {}

        result = process_homerun_data(df, batter_data, pitcher_data)
        expected_cols = [
            "date_dblhead", "b_id", "slot", "team", "opponent",
            "stand", "opp_throws", "park_hr_factor",
            "temp", "humidity", "wind_spd", "wind_out",
            "barrel_30", "ev_30", "hr_per_pa_30",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_empty_input_df(self):
        """Test with empty input DataFrame."""
        result = process_homerun_data(pd.DataFrame(), {}, {})
        assert result.empty

    def test_no_matching_batter_data(self):
        """Test with no matching batter data returns empty."""
        df = pd.DataFrame(
            {
                "date_dblhead": [GAME_DATE],
                "starting_pitcher_id_h": [100],
                "starting_pitcher_id_v": [200],
                "team_h": ["NYY"],
                "team_v": ["BOS"],
                "sp_throws_h": ["R"],
                "sp_throws_v": ["R"],
                "batter1_id_h": [101],
            }
        )
        result = process_homerun_data(df, {}, {})
        assert result.empty

    def test_includes_pitcher_features(self):
        """Test pitcher features are included when data available."""
        df = pd.DataFrame(
            {
                "date_dblhead": [GAME_DATE],
                "starting_pitcher_id_h": [100],
                "starting_pitcher_id_v": [200],
                "team_h": ["NYY"],
                "team_v": ["BOS"],
                "sp_throws_h": ["R"],
                "sp_throws_v": ["L"],
                "batter1_id_h": [101],
            }
        )
        batter_data = {"101": _make_batter_series()}
        pitcher_data = {"200": _make_pitcher_series()}

        result = process_homerun_data(df, batter_data, pitcher_data)
        assert not result.empty
        # Batter faces opposing pitcher (v for h team = sp_id_v = 200)
        assert "opp_hr_per_bf_10" in result.columns
        assert result["opp_hr_per_bf_10"].iloc[0] == 0.03


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
