import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.homerun import process_homerun_data


class TestProcessHomerunData:
    """Tests for home run data processing."""

    def test_returns_dataframe(self):
        """Test function returns a DataFrame."""
        df = pd.DataFrame(
            {
                "date_dblhead": [20240601],
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
        batter_data = {
            "101": pd.DataFrame(
                {
                    "BARREL_30": [0.05],
                    "EV_30": [90.0],
                    "HARDHIT_30": [0.4],
                    "SWSPOT_30": [0.35],
                    "SLG_30": [0.45],
                    "OBP_30": [0.35],
                    "OBS_30": [0.80],
                    "est_woba_30": [0.35],
                    "est_slg_30": [0.45],
                    "HR_per_PA_30": [0.04],
                    "HR_per_PA_vs_R_30": [0.05],
                    "HR_per_PA_vs_L_30": [0.03],
                    "age": [28],
                    "days_rest": [1],
                    "is_home": [1],
                    "stand": ["R"],
                },
                index=[20240601],
            )
        }
        pitcher_data = {}

        result = process_homerun_data(df, batter_data, pitcher_data)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_returns_expected_columns(self):
        """Test result has expected columns."""
        df = pd.DataFrame(
            {
                "date_dblhead": [20240601],
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
        batter_data = {
            "101": pd.DataFrame(
                {
                    "BARREL_30": [0.05],
                    "EV_30": [90.0],
                    "HARDHIT_30": [0.4],
                    "SWSPOT_30": [0.35],
                    "SLG_30": [0.45],
                    "OBP_30": [0.35],
                    "OBS_30": [0.80],
                    "est_woba_30": [0.35],
                    "est_slg_30": [0.45],
                    "HR_per_PA_30": [0.04],
                    "HR_per_PA_vs_R_30": [0.05],
                    "HR_per_PA_vs_L_30": [0.03],
                    "age": [28],
                    "days_rest": [1],
                    "is_home": [1],
                    "stand": ["R"],
                },
                index=[20240601],
            )
        }
        pitcher_data = {}

        result = process_homerun_data(df, batter_data, pitcher_data)
        expected_cols = [
            "date_dblhead", "b_id", "slot", "team", "opponent",
            "stand", "opp_throws", "park_hr_factor",
            "temp", "humidity", "wind_spd", "wind_out",
            "BARREL_30", "EV_30", "HR_per_PA_30",
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
                "date_dblhead": [20240601],
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
                "date_dblhead": [20240601],
                "starting_pitcher_id_h": [100],
                "starting_pitcher_id_v": [200],
                "team_h": ["NYY"],
                "team_v": ["BOS"],
                "sp_throws_h": ["R"],
                "sp_throws_v": ["L"],
                "batter1_id_h": [101],
            }
        )
        batter_data = {
            "101": pd.DataFrame(
                {
                    "BARREL_30": [0.05],
                    "EV_30": [90.0],
                    "HARDHIT_30": [0.4],
                    "SWSPOT_30": [0.35],
                    "SLG_30": [0.45],
                    "OBP_30": [0.35],
                    "OBS_30": [0.80],
                    "est_woba_30": [0.35],
                    "est_slg_30": [0.45],
                    "HR_per_PA_30": [0.04],
                    "HR_per_PA_vs_R_30": [0.05],
                    "HR_per_PA_vs_L_30": [0.03],
                    "age": [28],
                    "days_rest": [1],
                    "is_home": [0],
                    "stand": ["R"],
                },
                index=[20240601],
            )
        }
        pitcher_data = {
            "200": pd.DataFrame(
                {
                    "HR_per_BF_10": [0.03],
                    "FB_perc_10": [0.40],
                    "HR_per_BF_35": [0.025],
                    "FB_perc_35": [0.38],
                    "HR_per_BF_75": [0.02],
                    "FB_perc_75": [0.35],
                },
                index=[20240601],
            )
        }

        result = process_homerun_data(df, batter_data, pitcher_data)
        assert not result.empty
        # Batter faces opposing pitcher (v for h team = sp_id_v = 200)
        assert "opp_HR_per_BF_10" in result.columns
        assert result["opp_HR_per_BF_10"].iloc[0] == 0.03


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
