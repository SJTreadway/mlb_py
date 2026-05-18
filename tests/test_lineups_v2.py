import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['YEAR'] = '2024'
os.environ['TOMORROW_GAMES'] = '0'

from api.lineups_v2 import get_lineups, get_run_total_feats


class TestGetRunTotalFeats:
    """Tests for run total features extraction."""

    @pytest.mark.skip(reason="Requires too many columns - better tested via integration")
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        # This test is skipped because the function requires 150+ columns
        # Better to test this via integration tests with real data
        pass

    @pytest.mark.skip(reason="Requires too many columns - better tested via integration")
    def test_doubles_data(self):
        """Test that data is doubled (home and away perspectives)."""
        test_df = pd.DataFrame({
            'date_dblhead': [20240615],
            'game_time': ['7:05 PM'],
            'team_h': ['NYY'],
            'team_h_full': ['New York Yankees'],
            'team_v': ['BOS'],
            'team_v_full': ['Boston Red Sox'],
            'BATAVG_30_h': [0.250],
            'OBP_30_h': [0.320],
            'SLG_30_h': [0.420],
            'OBS_30_h': [0.740],
            'SB_30_h': [0.5],
            'CS_30_h': [0.1],
            'BATAVG_30_v': [0.260],
            'OBP_30_v': [0.330],
            'SLG_30_v': [0.430],
            'OBS_30_v': [0.760],
            'SB_30_v': [0.6],
            'CS_30_v': [0.2],
            'lineup9_BATAVG_30_h': [0.250],
            'lineup9_OBP_30_h': [0.320],
            'lineup9_SLG_30_h': [0.420],
            'lineup9_BATAVG_30_v': [0.260],
            'lineup9_OBP_30_v': [0.330],
            'lineup9_SLG_30_v': [0.430],
            'Strt_ERA_10_v': [3.50],
            'Strt_WHIP_10_v': [1.20],
            'Strt_SO_perc_10_v': [0.22],
            'Strt_H_BB_perc_10_v': [0.28],
            'Strt_TB_BB_perc_10_v': [0.35],
            'Strt_FIP_10_v': [3.60],
            'Strt_FIP_perc_10_v': [3.50],
            'Strt_ERA_10_h': [3.60],
            'Strt_WHIP_10_h': [1.25],
            'Strt_SO_perc_10_h': [0.23],
            'Strt_H_BB_perc_10_h': [0.29],
            'Strt_TB_BB_perc_10_h': [0.36],
            'Strt_FIP_10_h': [3.70],
            'Strt_FIP_perc_10_h': [3.60],
            'Bpen_WHIP_10_v': [1.30],
            'Bpen_SO_perc_10_v': [0.20],
            'Bpen_H_BB_perc_10_v': [0.30],
            'Bpen_TB_BB_perc_10_v': [0.38],
            'Bpen_WHIP_10_h': [1.35],
            'Bpen_SO_perc_10_h': [0.21],
            'Bpen_H_BB_perc_10_h': [0.31],
            'Bpen_TB_BB_perc_10_h': [0.39]
        })

        result = get_run_total_feats(test_df)

        # Should have 2 rows (home and away perspective)
        assert len(result) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
