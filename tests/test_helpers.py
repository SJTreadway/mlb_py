import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import (
    roll_column,
    safe_int,
    safe_float,
    get_team_league_map,
    get_park_factors_map,
)


class TestRollColumn:
    """Tests for rolling column calculation."""

    def test_basic_rolling_sum(self):
        """Test basic rolling sum with window size."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = roll_column(df, "val", 3)
        expected = np.array([0, 1, 3, 6, 9])
        np.testing.assert_array_equal(result, expected)

    def test_window_size_one(self):
        """Test with window size 1."""
        df = pd.DataFrame({"val": [10, 20, 30]})
        result = roll_column(df, "val", 1)
        expected = np.array([0, 10, 20])
        np.testing.assert_array_equal(result, expected)

    def test_large_window(self):
        """Test window larger than data length."""
        df = pd.DataFrame({"val": [1, 2, 3]})
        result = roll_column(df, "val", 10)
        expected = np.array([0, 1, 3])
        np.testing.assert_array_equal(result, expected)

    def test_single_row(self):
        """Test with single row of data."""
        df = pd.DataFrame({"val": [5]})
        result = roll_column(df, "val", 3)
        expected = np.array([0])
        np.testing.assert_array_equal(result, expected)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"val": []})
        result = roll_column(df, "val", 3)
        assert len(result) == 0


class TestSafeInt:
    """Tests for safe integer conversion."""

    def test_valid_int_string(self):
        """Test with valid integer string."""
        assert safe_int("42") == 42

    def test_valid_float_string(self):
        """Test with float string."""
        assert safe_int("42.5") == 0

    def test_non_numeric_string(self):
        """Test with non-numeric string."""
        assert safe_int("abc") == 0

    def test_empty_string(self):
        """Test with empty string."""
        assert safe_int("") == 0

    def test_none_value(self):
        """Test with None."""
        assert safe_int(None) == 0

    def test_numeric_string_with_whitespace(self):
        """Test numeric string with whitespace."""
        assert safe_int(" 42 ") == 42

    def test_zero(self):
        """Test with zero."""
        assert safe_int("0") == 0


class TestSafeFloat:
    """Tests for safe float conversion."""

    def test_valid_float(self):
        """Test with valid float string."""
        assert safe_float("3.14") == 3.14

    def test_valid_int_string(self):
        """Test with integer string."""
        assert safe_float("42") == 42.0

    def test_non_numeric_string(self):
        """Test with non-numeric string."""
        assert safe_float("abc") == 0

    def test_empty_string(self):
        """Test with empty string."""
        assert safe_float("") == 0

    def test_none_value(self):
        """Test with None."""
        assert safe_float(None) == 0


class TestGetTeamLeagueMap:
    """Tests for team league mapping."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_team_league_map()
        assert isinstance(result, dict)

    def test_al_east_teams(self):
        """Test AL East teams."""
        result = get_team_league_map()
        for team in ["BAL", "BOS", "NYY", "TB", "TBR", "TOR"]:
            assert result.get(team) == "A", f"{team} should be AL"

    def test_al_central_teams(self):
        """Test AL Central teams."""
        result = get_team_league_map()
        for team in ["CWS", "CHW", "CLE", "DET", "KC", "KCR", "MIN"]:
            assert result.get(team) == "A", f"{team} should be AL"

    def test_al_west_teams(self):
        """Test AL West teams."""
        result = get_team_league_map()
        for team in ["HOU", "LAA", "SEA", "TEX", "OAK", "ATH"]:
            assert result.get(team) == "A", f"{team} should be AL"

    def test_nl_east_teams(self):
        """Test NL East teams."""
        result = get_team_league_map()
        for team in ["ATL", "MIA", "NYM", "PHI", "WSN", "WAS"]:
            assert result.get(team) == "N", f"{team} should be NL"

    def test_nl_central_teams(self):
        """Test NL Central teams."""
        result = get_team_league_map()
        for team in ["CHC", "CIN", "MIL", "PIT", "STL"]:
            assert result.get(team) == "N", f"{team} should be NL"

    def test_nl_west_teams(self):
        """Test NL West teams."""
        result = get_team_league_map()
        for team in ["ARI", "AZ", "COL", "LAD", "SDP", "SD", "SFG", "SF"]:
            assert result.get(team) == "N", f"{team} should be NL"

    def test_unknown_team(self):
        """Test unknown team returns None."""
        result = get_team_league_map()
        assert result.get("XXX") is None


class TestGetParkFactorsMap:
    """Tests for park factors mapping."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_park_factors_map()
        assert isinstance(result, dict)

    def test_coors_field_highest(self):
        """Test Coors Field has the highest factor."""
        result = get_park_factors_map()
        assert result.get("COL", 0) == 112

    def test_known_parks(self):
        """Test a few known park factors."""
        result = get_park_factors_map()
        assert result.get("NYY") == 102
        assert result.get("BOS") == 102
        assert result.get("LAD") == 102
        assert result.get("SEA") == 92

    def test_unknown_park(self):
        """Test unknown park returns None."""
        result = get_park_factors_map()
        assert result.get("XXX") is None

    def test_all_factors_between_90_and_115(self):
        """Test all factors are in reasonable range."""
        result = get_park_factors_map()
        for team, factor in result.items():
            assert 90 <= factor <= 115, f"{team} has factor {factor}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
