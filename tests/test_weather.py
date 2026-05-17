import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.weather import compute_wind_out, process_weather_data, DOME_TEAMS


class TestComputeWindOut:
    """Tests for wind-out calculation."""

    def test_no_wind(self):
        """Test zero wind speed returns 0."""
        result = compute_wind_out(0, 90, 0)
        assert result == 0.0

    def test_wind_directly_out(self):
        """Test wind blowing directly out to CF."""
        result = compute_wind_out(10, 90, 90)
        assert result == 10.0

    def test_wind_directly_in(self):
        """Test wind blowing directly in from CF."""
        result = compute_wind_out(10, 270, 90)
        assert result == -10.0

    def test_wind_perpendicular(self):
        """Test cross-wind gives near-zero output."""
        result = compute_wind_out(10, 0, 90)
        assert abs(result) < 1

    def test_wind_45_degrees_out(self):
        """Test wind at 45 degrees out."""
        result = compute_wind_out(10, 45, 90)
        expected = round(10 * np.cos(np.radians(45)), 2)
        assert result == expected

    def test_angle_diff_wrapping(self):
        """Test angle difference wrapping around 360."""
        result = compute_wind_out(10, 350, 10)
        assert result > 0  # 20 deg diff, mostly out


class TestProcessWeatherData:
    """Tests for weather data processing."""

    def test_dome_team_gets_defaults(self):
        """Test dome teams get fixed values."""
        df = pd.DataFrame({"team_h": ["HOU"]})
        from datetime import date
        result = process_weather_data(df, date.today())
        assert result["temp"].iloc[0] == 72
        assert result["humidity"].iloc[0] == 50
        assert result["wind_spd"].iloc[0] == 0
        assert result["wind_out"].iloc[0] == 0

    def test_all_dome_teams(self):
        """Test all dome teams get defaults."""
        from datetime import date
        df = pd.DataFrame({"team_h": list(DOME_TEAMS)})
        result = process_weather_data(df, date.today())
        assert all(result["temp"] == 72)
        assert all(result["humidity"] == 50)

    def test_unknown_team_gets_defaults(self):
        """Test unknown team code gets defaults."""
        df = pd.DataFrame({"team_h": ["XXX"]})
        from datetime import date
        result = process_weather_data(df, date.today())
        assert result["temp"].iloc[0] == 72
        assert result["humidity"].iloc[0] == 50

    def test_returns_dataframe_with_weather_cols(self):
        """Test function adds weather columns to DataFrame."""
        df = pd.DataFrame({"team_h": ["HOU"]})
        from datetime import date
        result = process_weather_data(df, date.today())
        assert "temp" in result.columns
        assert "humidity" in result.columns
        assert "wind_spd" in result.columns
        assert "wind_out" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
