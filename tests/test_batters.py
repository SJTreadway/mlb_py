import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["YEAR"] = "2024"

from api.batters import get_position_defaults


class TestGetPositionDefaults:
    """Tests for position default statistics."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        defaults = get_position_defaults()
        assert isinstance(defaults, dict)

    def test_has_all_positions(self):
        """Test that all positions are included."""
        defaults = get_position_defaults()
        expected_positions = [
            "p",
            "ss",
            "c",
            "2b",
            "3b",
            "1b",
            "lf",
            "rf",
            "cf",
            "ph",
            "pr",
            "dh",
        ]
        for pos in expected_positions:
            assert pos in defaults

    def test_position_has_required_stats(self):
        """Test that each position has required stat fields."""
        defaults = get_position_defaults()
        required_stats = ["batavg", "obp", "slg", "slgmod", "obs", "sobat"]
        for pos, stats in defaults.items():
            for stat in required_stats:
                assert stat in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
