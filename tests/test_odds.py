import json
import os
import sys
import pytest
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["ODDS_API_KEY"] = "test_api_key"

from api.odds import (
    get_over_odds,
    get_under_odds,
    get_total_line,
    get_money_line_price,
    get_spread_line,
    get_spread_odds,
    line_to_bet,
    calculate_edge,
    line_to_prob,
    prob_to_line,
    get_stripped_team_val,
    extract_total_odds,
)


class TestCalculateEdge:
    """Tests for edge calculation."""

    def test_positive_edge(self):
        """Test calculation with positive edge."""
        # 60% probability, +150 odds
        result = calculate_edge(0.60, 150)
        # Result is a string like "10.0%"
        assert isinstance(result, str)
        assert "%" in result
        # Extract numeric value and check it's positive
        edge_val = float(result.replace("%", ""))
        assert edge_val > 0

    def test_negative_edge(self):
        """Test calculation with negative edge."""
        # 40% probability, -200 odds
        result = calculate_edge(0.40, -200)
        # Result is a string like "-10.0%"
        assert isinstance(result, str)
        assert "%" in result
        edge_val = float(result.replace("%", ""))
        assert edge_val < 0

    def test_zero_edge(self):
        """Test calculation with fair odds."""
        # 50% probability, +100 odds
        result = calculate_edge(0.50, 100)
        assert isinstance(result, str)
        edge_val = float(result.replace("%", ""))
        assert abs(edge_val) < 0.01  # Should be approximately 0

    def test_negative_odds(self):
        """Test with negative odds (favorite)."""
        result = calculate_edge(0.70, -150)
        assert isinstance(result, str)
        assert "%" in result

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

    @patch("api.odds.get_odds_results")
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {"Yankees": {"over_under_price_o": -110}}

        result = get_over_odds("New York Yankees")
        assert result == -110

    @patch("api.odds.get_odds_results")
    def test_returns_none_on_error(self, mock_get_results):
        """Test that function returns None when team not found."""
        mock_get_results.return_value = {}

        result = get_over_odds("New York Yankees")
        assert result is None


class TestGetUnderOdds:
    """Tests for under odds retrieval."""

    @patch("api.odds.get_odds_results")
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {"Yankees": {"over_under_price_u": -110}}

        result = get_under_odds("New York Yankees")
        assert result == -110


class TestGetTotalLine:
    """Tests for total line retrieval."""

    @patch("api.odds.get_odds_results")
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {"Yankees": {"over_under_line": 8.5}}

        result = get_total_line("New York Yankees")
        assert result == 8.5


class TestGetMoneyLinePrice:
    """Tests for money line price retrieval."""

    @patch("api.odds.get_odds_results")
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {"Yankees": {"moneyline_price": -150}}

        result = get_money_line_price("New York Yankees")
        assert result == -150


class TestProbToLine:
    """Tests for probability to line conversion."""

    def test_favorite_probability(self):
        """Test converting favorite probability to negative line."""
        result = prob_to_line(0.60)
        assert result is not None
        assert result < 0

    def test_underdog_probability(self):
        """Test converting underdog probability to positive line."""
        result = prob_to_line(0.40)
        assert result is not None
        assert result > 0

    def test_even_probability(self):
        """Test 50% probability returns even money."""
        result = prob_to_line(0.50)
        assert result == 100

    def test_invalid_probability_zero(self):
        """Test probability of 0 returns None."""
        result = prob_to_line(0)
        assert result is None

    def test_invalid_probability_one(self):
        """Test probability of 1 returns None."""
        result = prob_to_line(1)
        assert result is None

    def test_invalid_probability_negative(self):
        """Test negative probability returns None."""
        result = prob_to_line(-0.5)
        assert result is None

    def test_round_trip(self):
        """Test that prob_to_line and line_to_prob are inverses."""
        for prob in [0.25, 0.40, 0.50, 0.60, 0.75]:
            line = prob_to_line(prob)
            if line is not None:
                roundtrip = line_to_prob(line)
                assert abs(roundtrip - prob) < 0.01


class TestGetStrippedTeamVal:
    """Tests for team name stripping."""

    def test_single_word(self):
        """Test single word team name."""
        assert get_stripped_team_val("Yankees") == "Yankees"

    def test_two_words_last_not_sox(self):
        """Test two words where last is not Sox/Jays."""
        assert get_stripped_team_val("New York Yankees") == "Yankees"

    def test_red_sox(self):
        """Test Red Sox returns 'Red Sox'."""
        assert get_stripped_team_val("Boston Red Sox") == "Red Sox"

    def test_white_sox(self):
        """Test White Sox returns 'White Sox'."""
        assert get_stripped_team_val("Chicago White Sox") == "White Sox"

    def test_blue_jays(self):
        """Test Blue Jays returns 'Blue Jays'."""
        assert get_stripped_team_val("Toronto Blue Jays") == "Blue Jays"

    def test_single_word_sox(self):
        """Test 'Sox' alone returns 'Sox'."""
        assert get_stripped_team_val("Sox") == "Sox"


class TestExtractTotalOdds:
    """Tests for odds extraction from JSON data."""

    def test_extracts_home_and_away(self):
        """Test extraction of both home and away teams."""
        data = json.dumps(
            [
                {
                    "home_team": "New York Yankees",
                    "away_team": "Boston Red Sox",
                    "commence_time": "2024-06-01T18:00:00Z",
                    "bookmakers": [
                        {
                            "key": "fanduel",
                            "markets": [
                                {
                                    "key": "totals",
                                    "outcomes": [
                                        {"name": "Over", "point": 8.5, "price": -110},
                                        {"name": "Under", "price": -110},
                                    ],
                                },
                                {
                                    "key": "spreads",
                                    "outcomes": [
                                        {
                                            "name": "New York Yankees",
                                            "point": -1.5,
                                            "price": +150,
                                        },
                                    ],
                                },
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "New York Yankees", "price": -150},
                                        {"name": "Boston Red Sox", "price": +130},
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        )
        result = extract_total_odds(data)
        assert "Yankees" in result
        assert "Red Sox" in result
        assert result["Yankees"]["over_under_line"] == 8.5
        assert result["Yankees"]["over_under_price_o"] == -110
        assert result["Yankees"]["over_under_price_u"] == -110
        assert result["Yankees"]["moneyline_price"] == -150
        assert result["Red Sox"]["moneyline_price"] == +130

    def test_no_bookmakers(self):
        """Test game with no bookmakers."""
        data = json.dumps(
            [
                {
                    "home_team": "Los Angeles Dodgers",
                    "away_team": "San Francisco Giants",
                    "commence_time": "2024-06-01T20:00:00Z",
                    "bookmakers": [],
                }
            ]
        )
        result = extract_total_odds(data)
        assert "Dodgers" in result
        assert result["Dodgers"]["over_under_line"] is None

    def test_only_fanduel_is_used(self):
        """Test only FanDuel bookmaker is considered."""
        data = json.dumps(
            [
                {
                    "home_team": "Chicago Cubs",
                    "away_team": "St. Louis Cardinals",
                    "commence_time": "2024-06-01T18:00:00Z",
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "markets": [
                                {
                                    "key": "totals",
                                    "outcomes": [
                                        {"name": "Over", "point": 7.5, "price": -105}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        )
        result = extract_total_odds(data)
        assert result["Cubs"]["over_under_line"] is None

    def test_empty_data(self):
        """Test with empty JSON array."""
        result = extract_total_odds("[]")
        assert result == {}


class TestGetSpreadLine:
    """Tests for spread line retrieval."""

    @patch("api.odds.get_odds_results")
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {"Yankees": {"spread_line": -1.5}}

        result = get_spread_line("New York Yankees")
        assert result == -1.5

    @patch("api.odds.get_odds_results")
    def test_raises_on_missing_key(self, mock_get_results):
        """Test that function raises KeyError when spread not found."""
        mock_get_results.return_value = {"Yankees": {"over_under_line": 8.5}}

        with pytest.raises(KeyError):
            get_spread_line("New York Yankees")


class TestGetSpreadOdds:
    """Tests for spread odds retrieval."""

    @patch("api.odds.get_odds_results")
    def test_returns_value(self, mock_get_results):
        """Test that function returns a value."""
        mock_get_results.return_value = {"Yankees": {"spread_price": -110}}

        result = get_spread_odds("New York Yankees")
        assert result == -110


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
