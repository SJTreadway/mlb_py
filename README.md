# MLB Prediction Pipeline

An automated MLB game prediction system using machine learning and historical data from the **Statcast API** via [pybaseball](https://github.com/jldbc/pybaseball).

> **Note:** This project has been migrated from web scraping (Retrosheet/Baseball-Reference) to API-based data fetching for better reliability and performance.

## Overview

This project predicts MLB game outcomes (winners and run totals) using historical player and team statistics from the official MLB Statcast database. It fetches daily game schedules, calculates rolling performance metrics, and generates predictions with betting edge analysis.

## Architecture

```
mlb_py/
├── api/                      # Data processing modules
│   ├── batters_v2.py        # ✅ CURRENT: Statcast API version
│   ├── pitchers_v2.py       # ✅ CURRENT: Statcast API version  
│   ├── lineups.py           # Lineup construction and features
│   ├── teams.py             # Team-level aggregations
│   ├── odds.py              # Betting odds integration
│   ├── batters.py           # ⛔ LEGACY: Web scraping (deprecated)
│   └── pitchers.py          # ⛔ LEGACY: Web scraping (deprecated)
├── tests/                    # Unit tests
│   ├── test_batters_v2.py   # Tests for API version
│   ├── test_pitchers_v2.py  # Tests for API version
│   ├── test_lineups.py
│   ├── test_teams.py
│   ├── test_odds.py
│   └── test_pipeline.py
├── data/                     # Local data storage (gitignored)
│   ├── bat/                 # Batter historical data
│   ├── pitch/               # Pitcher historical data
│   ├── daily/               # Daily game data
│   └── results/             # Prediction outputs
├── models/                   # Trained ML models (gitignored)
├── helpers.py               # Utility functions
├── pipeline.py              # Main prediction pipeline
├── Makefile                # Build commands
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (gitignored)
```

## Key Features

### Data Sources ✅ API-Based
- **Statcast API** (via pybaseball) - Official MLB player statistics
- **MLB Stats API** - Team data and schedules
- **Odds API** - Betting lines and prices

> ⚡ **No Web Scraping:** The current implementation uses the Statcast API exclusively, eliminating timeouts and CAPTCHA issues associated with web scraping.

### Statistics Tracked

**Batters:**
- Rolling averages (30, 75, 162, 350 games)
- AVG, OBP, SLG, OPS
- Stolen bases, caught stealing
- Position-specific defaults

**Pitchers:**
- ERA, WHIP, FIP
- Strikeout percentage
- Walks + hits per inning
- Rolling windows (10, 35, 75 games)

**Teams:**
- Team hitting aggregates
- Bullpen vs starter splits
- Home/away splits

### Models

1. **Home Victory Model** - Predicts game winner
2. **Runs Scored Model** - Predicts total runs (over/under)

## Installation

```bash
# Clone repository
git clone <repository-url>
cd mlb_py

# Install dependencies
make install

# Or manually:
pip install -r requirements.txt
```

### Key Dependencies

- **[pybaseball](https://github.com/jldbc/pybaseball)** - Official MLB Statcast API client ⚾
  - `statcast_batter()` - Batter statistics
  - `statcast_pitcher()` - Pitcher statistics
  - `playerid_lookup()` - Player ID resolution
  
- **MLB-StatsAPI** - Official MLB stats and schedules
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning models
- **numpy** - Numerical computations

## Environment Setup

Create a `.env` file:

```bash
# Required
YEAR=2026
TOMORROW_GAMES=0
REFRESH_DATA=1

# Optional - for Twitter integration
X_ACCESS_KEY=your_key
X_ACCESS_SECRET=your_secret
X_CONSUMER_KEY=your_key
X_CONSUMER_SECRET=your_secret
X_BEARER_TOKEN=your_token

# Optional - for odds
ODDS_API_KEY=your_key
```

## Usage

### Run Full Pipeline

```bash
# Run for today's games
python pipeline.py

# Or use Makefile
make run
```

### Run Tests

```bash
# Run all tests
make test

# Run with verbose output
make test-verbose

# Run specific test file
make test-file FILE=test_batters_v2

# Run tests matching pattern
make test-pattern PATTERN=retro
```

### Available Commands

```bash
make test              # Run all tests
make test-verbose      # Run tests with verbose output
make test-file FILE=name    # Run specific test file
make test-pattern PATTERN=keyword  # Run matching tests
make run               # Run the pipeline
make clean             # Clean test cache
make help              # Show all commands
```

## How It Works

### 1. Data Collection
- Fetches today's game schedule from MLB Stats API
- Identifies starting pitchers and lineups
- Downloads historical player data from Statcast API

### 2. Feature Engineering
- Calculates rolling statistics for each player
- Aggregates team-level metrics
- Computes bullpen vs starter differentials

### 3. Prediction
- Loads trained ML models
- Generates predictions for each game:
  - Home team win probability
  - Total runs scored prediction
- Calculates betting edges against market lines

### 4. Output
- Saves predictions to `data/results/`
- Displays formatted results in console
- Optional: Posts to Twitter/X

## Pipeline Flow

```
1. Get Schedule → Fetch today's games from MLB API
2. Load Pitchers → Download pitcher stats (Statcast API)
3. Load Batters → Download batter stats (Statcast API)
4. Build Lineups → Construct lineups and features
5. Team Stats → Aggregate team-level metrics
6. Bullpen Data → Calculate relief pitcher stats
7. Predict → Run ML models
8. Odds → Compare to market lines
9. Output → Save and display results
```

## File Structure Details

### Core Pipeline

**`pipeline.py`** - Main execution script
- `get_games()` - Fetch daily schedule
- `process_pitching_data()` - Process pitcher features
- `process_batting_data()` - Process batter features  
- `get_lines()` - Build lineups and features
- `predict_winner()` - Run winner prediction model
- `predict_runs_scored()` - Run totals prediction model

### API Modules

**`api/batters_v2.py`** - Batter statistics
- `process_batting_data()` - Main processing function
- `load_batting_data()` - Fetch from Statcast API
- `transform_statcast_batter()` - Convert API data to format
- `process_batter_df()` - Calculate rolling features
- `get_batting_feats()` - Add features to main dataframe

**`api/pitchers_v2.py`** - Pitcher statistics
- `process_pitching_data()` - Main processing function
- `load_pitching_data()` - Fetch from Statcast API
- `transform_statcast_pitcher()` - Convert API data to format
- `load_and_process_pitch_df()` - Calculate rolling features
- `get_bullpen_data()` - Calculate bullpen stats (includes debug mode!)

**`api/lineups.py`** - Lineup construction
- `get_lineups()` - Main lineup function
- `get_run_total_feats()` - Features for totals model
- `agg_non_na()` - Aggregation helper

**`api/teams.py`** - Team statistics
- `process_team_data()` - Calculate team-level features
- `create_team_df()` - Build team-specific dataframe
- `generate_team_window_features()` - Rolling team stats

**`api/odds.py`** - Betting integration
- `get_over_odds()` / `get_under_odds()` - Fetch odds
- `calculate_edge()` - Calculate betting edge
- `line_to_bet()` - Convert probability to line

### Utilities

**`helpers.py`** - Shared utilities
- `roll_column()` - Calculate rolling sums
- `get_team_league_map()` - Team to league mapping
- `strip_suffix()` - Column name helper
- `safe_int()` / `safe_float()` - Safe type conversion

## Implementation Versions

### ✅ Current: V2 - Statcast API (Recommended)
**Files:** `api/batters_v2.py`, `api/pitchers_v2.py`

The current implementation uses the official MLB Statcast API via the [pybaseball](https://github.com/jldbc/pybaseball) library:
- No web scraping required
- No timeouts or CAPTCHA issues
- Faster data retrieval
- More reliable and consistent data
- Configured in `pipeline.py` by default

**Key functions:**
- `statcast_batter()` - Fetch batter data from Statcast
- `statcast_pitcher()` - Fetch pitcher data from Statcast
- `playerid_reverse_lookup()` - Convert player IDs

### ⛔ Legacy: V1 - Web Scraping (Deprecated)
**Files:** `api/batters.py`, `api/pitchers.py`

The original implementation scraped data from:
- Baseball-Reference (batters)
- Retrosheet (pitchers)

**Issues with V1:**
- Prone to timeouts
- CAPTCHA blocking
- Website structure changes break scrapers
- Slower data retrieval

**Migration:** V2 files are drop-in replacements. The pipeline automatically uses V2.

## Betting Integration

The system compares model predictions to market odds:

```python
# Example output
Game: NYY vs BOS
Predicted Total: 9.2 runs
Market Line: 8.5 runs (-110)
Edge: +12% (value on over)
```

## Troubleshooting

### Bullpen Features Always Default Values
**Fixed!** The issue was in `get_bullpen_team_df()` which only used home OR away games instead of both. Now uses `pd.concat()` to combine both datasets.

### API Performance
- ✅ **Statcast API** has no rate limits
- ✅ Built-in delays (0.1s) between requests to be polite
- ✅ Data is cached locally after first download
- ✅ No timeouts or connection issues

> **Note:** If you were experiencing timeouts with the old web scraping version (V1), please ensure you're using the V2 files (`api/batters_v2.py` and `api/pitchers_v2.py`), which are now the default in `pipeline.py`.

### Missing Data
- Check `data/bat/` and `data/pitch/` directories
- Run with `REFRESH_DATA=1` to re-download
- Models use position-specific defaults when data unavailable

### Tests Failing
```bash
# Run specific test with verbose output
python -m pytest tests/test_batters_v2.py::TestRetroToMlbam -v

# Run all tests
make test
```

## Development

### Adding New Features

1. Add to appropriate `api/` module
2. Update feature sets in `pipeline.py`:
   - `KS_FEAT_SET`
   - `HR_FEAT_SET`
3. Add tests in `tests/`
4. Run tests: `make test`

### Training New Models

```python
from sklearn.ensemble import RandomForestClassifier
import pickle

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
with open(f'models/winner_model_{YEAR}.pkl', 'wb') as f:
    pickle.dump(model, f)
```

## Data Sources

### Primary (Current Implementation)
- **Statcast API** (via pybaseball) - Official MLB player statistics ⚾
- **MLB Stats API** - Team data and game schedules
- **Odds API** - Betting lines and market prices (optional)

### Legacy (V1 Only - Deprecated)
- ~~Retrosheet~~ - Historical data (replaced by Statcast)
- ~~Baseball-Reference~~ - Player statistics (replaced by Statcast)

> **Why Statcast?** The Statcast database provides the most comprehensive and accurate MLB data available, including pitch-level data, exit velocity, launch angle, and more. It's the same data used by MLB teams and broadcasts.

## License

MIT License - See LICENSE file

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## Acknowledgments

- MLB Advanced Media for Statcast data
- pybaseball library for API access
- Retrosheet for historical data (V1)

## Support

For issues or questions:
- Check existing issues on GitHub
- Review `API_IMPLEMENTATION.md` for NHL version
- Run `make test` to verify setup
