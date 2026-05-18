# MLB Prediction Pipeline

A production-grade MLB prediction system built on Statcast data, featuring daily home run probability rankings and game winner predictions with betting edge calculations.

**Models:**
- **HR Model** — XGBoost classifier (AUC: 0.620, 290k rows, 2020-2025) using barrel rate, exit velocity, sweet spot%, hard hit%, platoon splits, pitcher fly ball rate, park factors, weather/wind
- **Win Model** — LightGBM classifier using starting pitcher, bullpen, and lineup rolling features. Profitable at 4%+ edge threshold

**Infrastructure:** Snowflake + GitHub Actions for daily automated ingestion and weekly model retraining

---

## Architecture

```
mlb_py/
├── api/                      # Data processing modules
│   ├── batters_v2.py         # Statcast batter pipeline
│   ├── pitchers_v2.py        # Statcast pitcher pipeline
│   ├── lineups_v2.py         # MLB Stats API lineup fetch
│   ├── teams.py              # Team-level rolling aggregations
│   ├── odds.py               # Betting odds integration
│   ├── homerun.py            # HR prediction feature engineering
│   └── weather.py            # Weather + wind-out calculation
├── ui/
│   └── dashboard.py          # HTML dashboard generator
├── data/                     # Local cache (gitignored)
│   ├── bat/                  # Batter historical CSVs
│   ├── pitch/                # Pitcher historical CSVs
│   ├── daily/                # Daily game data + odds cache
│   └── results/              # Prediction outputs
├── models/                   # Trained models (gitignored)
│   ├── homerun_model_2026v1.pkl
│   └── win_model_2026v1.pkl
├── cleanup.py
├── helpers.py
├── pipeline.py               # Main orchestration
├── Makefile
└── requirements.txt
```

---

## Models

### HR Prediction Model
XGBoost classifier with isotonic calibration trained on 290,000+ batter-game rows (2020-2025).

**Feature set:**
- Contact quality: barrel%, exit velocity, hard hit%, sweet spot% — rolling 7/14/30/75/162-game windows
- Power rates: HR/PA, HR/PA vs RHP, HR/PA vs LHP — same windows
- Rate stats: SLG, OBP, OBS, estimated wOBA, estimated SLG
- Pitcher matchup: HR/BF and FB% over 10/35/75-game windows
- Context: park HR factor, temperature, humidity, wind speed, wind-out (mph blowing toward CF)
- Player: age, days rest, home/away, batting slot

**Validation:** Mean AUC 0.620 (5-fold TimeSeriesSplit). Calibration shows actual HR rate 2x predicted at high confidence — model is conservative by design.

**Results (sample):**
- May 14, 2026 — Jordan Walker +375 — homered ✅
- May 15, 2026 — Brandon Lowe +440 — hit 2 HRs ✅
- May 15, 2026 — Nick Kurtz #1 ranked — homered ✅
- May 16, 2026 — Yordan Alvarez +290 — homered first AB ✅
- May 17, 2026 — Junior Caminero +270 — homered first AB ✅
- May 17, 2026 — Ben Rice +390 — homered ✅

### Win Prediction Model
LightGBM classifier trained on 2015-2025 game data (modern analytics era).

**Feature set:**
- Starting pitcher: WHIP, TB/BB%, H/BB%, SO% — 10/35/75-game windows
- Bullpen: WHIP, SO%, TB/BB%, H/BB% — 10/35/75-game windows
- Lineup: OBP, SLG — 75/162/350-game windows for home and away

**Results:** Profitable at 4%+ edge threshold vs moneyline market.

---

## Pipeline

```
1. Get Lineups          → MLB Stats API (confirmed starters + batting order)
2. Load Pitcher Data    → Statcast via pybaseball → rolling features
3. Load Batter Data     → Statcast via pybaseball → rolling features
4. Team Aggregations    → Rolling OBP/SLG/ERR per team
5. Weather              → Open-Meteo API (temp, humidity, wind speed + direction)
6. Win Prediction       → LightGBM → probability + edge vs FanDuel moneyline
7. HR Prediction        → XGBoost → probability per batter + odds from the-odds-api
8. Dashboard            → HTML page with color-coded tables, auto-opens in browser
```

---

## Dashboard

Auto-generated HTML dashboard with three sections:

- **VALUE** — HR edge plays with positive expected value
- **HR** — Top home run predictions with Statcast metrics and wind data
- **WIN** — Game winner predictions with probabilities and betting edges

Color coding:
- Statcast metrics: green (significantly above league average), yellow (near average), red (below)
- Edge: bright green (3%+), dim green (1-3%), red (negative)
- Wind Out: green (blowing toward CF), red (blowing in)

---

## Infrastructure

### Daily Pipeline (GitHub Actions)
Runs 3x daily (11am, 1pm, 3pm CST) on a self-hosted AWS EC2 runner:
- Fetches yesterday's boxscores → loads batter/pitcher game rows to Snowflake
- Computes rolling features → updates `BATTER_ROLLING_FEATURES` and `PITCHER_ROLLING_FEATURES`
- Logs lineup confirmation status for today's slate

### Snowflake Schema
```
BASEBALL.STATCAST.RAW_BATTER_GAMES
BASEBALL.STATCAST.RAW_PITCHER_GAMES
BASEBALL.STATCAST.GAME_RESULTS
BASEBALL.STATCAST.BATTER_ROLLING_FEATURES
BASEBALL.STATCAST.PITCHER_ROLLING_FEATURES
BASEBALL.HISTORICAL.RETROSHEET_EVENTS
```

### Weekly Retraining (GitHub Actions)
Mondays at 7am CST — pulls from Snowflake, retrains both models, saves updated pkl files.

---

## Installation

```bash
git clone https://github.com/SJTreadway/mlb_py.git
cd mlb_py
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```bash
YEAR=2026
REFRESH_DATA=0          # set to 1 to force re-download
ODDS_API_KEY=your_key   # the-odds-api.com
```

---

## Usage

```bash
# Run pipeline for today
make run

# Force refresh all data
make run:force

# Clean daily cache
make clean:daily

# Clean all cached data
make clean
```

---

## Data Sources

| Source | Usage |
|--------|-------|
| Statcast (pybaseball) | Pitch-level batter and pitcher data |
| MLB Stats API | Schedules, lineups, boxscores |
| Open-Meteo | Weather and wind forecasts |
| the-odds-api | Moneyline and HR prop odds |

---

## Stack

Python · XGBoost · LightGBM · pandas · numpy · Snowflake · GitHub Actions · AWS EC2 · Open-Meteo · pybaseball

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Follow [@MoneyballVo](https://x.com/MoneyballVo) on X for daily picks and model updates.*