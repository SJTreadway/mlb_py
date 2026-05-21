"""
homerun.py
──────────
Build one row per batter-game for HR model prediction.

Updated to work with the new Snowflake-backed batters.py and pitchers.py
which return pd.Series (latest row per player) instead of date-indexed DataFrames.
"""

import datetime

import numpy as np
import pandas as pd
from helpers import get_park_factors_map

CURRENT_YEAR = datetime.date.today().year
SEASON_START = datetime.date(CURRENT_YEAR, 4, 1)
CUTOFF_DATE = datetime.date.today() - datetime.timedelta(days=30)
DAYS_365_DATE = datetime.date.today() - datetime.timedelta(days=365)

PARK_HR_FACTORS = get_park_factors_map()

WINDOWS_BAT = [7, 14, 30, 75, 162, 350]
WINDOWS_PITCH = [10, 35, 75]


def _check_batter_active(brow: pd.Series) -> bool:
    """Return True if batter has recent enough activity to include in predictions.

    Mirrors original checks:
      1. Latest game is in the current season
      2. Latest game within last 30 days (not IL / minors)
      3. At least 5 regular season games → ab_162 >= 10
      4. At least 10 games in last 365 days → ab_162 >= 20
         (162-game rollsum is the best available proxy since we only
          store the latest row per batter, not full history)
    """
    game_date = brow.get("game_date")
    if game_date is None:
        return False

    # normalise to date object
    if hasattr(game_date, "date"):
        game_date = game_date.date()
    elif isinstance(game_date, str):
        try:
            game_date = datetime.date.fromisoformat(str(game_date)[:10])
        except Exception:
            return False

    # 1. must have played in current season
    if game_date < SEASON_START:
        return False

    # 2. must have played within last 30 days
    if game_date < CUTOFF_DATE:
        return False

    # 3 + 4. minimum games via rollsum proxy
    ab_162 = float(brow.get("rollsum_ab_162", 0) or 0)
    if ab_162 < 20:  # ~10 games × 2 AB/game
        return False

    return True


def process_homerun_data(df, batter_data_dict, pitcher_data_dict):
    """
    Build one row per batter-game for HR model prediction.

    df               : main game dataframe with weather, park, lineup columns
    batter_data_dict : dict of str(mlbam_id) → pd.Series (latest batter row)
    pitcher_data_dict: dict of str(mlbam_id) → pd.Series (latest pitcher row)
    """
    rows = []

    for _, game_row in df.iterrows():
        sp_id_h = game_row["starting_pitcher_id_h"]
        sp_id_v = game_row["starting_pitcher_id_v"]

        for hv in ["h", "v"]:
            opposing_sp = sp_id_v if hv == "h" else sp_id_h

            # ── opposing pitcher features ─────────────────────────────────
            pitcher_feats = {
                **{f"opp_hr_per_bf_{w}": 0.032 for w in WINDOWS_PITCH},
                **{f"opp_fb_perc_{w}": 0.35 for w in WINDOWS_PITCH},
                "opp_is_starter": 1,
            }
            if opposing_sp is not None and not pd.isna(opposing_sp):
                opp_sp_key = str(int(opposing_sp))
                prow = pitcher_data_dict.get(opp_sp_key)
                if prow is not None:
                    for w in WINDOWS_PITCH:
                        hr_bf = prow.get(f"hr_per_bf_{w}")
                        fb = prow.get(f"fb_perc_{w}")
                        if hr_bf is not None and not pd.isna(hr_bf):
                            pitcher_feats[f"opp_hr_per_bf_{w}"] = float(hr_bf)
                        if fb is not None and not pd.isna(fb):
                            pitcher_feats[f"opp_fb_perc_{w}"] = float(fb)
                    gs = prow.get("gs")
                    if gs is not None:
                        pitcher_feats["opp_is_starter"] = int(gs)

            # opposing pitcher handedness for platoon
            opp_throws = game_row.get(f'sp_throws_{"v" if hv == "h" else "h"}', "")

            # ── iterate lineup slots 1-9 ──────────────────────────────────
            for slot in range(1, 10):
                b_id_raw = game_row.get(f"batter{slot}_id_{hv}")
                if b_id_raw is None or pd.isna(b_id_raw):
                    continue
                b_id = str(int(b_id_raw))

                brow = batter_data_dict.get(b_id)
                if brow is None:
                    continue

                # activity / recency filter
                if not _check_batter_active(brow):
                    continue

                stand = str(brow.get("stand", ""))

                # matchup: stand + opp_throws → "RR", "RL", "LR", "LL"
                # must be lowercase to match model feature names
                matchup = (stand + str(opp_throws)) if stand and opp_throws else "??"

                # ── batter rolling features ───────────────────────────────
                # All keys lowercase to match XGBoost trained feature names
                batter_feats = {}
                for w in WINDOWS_BAT:
                    for stem in [
                        "barrel",
                        "ev",
                        "hardhit",
                        "swspot",
                        "slg",
                        "obp",
                        "obs",
                        "est_woba",
                        "est_slg",
                    ]:
                        val = brow.get(f"{stem}_{w}", np.nan)
                        batter_feats[f"{stem}_{w}"] = (
                            float(val)
                            if val is not None and not pd.isna(val)
                            else np.nan
                        )

                    for out_key, sf_col in [
                        (f"hr_per_pa_{w}", f"hr_per_pa_{w}"),
                        (f"hr_per_pa_vs_r_{w}", f"hr_per_pa_vs_r_{w}"),
                        (f"hr_per_pa_vs_l_{w}", f"hr_per_pa_vs_l_{w}"),
                    ]:
                        val = brow.get(sf_col, np.nan)
                        batter_feats[out_key] = (
                            float(val)
                            if val is not None and not pd.isna(val)
                            else np.nan
                        )

                batter_feats["age"] = float(brow.get("age", np.nan) or np.nan)
                batter_feats["is_home"] = int(hv == "h")
                batter_feats["matchup"] = matchup

                rows.append(
                    {
                        "date_dblhead": game_row.get("date_dblhead"),
                        "b_id": b_id,
                        "slot": slot,
                        "team": game_row.get(f"team_{hv}"),
                        "opponent": game_row.get(f'team_{"v" if hv == "h" else "h"}'),
                        "stand": stand,
                        "opp_throws": opp_throws,
                        "park_hr_factor": PARK_HR_FACTORS.get(
                            game_row.get("team_h"), 100
                        ),
                        "temp": game_row.get("temp", 72),
                        "humidity": game_row.get("humidity", 50),
                        "wind_spd": game_row.get("wind_spd", 0),
                        "wind_out": game_row.get("wind_out", 0),
                        **batter_feats,
                        **pitcher_feats,
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
