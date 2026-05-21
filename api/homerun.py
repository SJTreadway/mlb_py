import pandas as pd
import numpy as np
from helpers import get_park_factors_map

import datetime

CURRENT_YEAR = datetime.date.today().year
SEASON_START = int(f"{CURRENT_YEAR}0401")
CUTOFF = int((datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y%m%d"))

PARK_HR_FACTORS = get_park_factors_map()

WINDOWS_BAT = [7, 14, 30, 75, 162, 350]
WINDOWS_PITCH = [10, 35, 75]


def process_homerun_data(df, batter_data_dict, pitcher_data_dict):
    """
    Build one row per batter-game for HR model prediction.
    df: main game dataframe with weather, park, bullpen features
    batter_data_dict: dict of mlbam_id -> processed batter DataFrame
    pitcher_data_dict: dict of mlbam_id -> processed pitcher DataFrame
    """
    rows = []

    for _, game_row in df.iterrows():
        date_dblhead = game_row["date_dblhead"]
        sp_id_h = game_row["starting_pitcher_id_h"]
        sp_id_v = game_row["starting_pitcher_id_v"]

        for hv in ["h", "v"]:
            opposing_sp = sp_id_v if hv == "h" else sp_id_h

            # get opposing pitcher features
            pitcher_feats = {
                **{f"opp_HR_per_BF_{w}": 0 for w in WINDOWS_PITCH},
                **{f"opp_FB_perc_{w}": 0 for w in WINDOWS_PITCH},
            }
            if opposing_sp is not None and not pd.isna(opposing_sp):
                opp_sp_key = str(int(opposing_sp))
                pdf = pitcher_data_dict.get(opp_sp_key)
                if pdf is not None and not pdf.empty:
                    prev = pdf[pdf.index <= str(int(date_dblhead))]
                    if not prev.empty:
                        prow = prev.iloc[-1]
                        for winsize in WINDOWS_PITCH:
                            pitcher_feats[f"opp_HR_per_BF_{winsize}"] = prow.get(
                                f"HR_per_BF_{winsize}", 0
                            )
                            pitcher_feats[f"opp_FB_perc_{winsize}"] = prow.get(
                                f"FB_perc_{winsize}", 0
                            )

            # get opposing pitcher handedness for platoon
            opp_throws = game_row.get(f'sp_throws_{"v" if hv == "h" else "h"}', "")

            # iterate batters 1-9
            for slot in range(1, 10):
                b_id_raw = game_row.get(f"batter{slot}_id_{hv}")
                if b_id_raw is None or pd.isna(b_id_raw):
                    continue
                b_id = str(int(b_id_raw))

                bdf = batter_data_dict.get(b_id)
                if bdf is None or bdf.empty:
                    continue

                prev = bdf[bdf.index <= date_dblhead]
                if prev.empty:
                    continue
                # prevent guys from showing up with no recent activity
                regular_season = prev[prev["date"] >= SEASON_START]
                if regular_season.empty or regular_season["date"].iloc[-1] < CUTOFF:
                    continue
                if len(regular_season) < 5:  # minimum 5 regular season games
                    continue
                last_365_cutoff = int(
                    (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y%m%d")
                )
                last_365 = prev[prev["date"] >= last_365_cutoff]

                if len(last_365) < 10:  # minimum 10 games in last 365 days
                    continue
                brow = prev.iloc[-1]

                # batter features
                batter_feats = {}
                stand = str(brow.get("stand", ""))

                for winsize in WINDOWS_BAT:
                    for stem in [
                        "BARREL",
                        "EV",
                        "HARDHIT",
                        "SWSPOT",
                        "SLG",
                        "OBP",
                        "OBS",
                        "est_woba",
                        "est_slg",
                    ]:
                        batter_feats[f"{stem}_{winsize}"] = brow.get(
                            f"{stem}_{winsize}", np.nan
                        )
                    batter_feats[f"HR_per_PA_{winsize}"] = brow.get(
                        f"HR_per_PA_{winsize}", np.nan
                    )
                    batter_feats[f"HR_per_PA_vs_R_{winsize}"] = brow.get(
                        f"HR_per_PA_vs_R_{winsize}", np.nan
                    )
                    batter_feats[f"HR_per_PA_vs_L_{winsize}"] = brow.get(
                        f"HR_per_PA_vs_L_{winsize}", np.nan
                    )

                batter_feats["age"] = brow.get("age", np.nan)
                batter_feats["days_rest"] = brow.get("days_rest", np.nan)
                batter_feats["is_home"] = brow.get("is_home", np.nan)

                row = {
                    "date_dblhead": date_dblhead,
                    "b_id": b_id,
                    "slot": slot,
                    "team": game_row[f"team_{hv}"],
                    "opponent": game_row[f'team_{"v" if hv == "h" else "h"}'],
                    "stand": stand,
                    "opp_throws": opp_throws,
                    "park_hr_factor": PARK_HR_FACTORS.get(game_row["team_h"], 100),
                    "temp": game_row.get("temp", 72),
                    "humidity": game_row.get("humidity", 50),
                    "wind_spd": game_row.get("wind_spd", 0),
                    "wind_out": game_row.get("wind_out", 0),
                    **batter_feats,
                    **pitcher_feats,
                }
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
