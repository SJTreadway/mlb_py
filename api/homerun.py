import pandas as pd
from helpers import get_park_factors_map

PARK_HR_FACTORS = get_park_factors_map()


def process_homerun_data(df, batter_data_dict, pitcher_data_dict):
    """
    Build one row per batter-game for HR model.
    df: your main game dataframe (already has weather, park, bullpen features)
    """
    rows = []

    for _, game_row in df.iterrows():
        date_dblhead = game_row["date_dblhead"]
        sp_id_h = game_row["starting_pitcher_id_h"]
        sp_id_v = game_row["starting_pitcher_id_v"]

        for hv in ["h", "v"]:
            sp_id = sp_id_h if hv == "h" else sp_id_v
            opposing_sp = sp_id_v if hv == "h" else sp_id_h

            # get opposing pitcher features
            pitcher_feats = {}
            if opposing_sp in pitcher_data_dict:
                pdf = pitcher_data_dict[opposing_sp]
                if pdf is not None and not pdf.empty:
                    prev = pdf[pdf.index <= date_dblhead]
                    if not prev.empty:
                        prow = prev.iloc[-1]
                        for winsize in [10, 35, 75]:
                            pitcher_feats[f"opp_HR_per_BF_{winsize}"] = prow.get(
                                f"HR_per_BF_{winsize}", 0
                            )
                            pitcher_feats[f"opp_FB_perc_{winsize}"] = prow.get(
                                f"FB_perc_{winsize}", 0
                            )
                        pitcher_feats["opp_throws"] = prow.get("throws", "R")

            # iterate batters 1-9
            for slot in range(1, 10):
                b_id = game_row[f"batter{slot}_id_{hv}"]
                if not b_id or pd.isna(b_id):
                    continue

                bdf = batter_data_dict.get(b_id)
                if bdf is None or bdf.empty:
                    continue

                prev = bdf[bdf.index <= date_dblhead]
                if prev.empty:
                    continue
                brow = prev.iloc[-1]

                # batter features
                batter_feats = {}
                for winsize in [30, 75, 162, 350]:
                    for stem in [
                        "BARREL",
                        "EV",
                        "HARDHIT",
                        "SWSPOT",
                        "OBP",
                        "SLG",
                        "OBS",
                        "SObat_perc",
                    ]:
                        batter_feats[f"{stem}_{winsize}"] = brow.get(
                            f"{stem}_{winsize}", 0
                        )
                    batter_feats[f"rollsum_HR_{winsize}"] = brow.get(
                        f"rollsum_HR_{winsize}", 0
                    )
                    batter_feats[f"rollsum_AB_{winsize}"] = brow.get(
                        f"rollsum_AB_{winsize}", 0
                    )

                # HR/PA
                for winsize in [30, 75, 162, 350]:
                    ab = batter_feats[f"rollsum_AB_{winsize}"]
                    hr = batter_feats[f"rollsum_HR_{winsize}"]
                    batter_feats[f"HR_per_PA_{winsize}"] = hr / ab if ab > 0 else 0

                # platoon split
                stand = brow.get("stand", "R")
                opp_throws = pitcher_feats.get("opp_throws", "R")
                favorable_platoon = int(stand != opp_throws)

                row = {
                    "date_dblhead": date_dblhead,
                    "b_id": b_id,
                    "slot": slot,
                    "team": game_row[f"team_{hv}"],
                    "opponent": game_row[f'team_{"v" if hv == "h" else "h"}'],
                    "park_hr_factor": PARK_HR_FACTORS.get(game_row["team_h"], 100),
                    "temp": game_row.get("temp", 72),
                    "humidity": game_row.get("humidity", 50),
                    "favorable_platoon": favorable_platoon,
                    "batting_slot": slot,
                    **batter_feats,
                    **pitcher_feats,
                    # target
                    "hit_hr": int(brow.get("HR", 0) > 0),
                }
                rows.append(row)

    return pd.DataFrame(rows)
