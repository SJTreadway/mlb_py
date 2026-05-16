import warnings

warnings.filterwarnings("ignore")

import os
import time
import requests
import numpy as np
import pandas as pd
import joblib
from datetime import date
from pybaseball import statcast_batter

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/hr_model_2026v1.pkl"
ODDS_API_KEY = os.environ["ODDS_API_KEY"]
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MIN_EDGE = 0.04
API_SLEEP = float(os.environ.get("API_SLEEP", "0.5"))

WINDOWS_BAT = [30, 75, 162, 350]
WINDOWS_PITCH = [10, 35, 75]

NON_AB_EVENTS = [
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_bunt",
    "sac_fly",
    "sac_fly_error",
    "catcher_interf",
]

PARK_HR_FACTORS = {
    "COL": 112,
    "CIN": 103,
    "TEX": 92,
    "BAL": 103,
    "PHI": 102,
    "BOS": 102,
    "MIL": 97,
    "ATL": 100,
    "NYY": 102,
    "TOR": 101,
    "HOU": 101,
    "LAD": 102,
    "CHC": 95,
    "STL": 98,
    "MIN": 103,
    "ARI": 104,
    "DET": 101,
    "CLE": 98,
    "LAA": 100,
    "WSH": 101,
    "WSN": 101,
    "KCR": 100,
    "KC": 100,
    "MIA": 100,
    "NYM": 99,
    "ATH": 100,
    "OAK": 97,
    "PIT": 100,
    "SDP": 97,
    "SD": 97,
    "SEA": 92,
    "SFG": 98,
    "SF": 98,
    "TBR": 95,
    "TB": 95,
    "CHW": 99,
    "CWS": 99,
    "LAS": 100,
}

# ── Today's Slate ─────────────────────────────────────────────────────────────


def get_todays_slate():
    """Pull today's games, lineups, and probable pitchers from MLB Stats API."""
    today = date.today().strftime("%Y-%m-%d")
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": today,
        "hydrate": "team,probablePitcher,lineups",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    games = []
    for date_obj in resp.json().get("dates", []):
        for game in date_obj.get("games", []):
            home = game["teams"]["home"]
            away = game["teams"]["away"]

            home_lineup = [
                p["id"] for p in game.get("lineups", {}).get("homePlayers", [])
            ]
            away_lineup = [
                p["id"] for p in game.get("lineups", {}).get("awayPlayers", [])
            ]

            games.append(
                {
                    "game_pk": game["gamePk"],
                    "game_time": game["gameDate"],
                    "team_h": home["team"]["abbreviation"],
                    "team_v": away["team"]["abbreviation"],
                    "sp_mlbam_h": home.get("probablePitcher", {}).get("id"),
                    "sp_mlbam_v": away.get("probablePitcher", {}).get("id"),
                    "sp_name_h": home.get("probablePitcher", {}).get("fullName", ""),
                    "sp_name_v": away.get("probablePitcher", {}).get("fullName", ""),
                    "sp_throws_h": None,  # fetched separately
                    "sp_throws_v": None,
                    "lineup_h": home_lineup,
                    "lineup_v": away_lineup,
                }
            )

    # assign doubleheader numbers
    seen = {}
    for game in games:
        key = game["team_h"]
        count = seen.get(key, 0)
        game["dblhead_num"] = count
        game["date_dblhead"] = int(date.today().strftime("%Y%m%d") + str(count))
        seen[key] = count + 1

    print(f"Found {len(games)} games for {today}")
    return games


def get_pitcher_handedness(mlbam_id):
    """Fetch pitcher throws (L/R) from MLB Stats API."""
    if not mlbam_id:
        return ""
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}",
            params={"fields": "people,pitchHand,code"},
            timeout=10,
        )
        resp.raise_for_status()
        people = resp.json().get("people", [])
        if people:
            return people[0].get("pitchHand", {}).get("code", "")
    except Exception:
        pass
    return ""


def enrich_pitcher_handedness(slate):
    """Add pitcher handedness to each game."""
    pitcher_ids = set()
    for game in slate:
        if game["sp_mlbam_h"]:
            pitcher_ids.add(game["sp_mlbam_h"])
        if game["sp_mlbam_v"]:
            pitcher_ids.add(game["sp_mlbam_v"])

    throws_map = {}
    for pid in pitcher_ids:
        throws_map[pid] = get_pitcher_handedness(pid)
        time.sleep(0.1)

    for game in slate:
        game["sp_throws_h"] = throws_map.get(game["sp_mlbam_h"], "")
        game["sp_throws_v"] = throws_map.get(game["sp_mlbam_v"], "")

    return slate


# ── Statcast Batter Pull ──────────────────────────────────────────────────────


def transform_statcast_to_game_level(df):
    """Transform pitch-level Statcast to one row per game."""
    if df.empty:
        return pd.DataFrame()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    games = []

    for (game_date, game_pk), group in df.groupby(["game_date", "game_pk"]):
        pa_endings = group[group["events"].notna() & (group["events"] != "")].copy()
        if pa_endings.empty:
            continue

        pa_endings["runs_scored"] = (
            pa_endings["post_bat_score"] - pa_endings["bat_score"]
        ).clip(lower=0)

        is_home = group["inning_topbot"].iloc[0] == "Bot"
        opponent = group["away_team"].iloc[0] if is_home else group["home_team"].iloc[0]
        p_throws = group["p_throws"].iloc[0] if "p_throws" in group.columns else ""
        stand = group["stand"].iloc[0] if "stand" in group.columns else ""
        favorable_platoon = int(stand != p_throws and stand != "" and p_throws != "")

        ab = len(pa_endings[~pa_endings["events"].isin(NON_AB_EVENTS)])
        h = len(
            pa_endings[
                pa_endings["events"].isin(["single", "double", "triple", "home_run"])
            ]
        )
        x2b = len(pa_endings[pa_endings["events"] == "double"])
        x3b = len(pa_endings[pa_endings["events"] == "triple"])
        hr = len(pa_endings[pa_endings["events"] == "home_run"])
        bb = len(pa_endings[pa_endings["events"].isin(["walk", "intent_walk"])])
        hbp = len(pa_endings[pa_endings["events"] == "hit_by_pitch"])
        sf = len(pa_endings[pa_endings["events"] == "sac_fly"])

        batted = pa_endings[pa_endings["launch_speed"].notna()].copy()
        n_batted = len(batted)
        ev_sum = float(batted["launch_speed"].sum())
        hard_hits = int((batted["launch_speed"] >= 95).sum())
        sweet_spots = int(
            ((batted["launch_angle"] >= 8) & (batted["launch_angle"] <= 32)).sum()
        )
        barrels = (
            int((batted["launch_speed_angle"] == 6).sum())
            if "launch_speed_angle" in batted.columns
            else 0
        )
        est_woba = (
            float(batted["estimated_woba_using_speedangle"].dropna().mean())
            if "estimated_woba_using_speedangle" in batted.columns
            else np.nan
        )
        est_slg = (
            float(batted["estimated_slg_using_speedangle"].dropna().mean())
            if "estimated_slg_using_speedangle" in batted.columns
            else np.nan
        )

        age = (
            float(group["age_bat"].dropna().iloc[0])
            if "age_bat" in group.columns and group["age_bat"].notna().any()
            else np.nan
        )

        opp_pitcher_id = (
            int(group["pitcher"].iloc[0]) if "pitcher" in group.columns else None
        )

        games.append(
            {
                "game_date": game_date,
                "game_pk": game_pk,
                "opponent": opponent,
                "is_home": int(is_home),
                "stand": stand,
                "p_throws": p_throws,
                "favorable_platoon": favorable_platoon,
                "opp_pitcher_id": opp_pitcher_id,
                "age": age,
                "AB": ab,
                "H": h,
                "x2B": x2b,
                "x3B": x3b,
                "HR": hr,
                "BB": bb,
                "HBP": hbp,
                "SF": sf,
                "HR_vs_R": hr if p_throws == "R" else 0,
                "AB_vs_R": ab if p_throws == "R" else 0,
                "HR_vs_L": hr if p_throws == "L" else 0,
                "AB_vs_L": ab if p_throws == "L" else 0,
                "batted_balls": n_batted,
                "ev_sum": ev_sum,
                "hard_hits": hard_hits,
                "sweet_spots": sweet_spots,
                "barrels": barrels,
                "est_woba": est_woba,
                "est_slg": est_slg,
            }
        )

    return pd.DataFrame(games).sort_values("game_date").reset_index(drop=True)


def pull_batter_history(mlbam_id, years=2):
    """Pull recent Statcast history for a batter."""
    current_year = date.today().year
    all_seasons = []
    for year in range(current_year - years, current_year + 1):
        try:
            df = statcast_batter(f"{year}-03-01", f"{year}-11-30", mlbam_id)
            if df is not None and not df.empty:
                all_seasons.append(transform_statcast_to_game_level(df))
            time.sleep(API_SLEEP)
        except Exception:
            continue

    if not all_seasons:
        return pd.DataFrame()
    return pd.concat(all_seasons, ignore_index=True).sort_values("game_date")


# ── Rolling Features ──────────────────────────────────────────────────────────


def rolling_sum(df, col, winsize):
    return df[col].rolling(window=winsize, min_periods=1).sum().shift(1)


def add_batter_rolling_features(df):
    df = df.sort_values("game_date").reset_index(drop=True)

    bat_stat_cols = [
        "HR",
        "AB",
        "BB",
        "H",
        "HBP",
        "SF",
        "x2B",
        "x3B",
        "HR_vs_R",
        "AB_vs_R",
        "HR_vs_L",
        "AB_vs_L",
        "barrels",
        "ev_sum",
        "hard_hits",
        "sweet_spots",
        "batted_balls",
        "est_woba",
        "est_slg",
    ]

    new_cols = {}
    for winsize in WINDOWS_BAT:
        for col in bat_stat_cols:
            if col in df.columns:
                new_cols[f"rollsum_{col}_{winsize}"] = rolling_sum(
                    df, col, winsize
                ).values

    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    for winsize in WINDOWS_BAT:

        def g(col):
            return pd.Series(
                new_cols.get(f"rollsum_{col}_{winsize}", np.zeros(len(df))),
                index=df.index,
            )

        ab = g("AB")
        hr = g("HR")
        h = g("H")
        bb = g("BB")
        hbp = g("HBP")
        sf = g("SF")
        x2b = g("x2B")
        x3b = g("x3B")
        bbd = g("batted_balls")
        evs = g("ev_sum")
        hh = g("hard_hits")
        ss = g("sweet_spots")
        bar = g("barrels")
        hr_r = g("HR_vs_R")
        ab_r = g("AB_vs_R")
        hr_l = g("HR_vs_L")
        ab_l = g("AB_vs_L")

        ab_denom = ab.replace(0, np.nan)
        pa_denom = (ab + bb + hbp + sf).replace(0, np.nan)
        batted_denom = bbd.replace(0, np.nan)

        df[f"HR_per_PA_{winsize}"] = hr / pa_denom
        df[f"SLG_{winsize}"] = (h + x2b + 2 * x3b + 3 * hr) / ab_denom
        df[f"OBP_{winsize}"] = (h + bb + hbp) / pa_denom
        df[f"OBS_{winsize}"] = df[f"SLG_{winsize}"] + df[f"OBP_{winsize}"]
        df[f"EV_{winsize}"] = evs / batted_denom
        df[f"HARDHIT_{winsize}"] = hh / batted_denom
        df[f"SWSPOT_{winsize}"] = ss / batted_denom
        df[f"BARREL_{winsize}"] = bar / batted_denom
        df[f"HR_per_PA_vs_R_{winsize}"] = hr_r / ab_r.replace(0, np.nan)
        df[f"HR_per_PA_vs_L_{winsize}"] = hr_l / ab_l.replace(0, np.nan)
        df[f"est_woba_{winsize}"] = g("est_woba") / batted_denom
        df[f"est_slg_{winsize}"] = g("est_slg") / batted_denom

    return df


# ── Pitcher Feature Lookup ────────────────────────────────────────────────────


def get_pitcher_feats_from_statcast(mlbam_id, as_of_date):
    """Pull pitcher rolling stats from Statcast up to as_of_date."""
    defaults = {
        **{f"opp_HR_per_BF_{w}": 0 for w in WINDOWS_PITCH},
        **{f"opp_FB_perc_{w}": 0 for w in WINDOWS_PITCH},
    }
    if not mlbam_id:
        return defaults

    try:
        from pybaseball import statcast_pitcher

        current_year = date.today().year
        seasons = []
        for year in range(current_year - 2, current_year + 1):
            df = statcast_pitcher(f"{year}-03-01", f"{year}-11-30", int(mlbam_id))
            if df is not None and not df.empty:
                seasons.append(df)
            time.sleep(API_SLEEP)

        if not seasons:
            return defaults

        df = pd.concat(seasons, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df[df["game_date"] < pd.Timestamp(as_of_date)]
        if df.empty:
            return defaults

        games = []
        for (game_date, game_pk), group in df.groupby(["game_date", "game_pk"]):
            pa_endings = group[group["events"].notna() & (group["events"] != "")].copy()
            if pa_endings.empty:
                continue
            batted = pa_endings[pa_endings["launch_speed"].notna()]
            n_batted = len(batted)
            fly_balls = (
                int((batted["bb_type"] == "fly_ball").sum())
                if "bb_type" in batted.columns
                else 0
            )
            games.append(
                {
                    "game_date": game_date,
                    "BFP": len(pa_endings),
                    "HR": len(pa_endings[pa_endings["events"] == "home_run"]),
                    "fly_balls": fly_balls,
                    "batted_balls_allowed": n_batted,
                }
            )

        gdf = pd.DataFrame(games).sort_values("game_date").reset_index(drop=True)

        new_cols = {}
        for winsize in WINDOWS_PITCH:
            for col in ["HR", "BFP", "fly_balls", "batted_balls_allowed"]:
                if col in gdf.columns:
                    new_cols[f"rollsum_{col}_{winsize}"] = rolling_sum(
                        gdf, col, winsize
                    ).values

        result = {}
        for winsize in WINDOWS_PITCH:
            hr = new_cols.get(f"rollsum_HR_{winsize}", np.zeros(len(gdf)))
            bf = new_cols.get(f"rollsum_BFP_{winsize}", np.zeros(len(gdf)))
            fb = new_cols.get(f"rollsum_fly_balls_{winsize}", np.zeros(len(gdf)))
            bat = new_cols.get(
                f"rollsum_batted_balls_allowed_{winsize}", np.zeros(len(gdf))
            )

            bf_denom = np.where(bf == 0, np.nan, bf)
            bat_denom = np.where(bat == 0, np.nan, bat)

            result[f"opp_HR_per_BF_{winsize}"] = float(np.nanmean(hr / bf_denom))
            result[f"opp_FB_perc_{winsize}"] = float(np.nanmean(fb / bat_denom))

        return result

    except Exception as e:
        print(f"Error fetching pitcher {mlbam_id}: {e}")
        return defaults


# ── Build Prediction Rows ─────────────────────────────────────────────────────


def build_prediction_rows(slate):
    """Build one feature row per batter for today's slate."""
    today = date.today()
    rows = []
    all_mlbam_ids = set()

    for game in slate:
        all_mlbam_ids.update(game["lineup_h"])
        all_mlbam_ids.update(game["lineup_v"])

    print(f"Pulling Statcast history for {len(all_mlbam_ids)} batters...")
    batter_histories = {}
    for mlbam_id in all_mlbam_ids:
        hist = pull_batter_history(mlbam_id, years=2)
        if not hist.empty:
            batter_histories[mlbam_id] = add_batter_rolling_features(hist)

    print("Fetching pitcher features...")
    pitcher_cache = {}

    for game in slate:
        for hv in ["h", "v"]:
            lineup = game[f"lineup_{hv}"]
            opp_sp_id = game[f'sp_mlbam_{"v" if hv == "h" else "h"}']
            opp_throws = game[f'sp_throws_{"v" if hv == "h" else "h"}']
            home_team = game["team_h"]
            batting_team = game[f"team_{hv}"]

            # get pitcher features
            if opp_sp_id and opp_sp_id not in pitcher_cache:
                pitcher_cache[opp_sp_id] = get_pitcher_feats_from_statcast(
                    opp_sp_id, today
                )
            pitcher_feats = pitcher_cache.get(
                opp_sp_id,
                {
                    **{f"opp_HR_per_BF_{w}": 0 for w in WINDOWS_PITCH},
                    **{f"opp_FB_perc_{w}": 0 for w in WINDOWS_PITCH},
                },
            )

            for slot, mlbam_id in enumerate(lineup, 1):
                bdf = batter_histories.get(mlbam_id)
                if bdf is None or bdf.empty:
                    continue

                # use most recent row as pre-game features
                brow = bdf.iloc[-1]
                stand = str(brow.get("stand", ""))

                row = {
                    "mlbam_id": mlbam_id,
                    "batting_team": batting_team,
                    "opponent": game[f'team_{"v" if hv == "h" else "h"}'],
                    "home_team": home_team,
                    "slot": slot,
                    "stand": stand,
                    "opp_pitcher_id": opp_sp_id,
                    "opp_pitcher_name": game[f'sp_name_{"v" if hv == "h" else "h"}'],
                    "opp_throws": opp_throws,
                    "park_hr_factor": PARK_HR_FACTORS.get(home_team, 100),
                    "age": brow.get("age", np.nan),
                    **pitcher_feats,
                }

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
                        row[f"{stem}_{winsize}"] = brow.get(f"{stem}_{winsize}", np.nan)
                    row[f"HR_per_PA_{winsize}"] = brow.get(
                        f"HR_per_PA_{winsize}", np.nan
                    )
                    row[f"HR_per_PA_vs_R_{winsize}"] = brow.get(
                        f"HR_per_PA_vs_R_{winsize}", np.nan
                    )
                    row[f"HR_per_PA_vs_L_{winsize}"] = brow.get(
                        f"HR_per_PA_vs_L_{winsize}", np.nan
                    )

                rows.append(row)

    return pd.DataFrame(rows)


# ── Odds Pull ─────────────────────────────────────────────────────────────────


def get_hr_prop_odds():
    """Pull today's HR prop odds from the-odds-api.com."""
    # step 1: get event IDs
    events_resp = requests.get(
        f"{ODDS_API_BASE}/sports/baseball_mlb/events",
        params={"apiKey": ODDS_API_KEY, "dateFormat": "iso"},
        timeout=15,
    )
    events_resp.raise_for_status()
    events = events_resp.json()

    all_props = []
    for event in events:
        event_id = event["id"]
        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds",
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": "us",
                    "markets": "batter_home_runs",
                    "oddsFormat": "american",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            for bookmaker in data.get("bookmakers", []):
                book = bookmaker["key"]
                for market in bookmaker.get("markets", []):
                    if market["key"] != "batter_home_runs":
                        continue
                    for outcome in market.get("outcomes", []):
                        # only over 0.5 = hit at least 1 HR
                        if outcome.get("point", 0.5) != 0.5:
                            continue
                        if outcome.get("name", "").lower() != "over":
                            continue
                        all_props.append(
                            {
                                "player_name": outcome["description"],
                                "book": book,
                                "american_odds": outcome["price"],
                                "home_team": event["home_team"],
                                "away_team": event["away_team"],
                            }
                        )
        except Exception as e:
            print(f"Error fetching odds for event {event_id}: {e}")
            continue

    return pd.DataFrame(all_props)


def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def get_best_odds(odds_df):
    """Get best available odds per player across all books."""
    if odds_df.empty:
        return pd.DataFrame()
    odds_df["implied_prob"] = odds_df["american_odds"].apply(american_to_implied)
    # best odds = lowest implied probability (highest payout)
    return odds_df.sort_values("implied_prob").drop_duplicates(subset="player_name")[
        ["player_name", "book", "american_odds", "implied_prob"]
    ]


# ── Name Matching ─────────────────────────────────────────────────────────────


def get_player_name_map(mlbam_ids):
    """Fetch player names from MLB Stats API for MLBAM ID matching."""
    name_map = {}
    chunk_size = 200
    for i in range(0, len(mlbam_ids), chunk_size):
        chunk = mlbam_ids[i : i + chunk_size]
        try:
            resp = requests.get(
                "https://statsapi.mlb.com/api/v1/people",
                params={
                    "personIds": ",".join(str(x) for x in chunk),
                    "fields": "people,id,fullName",
                },
                timeout=15,
            )
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                name_map[person["id"]] = person["fullName"]
        except Exception as e:
            print(f"Error fetching names: {e}")
    return name_map


def normalize_name(name):
    """Normalize player name for fuzzy matching."""
    return name.lower().strip().replace(".", "").replace("'", "").replace("-", " ")


def match_odds_to_predictions(pred_df, odds_df, name_map):
    """Join odds to predictions by player name."""
    pred_df = pred_df.copy()
    pred_df["full_name"] = pred_df["mlbam_id"].map(name_map)
    pred_df["name_normalized"] = pred_df["full_name"].apply(
        lambda x: normalize_name(str(x)) if pd.notna(x) else ""
    )

    odds_df = odds_df.copy()
    odds_df["name_normalized"] = odds_df["player_name"].apply(
        lambda x: normalize_name(str(x)) if pd.notna(x) else ""
    )

    return pred_df.merge(odds_df, on="name_normalized", how="inner")


# ── Predict & Find Edges ──────────────────────────────────────────────────────


def predict_hr_probs(pred_df, model_path=MODEL_PATH):
    """Run HR probability predictions."""
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]

    # fill missing features with 0
    for col in features:
        if col not in pred_df.columns:
            pred_df[col] = 0
        pred_df[col] = pred_df[col].fillna(0)

    pred_df["hr_prob"] = model.predict_proba(pred_df[features])[:, 1]
    return pred_df


def find_edges(merged_df, min_edge=MIN_EDGE):
    """Find bets with positive edge vs implied odds."""
    merged_df = merged_df.copy()
    merged_df["edge"] = merged_df["hr_prob"] - merged_df["implied_prob"]
    edges = merged_df[merged_df["edge"] >= min_edge].copy()
    return edges.sort_values("edge", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print(f"HR Prediction Pipeline — {date.today()}")
    print("=" * 60)

    # step 1: get today's slate
    print("\n[1/6] Fetching today's slate...")
    slate = get_todays_slate()
    if not slate:
        print("No games today.")
        return

    # step 2: get pitcher handedness
    print("\n[2/6] Fetching pitcher handedness...")
    slate = enrich_pitcher_handedness(slate)

    # step 3: build prediction rows
    print("\n[3/6] Building batter feature rows...")
    pred_df = build_prediction_rows(slate)
    print(f"Built {len(pred_df)} batter-game rows")

    if pred_df.empty:
        print("No prediction rows built — lineups may not be posted yet.")
        return

    # step 4: run model
    print("\n[4/6] Running model...")
    pred_df = predict_hr_probs(pred_df)

    # step 5: get player names and odds
    print("\n[5/6] Fetching odds...")
    name_map = get_player_name_map(pred_df["mlbam_id"].unique().tolist())
    odds_df = get_hr_prop_odds()
    best_odds = get_best_odds(odds_df)
    print(f"Found HR props for {len(best_odds)} players")

    # step 6: find edges
    print("\n[6/6] Finding edges...")
    merged = match_odds_to_predictions(pred_df, best_odds, name_map)
    edges = find_edges(merged, min_edge=MIN_EDGE)

    # ── Output ────────────────────────────────────────────────────────────────

    print(f'\n{"=" * 60}')
    print(f"TODAY'S HR BETS (edge >= {MIN_EDGE*100:.0f}%)")
    print(f'{"=" * 60}')

    if edges.empty:
        print("No edges found today.")
    else:
        display_cols = [
            "full_name",
            "batting_team",
            "opponent",
            "opp_pitcher_name",
            "opp_throws",
            "stand",
            "hr_prob",
            "implied_prob",
            "edge",
            "american_odds",
            "book",
            "park_hr_factor",
            "BARREL_162",
            "HR_per_PA_350",
        ]
        display_cols = [c for c in display_cols if c in edges.columns]
        print(edges[display_cols].to_string(index=False))

    # also print full ranked list
    print(f'\n{"=" * 60}')
    print("FULL RANKED LIST (top 20)")
    print(f'{"=" * 60}')

    ranked = pred_df.copy()
    ranked["full_name"] = ranked["mlbam_id"].map(name_map)
    ranked = ranked.sort_values("hr_prob", ascending=False)

    rank_cols = [
        "full_name",
        "batting_team",
        "opponent",
        "opp_pitcher_name",
        "stand",
        "opp_throws",
        "hr_prob",
        "park_hr_factor",
        "BARREL_162",
        "HR_per_PA_350",
    ]
    rank_cols = [c for c in rank_cols if c in ranked.columns]
    print(ranked[rank_cols].head(20).to_string(index=False))

    # save outputs
    pred_df["full_name"] = pred_df["mlbam_id"].map(name_map)
    pred_df.to_csv(f"data/hr_predictions_{date.today()}.csv", index=False)
    if not edges.empty:
        edges.to_csv(f"data/hr_edges_{date.today()}.csv", index=False)
    print(f"\nSaved predictions to data/hr_predictions_{date.today()}.csv")


if __name__ == "__main__":
    main()
