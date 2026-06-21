#!/usr/bin/env python3

import os
import math
import warnings

# Silence all performance warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import pandas as pd
import requests

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# Set options to display all columns and adjust the width
pd.set_option("display.max_columns", None)  # Display all columns
pd.set_option("display.width", 0)  # Automatically adjust to terminal width

import logging

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

from datetime import date, timedelta, datetime, timezone
import pytz
import pickle
import webbrowser
import pathlib

from ui.dashboard import generate_html

from api.teams import (
    generate_team_window_features,
    add_team_form_features,
    assemble_win_model_features,
)
from api.lineups import get_lineups, get_run_total_feats
from api.odds import (
    get_over_odds,
    get_under_odds,
    get_total_line,
    get_money_line_price,
    calculate_edge,
)
from api.pitchers import process_pitching_data
from api.batters import process_batting_data
from api.weather import process_weather_data
from api.homerun import process_homerun_data


from cleanup import cleanup_directory
from simulator_inference import predict_game

import tweepy

from dotenv import load_dotenv

load_dotenv()

DISPLAY_EDGE_ONLY = 1
EDGE_THRESHOLD_H = 4.0
EDGE_THRESHOLD_V = 4.0
HR_PROB_THRESHOLD = 0.20  # model must believe in the guy
HR_EDGE_THRESHOLD = 2.0  # at least 2pp of edge over implied

# Force Refresh Data
REFRESH_DATA = int(os.environ.get("REFRESH_DATA", 0))

# X essentials
ACCESS_KEY = os.environ["X_ACCESS_KEY"]
ACCESS_SECRET = os.environ["X_ACCESS_SECRET"]
CONSUMER_KEY = os.environ["X_CONSUMER_KEY"]
CONSUMER_SECRET = os.environ["X_CONSUMER_SECRET"]
BEARER_TOKEN = os.environ["X_BEARER_TOKEN"]

# Flags for Settings
TOMORROW_GAMES = int(os.environ["TOMORROW_GAMES"])

RUN_DATE = date.today() if TOMORROW_GAMES == 0 else date.today() + timedelta(days=1)

# File locations
WINS_MODEL_FILE = "models/win_model_2026v1.pkl"
RUNS_MODEL_FILE = "models/runs_scored_model_v1.pkl"
HR_MODEL_FILE = "models/homerun_model_2026v1.pkl"
BATTER_DICT_FILE = f"data/daily/{RUN_DATE}_batter_dict.pkl"
PITCHER_DICT_FILE = f"data/daily/{RUN_DATE}_pitcher_dict.pkl"
BVP_DICT_FILE = f"data/daily/{RUN_DATE}_bvp_dict.pkl"
HR_ODDS_CACHE_FILE = f"data/daily/{RUN_DATE}_hr_odds_cache.pkl"
PREDS_FILE = f"data/results/{RUN_DATE}_home_victory_preds.csv"
DASHBOARD_FILE = f"data/results/{RUN_DATE}_dashboard.html"

# Simulator is now the PRIMARY win model — prob/picks/edge all come from it.
# XGBoost still runs every time too (cheap — one predict_proba call, not a
# simulation) so its number is visible for comparison via results:sim, but
# results_tracker.py only ever scores the simulator's prob now.
SIMULATOR_N_SIMS = int(os.environ.get("SIMULATOR_N_SIMS", 30000))

# Set of features we will predict on
RUNS_SCORED_FEAT_SET = [
    "OBP_162",
    "SLG_162",
    "Strt_WHIP_35",
    "Strt_TB_BB_perc_35",
    "Strt_H_BB_perc_35",
    "Strt_SO_perc_10",
    "Bpen_WHIP_75",
    "Bpen_TB_BB_perc_75",
    "Bpen_SO_perc_75",
    "Bpen_TB_BB_perc_35",
    "lineup8_OBP_162",
    "lineup8_SLG_162",
    "lineup9_OBP_162",
    "lineup9_SLG_162",
    "home_hitting",
    "Bpen_H_BB_perc_75",
    "Bpen_WHIP_35",
    "Bpen_H_BB_perc_35",
    "Bpen_SO_perc_35",
    "Bpen_WHIP_10",
    "Bpen_TB_BB_perc_10",
    "Bpen_H_BB_perc_10",
    "Bpen_SO_perc_10",
]

HOME_VICTORY_FEAT_SET = [
    "OBP_162_h",
    "OBP_162_v",
    "SLG_162_h",
    "SLG_162_v",
    "Strt_WHIP_35_h",
    "Strt_WHIP_35_v",
    "Strt_TB_BB_perc_35_h",
    "Strt_TB_BB_perc_35_v",
    "Strt_H_BB_perc_35_h",
    "Strt_H_BB_perc_35_v",
    "Strt_SO_perc_10_h",
    "Strt_SO_perc_10_v",
    "Bpen_WHIP_75_h",
    "Bpen_WHIP_75_v",
    "Bpen_TB_BB_perc_75_h",
    "Bpen_TB_BB_perc_75_v",
    "Bpen_H_BB_perc_75_h",
    "Bpen_H_BB_perc_75_v",
    "Bpen_SO_perc_75_h",
    "Bpen_SO_perc_75_v",
    "Bpen_WHIP_35_h",
    "Bpen_WHIP_35_v",
    "Bpen_TB_BB_perc_35_h",
    "Bpen_TB_BB_perc_35_v",
    "Bpen_H_BB_perc_35_h",
    "Bpen_H_BB_perc_35_v",
    "Bpen_SO_perc_35_h",
    "Bpen_SO_perc_35_v",
    "Bpen_WHIP_10_h",
    "Bpen_WHIP_10_v",
    "Bpen_TB_BB_perc_10_h",
    "Bpen_TB_BB_perc_10_v",
    "Bpen_H_BB_perc_10_h",
    "Bpen_H_BB_perc_10_v",
    "Bpen_SO_perc_10_h",
    "Bpen_SO_perc_10_v",
    "lineup9_OBP_350_h",
    "lineup9_OBP_350_v",
    "lineup9_SLG_350_h",
    "lineup9_SLG_350_v",
    "lineup9_OBP_162_h",
    "lineup9_OBP_162_v",
    "lineup9_SLG_162_h",
    "lineup9_SLG_162_v",
    "lineup9_OBP_75_h",
    "lineup9_OBP_75_v",
    "lineup9_SLG_75_h",
    "lineup9_SLG_75_v",
]


def format_game_time(utc_str):
    """Convert ISO UTC game time to CST display string."""
    try:
        utc_dt = pd.to_datetime(utc_str, utc=True)
        cst = pytz.timezone("America/Chicago")
        cst_dt = utc_dt.astimezone(cst)
        return cst_dt.strftime("%-I:%M %p CST")
    except Exception:
        return utc_str


def predict_winner(X):
    with open(WINS_MODEL_FILE, "rb") as f:
        artifact = pickle.load(f)

    # handle both old-style (raw model) and new-style (dict with model + features)
    if isinstance(artifact, dict):
        model = artifact["model"]
        feat_cols = artifact["features"]
    else:
        model = artifact
        feat_cols = HOME_VICTORY_FEAT_SET  # fall back to hardcoded list

    pred = model.predict(X[feat_cols])
    prob = model.predict_proba(X[feat_cols])[:, 1]
    return pred, prob


def predict_runs_scored(X):
    with open(RUNS_MODEL_FILE, "rb") as f:
        artifact = pickle.load(f)
    # handle both old-style (raw model) and new-style (dict with model + features)
    if isinstance(artifact, dict):
        model = artifact["model"]
        feat_cols = artifact["features"]
    else:
        model = artifact
        feat_cols = RUNS_SCORED_FEAT_SET  # fall back to hardcoded list

    probs = model.predict_proba(X[feat_cols])[:, 1]
    return probs


def predict_homerun_hitter(X):
    with open(HR_MODEL_FILE, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feat_cols = artifact["features"]
    X["matchup"] = pd.Categorical(X["matchup"])

    probs = model.predict_proba(X[feat_cols])[:, 1]
    return probs


def get_runs_scored_prob(probs, line):
    line = pd.to_numeric(line, errors="coerce")
    return round(probs[math.ceil(line) :].sum()) if not pd.isna(line) else None


def post_to_X():
    log.info("\nPosting picks to X")
    client = tweepy.Client(
        bearer_token=BEARER_TOKEN,
        access_token=ACCESS_KEY,
        access_token_secret=ACCESS_SECRET,
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
    )

    tweet = f"""⚾️ MLB Predictions | {date.today().strftime('%Y.%m.%d')} ⚾️"""

    post_result = client.create_tweet(text=tweet)
    log.info("\nTweet Posted to @MoneyballVo!")


def filter_games_by_edge(df):
    filtered_df = df.copy()
    filtered_df["edge_h"] = filtered_df["edge_h"].str.replace("%", "").astype(float)
    filtered_df["edge_v"] = filtered_df["edge_v"].str.replace("%", "").astype(float)
    filtered_df["prob"] = filtered_df["prob"].astype(float)
    filtered_df = filtered_df[
        ((filtered_df["edge_h"] > EDGE_THRESHOLD_H) & (filtered_df["prob"] > 0.50))
        | (
            (filtered_df["edge_v"] > EDGE_THRESHOLD_V)
            & ((1 - filtered_df["prob"]) > 0.50)
        )
    ]
    return filtered_df


def filter_hr_by_edge(df):
    filtered = df.copy()

    filtered = filtered[
        (filtered["hr_prob_numeric"] >= HR_PROB_THRESHOLD)
        & ~(
            (filtered["opp_hr_per_bf_10"] == 0.032)
            & (filtered["opp_hr_per_bf_35"] == 0.032)
            & (filtered["opp_hr_per_bf_75"] == 0.032)
            & (filtered["opp_fb_perc_10"] == 0.35)
            & (filtered["opp_fb_perc_35"] == 0.35)
            & (filtered["opp_fb_perc_75"] == 0.35)
        )
    ]
    return filtered.sort_values("hr_prob_numeric", ascending=False)


def get_player_name_map(mlbam_ids):
    """Fetch player names from MLB Stats API."""
    name_map = {}
    chunk_size = 200
    ids = [i for i in mlbam_ids if i is not None]
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
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
                name_map[str(person["id"])] = person["fullName"]
        except Exception as e:
            log.error(f"Error fetching names: {e}")
    return name_map


def print_todays_home_victory_preds(df):
    if DISPLAY_EDGE_ONLY == 1:
        filtered_df = filter_games_by_edge(df)
    else:
        filtered_df = df.copy()
    filtered_df = filtered_df.rename(
        columns={
            "date_dblhead": "Date",
            "game_time": "Time",
            "temp": "Temp",
            "humidity": "Humidity",
            "team_h_full": "Home",
            "team_v_full": "Visitor",
            "starting_pitcher_name_h": "Probable Starter (H)",
            "starting_pitcher_name_v": "Probable Starter (V)",
            "moneyline_h": "ML (H)",
            "prob": "Sim Prob (H)",
            "xgb_prob": "XGB Prob (H)",
            "edge_h": "Edge (H)",
            "xgb_edge_h": "XGB Edge (H)",
            "moneyline_v": "ML (V)",
            "edge_v": "Edge (V)",
            "xgb_edge_v": "XGB Edge (V)",
        }
    )

    filtered_df["Time"] = filtered_df["Time"].apply(format_game_time)
    filtered_df.sort_values("Date", ascending=True, inplace=True)
    if "Sim Prob (H)" in filtered_df.columns:
        filtered_df["Sim Prob (H)"] = filtered_df["Sim Prob (H)"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    if "XGB Prob (H)" in filtered_df.columns:
        filtered_df["XGB Prob (H)"] = filtered_df["XGB Prob (H)"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )

    # add reliever flag
    filtered_df["Probable Starter (H)"] = filtered_df.apply(
        lambda r: (
            f"{r['Probable Starter (H)']} ⚠️"
            if r.get("Strt_is_reliever_h", 0) == 1
            else r["Probable Starter (H)"]
        ),
        axis=1,
    )
    filtered_df["Probable Starter (V)"] = filtered_df.apply(
        lambda r: (
            f"{r['Probable Starter (V)']} ⚠️"
            if r.get("Strt_is_reliever_v", 0) == 1
            else r["Probable Starter (V)"]
        ),
        axis=1,
    )

    cols = [
        "Date",
        "Time",
        "Temp",
        "Humidity",
        "Visitor",
        "Probable Starter (V)",
        "Home",
        "Probable Starter (H)",
        "ML (H)",
        "Edge (H)",
        "XGB Edge (H)",
        "Sim Prob (H)",
        "XGB Prob (H)",
        "ML (V)",
        "Edge (V)",
        "XGB Edge (V)",
    ]

    # filter out games without confirmed lineups on both sides
    filtered_df = filtered_df[filtered_df["lineups_confirmed"] == True]

    # filter out games without starting pitcher bc it may be predicting off default values
    filtered_df = filtered_df[
        (filtered_df["Probable Starter (H)"].notna())
        & (filtered_df["Probable Starter (H)"] != "")
        & (filtered_df["Probable Starter (V)"].notna())
        & (filtered_df["Probable Starter (V)"] != "")
    ]

    return filtered_df.loc[:, cols]


def print_todays_homerun_preds(df):
    df = df.copy()

    df["Platoon"] = df.apply(
        lambda r: (
            r.get("hr_per_pa_vs_r_162")
            if r.get("opp_throws") == "R"
            else r.get("hr_per_pa_vs_l_162")
        ),
        axis=1,
    )

    df = df.rename(
        columns={
            "date_dblhead": "Date",
            "team": "Team",
            "opponent": "Opponent",
            "slot": "Slot",
            "stand": "Bats",
            "opp_throws": "P_Throws",
            "park_hr_factor": "Park",
            "temp": "Temp",
            "humidity": "Humidity",
            "hr_prob": "HR Prob",
            "player_name": "Player",
        }
    )

    # Format for presentation
    if "opp_hr_per_bf_75" in df.columns:
        df["P-HR/BF"] = df["opp_hr_per_bf_75"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    if "ev_162" in df.columns:
        df["EV"] = df["ev_162"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    if "wind_spd" in df.columns:
        df["Wind"] = df["wind_spd"].map(
            lambda x: f"{x:.1f} mph" if pd.notna(x) else "N/A"
        )
    if "wind_out" in df.columns:
        df["Wind Out"] = df["wind_out"].map(
            lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A"
        )

    # all % cols
    pct_cols = {
        "Implied": "Implied",
        "barrel_162": "Barrel%",
        "hardhit_162": "HARDHIT%",
        "swspot_162": "SWSPOT%",
        "hr_per_pa_162": "HR/PA",
        "opp_fb_perc_35": "FB%",
        "Platoon": "Platoon",
    }
    for k, v in pct_cols.items():
        if k in df.columns:
            df[v] = df[k].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    def _format_bvp(row):
        pa = row.get("bvp_pa", 0) or 0
        hr = row.get("bvp_hr", 0) or 0
        if pa == 0:
            return "-"
        elif hr > 0:
            return f"{int(hr)}/{int(pa)}"
        else:
            return f"{int(pa)} PA"

    df["BvP"] = df.apply(_format_bvp, axis=1)

    cols = [
        "Player",
        "Team",
        "BvP",
        "Barrel%",
        "EV",
        "SWSPOT%",
        "HARDHIT%",
        "HR/PA",
        "Park",
        "Temp",
        "Humidity",
        "Wind",
        "Wind Out",
        "Platoon",
        "FB%",
        "P-HR/BF",
        "HR Prob",
    ]
    cols = [c for c in cols if c in df.columns]
    df["hr_prob_numeric"] = df["HR Prob"]  # save numeric before formatting
    df["HR Prob"] = df["HR Prob"].map(lambda x: f"{x:.1%}")

    # filter out players with missing names
    df = df[(df["Player"].notna()) & (df["Player"] != "")]

    df.to_csv(f"data/results/{RUN_DATE}_homerun_preds.csv", index=False)

    top_hr_df = filter_hr_by_edge(df)

    return top_hr_df[cols]


def print_todays_totals_preds(df):
    df = df.rename(
        columns={
            "date_dblhead": "Date",
            "game_time": "Time",
            "temp": "Temp",
            "humidity": "Humidity",
            "team_h_full": "Home",
            "team_v_full": "Visitor",
            "over_under_line": "O/U Line",
            "over_under_price_o": "Over Price",
            "over_under_price_u": "Under Price",
            "total_runs_predicted": "Total Runs Predicted",
        }
    )
    df.sort_values("Date", ascending=True, inplace=True)
    cols = [
        "Date",
        "Time",
        "Temp",
        "Humidity",
        "Visitor",
        "Home",
        "O/U Line",
        "Over Price",
        "Under Price",
        "Total Runs Predicted",
    ]


def run_pipeline(run_date_str):
    log.info("--- TIME TO COOK 👨🏻‍🍳 ⚾️ 🚀 💰 ---")
    if REFRESH_DATA == 1:
        log.info(f"\nEmptying Data Directories")
        cleanup_directory()

    log.info(f"\nGetting Starting Lineups for {RUN_DATE}")
    fname = f"data/daily/{RUN_DATE}_lineup_data.csv"
    files_exist = (
        os.path.exists(fname)
        and os.path.exists(BATTER_DICT_FILE)
        and os.path.exists(PITCHER_DICT_FILE)
        and os.path.exists(BVP_DICT_FILE)
    )

    if files_exist and REFRESH_DATA != 1:
        log.info(f"\nLoading Data From File: {fname}")
        lineup_w_pitching_batting_df = pd.read_csv(fname, index_col=False)
        with open(BATTER_DICT_FILE, "rb") as f:
            batter_data_dict = pickle.load(f)
        with open(PITCHER_DICT_FILE, "rb") as f:
            pitcher_data_dict = pickle.load(f)
        with open(BVP_DICT_FILE, "rb") as f:
            bvp_dict = pickle.load(f)
        pitcher_warnings = []
        batter_warnings = []

    else:
        log.info("\nLoading Lineup Data")
        df = get_lineups(RUN_DATE)

        if df.empty:
            log.info("No lineups posted yet — try again later")
            return {}

        # Add Pitching Data
        log.info("\nLoading Pitching Data")
        lineup_w_pitching_df, pitcher_data_dict, pitcher_warnings = (
            process_pitching_data(df)
        )

        # Add Batting Data
        log.info("\nLoading Batting Data")
        lineup_w_pitching_batting_df, batter_data_dict, bvp_dict, batter_warnings = (
            process_batting_data(lineup_w_pitching_df)
        )

        log.info(f"\nSaving Lineup Data to CSV")
        lineup_w_pitching_batting_df.to_csv(fname, index=False)

        # Save Dicts
        with open(BATTER_DICT_FILE, "wb") as f:
            pickle.dump(batter_data_dict, f)
        with open(PITCHER_DICT_FILE, "wb") as f:
            pickle.dump(pitcher_data_dict, f)
        with open(BVP_DICT_FILE, "wb") as f:
            pickle.dump(bvp_dict, f)

    log.info(f"\nLoading Team Data")
    lineup_w_pitching_batting_team_df = generate_team_window_features(
        lineup_w_pitching_batting_df
    )
    lineup_w_pitching_batting_team_df = add_team_form_features(
        lineup_w_pitching_batting_team_df
    )

    # Add Weather Data
    log.info("\nLoading Weather Data")
    lineup_w_pitching_batting_team_weather_df = process_weather_data(
        lineup_w_pitching_batting_team_df, RUN_DATE
    )

    log.info(f"\nGetting Features for Run Total Predictions")
    df_runs = get_run_total_feats(lineup_w_pitching_batting_team_df)
    df_runs.drop_duplicates(subset=["date_dblhead", "team_h", "team_v"], inplace=True)
    df_runs.reset_index(drop=True, inplace=True)

    log.info(f"\nGetting Odds Data")
    lineup_w_pitching_batting_team_weather_df["moneyline_h"] = (
        lineup_w_pitching_batting_team_weather_df.apply(
            lambda row: get_money_line_price(row["team_h_full"]), axis=1
        )
    )
    lineup_w_pitching_batting_team_weather_df["moneyline_v"] = (
        lineup_w_pitching_batting_team_weather_df.apply(
            lambda row: get_money_line_price(row["team_v_full"]), axis=1
        )
    )

    df_runs["over_under_price_o"] = df_runs.apply(
        lambda row: get_over_odds(row["team_h_full"]), axis=1
    )
    df_runs["over_under_price_u"] = df_runs.apply(
        lambda row: get_under_odds(row["team_h_full"]), axis=1
    )
    df_runs["over_under_line"] = df_runs.apply(
        lambda row: get_total_line(row["team_h_full"]), axis=1
    )

    log.info(f"\nMaking Predictions")

    # XGBoost — always runs, kept for side-by-side comparison via
    # results:sim, no longer the model results_tracker.py scores.
    win_model_df = assemble_win_model_features(lineup_w_pitching_batting_team_df.copy())
    xgb_pred, xgb_prob = predict_winner(win_model_df)
    lineup_w_pitching_batting_team_weather_df["xgb_prob"] = xgb_prob

    # Simulator — always runs, this is the model that drives prob/picks/edge
    # and the one results_tracker.py scores going forward.
    # Cache of full predict_game() summaries, keyed by (date_dblhead,
    # team_h, team_v) — NOT game_pk, which does not survive the upstream
    # pipeline transforms (confirmed empirically: game_row.get("game_pk")
    # returns None at this point). Reused by the HR block below so the
    # simulator only runs once per game, not twice.
    sim_summaries_by_game_key = {}

    def _game_key(row):
        return (row.get("date_dblhead"), row.get("team_h"), row.get("team_v"))

    log.info(f"  Running simulator (n_sims={SIMULATOR_N_SIMS}) ...")
    preds, probs, simmable = [], [], []
    for _, game_row in lineup_w_pitching_batting_team_weather_df.iterrows():
        summary = predict_game(
            game_row,
            batter_data_dict,
            pitcher_data_dict,
            n_sims=SIMULATOR_N_SIMS,
        )
        sim_summaries_by_game_key[_game_key(game_row)] = summary  # may be None
        if summary is None:
            log.warning(
                f"  Simulator could not predict "
                f"{game_row.get('team_v_full', '?')} @ {game_row.get('team_h_full', '?')} "
                f"(missing starter or lineup data) — DROPPING this game (no win "
                f"pick shown for it today; xgb_prob is still computed separately "
                f"and is not used as a fallback)"
            )
            simmable.append(False)
            preds.append(None)
            probs.append(None)
        else:
            simmable.append(True)
            prob = summary["home_win_prob"]
            preds.append(int(prob > 0.5))
            probs.append(prob)

    # drop rather than fill — a fabricated 0.5 in the prob column would still
    # produce a real (meaningless) edge_h/edge_v via calculate_edge below and
    # could surface as a false pick; dropping the row entirely is the only
    # way to guarantee an unsimmable game never reaches filter_games_by_edge.
    lineup_w_pitching_batting_team_weather_df = (
        lineup_w_pitching_batting_team_weather_df[simmable].copy()
    )
    preds = [p for p, ok in zip(preds, simmable) if ok]
    probs = [p for p, ok in zip(probs, simmable) if ok]

    lineup_w_pitching_batting_team_weather_df["home_victory"] = preds
    lineup_w_pitching_batting_team_weather_df["prob"] = probs

    # calculate our edge (drives actual picks via filter_games_by_edge below)
    lineup_w_pitching_batting_team_weather_df["edge_h"] = (
        lineup_w_pitching_batting_team_weather_df.apply(
            lambda row: calculate_edge(row["prob"], row["moneyline_h"]), axis=1
        )
    )
    lineup_w_pitching_batting_team_weather_df["edge_v"] = (
        lineup_w_pitching_batting_team_weather_df.apply(
            lambda row: calculate_edge(1 - row["prob"], row["moneyline_v"]), axis=1
        )
    )

    # XGBoost edge — COMPARISON DISPLAY ONLY, never used for pick filtering.
    # Same calculate_edge() math, just fed xgb_prob instead of the
    # simulator's prob, so the dashboard can show what XGBoost would have
    # called edge on the same game/line.
    lineup_w_pitching_batting_team_weather_df["xgb_edge_h"] = (
        lineup_w_pitching_batting_team_weather_df.apply(
            lambda row: calculate_edge(row["xgb_prob"], row["moneyline_h"]), axis=1
        )
    )
    lineup_w_pitching_batting_team_weather_df["xgb_edge_v"] = (
        lineup_w_pitching_batting_team_weather_df.apply(
            lambda row: calculate_edge(1 - row["xgb_prob"], row["moneyline_v"]), axis=1
        )
    )

    run_total_probs = predict_runs_scored(df_runs)
    df_runs["total_runs_predicted"] = df_runs.apply(
        lambda row: get_runs_scored_prob(run_total_probs, row["over_under_line"]),
        axis=1,
    )

    lineup_w_pitching_batting_team_weather_df.reset_index(drop=True, inplace=True)

    now = datetime.now(timezone.utc)
    lineup_w_pitching_batting_team_weather_df = (
        lineup_w_pitching_batting_team_weather_df[
            pd.to_datetime(
                lineup_w_pitching_batting_team_weather_df["game_time"], utc=True
            )
            > now
        ].copy()
    )

    wins_display_df = print_todays_home_victory_preds(
        lineup_w_pitching_batting_team_weather_df
    )

    df_hr = process_homerun_data(
        lineup_w_pitching_batting_team_weather_df,
        batter_data_dict,
        pitcher_data_dict,
        bvp_dict,
    )

    hr_display_df = pd.DataFrame(), pd.DataFrame()

    if not df_hr.empty:
        xgb_hr_probs = predict_homerun_hitter(df_hr)
        df_hr["xgb_hr_prob"] = xgb_hr_probs

        log.info(
            "  Reusing simulator results for HR probabilities (no re-simulation) ..."
        )
        sim_hr_probs_by_batter = {}
        for summary in sim_summaries_by_game_key.values():
            if summary is None:
                continue
            for slot, mlbam_id in summary["home_slot_to_mlbam_id"].items():
                if mlbam_id is not None:
                    sim_hr_probs_by_batter[mlbam_id] = summary["home_hr_prob_by_slot"][
                        slot
                    ]
            for slot, mlbam_id in summary["away_slot_to_mlbam_id"].items():
                if mlbam_id is not None:
                    sim_hr_probs_by_batter[mlbam_id] = summary["away_hr_prob_by_slot"][
                        slot
                    ]

        df_hr["sim_hr_prob"] = df_hr["b_id"].astype(str).map(sim_hr_probs_by_batter)
        missing = df_hr["sim_hr_prob"].isna().sum()
        if missing:
            log.warning(
                f"  {missing}/{len(df_hr)} batters had no simulator HR prediction "
                f"(game skipped or batter not in lineup) — DROPPING those rows "
                f"(no HR pick shown for them today; xgb_hr_prob is still computed "
                f"and available but is not used as a fallback)"
            )
        # drop rather than fill — a silently XGBoost-sourced number in a
        # simulator-driven column would be invisible in the output and
        # would corrupt HR calibration tracking without any visible marker.
        df_hr = df_hr[df_hr["sim_hr_prob"].notna()].copy()
        hr_probs = df_hr["sim_hr_prob"]

        NAME_MAP_FILE = f"data/daily/{RUN_DATE}_name_map.pkl"

        if os.path.exists(NAME_MAP_FILE):
            with open(NAME_MAP_FILE, "rb") as f:
                name_map = pickle.load(f)
        else:
            name_map = get_player_name_map(df_hr["b_id"].unique().tolist())
            with open(NAME_MAP_FILE, "wb") as f:
                pickle.dump(name_map, f)
        df_hr["hr_prob"] = hr_probs
        df_hr["player_name"] = df_hr["b_id"].map(name_map)
        hr_display_df = print_todays_homerun_preds(df_hr)
    else:
        log.info("\nHR df is empty — no batter rows built")

    # not printing df output until o/u preds are fixed
    # print_todays_totals_preds(df_runs)

    # log.info(f'\nHOME VICTORY FEATS:\n{lineup_w_pitching_batting_team_weather_df.loc[:, HOME_VICTORY_FEAT_SET]}')
    # log.info(f'\nRUNS SCORED FEATS:\n{df_runs.loc[:, RUNS_SCORED_FEAT_SET]}')

    log.info(f"\nSaving Predictions DataFrames to CSV")
    preds_cols = [
        "date_dblhead",
        "game_time",
        "team_h_full",
        "team_v_full",
        "prob",
        "xgb_prob",
        "moneyline_h",
        "edge_h",
        "xgb_edge_h",
        "moneyline_v",
        "edge_v",
        "xgb_edge_v",
    ]
    new_preds = lineup_w_pitching_batting_team_weather_df.loc[:, preds_cols]

    if os.path.exists(PREDS_FILE):
        existing = pd.read_csv(PREDS_FILE)
        combined = pd.concat([existing, new_preds], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["team_h_full", "team_v_full"], keep="last"
        )
        combined.to_csv(PREDS_FILE, index=False)
    else:
        new_preds.to_csv(PREDS_FILE, index=False)

    pipeline_warnings = pitcher_warnings + batter_warnings

    return hr_display_df, wins_display_df, pipeline_warnings


def handler(event, context):
    hr_display_df, wins_display_df, pipeline_warnings = run_pipeline(str(RUN_DATE))

    html = generate_html(
        hr_df=hr_display_df,
        wins_df=wins_display_df,
        run_date=str(RUN_DATE),
        warnings=pipeline_warnings,
    )
    with open(DASHBOARD_FILE, "w") as f:
        f.write(html)
    webbrowser.open(pathlib.Path(DASHBOARD_FILE).resolve().as_uri())
    # post_to_X()
    return {}


if __name__ == "__main__":
    handler({}, {})
