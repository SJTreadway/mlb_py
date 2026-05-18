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

from datetime import date, timedelta, datetime, timezone
import pytz
import pickle

from api.teams import generate_team_window_features
from api.lineups import get_lineups, get_run_total_feats
from api.odds import (
    get_over_odds,
    get_under_odds,
    get_total_line,
    get_money_line_price,
    calculate_edge,
    get_hr_prop_odds,
    get_best_hr_odds,
    match_hr_odds,
)
from api.pitchers import process_pitching_data
from api.batters import process_batting_data
from api.weather import process_weather_data
from api.homerun import process_homerun_data

from ui.dashboard import display_dashboard

from cleanup import cleanup_directory

import tweepy

from dotenv import load_dotenv

load_dotenv()

DISPLAY_EDGE_ONLY = 1
EDGE_THRESHOLD = 4.0
HR_EDGE_THRESHOLD = 1.0

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
HR_ODDS_CACHE_FILE = f"data/daily/{RUN_DATE}_hr_odds_cache.pkl"


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

HR_FEAT_SET = [
    "BARREL_7",
    "BARREL_14",
    "BARREL_30",
    "BARREL_75",
    "BARREL_162",
    "EV_7",
    "EV_14",
    "EV_30",
    "EV_75",
    "EV_162",
    "HARDHIT_7",
    "HARDHIT_14",
    "HARDHIT_30",
    "HARDHIT_75",
    "HARDHIT_162",
    "SWSPOT_7",
    "SWSPOT_14",
    "SWSPOT_30",
    "SWSPOT_75",
    "SWSPOT_162",
    "HR_per_PA_7",
    "HR_per_PA_14",
    "HR_per_PA_30",
    "HR_per_PA_75",
    "HR_per_PA_162",
    "HR_per_PA_350",
    "HR_per_PA_vs_R_7",
    "HR_per_PA_vs_R_14",
    "HR_per_PA_vs_R_30",
    "HR_per_PA_vs_R_75",
    "HR_per_PA_vs_R_162",
    "HR_per_PA_vs_L_7",
    "HR_per_PA_vs_L_14",
    "HR_per_PA_vs_L_30",
    "HR_per_PA_vs_L_75",
    "HR_per_PA_vs_L_162",
    "SLG_7",
    "SLG_14",
    "SLG_30",
    "SLG_75",
    "OBP_7",
    "OBP_14",
    "OBP_30",
    "OBP_75",
    "OBS_30",
    "OBS_75",
    "opp_HR_per_BF_10",
    "opp_HR_per_BF_35",
    "opp_HR_per_BF_75",
    "opp_FB_perc_10",
    "opp_FB_perc_35",
    "opp_FB_perc_75",
    "park_hr_factor",
    "age",
    # "bat_speed_30",
    # "bat_speed_75",
    # "bat_speed_162",
    "est_woba_7",
    "est_woba_14",
    "est_woba_30",
    "est_woba_75",
    "est_woba_162",
    "est_slg_7",
    "est_slg_14",
    "est_slg_30",
    "est_slg_75",
    "est_slg_162",
    "is_home",
    "temp",
    "humidity",
    "wind_spd",
    "wind_out",
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
    with open(WINS_MODEL_FILE, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return pred, prob


def predict_runs_scored(X):
    with open(RUNS_MODEL_FILE, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    probs = model.predict_proba(X)
    return probs


def predict_homerun_hitter(X):
    with open(HR_MODEL_FILE, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    probs = model.predict_proba(X)[:, 1]
    return probs


def get_runs_scored_prob(probs, line):
    line = pd.to_numeric(line, errors="coerce")
    return round(probs[math.ceil(line) :].sum()) if not pd.isna(line) else None


def post_to_X():
    print("\nPosting picks to X")
    client = tweepy.Client(
        bearer_token=BEARER_TOKEN,
        access_token=ACCESS_KEY,
        access_token_secret=ACCESS_SECRET,
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
    )

    tweet = f"""xxx"""

    post_result = client.create_tweet(text=tweet)
    print("\nTweet Posted to @MoneyballVo!")


def filter_games_by_edge(df):
    filtered_df = df.copy()
    filtered_df["edge_h"] = filtered_df["edge_h"].str.replace("%", "").astype(float)
    filtered_df["edge_v"] = filtered_df["edge_v"].str.replace("%", "").astype(float)
    filtered_df["prob"] = filtered_df["prob"].astype(float)
    filtered_df = filtered_df[
        ((filtered_df["edge_h"] > EDGE_THRESHOLD) & (filtered_df["prob"] > 0.50))
        | (
            (filtered_df["edge_v"] > EDGE_THRESHOLD)
            & ((1 - filtered_df["prob"]) > 0.50)
        )
    ]
    return filtered_df


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
            print(f"Error fetching names: {e}")
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
            "prob": "Prob Win (H)",
            "edge_h": "Edge (H)",
            "moneyline_v": "ML (V)",
            "edge_v": "Edge (V)",
        }
    )
    filtered_df["Time"] = filtered_df["Time"].apply(format_game_time)
    filtered_df.sort_values("Date", ascending=True, inplace=True)
    if "Prob Win (H)" in filtered_df.columns:
        filtered_df["Prob Win (H)"] = filtered_df["Prob Win (H)"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
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
        "Prob Win (H)",
        "ML (V)",
        "Edge (V)",
    ]
    print("\n── Game Winner Predictions ──────────────────────────────────")
    print(f"\n{filtered_df.loc[:,cols]}")
    return filtered_df.loc[:, cols]


def print_todays_homerun_preds(df):
    df = df.copy()

    # add odds if available
    # get or load cached HR odds
    if os.path.exists(HR_ODDS_CACHE_FILE):
        with open(HR_ODDS_CACHE_FILE, "rb") as f:
            best_odds = pickle.load(f)
    else:
        odds_df = get_hr_prop_odds()
        best_odds = get_best_hr_odds(odds_df)
        with open(HR_ODDS_CACHE_FILE, "wb") as f:
            pickle.dump(best_odds, f)

    df = match_hr_odds(df, best_odds)

    if "american_odds" in df.columns:
        df["edge"] = df.apply(
            lambda r: (
                calculate_edge(r["hr_prob"], r["american_odds"])
                if pd.notna(r.get("american_odds"))
                else None
            ),
            axis=1,
        )

    df["Platoon"] = df.apply(
        lambda r: (
            r.get("HR_per_PA_vs_R_162")
            if r.get("opp_throws") == "R"
            else r.get("HR_per_PA_vs_L_162")
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
            "american_odds": "Odds",
            "implied_prob": "Implied",
            "edge": "Edge",
            # "book": "Book",
        }
    )

    # Format for presentation
    if "opp_HR_per_BF_75" in df.columns:
        df["P-HR/BF"] = df["opp_HR_per_BF_75"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    if "EV_162" in df.columns:
        df["EV"] = df["EV_162"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
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
        "BARREL_162": "Barrel%",
        "HARDHIT_162": "HARDHIT%",
        "SWSPOT_162": "SWSPOT%",
        "HR_per_PA_162": "HR/PA",
        "opp_FB_perc_35": "FB%",
        "Platoon": "Platoon",
    }
    for k, v in pct_cols.items():
        if k in df.columns:
            df[v] = df[k].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    df["Game"] = df.apply(
        lambda r: (
            f'{r["Opponent"]} @ {r["Team"]}'
            if r["is_home"]
            else f'{r["Team"]} @ {r["Opponent"]}'
        ),
        axis=1,
    )

    cols = [
        "Player",
        "Game",
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
        "Odds",
        "Edge",
        # "Book",
    ]
    cols = [c for c in cols if c in df.columns]
    df["hr_prob_numeric"] = df["HR Prob"]  # save numeric before formatting
    df["HR Prob"] = df["HR Prob"].map(lambda x: f"{x:.1%}")
    df.to_csv(f"data/results/{RUN_DATE}_homerun_preds.csv", index=False)

    # show top 7 for reference
    print("\n── Top HR Predictions ──────────────────────────────────")
    top_hr_df = df.nlargest(7, "hr_prob_numeric")[cols]
    print(df.nlargest(7, "hr_prob_numeric")[cols].to_string(index=False))

    return top_hr_df


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
    print(f"\n{df.loc[:,cols]}")


def handler(event, context):
    print("--- TIME TO COOK 👨🏻‍🍳 ⚾️ 🚀 💰 ---")

    if REFRESH_DATA == 1:
        print(f"\nEmptying Data Directories")
        cleanup_directory()

    print(f"\nGetting Starting Lineups for {RUN_DATE}")
    fname = f"data/daily/{RUN_DATE}_lineup_data.csv"
    files_exist = (
        os.path.exists(fname)
        and os.path.exists(BATTER_DICT_FILE)
        and os.path.exists(PITCHER_DICT_FILE)
    )

    if files_exist and REFRESH_DATA != 1:
        print(f"\nLoading Data From File: {fname}")
        lineup_w_pitching_batting_df = pd.read_csv(fname, index_col=False)
        with open(BATTER_DICT_FILE, "rb") as f:
            batter_data_dict = pickle.load(f)
        with open(PITCHER_DICT_FILE, "rb") as f:
            pitcher_data_dict = pickle.load(f)

    else:
        print("\nLoading Lineup Data")
        df = get_lineups(RUN_DATE)

        if df.empty:
            print("No lineups posted yet — try again later")
            return {}

        # Add Pitching Data
        print("\nLoading Pitching Data")
        lineup_w_pitching_df, pitcher_data_dict = process_pitching_data(df)

        # Add Batting Data
        print("\nLoading Batting Data")
        lineup_w_pitching_batting_df, batter_data_dict = process_batting_data(
            lineup_w_pitching_df
        )

        print(f"\nSaving Lineup Data to CSV")
        lineup_w_pitching_batting_df.to_csv(fname, index=False)

        # Save Dicts
        with open(BATTER_DICT_FILE, "wb") as f:
            pickle.dump(batter_data_dict, f)
        with open(PITCHER_DICT_FILE, "wb") as f:
            pickle.dump(pitcher_data_dict, f)

    print(f"\nLoading Team Data")
    lineup_w_pitching_batting_team_df = generate_team_window_features(
        lineup_w_pitching_batting_df
    )

    # Add Weather Data
    print("\nLoading Weather Data")
    lineup_w_pitching_batting_team_weather_df = process_weather_data(
        lineup_w_pitching_batting_team_df, RUN_DATE
    )

    print(f"\nGetting Features for Run Total Predictions")
    df_runs = get_run_total_feats(lineup_w_pitching_batting_team_df)
    df_runs.drop_duplicates(subset=["date_dblhead", "team_h", "team_v"], inplace=True)
    df_runs.reset_index(drop=True, inplace=True)

    print(f"\nGetting Odds Data")
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

    print(f"\nMaking Predictions")
    (
        lineup_w_pitching_batting_team_weather_df["home_victory"],
        lineup_w_pitching_batting_team_weather_df["prob"],
    ) = predict_winner(lineup_w_pitching_batting_df.loc[:, HOME_VICTORY_FEAT_SET])

    # calculate our edge
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

    run_total_probs = predict_runs_scored(df_runs.loc[:, RUNS_SCORED_FEAT_SET])
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
    )

    hr_display_df = pd.DataFrame(), pd.DataFrame()

    if not df_hr.empty:
        hr_probs = predict_homerun_hitter(df_hr.loc[:, HR_FEAT_SET])
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
        print("\nHR df is empty — no batter rows built")

    display_dashboard(
        hr_df=hr_display_df,  # your formatted top 7 HR df
        wins_df=wins_display_df,  # your formatted wins df
        run_date=str(RUN_DATE),
    )

    # not printing df output until o/u preds are fixed
    # print_todays_totals_preds(df_runs)

    # print(f'\nHOME VICTORY FEATS:\n{lineup_w_pitching_batting_team_weather_df.loc[:, HOME_VICTORY_FEAT_SET]}')
    # print(f'\nRUNS SCORED FEATS:\n{df_runs.loc[:, RUNS_SCORED_FEAT_SET]}')

    print(f"\nSaving Predictions DataFrames to CSV")
    lineup_w_pitching_batting_team_weather_df.loc[
        :,
        [
            "date_dblhead",
            "game_time",
            "team_h_full",
            "team_v_full",
            "prob",
            "moneyline_h",
            "edge_h",
            "moneyline_v",
            "edge_v",
        ],
    ].to_csv(f"data/results/{RUN_DATE}_home_victory_preds.csv", index=False)
    """
    df_runs.loc[
        :,
        [
            "date_dblhead",
            "game_time",
            "team_h_full",
            "team_v_full",
            "over_under_line",
            "total_runs_predicted",
        ],
    ].to_csv(f"data/results/{RUN_DATE}_run_total_preds.csv", index=False)
    """

    # post_to_X()
    return {}


if __name__ == "__main__":
    handler({}, {})
