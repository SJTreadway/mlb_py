import warnings

warnings.filterwarnings("ignore")

import os
import requests
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pybaseball import (
    statcast_batter,
    statcast_pitcher,
)
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib
import pickle

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_PATH = "data/hr_training_data.csv"
MODEL_PATH = "models/homerun_model_2026v1.pkl"

MAX_API_WORKERS = int(os.environ.get("MAX_API_WORKERS", "8"))
API_SLEEP = float(os.environ.get("API_SLEEP", "0.5"))

START_YEAR = 2020
END_YEAR = 2024
MIN_PA = 150

WINDOWS_BAT = [30, 75, 162, 350]
WINDOWS_PITCH = [10, 35, 75]

FEATURE_COLS = [
    "BARREL_30",
    "BARREL_75",
    "BARREL_162",
    "EV_30",
    "EV_75",
    "EV_162",
    "HARDHIT_30",
    "HARDHIT_75",
    "HARDHIT_162",
    "SWSPOT_30",
    "SWSPOT_75",
    "SWSPOT_162",
    "HR_per_PA_30",
    "HR_per_PA_75",
    "HR_per_PA_162",
    "HR_per_PA_350",
    "HR_per_PA_vs_R_30",
    "HR_per_PA_vs_R_75",
    "HR_per_PA_vs_R_162",
    "HR_per_PA_vs_L_30",
    "HR_per_PA_vs_L_75",
    "HR_per_PA_vs_L_162",
    "SLG_30",
    "SLG_75",
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
    "batting_slot",
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
    "OAK": 100,
    "LAS": 100,
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
}

NON_AB_EVENTS = [
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_bunt",
    "sac_fly",
    "sac_fly_error",
    "catcher_interf",
]

# ── Player ID Helpers ─────────────────────────────────────────────────────────


def get_qualified_player_ids(min_pa=150, start_year=2020, end_year=2024):
    """Get MLBAM IDs directly from MLB Stats API — no FanGraphs needed."""
    all_mlbam_ids = set()
    for year in range(start_year, end_year + 1):
        try:
            url = "https://statsapi.mlb.com/api/v1/stats"
            params = {
                "stats": "season",
                "group": "hitting",
                "season": year,
                "playerPool": "All",
                "limit": 1000,
                "offset": 0,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            splits = resp.json()["stats"][0]["splits"]
            ids = [
                s["player"]["id"]
                for s in splits
                if s.get("stat", {}).get("plateAppearances", 0) >= min_pa
            ]
            all_mlbam_ids.update(ids)
            print(f"  {year}: {len(ids)} qualified batters")
        except Exception as e:
            print(f"  Error fetching {year}: {e}")
    return list(all_mlbam_ids)


# ── Statcast Pull & Transform ─────────────────────────────────────────────────


def transform_statcast_to_game_level(df):
    """Transform pitch-level Statcast pull to one row per game."""
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
        so = len(pa_endings[pa_endings["events"] == "strikeout"])
        rbi = pa_endings[~pa_endings["events"].isin(["field_error", "catcher_interf"])][
            "runs_scored"
        ].sum()

        # contact quality — batted ball rows only
        batted = pa_endings[pa_endings["launch_speed"].notna()].copy()

        # pitcher data points
        p_throws = group["p_throws"].iloc[0] if "p_throws" in group.columns else ""
        stand = group["stand"].iloc[0] if "stand" in group.columns else ""
        opp_pitcher_id = int(group["pitcher"].iloc[0])
        # this is from the BATTER's perspective
        # group is all pitches the batter saw in that game
        # so inning.min() == 1 means the batter faced this pitcher in inning 1
        # which means the pitcher was starting
        opp_is_starter = int(group["inning"].min() == 1)

        n_batted = len(batted)
        ev_sum = float(batted["launch_speed"].sum())
        hard_hits = int((batted["launch_speed"] >= 95).sum())
        sweet_spots = int(
            ((batted["launch_angle"] >= 8) & (batted["launch_angle"] <= 32)).sum()
        )

        # launch_speed_angle codes:
        # 6 = Barrel, 5 = Solid Contact, 4 = Flare/Burner,
        # 3 = Under, 2 = Topped, 1 = Weak
        barrels = int((batted["launch_speed_angle"] == 6).sum())

        stand = group["stand"].iloc[0] if "stand" in group.columns else ""
        bp_vals = group["batting_order"].dropna()
        batting_slot = int(bp_vals.mode().iloc[0]) if not bp_vals.empty else 0

        print(list(group.columns))
        print(group[["batting_order"]].value_counts(dropna=False).head())

        games.append(
            {
                "game_date": game_date,
                "game_pk": game_pk,
                "opponent": opponent,
                "is_home": int(is_home),
                "stand": stand,
                "batting_slot": batting_slot,
                "AB": ab,
                "H": h,
                "x2B": x2b,
                "x3B": x3b,
                "HR": hr,
                "BB": bb,
                "HBP": hbp,
                "SF": sf,
                "SO": so,
                "RBI": rbi,
                "batted_balls": n_batted,
                "ev_sum": ev_sum,
                "hard_hits": hard_hits,
                "sweet_spots": sweet_spots,
                "barrels": barrels,
                "p_throws": p_throws,
                "opp_pitcher_id": opp_pitcher_id,
                "opp_is_starter": opp_is_starter,
                "HR_vs_R": hr if p_throws == "R" else 0,
                "AB_vs_R": ab if p_throws == "R" else 0,
                "HR_vs_L": hr if p_throws == "L" else 0,
                "AB_vs_L": ab if p_throws == "L" else 0,
            }
        )

    return pd.DataFrame(games).sort_values("game_date").reset_index(drop=True)


def pull_statcast_for_player_year(mlbam_id, year):
    """Pull and transform one player-season from Statcast."""
    try:
        df = statcast_batter(f"{year}-03-01", f"{year}-11-30", mlbam_id)
        if df is None or df.empty:
            return None
        return transform_statcast_to_game_level(df)
    except Exception as e:
        print(f"    Error pulling {mlbam_id} / {year}: {e}")
        return None


def build_all_player_games(start_year=START_YEAR, end_year=END_YEAR, min_pa=MIN_PA):
    print("Fetching qualified player list from MLB Stats API...")
    mlbam_ids = get_qualified_player_ids(
        min_pa=min_pa, start_year=start_year, end_year=end_year
    )
    print(f"{len(mlbam_ids)} unique players found")

    checkpoint_path = CACHE_PATH.replace(".csv", "_batter_checkpoint.pkl")

    # resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            all_player_games = pickle.load(f)
        print(
            f"Resumed from checkpoint — {len(all_player_games)} batters already pulled"
        )
        remaining = [mid for mid in mlbam_ids if mid not in all_player_games]
        print(f"{len(remaining)} batters remaining")
    else:
        all_player_games = {}
        remaining = mlbam_ids

    def _fetch_batter(mlbam_id):
        player_seasons = []
        for year in range(start_year, end_year + 1):
            gdf = pull_statcast_for_player_year(mlbam_id, year)
            if gdf is not None and not gdf.empty:
                gdf["year"] = year
                gdf["mlbam_id"] = mlbam_id
                player_seasons.append(gdf)
            time.sleep(API_SLEEP)
        if player_seasons:
            combined = pd.concat(player_seasons, ignore_index=True)
            return mlbam_id, combined
        return mlbam_id, None

    with ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(_fetch_batter, mid): mid for mid in remaining}
        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Loading batter data"),
            1,
        ):
            mlbam_id, result = future.result()
            if result is not None:
                all_player_games[mlbam_id] = result

            # save checkpoint every 50 batters
            if i % 50 == 0:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(all_player_games, f)

    print(f"Done — pulled data for {len(all_player_games)} players")
    return all_player_games, checkpoint_path


# ── Rolling Feature Computation ───────────────────────────────────────────────


def rolling_sum(df, col, winsize):
    """Shift-1 rolling sum — no lookahead leakage."""
    return df[col].rolling(window=winsize, min_periods=1).sum().shift(1)


def add_batter_rolling_features(df):
    """Add rolling rate features to a single-player game-level DataFrame."""
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
        "SO",
        "barrels",
        "ev_sum",
        "hard_hits",
        "sweet_spots",
        "batted_balls",
        "HR_vs_R",
        "AB_vs_R",
        "HR_vs_L",
        "AB_vs_L",
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
        ab_r_denom = ab_r.replace(0, np.nan)
        ab_l_denom = ab_l.replace(0, np.nan)
        pa_denom = (ab + bb + hbp + sf).replace(0, np.nan)
        batted_denom = bbd.replace(0, np.nan)

        df[f"HR_per_PA_{winsize}"] = hr / pa_denom
        df[f"HR_per_PA_vs_R_{winsize}"] = hr_r / ab_r_denom
        df[f"HR_per_PA_vs_L_{winsize}"] = hr_l / ab_l_denom
        df[f"SLG_{winsize}"] = (h + x2b + 2 * x3b + 3 * hr) / ab_denom
        df[f"OBP_{winsize}"] = (h + bb + hbp) / pa_denom
        df[f"OBS_{winsize}"] = df[f"SLG_{winsize}"] + df[f"OBP_{winsize}"]
        df[f"EV_{winsize}"] = evs / batted_denom
        df[f"HARDHIT_{winsize}"] = hh / batted_denom
        df[f"SWSPOT_{winsize}"] = ss / batted_denom
        df[f"BARREL_{winsize}"] = bar / batted_denom

    return df


# ── Build Training Rows ───────────────────────────────────────────────────────


def compute_training_rows(all_player_games, pitcher_dict):
    rows = []
    for mlbam_id, df in all_player_games.items():
        df = add_batter_rolling_features(df)
        if len(df) < 2:
            continue

        for i in range(1, len(df)):
            current = df.iloc[i]
            prior = df.iloc[i - 1]

            pitcher_feats = get_pitcher_feats(
                current.get("opp_pitcher_id"),
                current["game_date"],
                pitcher_dict,
            )

            row = {
                "mlbam_id": mlbam_id,
                "game_date": current["game_date"],
                "year": current.get("year", 0),
                "opponent": current["opponent"],
                "stand": current.get("stand", ""),
                "batting_slot": current.get("batting_slot", 0),
                "park_hr_factor": PARK_HR_FACTORS.get(
                    str(current.get("opponent", "")), 100
                ),
                "hit_hr": int(current.get("HR", 0) > 0),
                **pitcher_feats,
            }

            p_throws = str(current.get("p_throws", ""))

            for winsize in WINDOWS_BAT:
                for stem in ["BARREL", "EV", "HARDHIT", "SWSPOT", "SLG", "OBP", "OBS"]:
                    row[f"{stem}_{winsize}"] = prior.get(f"{stem}_{winsize}", np.nan)
                row[f"HR_per_PA_{winsize}"] = prior.get(f"HR_per_PA_{winsize}", np.nan)
                row[f"HR_per_PA_vs_R_{winsize}"] = prior.get(
                    f"HR_per_PA_vs_R_{winsize}", np.nan
                )
                row[f"HR_per_PA_vs_L_{winsize}"] = prior.get(
                    f"HR_per_PA_vs_L_{winsize}", np.nan
                )

            rows.append(row)

    if not rows:
        print("No training rows built — check data pull")
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)


# ── Pitcher Pull & Transform ───────────────────────────────────────────────────


def transform_statcast_pitcher_to_game_level(df):
    """Transform pitch-level Statcast pitcher pull to one row per game."""
    if df.empty:
        return pd.DataFrame()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    games = []

    for (game_date, game_pk), group in df.groupby(["game_date", "game_pk"]):
        pa_endings = group[group["events"].notna() & (group["events"] != "")].copy()
        if pa_endings.empty:
            continue

        bfp = len(pa_endings)
        hr = len(pa_endings[pa_endings["events"] == "home_run"])

        # fly balls from batted ball type
        batted = pa_endings[pa_endings["launch_speed"].notna()].copy()
        n_batted = len(batted)
        fly_balls = int((batted["bb_type"] == "fly_ball").sum())

        games.append(
            {
                "game_date": game_date,
                "game_pk": game_pk,
                "BFP": bfp,
                "HR": hr,
                "fly_balls": fly_balls,
                "batted_balls_allowed": n_batted,
            }
        )

    return pd.DataFrame(games).sort_values("game_date").reset_index(drop=True)


def add_pitcher_rolling_features(df):
    """Add rolling HR/BF and FB% to a single-pitcher game-level DataFrame."""
    df = df.sort_values("game_date").reset_index(drop=True)

    new_cols = {}
    for winsize in WINDOWS_PITCH:
        for col in ["HR", "BFP", "fly_balls", "batted_balls_allowed"]:
            if col in df.columns:
                new_cols[f"rollsum_{col}_{winsize}"] = rolling_sum(
                    df, col, winsize
                ).values

    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    for winsize in WINDOWS_PITCH:
        hr = pd.Series(
            new_cols.get(f"rollsum_HR_{winsize}", np.zeros(len(df))), index=df.index
        )
        bf = pd.Series(
            new_cols.get(f"rollsum_BFP_{winsize}", np.zeros(len(df))), index=df.index
        )
        fb = pd.Series(
            new_cols.get(f"rollsum_fly_balls_{winsize}", np.zeros(len(df))),
            index=df.index,
        )
        bat = pd.Series(
            new_cols.get(f"rollsum_batted_balls_allowed_{winsize}", np.zeros(len(df))),
            index=df.index,
        )

        df[f"HR_per_BF_{winsize}"] = hr / bf.replace(0, np.nan)
        df[f"FB_perc_{winsize}"] = fb / bat.replace(0, np.nan)

    return df


def pull_statcast_for_pitcher_year(mlbam_id, year):
    """Pull and transform one pitcher-season from Statcast."""

    try:
        df = statcast_pitcher(f"{year}-03-01", f"{year}-11-30", mlbam_id)
        if df is None or df.empty:
            return None
        return transform_statcast_pitcher_to_game_level(df)
    except Exception as e:
        print(f"    Error pulling pitcher {mlbam_id} / {year}: {e}")
        return None


def get_qualified_starter_ids(min_gs=10, start_year=START_YEAR, end_year=END_YEAR):
    """Get MLBAM IDs for pitchers with significant starts from MLB Stats API."""
    all_ids = set()
    for year in range(start_year, end_year + 1):
        try:
            url = "https://statsapi.mlb.com/api/v1/stats"
            params = {
                "stats": "season",
                "group": "pitching",
                "season": year,
                "playerPool": "All",
                "limit": 1000,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            splits = resp.json()["stats"][0]["splits"]
            ids = [
                s["player"]["id"]
                for s in splits
                if s.get("stat", {}).get("gamesStarted", 0) >= min_gs
            ]
            all_ids.update(ids)
            print(f"  {year}: {len(ids)} qualified starters")
        except Exception as e:
            print(f"  Error fetching {year}: {e}")
    return all_ids


def build_pitcher_dict(start_year=START_YEAR, end_year=END_YEAR, min_gs=10):
    checkpoint_path = CACHE_PATH.replace(".csv", "_pitcher_checkpoint.pkl")

    print("Fetching qualified starter list from MLB Stats API...")
    qualified = get_qualified_starter_ids(
        min_gs=min_gs, start_year=start_year, end_year=end_year
    )
    print(f"{len(qualified)} qualified starting pitchers")

    # resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        import pickle

        with open(checkpoint_path, "rb") as f:
            pitcher_dict = pickle.load(f)
        print(f"Resumed from checkpoint — {len(pitcher_dict)} pitchers already pulled")
        remaining = [pid for pid in qualified if pid not in pitcher_dict]
        print(f"{len(remaining)} pitchers remaining")
    else:
        pitcher_dict = {}
        remaining = list(qualified)

    def _fetch_pitcher(mlbam_id):
        pitcher_seasons = []
        for year in range(start_year, end_year + 1):
            gdf = pull_statcast_for_pitcher_year(mlbam_id, year)
            if gdf is not None and not gdf.empty:
                gdf["year"] = year
                pitcher_seasons.append(gdf)
            time.sleep(API_SLEEP)
        if pitcher_seasons:
            combined = pd.concat(pitcher_seasons, ignore_index=True)
            return mlbam_id, add_pitcher_rolling_features(combined)
        return mlbam_id, None

    with ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(_fetch_pitcher, pid): pid for pid in remaining}
        for i, future in enumerate(
            tqdm(
                as_completed(futures), total=len(futures), desc="Loading pitcher data"
            ),
            1,
        ):
            mlbam_id, result = future.result()
            if result is not None:
                pitcher_dict[mlbam_id] = result

            if i % 50 == 0:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(pitcher_dict, f)

    return pitcher_dict, checkpoint_path


# ── Pitcher Feature Lookup ────────────────────────────────────────────────────


def get_pitcher_feats(mlbam_id, game_date, pitcher_dict):
    """Look up pre-game rolling pitcher stats."""
    defaults = {
        **{f"opp_HR_per_BF_{w}": 0 for w in WINDOWS_PITCH},
        **{f"opp_FB_perc_{w}": 0 for w in WINDOWS_PITCH},
    }

    if not mlbam_id or pd.isna(mlbam_id):
        return defaults

    mlbam_id = int(mlbam_id)
    if mlbam_id not in pitcher_dict:
        return defaults

    pdf = pitcher_dict[mlbam_id]
    prior = pdf[pdf["game_date"] < game_date]
    if prior.empty:
        return defaults

    prow = prior.iloc[-1]
    return {
        **{f"opp_HR_per_BF_{w}": prow.get(f"HR_per_BF_{w}", 0) for w in WINDOWS_PITCH},
        **{f"opp_FB_perc_{w}": prow.get(f"FB_perc_{w}", 0) for w in WINDOWS_PITCH},
    }


# ── Model Training ────────────────────────────────────────────────────────────


def check_feature_coverage(hr_df):
    print("\n── Feature Coverage ──────────────────────────────")
    contact_cols = ["BARREL_30", "EV_30", "HARDHIT_30", "SWSPOT_30"]
    has_contact = hr_df[contact_cols].notna().all(axis=1).sum()
    print(
        f"Rows with contact quality data: "
        f"{has_contact:,} / {len(hr_df):,} "
        f"({100*has_contact/len(hr_df):.1f}%)"
    )
    print(hr_df[FEATURE_COLS].describe().round(3))


def train_model(hr_df):
    hr_df = hr_df.sort_values("game_date").reset_index(drop=True)

    # fill missing pitcher/platoon features with 0
    for col in FEATURE_COLS:
        if col not in hr_df.columns:
            hr_df[col] = 0
        hr_df[col] = hr_df[col].fillna(0)

    # drop rows missing core batter features
    core_cols = ["HR_per_PA_30", "SLG_30", "OBP_30"]
    hr_df = hr_df.dropna(subset=core_cols)

    print(f"\nTraining on {len(hr_df):,} rows — HR rate: {hr_df.hit_hr.mean():.3f}")

    X = hr_df[FEATURE_COLS]
    y = hr_df["hit_hr"]

    # ── Cross-validation ──
    print("\n── TimeSeriesSplit CV ────────────────────────────")
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores, brier_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        m = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            eval_metric="auc",
            random_state=42,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        probs = m.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, probs)
        brier = brier_score_loss(y_va, probs)
        auc_scores.append(auc)
        brier_scores.append(brier)
        print(f"Fold {fold} — AUC: {auc:.4f}  Brier: {brier:.4f}")

    print(f"\nMean AUC:   {np.mean(auc_scores):.4f}")
    print(f"Mean Brier: {np.mean(brier_scores):.4f}")

    # ── Feature importance ──
    base_for_importance = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=42,
    )
    base_for_importance.fit(X, y)
    importance = pd.Series(
        base_for_importance.feature_importances_, index=FEATURE_COLS
    ).sort_values()
    print("\n── Top Features ──────────────────────────────────")
    print(importance.tail(10).round(4))

    # ── Final calibrated model ──
    print("\nTraining final calibrated model...")
    base_final = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_final, cv=5, method="isotonic")
    calibrated.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": calibrated, "features": FEATURE_COLS}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return calibrated


def plot_calibration(model, hr_df):
    """Plot predicted probability vs actual HR rate."""
    X = hr_df[FEATURE_COLS].fillna(0)
    y = hr_df["hit_hr"]
    probs = model.predict_proba(X)[:, 1]

    df_cal = hr_df[["hit_hr"]].copy()
    df_cal["pred_prob"] = probs
    df_cal["bin"] = pd.cut(df_cal["pred_prob"], bins=np.linspace(0, 0.4, 11))
    cal_summary = df_cal.groupby("bin").agg(
        mean_pred=("pred_prob", "mean"),
        actual_rate=("hit_hr", "mean"),
        count=("hit_hr", "count"),
    )

    plt.figure(figsize=(7, 5))
    plt.plot(cal_summary.mean_pred, cal_summary.actual_rate, "o-", label="Model")
    plt.plot([0, 0.4], [0, 0.4], "k--", label="Perfect calibration")
    plt.xlabel("Predicted HR probability")
    plt.ylabel("Actual HR rate")
    plt.title("Calibration Plot")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached training data from {CACHE_PATH}")
        hr_df = pd.read_csv(CACHE_PATH, parse_dates=["game_date"])
        print(f"Loaded {len(hr_df):,} rows")
    else:
        print("No cache found — pulling from Statcast...")
        all_player_games, b_ckpt = build_all_player_games(
            start_year=START_YEAR,
            end_year=END_YEAR,
            min_pa=MIN_PA,
        )

        print("\nPulling pitcher data...")
        pitcher_dict, p_ckpt = build_pitcher_dict(
            start_year=START_YEAR, end_year=END_YEAR, min_gs=10
        )

        hr_df = compute_training_rows(all_player_games, pitcher_dict)
        print(f"Built {len(hr_df):,} rows — HR rate: {hr_df.hit_hr.mean():.3f}")

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        hr_df.to_csv(CACHE_PATH, index=False)
        print(f"Cached to {CACHE_PATH}")

        # now safe to clean up checkpoints
        for ckpt in [b_ckpt, p_ckpt]:
            if os.path.exists(ckpt):
                os.remove(ckpt)
        print("Checkpoints cleaned up")

    check_feature_coverage(hr_df)
    model = train_model(hr_df)
    plot_calibration(model, hr_df)
