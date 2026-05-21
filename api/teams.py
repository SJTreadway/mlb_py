from pybaseball import team_game_logs, season_game_logs
import pandas as pd
import numpy as np
import os
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

from helpers import strip_suffix, agg_non_na
from tqdm import tqdm

load_dotenv()

pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)

TEAMS = [
    "LAA",
    "MIL",
    "HOU",
    "BAL",
    "BOS",
    "CHW",
    "CLE",
    "DET",
    "KCR",
    "MIN",
    "NYY",
    "ATH",
    "SEA",
    "TBR",
    "TEX",
    "TOR",
    "ARI",
    "ATL",
    "CIN",
    "COL",
    "SDP",
    "MIA",
    "NYM",
    "PHI",
    "PIT",
    "SFG",
    "STL",
    "WSN",
    "LAD",
    "CHC",
]

# Game Windows for Prev Data Lookup
WINDOWS = [162, 90, 30]

# Form windows for win model
FORM_WINDOWS = [5, 10, 20, 30]
PYTH_WINDOWS = [30, 162]


# ── Snowflake connection ───────────────────────────────────────────────────────


def _load_private_key() -> bytes:
    key_path = os.path.expanduser(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"])
    passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    pw_bytes = passphrase.encode() if passphrase else None
    with open(key_path, "rb") as f:
        p_key = serialization.load_pem_private_key(
            f.read(),
            password=pw_bytes,
            backend=default_backend(),
        )
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _sf_cfg() -> dict:
    cfg = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "private_key": _load_private_key(),
        "database": os.environ.get("SNOWFLAKE_DATABASE", "BASEBALL"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA", "STATCAST"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    }
    role = os.environ.get("SNOWFLAKE_ROLE", "")
    if role:
        cfg["role"] = role
    return cfg


# ── Team form features (win model) ────────────────────────────────────────────


def add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach team form features for each game in df.

    Pulls GAME_RESULTS from Snowflake, computes rolling win%, run_diff_avg,
    pythagorean win%, home/away splits, and h2h win% for today's matchups.
    All windows are computed on games PRIOR to the current game (no leakage).

    Adds columns matching the win model's FORM_FEATURES:
        win_pct_{5/10/20/30}_{h/v}
        run_diff_avg_{5/10/20/30}_{h/v}
        pyth_win_pct_{30/162}_{h/v}
        home_win_pct_h
        away_win_pct_v
        h2h_win_pct_h
        h2h_games
    """
    db = os.environ.get("SNOWFLAKE_DATABASE", "BASEBALL")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "STATCAST")

    conn = snowflake.connector.connect(**_sf_cfg())
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT GAME_PK, GAME_DATE, TEAM_H, TEAM_V,
               HOME_VICTORY, RUN_DIFF, RUNS_H, RUNS_V
        FROM {db}.{schema}.GAME_RESULTS
        WHERE HOME_VICTORY IS NOT NULL
        ORDER BY GAME_DATE
    """
    )
    cols = [d[0].lower() for d in cursor.description]
    history = pd.DataFrame(cursor.fetchall(), columns=cols)
    cursor.close()
    conn.close()

    history["game_date"] = pd.to_datetime(history["game_date"])

    # ── build long-form team game log ─────────────────────────────────────────
    home_rows = history[
        [
            "game_date",
            "game_pk",
            "team_h",
            "team_v",
            "home_victory",
            "run_diff",
            "runs_h",
            "runs_v",
        ]
    ].copy()
    home_rows.columns = [
        "game_date",
        "game_pk",
        "team",
        "opponent",
        "won",
        "run_diff",
        "runs_scored",
        "runs_allowed",
    ]
    home_rows["is_home"] = 1

    away_rows = history[
        [
            "game_date",
            "game_pk",
            "team_v",
            "team_h",
            "home_victory",
            "run_diff",
            "runs_v",
            "runs_h",
        ]
    ].copy()
    away_rows.columns = [
        "game_date",
        "game_pk",
        "team",
        "opponent",
        "won",
        "run_diff",
        "runs_scored",
        "runs_allowed",
    ]
    away_rows["won"] = 1 - away_rows["won"]
    away_rows["run_diff"] = -away_rows["run_diff"]
    away_rows["is_home"] = 0

    team_log = pd.concat([home_rows, away_rows], ignore_index=True)
    team_log = team_log.sort_values(["team", "game_date"]).reset_index(drop=True)

    # ── compute rolling form stats ────────────────────────────────────────────
    grp = team_log.groupby("team")

    for w in FORM_WINDOWS:
        team_log[f"win_pct_{w}"] = grp["won"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        team_log[f"run_diff_avg_{w}"] = grp["run_diff"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

    # pythagorean win%: RS^2 / (RS^2 + RA^2)
    for w in PYTH_WINDOWS:
        rs = grp["runs_scored"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).sum()
        )
        ra = grp["runs_allowed"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).sum()
        )
        team_log[f"pyth_win_pct_{w}"] = rs**2 / (rs**2 + ra**2).replace(0, np.nan)

    # home win% (home games only)
    home_log = team_log[team_log["is_home"] == 1].copy()
    home_log["home_win_pct"] = home_log.groupby("team")["won"].transform(
        lambda x: x.shift(1).rolling(50, min_periods=1).mean()
    )

    # away win% (away games only)
    away_log = team_log[team_log["is_home"] == 0].copy()
    away_log["away_win_pct"] = away_log.groupby("team")["won"].transform(
        lambda x: x.shift(1).rolling(50, min_periods=1).mean()
    )

    # merge home/away splits back
    team_log = team_log.merge(
        home_log[["game_pk", "team", "home_win_pct"]],
        on=["game_pk", "team"],
        how="left",
    )
    team_log = team_log.merge(
        away_log[["game_pk", "team", "away_win_pct"]],
        on=["game_pk", "team"],
        how="left",
    )

    # ── for each game in today's df, look up form stats ───────────────────────
    today = (
        pd.to_datetime(df["game_date"].iloc[0])
        if "game_date" in df.columns
        else pd.Timestamp.now()
    )

    # use most recent row per team as today's form
    latest = (
        team_log[team_log["game_date"] < today].groupby("team").last().reset_index()
    )
    form_cols = (
        [f"win_pct_{w}" for w in FORM_WINDOWS]
        + [f"run_diff_avg_{w}" for w in FORM_WINDOWS]
        + [f"pyth_win_pct_{w}" for w in PYTH_WINDOWS]
        + ["home_win_pct", "away_win_pct"]
    )
    latest_form = latest[["team"] + [c for c in form_cols if c in latest.columns]]

    # merge for home team
    df = df.merge(
        latest_form.rename(
            columns={c: f"{c}_h" for c in form_cols if c in latest.columns}
            | {"team": "team_h"}
        ),
        on="team_h",
        how="left",
    )
    # merge for away team
    df = df.merge(
        latest_form.rename(
            columns={
                **{c: f"{c}_v" for c in form_cols if c in latest.columns},
                "team": "team_v",
                "home_win_pct_v": "home_win_pct_v",
                "away_win_pct_v": "away_win_pct_v",
            }
        ),
        on="team_v",
        how="left",
    )

    # rename to match model feature names
    df = df.rename(
        columns={
            "home_win_pct_h": "home_win_pct_h",
            "away_win_pct_v": "away_win_pct_v",
        }
    )

    # ── h2h win% (home team vs this specific away team) ───────────────────────
    h2h_records = []
    for _, row in df.iterrows():
        team_h = row["team_h"]
        team_v = row["team_v"]
        prior = history[
            (history["team_h"] == team_h)
            & (history["team_v"] == team_v)
            & (history["game_date"] < today)
        ].tail(
            30
        )  # last 30 matchups
        games = len(prior)
        win_pct = prior["home_victory"].mean() if games > 0 else 0.5
        h2h_records.append({"h2h_win_pct_h": win_pct, "h2h_games": games})

    h2h_df = pd.DataFrame(h2h_records, index=df.index)
    df = pd.concat([df, h2h_df], axis=1)

    # fill NaNs with neutral priors
    fills = {
        **{f"win_pct_{w}_h": 0.50 for w in FORM_WINDOWS},
        **{f"win_pct_{w}_v": 0.50 for w in FORM_WINDOWS},
        **{f"run_diff_avg_{w}_h": 0.0 for w in FORM_WINDOWS},
        **{f"run_diff_avg_{w}_v": 0.0 for w in FORM_WINDOWS},
        **{f"pyth_win_pct_{w}_h": 0.50 for w in PYTH_WINDOWS},
        **{f"pyth_win_pct_{w}_v": 0.50 for w in PYTH_WINDOWS},
        "home_win_pct_h": 0.54,  # MLB home teams win ~54%
        "away_win_pct_v": 0.46,
        "h2h_win_pct_h": 0.50,
        "h2h_games": 0,
    }
    for col, val in fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


def assemble_win_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map pipeline column names to win model feature names.

    Pipeline has:
      - Per-slot batter cols: OBP_162_b1_h, OBP_162_b2_h ... OBP_162_b9_h
      - Starter cols:  Strt_WHIP_35_h, Strt_ERA_75_v etc.
      - Bullpen cols:  Bpen_WHIP_35_h, Bpen_SO_perc_75_v etc.

    Win model expects:
      - Averaged lineup: obp_162_h, slg_30_v etc.
      - Starters: strt_whip_35_h, strt_era_75_v etc.
      - Bullpen:  bpen_whip_35_h, bpen_so_perc_75_v etc.
    """
    BAT_WINDOWS = [14, 30, 75, 162, 350]
    BAT_STEMS = ["obp", "slg", "barrel", "hr_per_pa"]
    STEM_COL_MAP = {
        "obp": "OBP",
        "slg": "SLG",
        "barrel": "BARREL",
        "hr_per_pa": "HR_per_PA",
    }
    SLOTS = range(1, 10)

    # ── 1. average per-slot batter cols → lineup avg ──────────────────────────
    for stem in BAT_STEMS:
        upper = STEM_COL_MAP[stem]
        for w in BAT_WINDOWS:
            for hv in ["h", "v"]:
                slot_cols = [
                    f"{upper}_{w}_b{slot}_{hv}"
                    for slot in SLOTS
                    if f"{upper}_{w}_b{slot}_{hv}" in df.columns
                ]
                if slot_cols:
                    df[f"{stem}_{w}_{hv}"] = df[slot_cols].mean(axis=1)
                else:
                    # fallback: try the lineup9 pre-averaged column
                    ln_col = f"lineup9_{upper}_{w}_{hv}"
                    if ln_col in df.columns:
                        df[f"{stem}_{w}_{hv}"] = df[ln_col]
                    else:
                        df[f"{stem}_{w}_{hv}"] = np.nan

    # ── 2. rename Strt_ → strt_ ───────────────────────────────────────────────
    strt_rename = {
        col: col.replace("Strt_", "strt_").lower()
        for col in df.columns
        if col.startswith("Strt_")
    }
    df = df.rename(columns=strt_rename)

    # ── 3. rename Bpen_ → bpen_ ───────────────────────────────────────────────
    bpen_rename = {
        col: col.replace("Bpen_", "bpen_").lower()
        for col in df.columns
        if col.startswith("Bpen_")
    }
    df = df.rename(columns=bpen_rename)

    return df


def get_team_cols(df):
    visiting_cols = [col for col in df.columns if not col.endswith("_h")]
    visiting_cols_stripped = [strip_suffix(col, "_v") for col in visiting_cols]
    home_cols = [col for col in df.columns if not col.endswith("_v")]
    home_cols_stripped = [strip_suffix(col, "_h") for col in home_cols]
    return home_cols, home_cols_stripped, visiting_cols, visiting_cols_stripped


def create_team_df(df, team):
    cols = ["AB", "H", "x2B", "x3B", "HR", "BB", "SB", "CS"]
    cols_h = ["AB_h", "H_h", "x2B_h", "x3B_h", "HR_h", "BB_h", "SB_h", "CS_h"]
    cols_v = ["AB_v", "H_v", "x2B_v", "x3B_v", "HR_v", "BB_v", "SB_v", "CS_v"]

    df_team_v = df[(df.team_v == team)]
    opp = df_team_v["team_h"]
    df_team_v = df_team_v[cols_v + ["date_dblhead"]]
    df_team_v.rename(
        columns={
            "AB_v": "AB",
            "H_v": "H",
            "x2B_v": "x2B",
            "x3B_v": "x3B",
            "HR_v": "HR",
            "BB_v": "BB",
            "SB_v": "SB",
            "CS_v": "CS",
        },
        inplace=True,
    )
    df_team_v["home_game"] = 0
    df_team_v["opponent"] = opp

    df_team_h = df[(df.team_h == team)]
    opp = df_team_h["team_v"]
    df_team_h = df_team_h[cols_h + ["date_dblhead"]]
    df_team_h.rename(
        columns={
            "AB_h": "AB",
            "H_h": "H",
            "x2B_h": "x2B",
            "x3B_h": "x3B",
            "HR_h": "HR",
            "BB_h": "BB",
            "SB_h": "SB",
            "CS_h": "CS",
        },
        inplace=True,
    )
    df_team_h["home_game"] = 1
    df_team_h["opponent"] = opp

    df_team = df_team_h if not df_team_h.empty else df_team_v
    df_team = df_team.set_index("date_dblhead")

    for winsize in WINDOWS:
        suff = str(winsize)
        for raw_col in cols:
            new_col = "rollsum_" + raw_col + "_" + suff
            df_team[new_col] = df_team[raw_col].rolling(winsize, closed="left").sum()

        df_team["rollsum_BATAVG_" + suff] = (
            df_team["rollsum_H_" + suff] / df_team["rollsum_AB_" + suff]
        )
        df_team["rollsum_OBP_" + suff] = (
            df_team["rollsum_H_" + suff] + df_team["rollsum_BB_" + suff]
        ) / (df_team["rollsum_BB_" + suff] + df_team["rollsum_AB_" + suff])
        df_team["rollsum_SLG_" + suff] = (
            df_team["rollsum_H_" + suff]
            + df_team["rollsum_x2B_" + suff]
            + 2 * df_team["rollsum_x3B_" + suff]
            + 3 * df_team["rollsum_HR_" + suff]
        ) / df_team["rollsum_AB_" + suff]
        df_team["rollsum_OBS_" + suff] = (
            df_team["rollsum_OBP_" + suff] + df_team["rollsum_SLG_" + suff]
        )

    return df_team


def generate_team_window_features(df):
    team_data_dict = {}
    teams = df[["team_h", "team_v"]].stack().unique().tolist()
    for team in teams:
        team_data_dict[team] = create_team_df(df, team)

    stats = ["BATAVG", "OBP", "SLG", "OBS", "SB", "CS"]
    teams_ = ["h", "v"]

    arrays = {
        f"{stat}_{window}_{team}": np.zeros(df.shape[0])
        for stat in stats
        for window in WINDOWS
        for team in teams_
    }

    for i, (index, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
        home_team = row["team_h"]
        visit_team = row["team_v"]
        game_index = row["date_dblhead"]

        for window in WINDOWS:
            for stat in stats:
                arrays[f"{stat}_{window}_h"][i] = team_data_dict[home_team].loc[
                    game_index, f"rollsum_{stat}_{window}"
                ]
                arrays[f"{stat}_{window}_v"][i] = team_data_dict[visit_team].loc[
                    game_index, f"rollsum_{stat}_{window}"
                ]

    for key, value in arrays.items():
        df[key] = value

    return df
