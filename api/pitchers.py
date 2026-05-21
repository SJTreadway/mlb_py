"""
pitchers.py  —  Snowflake edition
───────────────────────────────────
Replaces pybaseball API + CSV approach with direct Snowflake queries
against PITCHER_ROLLING_FEATURES.

Pre-condition: compute_rolling_features() must have run today so
PITCHER_ROLLING_FEATURES has up-to-date WHIP / SO_PERC / ERA etc.

Returns same signature as original: (df, pitcher_data_dict)
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import snowflake.connector
import streamlit as st
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

log = logging.getLogger(__name__)

WINDOWS = [10, 35, 75]

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


# ── column mapping ─────────────────────────────────────────────────────────────

STARTER_COL_MAP = {
    # smoothed rate features
    "whip_10": "WHIP_10",
    "whip_35": "WHIP_35",
    "whip_75": "WHIP_75",
    "fip_10": "FIP_10",
    "fip_35": "FIP_35",
    "fip_75": "FIP_75",
    "fip_perc_10": "FIP_perc_10",
    "fip_perc_35": "FIP_perc_35",
    "fip_perc_75": "FIP_perc_75",
    "so_perc_10": "SO_perc_10",
    "so_perc_35": "SO_perc_35",
    "so_perc_75": "SO_perc_75",
    "h_bb_perc_10": "H_BB_perc_10",
    "h_bb_perc_35": "H_BB_perc_35",
    "h_bb_perc_75": "H_BB_perc_75",
    "tb_bb_perc_10": "TB_BB_perc_10",
    "tb_bb_perc_35": "TB_BB_perc_35",
    "tb_bb_perc_75": "TB_BB_perc_75",
    "era_10": "ERA_10",
    "era_35": "ERA_35",
    "era_75": "ERA_75",
    "hr_per_bf_10": "HR_per_BF_10",
    "hr_per_bf_35": "HR_per_BF_35",
    "hr_per_bf_75": "HR_per_BF_75",
    "fb_perc_10": "FB_perc_10",
    "fb_perc_35": "FB_perc_35",
    "fb_perc_75": "FB_perc_75",
    # rolling sums
    "rollsum_h_10": "rollsum_H_10",
    "rollsum_h_35": "rollsum_H_35",
    "rollsum_h_75": "rollsum_H_75",
    "rollsum_bb_10": "rollsum_BB_10",
    "rollsum_bb_35": "rollsum_BB_35",
    "rollsum_bb_75": "rollsum_BB_75",
    "rollsum_so_10": "rollsum_SO_10",
    "rollsum_so_35": "rollsum_SO_35",
    "rollsum_so_75": "rollsum_SO_75",
    "rollsum_ip_10": "rollsum_IP_real_10",
    "rollsum_ip_35": "rollsum_IP_real_35",
    "rollsum_ip_75": "rollsum_IP_real_75",
    "rollsum_bfp_10": "rollsum_BFP_10",
    "rollsum_bfp_35": "rollsum_BFP_35",
    "rollsum_bfp_75": "rollsum_BFP_75",
    "rollsum_hr_10": "rollsum_HR_10",
    "rollsum_hr_35": "rollsum_HR_35",
    "rollsum_hr_75": "rollsum_HR_75",
    "rollsum_er_10": "rollsum_ER_10",
    "rollsum_er_35": "rollsum_ER_35",
    "rollsum_er_75": "rollsum_ER_75",
    # raw game stats
    "gs": "GS",
    "ip": "IP_real",
    "bfp": "BFP",
    "hr": "HR",
    "r": "R",
}

BULLPEN_COL_RENAME = {
    "whip_10": "Bpen_WHIP_10",
    "whip_35": "Bpen_WHIP_35",
    "whip_75": "Bpen_WHIP_75",
    "so_perc_10": "Bpen_SO_perc_10",
    "so_perc_35": "Bpen_SO_perc_35",
    "so_perc_75": "Bpen_SO_perc_75",
    "h_bb_perc_10": "Bpen_H_BB_perc_10",
    "h_bb_perc_35": "Bpen_H_BB_perc_35",
    "h_bb_perc_75": "Bpen_H_BB_perc_75",
    "tb_bb_perc_10": "Bpen_TB_BB_perc_10",
    "tb_bb_perc_35": "Bpen_TB_BB_perc_35",
    "tb_bb_perc_75": "Bpen_TB_BB_perc_75",
}


# ── Snowflake data loads ───────────────────────────────────────────────────────


def load_starter_data_from_snowflake(pitcher_ids: list[str]) -> dict[str, pd.Series]:
    """Fetch the most recent PITCHER_ROLLING_FEATURES row per starting pitcher."""
    if not pitcher_ids:
        return {}

    db = os.environ.get("SNOWFLAKE_DATABASE", "BASEBALL")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "STATCAST")
    ids = ", ".join(pitcher_ids)

    query = f"""
        SELECT p.*
        FROM {db}.{schema}.PITCHER_ROLLING_FEATURES p
        INNER JOIN (
            SELECT MLBAM_ID, MAX(GAME_DATE) AS latest_date
            FROM {db}.{schema}.PITCHER_ROLLING_FEATURES
            WHERE MLBAM_ID IN ({ids})
            GROUP BY MLBAM_ID
        ) latest
            ON p.MLBAM_ID   = latest.MLBAM_ID
            AND p.GAME_DATE = latest.latest_date
    """

    log.info(f"Querying Snowflake for {len(pitcher_ids)} starters …")
    conn = snowflake.connector.connect(**_sf_cfg())
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        cols = [d[0].lower() for d in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=cols)
    finally:
        cursor.close()
        conn.close()

    log.info(f"Pulled {len(df)} starter rows from Snowflake")
    return {str(int(row["mlbam_id"])): row for _, row in df.iterrows()}


def load_bullpen_data_from_snowflake(teams: list[str]) -> dict[str, dict]:
    """Fetch BFP-weighted bullpen rolling stats per team."""
    if not teams:
        return {}

    db = os.environ.get("SNOWFLAKE_DATABASE", "BASEBALL")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "STATCAST")
    team_list = ", ".join(f"'{t}'" for t in teams)

    rate_cols_10 = ["whip_10", "so_perc_10", "h_bb_perc_10", "tb_bb_perc_10"]
    rate_cols_35 = ["whip_35", "so_perc_35", "h_bb_perc_35", "tb_bb_perc_35"]
    rate_cols_75 = ["whip_75", "so_perc_75", "h_bb_perc_75", "tb_bb_perc_75"]

    def weighted_agg(col, weight):
        return f"SUM({col} * {weight}) / NULLIF(SUM({weight}), 0) AS {col}"

    agg_exprs = ", ".join(
        [
            *[weighted_agg(c, "rollsum_bfp_10") for c in rate_cols_10],
            *[weighted_agg(c, "rollsum_bfp_35") for c in rate_cols_35],
            *[weighted_agg(c, "rollsum_bfp_75") for c in rate_cols_75],
        ]
    )

    query = f"""
        WITH latest_relief AS (
            SELECT
                p.mlbam_id,
                CASE WHEN p.is_home_pitcher = 1 THEN gr.team_h ELSE gr.team_v END AS team,
                p.whip_10, p.so_perc_10, p.h_bb_perc_10, p.tb_bb_perc_10,
                p.whip_35, p.so_perc_35, p.h_bb_perc_35, p.tb_bb_perc_35,
                p.whip_75, p.so_perc_75, p.h_bb_perc_75, p.tb_bb_perc_75,
                p.rollsum_bfp_10, p.rollsum_bfp_35, p.rollsum_bfp_75,
                ROW_NUMBER() OVER (
                    PARTITION BY p.mlbam_id ORDER BY p.game_date DESC
                ) AS rn
            FROM {db}.{schema}.PITCHER_ROLLING_FEATURES p
            JOIN {db}.{schema}.GAME_RESULTS gr ON p.game_pk = gr.game_pk
            WHERE p.gs = 0
              AND p.game_date >= DATEADD(day, -90, CURRENT_DATE)
              AND CASE WHEN p.is_home_pitcher = 1 THEN gr.team_h ELSE gr.team_v END
                  IN ({team_list})
        )
        SELECT
            team,
            {agg_exprs}
        FROM latest_relief
        WHERE rn = 1
          AND rollsum_bfp_35 > 0
        GROUP BY team
    """

    log.info(f"Querying bullpen stats for {len(teams)} teams …")
    conn = snowflake.connector.connect(**_sf_cfg())
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        cols = [d[0].lower() for d in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=cols)
    finally:
        cursor.close()
        conn.close()

    log.info(f"Pulled bullpen stats for {len(df)} teams")
    return {row["team"]: row.to_dict() for _, row in df.iterrows()}


# ── feature assembly ───────────────────────────────────────────────────────────


def _get_starter_defaults() -> dict:
    return {
        "WHIP_10": 1.35,
        "WHIP_35": 1.35,
        "WHIP_75": 1.35,
        "FIP_10": 4.20,
        "FIP_35": 4.20,
        "FIP_75": 4.20,
        "FIP_perc_10": 0.35,
        "FIP_perc_35": 0.35,
        "FIP_perc_75": 0.35,
        "SO_perc_10": 0.22,
        "SO_perc_35": 0.22,
        "SO_perc_75": 0.22,
        "H_BB_perc_10": 0.37,
        "H_BB_perc_35": 0.37,
        "H_BB_perc_75": 0.37,
        "TB_BB_perc_10": 0.45,
        "TB_BB_perc_35": 0.45,
        "TB_BB_perc_75": 0.45,
        "ERA_10": 4.50,
        "ERA_35": 4.50,
        "ERA_75": 4.50,
        "HR_per_BF_10": 0.032,
        "HR_per_BF_35": 0.032,
        "HR_per_BF_75": 0.032,
        "FB_perc_10": 0.35,
        "FB_perc_35": 0.35,
        "FB_perc_75": 0.35,
        "GS": 1,
        "IP_real": 5.5,
        "BFP": 22,
        "HR": 1,
        "R": 3,
    }


def _get_bullpen_defaults() -> dict:
    return {
        "whip_10": 1.35,
        "whip_35": 1.35,
        "whip_75": 1.35,
        "so_perc_10": 0.22,
        "so_perc_35": 0.22,
        "so_perc_75": 0.22,
        "h_bb_perc_10": 0.37,
        "h_bb_perc_35": 0.37,
        "h_bb_perc_75": 0.37,
        "tb_bb_perc_10": 0.45,
        "tb_bb_perc_35": 0.45,
        "tb_bb_perc_75": 0.45,
    }


def assemble_starter_features(
    df: pd.DataFrame,
    starter_data_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    defaults = _get_starter_defaults()
    strt_cols = {
        f"Strt_{pipeline_name}_{hv}": np.zeros(len(df))
        for _, pipeline_name in STARTER_COL_MAP.items()
        for hv in ["h", "v"]
    }

    for i, (_, row) in enumerate(df.iterrows()):
        for hv, id_col in [
            ("h", "starting_pitcher_id_h"),
            ("v", "starting_pitcher_id_v"),
        ]:
            p_id_raw = row.get(id_col)
            if p_id_raw is None or pd.isna(p_id_raw):
                continue
            p_id = str(int(p_id_raw))

            if p_id in starter_data_dict:
                sf_row = starter_data_dict[p_id]
                feat_map = {
                    pipeline_name: float(sf_row.get(sf_col, np.nan) or 0)
                    for sf_col, pipeline_name in STARTER_COL_MAP.items()
                }
            else:
                log.debug(f"No Snowflake data for starter {p_id} — using defaults")
                feat_map = defaults

            for _, pipeline_name in STARTER_COL_MAP.items():
                strt_cols[f"Strt_{pipeline_name}_{hv}"][i] = feat_map.get(
                    pipeline_name, 0
                )

    return pd.concat([df, pd.DataFrame(strt_cols, index=df.index)], axis=1)


def assemble_bullpen_features(
    df: pd.DataFrame,
    bullpen_data_dict: dict[str, dict],
) -> pd.DataFrame:
    defaults = _get_bullpen_defaults()
    bpen_cols = {
        f"{pipeline_name}_{hv}": np.zeros(len(df))
        for _, pipeline_name in BULLPEN_COL_RENAME.items()
        for hv in ["h", "v"]
    }

    for i, (_, row) in enumerate(df.iterrows()):
        for hv, team_col in [("h", "team_h"), ("v", "team_v")]:
            team = row.get(team_col)
            bp_data = bullpen_data_dict.get(team, defaults)
            for sf_col, pipeline_name in BULLPEN_COL_RENAME.items():
                bpen_cols[f"{pipeline_name}_{hv}"][i] = float(
                    bp_data.get(sf_col, defaults.get(sf_col, 0)) or 0
                )

    return pd.concat([df, pd.DataFrame(bpen_cols, index=df.index)], axis=1)


# ── public entry point ─────────────────────────────────────────────────────────


def process_pitching_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Pull pitcher rolling features from Snowflake and assemble starter + bullpen stats.

    Drop-in replacement for the old pybaseball/CSV version.
    Returns (enriched_df, pitcher_data_dict) — same signature as before.
    """
    # ── 1. extract starting pitcher IDs ──────────────────────────────────────
    start_pitchers_h = [
        str(int(p))
        for p in df["starting_pitcher_id_h"].dropna().unique()
        if str(p) != "nan"
    ]
    start_pitchers_v = [
        str(int(p))
        for p in df["starting_pitcher_id_v"].dropna().unique()
        if str(p) != "nan"
    ]
    all_starter_ids = list(set(start_pitchers_h + start_pitchers_v))
    log.info(f"Found {len(all_starter_ids)} unique starters in today's games")

    # ── 2. pull starter features from Snowflake ───────────────────────────────
    starter_data_dict = load_starter_data_from_snowflake(all_starter_ids)
    missing = len(all_starter_ids) - len(starter_data_dict)
    if missing:
        missing_ids = [p for p in all_starter_ids if p not in starter_data_dict]
        log.warning(
            f"{missing} starters not found in Snowflake — using league-average defaults"
        )
        st.warning(
            f"⚠️ {missing} starter(s) not found in Snowflake — league-average defaults used. "
            f"IDs: {', '.join(missing_ids)}",
            icon=None,
        )

    # ── 3. assemble Strt_ columns ─────────────────────────────────────────────
    df = assemble_starter_features(df, starter_data_dict)

    # ── 4. pull bullpen features per team ─────────────────────────────────────
    teams = list(set(df["team_h"].tolist() + df["team_v"].tolist()))
    bullpen_data_dict = load_bullpen_data_from_snowflake(teams)
    missing_teams = len(teams) - len(bullpen_data_dict)
    if missing_teams:
        missing_team_list = [t for t in teams if t not in bullpen_data_dict]
        log.warning(
            f"{missing_teams} teams missing bullpen data — using league-average defaults"
        )
        st.warning(
            f"⚠️ {missing_teams} team(s) missing bullpen data — league-average defaults used. "
            f"Teams: {', '.join(missing_team_list)}",
            icon=None,
        )

    # ── 5. assemble Bpen_ columns ─────────────────────────────────────────────
    df = assemble_bullpen_features(df, bullpen_data_dict)

    return df, starter_data_dict
