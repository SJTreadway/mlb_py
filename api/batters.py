"""
batters.py  —  Snowflake edition
──────────────────────────────────
Replaces pybaseball API + CSV approach with a direct Snowflake query
against BATTER_ROLLING_FEATURES. Rolling features are pre-computed by
compute_rolling_features() and stored in Snowflake before this runs.

Key improvements:
  - No Statcast API calls
  - No CSV reads or writes
  - No per-batter rolling computation in Python
  - One batched Snowflake query for all lineup batters
  - Same return signature as original — pipeline.py unchanged

TODO: Add SO to bat_stat_cols in compute_rolling_features() so
      SLGmod and SObat_perc can be computed precisely rather than
      approximated from position defaults.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import requests
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

load_dotenv()

from helpers import resolve_names

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

log = logging.getLogger(__name__)

WINDOWS = [7, 14, 30, 75, 162, 350]
MLB_API_PEOPLE = "https://statsapi.mlb.com/api/v1/people"


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


# ── position defaults (unchanged from original) ────────────────────────────────


def get_position_defaults() -> dict:
    dd_p = {
        "batavg": 0.100,
        "obp": 0.150,
        "slg": 0.180,
        "slgmod": 0.220,
        "obs": 0.330,
        "sobat": 0.30,
    }
    dd_ss_c = {
        "batavg": 0.205,
        "obp": 0.260,
        "slg": 0.300,
        "slgmod": 0.320,
        "obs": 0.540,
        "sobat": 0.25,
    }
    dd_2b_3b = {
        "batavg": 0.240,
        "obp": 0.280,
        "slg": 0.350,
        "slgmod": 0.355,
        "obs": 0.630,
        "sobat": 0.20,
    }
    dd_rest = {
        "batavg": 0.255,
        "obp": 0.310,
        "slg": 0.380,
        "slgmod": 0.430,
        "obs": 0.690,
        "sobat": 0.20,
    }
    return {
        "p": dd_p,
        "ss": dd_ss_c,
        "c": dd_ss_c,
        "2b": dd_2b_3b,
        "3b": dd_2b_3b,
        "1b": dd_rest,
        "lf": dd_rest,
        "rf": dd_rest,
        "cf": dd_rest,
        "ph": dd_rest,
        "dh": dd_rest,
        "pr": dd_ss_c,
    }


# ── MLB people API (position lookup) ──────────────────────────────────────────


def _fetch_positions_from_mlb_api(mlbam_ids: list[str]) -> dict:
    chunk_size = 200
    pos_map = {}
    for i in range(0, len(mlbam_ids), chunk_size):
        chunk = mlbam_ids[i : i + chunk_size]
        params = {
            "personIds": ",".join(str(x) for x in chunk),
            "fields": "people,id,primaryPosition,abbreviation",
        }
        try:
            resp = requests.get(MLB_API_PEOPLE, params=params, timeout=10)
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                pos_map[person["id"]] = person.get("primaryPosition", {}).get(
                    "abbreviation", ""
                )
        except Exception as e:
            log.warning(f"MLB API error for chunk starting at {i}: {e}")
    return pos_map


def build_position_map(batter_ids: list[str]) -> dict[str, str]:
    valid_ids = [b for b in batter_ids if b]
    pos_by_id = _fetch_positions_from_mlb_api(valid_ids)
    return {str(mid): pos.lower() for mid, pos in pos_by_id.items()}


# ── Snowflake data load ────────────────────────────────────────────────────────


def load_batter_data_from_snowflake(
    batter_ids: list[str],
) -> dict[str, pd.Series]:
    """Fetch the most recent BATTER_ROLLING_FEATURES row per batter in one query.

    Returns a dict: str(mlbam_id) → pd.Series of that batter's latest features.
    Batters with no data in Snowflake are absent from the dict — callers fall
    back to position defaults for those slots.
    """
    if not batter_ids:
        return {}

    db = os.environ.get("SNOWFLAKE_DATABASE", "BASEBALL")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "STATCAST")
    ids = ", ".join(batter_ids)

    query = f"""
        SELECT b.*
        FROM {db}.{schema}.BATTER_ROLLING_FEATURES b
        INNER JOIN (
            SELECT MLBAM_ID, MAX(GAME_DATE) AS latest_date
            FROM {db}.{schema}.BATTER_ROLLING_FEATURES
            WHERE MLBAM_ID IN ({ids})
            GROUP BY MLBAM_ID
        ) latest
            ON b.MLBAM_ID    = latest.MLBAM_ID
            AND b.GAME_DATE  = latest.latest_date
    """

    log.info(f"Querying Snowflake for {len(batter_ids)} batters …")
    conn = snowflake.connector.connect(**_sf_cfg())
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        cols = [d[0].lower() for d in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=cols)
    finally:
        cursor.close()
        conn.close()

    log.info(f"Pulled {len(df)} batter rows from Snowflake")

    # Build dict: mlbam_id (str) → Series
    result = {}
    for _, row in df.iterrows():
        mid = str(int(row["mlbam_id"]))
        result[mid] = row
    return result


def load_bvp_data_from_snowflake(
    batter_pitcher_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict]:
    """Fetch cumulative BvP stats for batter-pitcher pairs from BVP_HISTORY.

    Args:
        batter_pitcher_pairs: list of (batter_id, pitcher_id) tuples

    Returns:
        dict: (batter_id, pitcher_id) → dict with bvp_pa, bvp_hr, bvp_hr_rate_smoothed
    """
    if not batter_pitcher_pairs:
        return {}

    db = os.environ.get("SNOWFLAKE_DATABASE", "BASEBALL")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "STATCAST")

    # Build IN clause for batter-pitcher pairs
    pairs_sql = ", ".join(f"({b}, {p})" for b, p in batter_pitcher_pairs if b and p)

    query = f"""
        SELECT
            BATTER,
            PITCHER,
            BVP_PA_PRIOR AS total_pa,
            BVP_HR_PRIOR AS total_hr
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY BATTER, PITCHER
                    ORDER BY GAME_DATE DESC
                ) AS rn
            FROM {db}.{schema}.BVP_HISTORY
            WHERE (BATTER, PITCHER) IN ({pairs_sql})
        )
        WHERE rn = 1
    """

    log.info(
        f"Querying BVP_HISTORY for {len(batter_pitcher_pairs)} batter-pitcher pairs …"
    )
    conn = snowflake.connector.connect(**_sf_cfg())
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        cols = [d[0].lower() for d in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=cols)
    finally:
        cursor.close()
        conn.close()

    log.info(f"Pulled BvP data for {len(df)} pairs")

    result = {}
    for _, row in df.iterrows():
        b_id = str(int(row["batter"]))
        p_id = str(int(row["pitcher"]))
        pa = int(row["total_pa"] or 0)
        hr = int(row["total_hr"] or 0)
        result[(b_id, p_id)] = {
            "bvp_pa": pa,
            "bvp_hr": hr,
            "bvp_hr_rate_smoothed": (hr + 50 * 0.034) / (pa + 50),
        }
    return result


def _derive_batter_features(row: pd.Series, pos: str) -> dict:
    """Compute the full colstem feature set for one batter from a Snowflake row.

    Snowflake has: OBP, SLG, OBS, BARREL, HARDHIT, SWSPOT, EV at all windows.
    Missing: BATAVG (derived from rollsums), SLGmod & SObat_perc (approximated).

    Returns a flat dict keyed by the colstem format used in get_lineup_averages():
      {OBP_162: 0.341, SLG_162: 0.478, BATAVG_162: 0.271, ...}
    """
    defaults = get_position_defaults()
    pos_defs = defaults.get(pos, defaults["dh"])
    features = {}

    ab_per_game = 2
    pa_per_game = 2

    for w in WINDOWS:
        sw = str(w)

        # ── direct from Snowflake ──────────────────────────────────────────
        for stem, col in [
            ("OBP", f"obp_{sw}"),
            ("SLG", f"slg_{sw}"),
            ("OBS", f"obs_{sw}"),
            ("EV", f"ev_{sw}"),
            ("HARDHIT", f"hardhit_{sw}"),
            ("SWSPOT", f"swspot_{sw}"),
            ("BARREL", f"barrel_{sw}"),
        ]:
            val = row.get(col, np.nan)
            features[f"{stem}_{w}"] = (
                float(val) if pd.notna(val) else pos_defs.get(stem.lower(), np.nan)
            )

        # ── BATAVG: derived from rollsums with position smoothing ──────────
        rollsum_h = float(row.get(f"rollsum_h_{sw}", 0) or 0)
        rollsum_ab = float(row.get(f"rollsum_ab_{sw}", 0) or 0)
        abmod = max(rollsum_ab, w * ab_per_game)
        fakeab = min(abmod - rollsum_ab, 0)
        features[f"BATAVG_{w}"] = (
            (rollsum_h + fakeab * pos_defs["batavg"]) / abmod
            if abmod > 0
            else pos_defs["batavg"]
        )

        # ── SLGmod: approximated (no SO rollsum in Snowflake yet)  ─────────
        # Uses SLG as a proxy. Add SO to compute_rolling_features() bat_stat_cols
        # to enable exact computation.
        features[f"SLGmod_{w}"] = features[f"SLG_{w}"]

        # ── SObat_perc: position default (no SO rollsum available) ─────────
        features[f"SObat_perc_{w}"] = pos_defs["sobat"]

    return features


# ── feature assembly ───────────────────────────────────────────────────────────


def get_batting_feats(
    df: pd.DataFrame,
    batter_data_dict: dict[str, pd.Series],
    pos_map: dict[str, str],
) -> pd.DataFrame:
    """Map per-batter Snowflake features to per-slot columns on the game df.

    Output columns follow the existing convention:
        OBP_162_b1_h, SLG_162_b3_v, BARREL_75_b9_h, ...

    Batters absent from Snowflake (no prior history) receive position defaults.
    """
    defaults = get_position_defaults()

    colstems = [
        "BATAVG",
        "OBP",
        "SLG",
        "OBS",
        "SLGmod",
        "SObat_perc",
        "EV",
        "HARDHIT",
        "SWSPOT",
        "BARREL",
    ]

    # Pre-allocate all slot columns as NaN
    new_cols: dict[str, np.ndarray] = {
        f"{stem}_{w}_b{slot}{hv}": np.full(len(df), np.nan)
        for stem in colstems
        for w in WINDOWS
        for slot in range(1, 10)
        for hv in ["_h", "_v"]
    }

    for i, (_, row) in enumerate(df.iterrows()):
        for hv in ["_h", "_v"]:
            for slot in range(1, 10):
                b_id_raw = row.get(f"batter{slot}_id{hv}")
                if b_id_raw is None or pd.isna(b_id_raw):
                    continue
                b_id = str(int(b_id_raw))

                if b_id in batter_data_dict:
                    pos = pos_map.get(b_id, "dh")
                    sf_row = batter_data_dict[b_id]
                    feat_map = _derive_batter_features(sf_row, pos)
                else:
                    # No Snowflake data — fill with position defaults
                    pos = pos_map.get(b_id, "dh")
                    pos_defs = defaults.get(pos, defaults["dh"])
                    feat_map = {
                        f"{stem}_{w}": pos_defs.get(stem.lower(), np.nan)
                        for stem in colstems
                        for w in WINDOWS
                    }
                    log.debug(
                        f"No Snowflake data for batter {b_id} — using {pos} defaults"
                    )

                for stem in colstems:
                    for w in WINDOWS:
                        col = f"{stem}_{w}_b{slot}{hv}"
                        new_cols[col][i] = feat_map.get(f"{stem}_{w}", np.nan)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ── lineup averages (unchanged from original) ──────────────────────────────────


def get_lineup_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-slot batter features to lineup-level weighted averages.

    Produces: lineup9_OBP_162_h, lineup8_SLG_75_w_v, etc.
    Identical logic to the original — no changes needed here.
    """
    default_dict = get_position_defaults()
    colstems = ["BATAVG", "OBP", "SLG", "OBS", "SLGmod", "SObat_perc"]
    stem_key_map = {
        "BATAVG": "batavg",
        "OBP": "obp",
        "SLG": "slg",
        "OBS": "obs",
        "SLGmod": "slgmod",
        "SObat_perc": "sobat",
    }

    # Fill NaN slot values with position pitcher default before averaging
    all_slot_cols = [
        f"{stem}_{w}_b{i}{hv}"
        for stem in colstems
        for w in WINDOWS
        for hv in ["_h", "_v"]
        for i in range(1, 10)
    ]
    for col in all_slot_cols:
        if col in df.columns:
            stem = col.split("_")[0]
            dict_key = stem_key_map.get(stem, "batavg")
            df[col] = df[col].fillna(default_dict["p"][dict_key])

    # Batting order weights (slot 1 gets most weight, slot 9 least)
    w9 = np.array(
        [
            0.12541131,
            0.12159052,
            0.11787189,
            0.11434144,
            0.11096691,
            0.10772781,
            0.10430724,
            0.10078822,
            0.09699465,
        ]
    )
    w8 = w9[:-1] / w9[:-1].sum()

    for stem in colstems:
        for w in WINDOWS:
            for hv in ["_h", "_v"]:
                b9 = [f"{stem}_{w}_b{i}{hv}" for i in range(1, 10)]
                b8 = [f"{stem}_{w}_b{i}{hv}" for i in range(1, 9)]
                df[f"lineup9_{stem}_{w}{hv}"] = df[b9].to_numpy().mean(axis=1)
                df[f"lineup8_{stem}_{w}{hv}"] = df[b8].to_numpy().mean(axis=1)
                df[f"lineup9_{stem}_{w}_w{hv}"] = df[b9].to_numpy().dot(w9)
                df[f"lineup8_{stem}_{w}_w{hv}"] = df[b8].to_numpy().dot(w8)

    return df


# ── public entry point ─────────────────────────────────────────────────────────


def process_batting_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict]:
    """Pull batter rolling features from Snowflake and assemble lineup stats.

    Returns (enriched_df, batter_data_dict, bvp_dict).
    bvp_dict: (batter_id, pitcher_id) → {bvp_pa, bvp_hr, bvp_hr_rate_smoothed}
    """
    # ── 1. extract unique batter IDs from lineup columns ──────────────────
    batter_ids: set[str] = set()
    for slot in range(1, 10):
        for hv in ["_h", "_v"]:
            col = f"batter{slot}_id{hv}"
            if col in df.columns:
                for val in df[col].dropna().unique():
                    try:
                        batter_ids.add(str(int(val)))
                    except (ValueError, TypeError):
                        pass

    batter_ids_list = list(batter_ids)
    log.info(f"Found {len(batter_ids_list)} unique batters in today's lineups")

    # ── 2. build position map (MLB API — fast, ~1s for 100 batters) ───────
    pos_map = build_position_map(batter_ids_list)

    # ── 3. pull latest rolling features from Snowflake ────────────────────
    batter_data_dict = load_batter_data_from_snowflake(batter_ids_list)
    missing = len(batter_ids_list) - len(batter_data_dict)
    warnings = []
    if missing:
        missing_ids = [b for b in batter_ids_list if b not in batter_data_dict]
        msg = (
            f"{missing} batter(s) not found in Snowflake — position defaults used. "
            f"{', '.join(list(resolve_names(missing_ids).values()))}"
        )
        log.warning(msg)
        warnings.append(msg)

    # ── 4. pull BvP history for all batter-pitcher matchups ───────────────
    batter_pitcher_pairs: set[tuple[str, str]] = set()
    for slot in range(1, 10):
        for hv, opp_hv in [("_h", "v"), ("_v", "h")]:
            bat_col = f"batter{slot}_id{hv}"
            sp_col = f"starting_pitcher_id_{opp_hv}"
            if bat_col in df.columns and sp_col in df.columns:
                for _, row in df.iterrows():
                    b_id = row.get(bat_col)
                    p_id = row.get(sp_col)
                    if b_id and p_id and not pd.isna(b_id) and not pd.isna(p_id):
                        batter_pitcher_pairs.add((str(int(b_id)), str(int(p_id))))

    bvp_dict = load_bvp_data_from_snowflake(list(batter_pitcher_pairs))
    log.info(f"Loaded BvP data for {len(bvp_dict)} batter-pitcher pairs")

    # ── 5. assemble per-slot feature columns ──────────────────────────────
    df = get_batting_feats(df, batter_data_dict, pos_map)

    # ── 6. aggregate to lineup-level averages ─────────────────────────────
    df = get_lineup_averages(df)

    return df, batter_data_dict, bvp_dict, warnings
