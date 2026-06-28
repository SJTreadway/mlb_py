#!/usr/bin/env python3

import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TOMORROW_GAMES = int(os.environ["TOMORROW_GAMES"])
YEAR = int(os.environ["YEAR"])
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


# ── Team Stats ────────────────────────────────────────────────────────────────


def get_team_hitting_stats(team_id, year=None):
    """Pull last 10 games hitting stats for a team via MLB Stats API."""
    year = year or YEAR
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/teams/{team_id}/stats",
            params={
                "season": year,
                "group": "hitting",
                "stats": "season",
            },
            timeout=15,
        )
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
        if not splits:
            return {}
        stat = splits[0]["stat"]
        gp = stat.get("gamesPlayed", 1) or 1
        return {
            "atBats": stat.get("atBats", 0) / gp,
            "baseOnBalls": stat.get("baseOnBalls", 0) / gp,
            "hits": stat.get("hits", 0) / gp,
            "runs": stat.get("runs", 0) / gp,
            "doubles": stat.get("doubles", 0) / gp,
            "triples": stat.get("triples", 0) / gp,
            "homeRuns": stat.get("homeRuns", 0) / gp,
            "hitByPitch": stat.get("hitByPitch", 0) / gp,
            "strikeOuts": stat.get("strikeOuts", 0) / gp,
            "stolenBases": stat.get("stolenBases", 0) / gp,
            "caughtStealing": stat.get("caughtStealing", 0) / gp,
        }
    except Exception as e:
        print(f"Error fetching team stats for {team_id}: {e}")
        return {}


# ── Suspended Games Filter ─────────────────────────────────────────────────────────────────────

SUSPENDED_RESUMED_STATES = {
    "Suspended",
    "Suspended: Rain",
    "Suspended: Darkness",
    "Suspended: Weather",
    "Resumed",  # game continuing from a prior date
    "Completed Early",  # often paired with a suspension that finished without resuming
}


def _is_carryover_suspended_game(game: dict, run_date: datetime.date) -> bool:
    """Return True if this game is a suspended game resuming/continuing
    from a previous day (i.e. NOT a fresh game for run_date)."""
    status = game.get("status", {})
    detailed_state = status.get("detailedState", "")

    if detailed_state in SUSPENDED_RESUMED_STATES:
        return True

    # Some resumed games carry the original game date explicitly.
    # If present and it doesn't match run_date, this game started earlier.
    resume_date_str = (
        game.get("resumeDate")
        or game.get("resumeGameDate")
        or game.get("resumedFromDate")
    )
    if resume_date_str:
        try:
            resume_date = datetime.fromisoformat(resume_date_str[:10]).date()
            if resume_date != run_date:
                return True
        except (ValueError, TypeError):
            pass

    return False


# ── Slate ─────────────────────────────────────────────────────────────────────


def get_games_slate(run_date):
    """Pull today's or tomorrow's games from MLB Stats API."""
    resp = requests.get(
        f"{MLB_API_BASE}/schedule",
        params={
            "sportId": 1,
            "date": run_date.strftime("%Y-%m-%d"),
            "hydrate": "team,probablePitcher,lineups",
        },
        timeout=15,
    )
    resp.raise_for_status()

    games = []
    for date_obj in resp.json().get("dates", []):
        for game in date_obj.get("games", []):
            if _is_carryover_suspended_game(game, run_date):
                continue
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
                    "team_h_full": home["team"]["name"],
                    "team_h_id": home["team"]["id"],
                    "team_v": away["team"]["abbreviation"],
                    "team_v_full": away["team"]["name"],
                    "team_v_id": away["team"]["id"],
                    "sp_mlbam_h": home.get("probablePitcher", {}).get("id"),
                    "sp_name_h": home.get("probablePitcher", {}).get("fullName", ""),
                    "sp_mlbam_v": away.get("probablePitcher", {}).get("id"),
                    "sp_name_v": away.get("probablePitcher", {}).get("fullName", ""),
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
        game["date_dblhead"] = int(run_date.strftime("%Y%m%d") + str(count))
        seen[key] = count + 1

    print(f"Found {len(games)} games for {run_date}")
    return games


def get_pitcher_handedness(mlbam_id):
    """Fetch pitcher throws (L/R) from MLB Stats API."""
    if not mlbam_id:
        return ""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}",
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


def get_lineups(run_date):
    """
    Main entry point — replaces Rotowire scraper.
    Returns a DataFrame with one row per game matching the
    structure expected by process_pitching_data and process_batting_data.
    """
    slate = get_games_slate(run_date)
    if not slate:
        return pd.DataFrame()

    """
    # filter out games that have already started
    now = datetime.now(timezone.utc)
    active = [g for g in slate if pd.Timestamp(g["game_time"], tz="UTC") > now]
    skipped = len(slate) - len(active)
    if skipped > 0:
        print(f"Skipping {skipped} games that have already started")
    slate = active
    """

    # fetch pitcher handedness
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

    # fetch team hitting stats
    team_ids = set()
    for game in slate:
        team_ids.add(game["team_h_id"])
        team_ids.add(game["team_v_id"])

    team_stats_map = {}
    for tid in team_ids:
        team_stats_map[tid] = get_team_hitting_stats(tid)
        time.sleep(0.1)

    rows = []
    for game in slate:
        if not game["lineup_h"] or not game["lineup_v"]:
            print(f'Lineups not yet posted for {game["team_h"]} vs {game["team_v"]}')
        h_stats = team_stats_map.get(game["team_h_id"], {})
        v_stats = team_stats_map.get(game["team_v_id"], {})

        # flatten lineup to batter1_id_h ... batter9_id_h
        batter_cols = {}
        for i, mlbam_id in enumerate(game["lineup_h"][:9], 1):
            batter_cols[f"batter{i}_id_h"] = mlbam_id
            batter_cols[f"batter{i}_name_h"] = ""  # names fetched separately if needed
        for i, mlbam_id in enumerate(game["lineup_v"][:9], 1):
            batter_cols[f"batter{i}_id_v"] = mlbam_id
            batter_cols[f"batter{i}_name_v"] = ""

        # pad missing batters with None
        for i in range(len(game["lineup_h"]) + 1, 10):
            batter_cols[f"batter{i}_id_h"] = None
            batter_cols[f"batter{i}_name_h"] = ""
        for i in range(len(game["lineup_v"]) + 1, 10):
            batter_cols[f"batter{i}_id_v"] = None
            batter_cols[f"batter{i}_name_v"] = ""

        row = {
            "date_dblhead": game["date_dblhead"],
            "game_time": game["game_time"],
            "team_h": game["team_h"],
            "team_h_full": game["team_h_full"],
            "team_v": game["team_v"],
            "team_v_full": game["team_v_full"],
            "starting_pitcher_id_h": game["sp_mlbam_h"],
            "starting_pitcher_name_h": game["sp_name_h"],
            "starting_pitcher_id_v": game["sp_mlbam_v"],
            "starting_pitcher_name_v": game["sp_name_v"],
            "sp_throws_h": throws_map.get(game["sp_mlbam_h"], ""),
            "sp_throws_v": throws_map.get(game["sp_mlbam_v"], ""),
            # team hitting stats
            "AB_h": h_stats.get("atBats", 0),
            "BB_h": h_stats.get("baseOnBalls", 0),
            "H_h": h_stats.get("hits", 0),
            "R_h": h_stats.get("runs", 0),
            "x2B_h": h_stats.get("doubles", 0),
            "x3B_h": h_stats.get("triples", 0),
            "HR_h": h_stats.get("homeRuns", 0),
            "HBP_h": h_stats.get("hitByPitch", 0),
            "SO_h": h_stats.get("strikeOuts", 0),
            "SB_h": h_stats.get("stolenBases", 0),
            "CS_h": h_stats.get("caughtStealing", 0),
            "AB_v": v_stats.get("atBats", 0),
            "BB_v": v_stats.get("baseOnBalls", 0),
            "H_v": v_stats.get("hits", 0),
            "R_v": v_stats.get("runs", 0),
            "x2B_v": v_stats.get("doubles", 0),
            "x3B_v": v_stats.get("triples", 0),
            "HR_v": v_stats.get("homeRuns", 0),
            "HBP_v": v_stats.get("hitByPitch", 0),
            "SO_v": v_stats.get("strikeOuts", 0),
            "SB_v": v_stats.get("stolenBases", 0),
            "CS_v": v_stats.get("caughtStealing", 0),
            **batter_cols,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["game_id"] = df["date_dblhead"].astype(str) + df["team_h"] + df["team_v"]
    df["lineups_confirmed"] = df["batter1_id_h"].notna() & df["batter1_id_v"].notna()
    df.set_index("game_id", inplace=True)
    return df


# ── Run Total Features ────────────────────────────────────────────────────────


def get_run_total_feats(df):
    cols_ref = [
        "date_dblhead",
        "game_time",
        "team_h",
        "team_h_full",
        "team_v",
        "team_v_full",
    ]

    team_hit_stems = ["BATAVG", "OBP", "SLG", "OBS", "SB", "CS"]
    lineup_hit_stems = ["BATAVG", "OBP", "SLG", "OBS", "SLGmod", "SObat_perc"]
    strt_pitch_stems = [
        "ERA",
        "WHIP",
        "SO_perc",
        "H_BB_perc",
        "TB_BB_perc",
        "FIP",
        "FIP_perc",
    ]
    bpen_pitch_stems = ["WHIP", "SO_perc", "H_BB_perc", "TB_BB_perc"]

    team_hit_winsizes = [30, 90, 162]
    lineup_hit_winsizes = [30, 75, 162, 350]
    strt_pitch_winsizes = [10, 35, 75]
    bpen_pitch_winsizes = [10, 35, 75]

    team_hit_features_a = [
        f"{x}_{w}_h" for w in team_hit_winsizes for x in team_hit_stems
    ]
    lineup_hit_features_a = [
        f"lineup{n}_{x}_{w}{ow}_h"
        for w in lineup_hit_winsizes
        for x in lineup_hit_stems
        for ow in ["", "_w"]
        for n in ["8", "9"]
    ]
    start_pitch_features_a = [
        f"Strt_{x}_{w}_v" for w in strt_pitch_winsizes for x in strt_pitch_stems
    ]
    bpen_pitch_features_a = [
        f"Bpen_{x}_{w}_v" for w in bpen_pitch_winsizes for x in bpen_pitch_stems
    ]

    team_hit_features_b = [
        f"{x}_{w}_v" for w in team_hit_winsizes for x in team_hit_stems
    ]
    lineup_hit_features_b = [
        f"lineup{n}_{x}_{w}{ow}_v"
        for w in lineup_hit_winsizes
        for x in lineup_hit_stems
        for ow in ["", "_w"]
        for n in ["8", "9"]
    ]
    start_pitch_features_b = [
        f"Strt_{x}_{w}_h" for w in strt_pitch_winsizes for x in strt_pitch_stems
    ]
    bpen_pitch_features_b = [
        f"Bpen_{x}_{w}_h" for w in bpen_pitch_winsizes for x in bpen_pitch_stems
    ]

    cols_a = (
        cols_ref
        + team_hit_features_a
        + lineup_hit_features_a
        + start_pitch_features_a
        + bpen_pitch_features_a
    )
    cols_b = (
        cols_ref
        + team_hit_features_b
        + lineup_hit_features_b
        + start_pitch_features_b
        + bpen_pitch_features_b
    )

    df_a = df.loc[:, cols_a].copy()
    df_b = df.loc[:, cols_b].copy()
    df_a["home_hitting"] = 1
    df_b["home_hitting"] = 0

    stripped = [
        x[:-2]
        for x in team_hit_features_a
        + lineup_hit_features_a
        + start_pitch_features_a
        + bpen_pitch_features_a
    ]
    final_cols = cols_ref + stripped + ["home_hitting"]

    df_a.columns = final_cols
    df_b.columns = final_cols

    df_runs = pd.concat([df_a, df_b])
    df_runs["game_id"] = (
        df_runs["date_dblhead"].astype(str) + df_runs["team_h"] + df_runs["team_v"]
    )
    df_runs.set_index("game_id", inplace=True)
    return df_runs
