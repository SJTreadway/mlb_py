"""
hr_results_tracker.py
─────────────────────────
HR-pick tracking, modeled on results_tracker.py's win-tracking conventions —
same MLB Stats API source, same append-to-master-CSV pattern, same
make results / make results:today usage shape.

This is the missing piece flagged in conversation: HR picks have never been
tracked the way win picks are, so there's no real data behind decisions like
"is a 23-24% threshold reasonable" for HR_PROB_THRESHOLD. This closes that
gap going forward — it does NOT backfill past picks, since pipeline.py's
*_homerun_preds.csv exports are the only record of what was actually
predicted each day, and tracking starts from whenever this is wired in.

Differs from results_tracker.py in one structural way: win results come
straight from box-score linescore data (one hydrate call per date). HR
results need per-PLAYER event data, which requires pulling each game's
boxscore (batting stats) rather than just the linescore — MLB Stats API's
/game/{gamePk}/boxscore endpoint includes each batter's HR count for that
game directly (stats.batting.homeRuns per player), no play-by-play parsing
needed.

Usage
─────
    python3.11 hr_results_tracker.py              # today's games
    python3.11 hr_results_tracker.py 2026-06-20    # specific date

Makefile (mirroring results:/results:today):
    make hr-results
    make hr-results DATE=2026-06-20
"""

import logging
import os
import sys
from datetime import date

import boto3
import botocore.exceptions
import pandas as pd
import requests

from helpers import parse_date

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

MLB_API = "https://statsapi.mlb.com/api/v1"

S3_BUCKET = os.environ.get("S3_BUCKET", "moneyballvo-results")
S3_KEY = "hr_model_results.csv"
LOCAL_CACHE = "/tmp/hr_model_results.csv"


def _download_from_s3() -> pd.DataFrame | None:
    s3 = boto3.client("s3")
    try:
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_CACHE)
        return pd.read_csv(LOCAL_CACHE)
    except botocore.exceptions.ClientError:
        return None
    except Exception as e:
        log.warning(f"S3 download failed: {e}")
        return None


def _upload_to_s3(df: pd.DataFrame) -> None:
    df.to_csv(LOCAL_CACHE, index=False)
    boto3.client("s3").upload_file(LOCAL_CACHE, S3_BUCKET, S3_KEY)
    log.info(f"Uploaded {len(df)} rows to s3://{S3_BUCKET}/{S3_KEY}")


def fetch_game_pks_for_date(run_date: str) -> list[dict]:
    """Pull final games for a date — same pattern as results_tracker.fetch_results,
    but we only need gamePk + final status here (boxscore pulled separately
    per game, since that's where per-batter HR counts live)."""
    resp = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": run_date, "hydrate": "team,linescore"},
        timeout=15,
    )
    resp.raise_for_status()

    games = []
    for date_obj in resp.json().get("dates", []):
        for game in date_obj.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            games.append(
                {
                    "game_pk": game["gamePk"],
                    "team_h": game["teams"]["home"]["team"]["abbreviation"],
                    "team_v": game["teams"]["away"]["team"]["abbreviation"],
                }
            )
    return games


def fetch_actual_hrs_for_game(game_pk: int) -> dict:
    """Pull per-batter actual HR counts for one final game via the boxscore
    endpoint. Returns {str(mlbam_id): hr_count} covering both teams.

    ASSUMED SHAPE (NOT independently verified against a live response — the
    structure below is the standard documented MLB Stats API boxscore shape,
    but verify it once against a real game before trusting this in
    production; see debug_print_boxscore_shape() below for a quick check):
        data["teams"]["home"]["players"]["ID<mlbam_id>"]["person"]["id"]
        data["teams"]["home"]["players"]["ID<mlbam_id>"]["stats"]["batting"]["homeRuns"]
    same shape for "away".
    """
    resp = requests.get(f"{MLB_API}/game/{game_pk}/boxscore", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hr_by_batter = {}
    for side in ("home", "away"):
        players = data.get("teams", {}).get(side, {}).get("players", {})
        for player_key, player in players.items():
            mlbam_id = player.get("person", {}).get("id")
            hr = player.get("stats", {}).get("batting", {}).get("homeRuns")
            if mlbam_id is not None and hr is not None:
                hr_by_batter[str(mlbam_id)] = hr
    return hr_by_batter


def debug_print_boxscore_shape(game_pk: int) -> None:
    """Run this once against a real recent game_pk BEFORE trusting
    fetch_actual_hrs_for_game in production:
        python3.11 -c "from hr_results_tracker import debug_print_boxscore_shape as d; d(745313)"
    Confirms the assumed JSON shape matches reality, and prints a sample
    parsed result so you can eyeball it against the real box score.
    """
    resp = requests.get(f"{MLB_API}/game/{game_pk}/boxscore", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    home_players = data.get("teams", {}).get("home", {}).get("players", {})
    if not home_players:
        print("No players found under teams.home.players — shape assumption is WRONG")
        return

    sample_key = next(iter(home_players))
    sample_player = home_players[sample_key]
    print(f"Sample player key: {sample_key}")
    print(f"Top-level keys on player: {list(sample_player.keys())}")
    print(f"person.id: {sample_player.get('person', {}).get('id')}")
    print(
        f"stats.batting keys: {list(sample_player.get('stats', {}).get('batting', {}).keys())}"
    )
    print(
        f"stats.batting.homeRuns: {sample_player.get('stats', {}).get('batting', {}).get('homeRuns')}"
    )

    print("\nFull parsed result:")
    result = fetch_actual_hrs_for_game(game_pk)
    print(f"  {len(result)} batters parsed, sample: {dict(list(result.items())[:5])}")


def fetch_results(run_date: str) -> pd.DataFrame:
    """Pull actual per-batter HR counts for every final game on run_date.
    Returns a DataFrame with columns: game_pk, b_id, actual_hr_count.
    """
    games = fetch_game_pks_for_date(run_date)
    if not games:
        return pd.DataFrame(columns=["game_pk", "b_id", "actual_hr_count"])

    rows = []
    for g in games:
        hr_by_batter = fetch_actual_hrs_for_game(g["game_pk"])
        for b_id, hr_count in hr_by_batter.items():
            rows.append(
                {
                    "game_pk": g["game_pk"],
                    "b_id": b_id,
                    "actual_hr_count": hr_count,
                }
            )
    return pd.DataFrame(rows)


def load_predictions(run_date: str) -> pd.DataFrame | None:
    """Load that day's HR predictions CSV — written by pipeline.py's
    print_todays_homerun_preds() via df_hr.to_csv(...).
    """
    path = f"data/results/{run_date}_homerun_preds.csv"
    if not os.path.exists(path):
        log.warning(f"No HR predictions file found: {path}")
        return None
    df = pd.read_csv(path)
    df["b_id"] = df["b_id"].astype(str)
    return df


def _append_to_master(df: pd.DataFrame) -> None:
    existing = _download_from_s3()
    if existing is not None:
        combined = pd.concat([existing, df], ignore_index=True)
        # de-dupe on (run_date, b_id) so re-running the same date is safe
        combined = combined.drop_duplicates(subset=["run_date", "b_id"], keep="last")
    else:
        combined = df
    _upload_to_s3(combined)


def update_hr_results(run_date: str = None) -> None:
    """Fetch actual HR outcomes for run_date, join against that day's HR
    predictions, append tracking columns to the master CSV. Mirrors
    results_tracker.update_results()'s structure for the win model.
    """
    run_date = run_date or str(date.today())
    log.info(f"Fetching HR results for {run_date} …")

    actual_df = fetch_results(run_date)
    if actual_df.empty:
        log.info("No final games with batting data found — try again later")
        return
    log.info(f"  {len(actual_df)} batter-game HR results found")

    preds_df = load_predictions(run_date)
    if preds_df is None:
        log.info("No HR predictions to match against — nothing to track for this date")
        return

    # hr_prob_numeric is the raw probability saved before % formatting (see
    # pipeline.py's print_todays_homerun_preds: df["hr_prob_numeric"] = ...)
    prob_col = "hr_prob_numeric" if "hr_prob_numeric" in preds_df.columns else "hr_prob"

    merged = preds_df.merge(actual_df, on="b_id", how="left")
    merged["actual_hr_count"] = merged["actual_hr_count"].fillna(0).astype(int)
    merged["actual_hit_hr"] = (merged["actual_hr_count"] >= 1).astype(int)
    merged["run_date"] = run_date
    merged["predicted_prob"] = merged[prob_col]

    # threshold-clearing flag at TODAY's HR_PROB_THRESHOLD value, so historical
    # rows preserve what threshold was active when tracked — pass explicitly
    # rather than importing from pipeline.py to avoid a heavy import chain
    # here; update this constant if HR_PROB_THRESHOLD changes.
    HR_PROB_THRESHOLD = 0.20
    merged["cleared_threshold"] = (
        merged["predicted_prob"] >= HR_PROB_THRESHOLD
    ).astype(int)

    keep_cols = [
        "run_date",
        "b_id",
        "Player" if "Player" in merged.columns else "player_name",
        "predicted_prob",
        "cleared_threshold",
        "actual_hr_count",
        "actual_hit_hr",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    out_df = merged[keep_cols]

    n_matched = merged["actual_hr_count"].notna().sum()
    log.info(f"  {n_matched}/{len(merged)} predicted batters matched to actual results")

    _append_to_master(out_df)

    # quick calibration snapshot for threshold-clearing picks specifically —
    # the number that actually answers "is 0.20 (or 23-24%) doing its job"
    cleared = out_df[out_df["cleared_threshold"] == 1]
    if len(cleared) > 0:
        hit_rate = cleared["actual_hit_hr"].mean()
        log.info(
            f"  Threshold-clearing picks today: {len(cleared)}, "
            f"actual HR rate among them: {hit_rate:.3f}"
        )


def print_summary() -> None:
    """Print HR model performance summary from the master tracking CSV —
    mirrors results_tracker's print_summary() for the win model. This is the
    actual answer to "is HR_PROB_THRESHOLD=0.20 (or 23-24%) reasonable": it
    shows the real accumulated hit rate at the current threshold AND at a
    few candidate alternatives, plus a full calibration-by-bucket breakdown
    so you can see where the model over/under-predicts without guessing
    from a single day's output.

    Usage:
        python3.11 -c "from hr_results_tracker import print_summary; print_summary()"
        make hr-results:summary
    """
    df = _download_from_s3()
    if df is None or df.empty:
        log.info(
            f"No tracking data yet at s3://{S3_BUCKET}/{S3_KEY} — run hr-results for a few days first"
        )
        return

    if df.empty:
        log.info("No tracking rows in range")
        return

    n_total = len(df)
    n_days = df["run_date"].nunique()
    overall_actual_rate = df["actual_hit_hr"].mean()
    print(f"\n=== HR Tracking Summary ===")
    print(
        f"Date range: {df['run_date'].min()} to {df['run_date'].max()} ({n_days} days)"
    )
    print(f"Total predicted batter-games tracked: {n_total}")
    print(f"Overall actual HR rate (all tracked batters): {overall_actual_rate:.4f}\n")

    # ── current threshold performance ─────────────────────────────────────
    cleared = df[df["cleared_threshold"] == 1]
    if len(cleared) > 0:
        print(
            f"Current threshold-clearing picks: {len(cleared)} "
            f"({len(cleared)/n_days:.1f}/day avg)"
        )
        print(f"  Actual HR rate among them: {cleared['actual_hit_hr'].mean():.4f}")
    else:
        print("No picks have cleared the threshold in this range yet.")

    # ── candidate threshold comparison — the number this whole tracker
    # exists to answer ───────────────────────────────────────────────────
    print(f"\nCandidate threshold comparison (using predicted_prob, not the")
    print(f"threshold active when each row was originally tracked):")
    print(
        f"{'threshold':>10s} {'n_picks':>10s} {'picks/day':>10s} {'actual_hr_rate':>15s}"
    )
    for thresh in [0.15, 0.18, 0.20, 0.22, 0.23, 0.24, 0.25, 0.28, 0.30]:
        subset = df[df["predicted_prob"] >= thresh]
        if len(subset) == 0:
            continue
        print(
            f"{thresh:>10.2f} {len(subset):>10d} {len(subset)/n_days:>10.2f} "
            f"{subset['actual_hit_hr'].mean():>15.4f}"
        )

    # ── calibration by bucket — same diagnostic shape used in the offline
    # validation work, now live on production picks ────────────────────────
    print(f"\nCalibration by predicted-probability bucket:")
    print(f"{'bucket':>10s} {'n':>8s} {'actual_rate':>12s}")
    df["bucket"] = (df["predicted_prob"] / 0.05).round() * 0.05
    for bucket in sorted(df["bucket"].unique()):
        subset = df[df["bucket"] == bucket]
        print(
            f"{bucket:>10.2f} {len(subset):>8d} {subset['actual_hit_hr'].mean():>12.4f}"
        )
    print("  (buckets with small n are noisy, same caveat as the offline validation —")
    print("   read directionally until enough days accumulate)")


if __name__ == "__main__":
    raw_date = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
    run_date = parse_date(raw_date)
    update_hr_results(run_date)
    print_summary()
