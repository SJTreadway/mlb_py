"""
results_tracker.py
──────────────────
Pull actual game outcomes from MLB Stats API and append to a master
results CSV for model performance tracking.

Usage:
    python3.11 results_tracker.py              # today's games
    python3.11 results_tracker.py 2026-05-21   # specific date (YYYY-MM-DD)
    python3.11 results_tracker.py 05/21/2026   # specific date (MM/DD/YYYY)

Makefile:
    make results            # today
    make results DATE=2026-05-21
"""

import logging
import os
import sys
from datetime import date, datetime

import pandas as pd
import requests

import boto3

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

MLB_API = "https://statsapi.mlb.com/api/v1"

S3_BUCKET = os.environ.get("S3_BUCKET", "moneyballvo-results")
S3_KEY = "model_results.csv"
LOCAL_CACHE = "/tmp/model_results.csv"


def parse_date(raw: str) -> str:
    """Normalize any reasonable date string to YYYY-MM-DD."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(
        f"Unrecognized date format: '{raw}'. Use YYYY-MM-DD or MM/DD/YYYY."
    )


def _download_from_s3() -> pd.DataFrame | None:
    s3 = boto3.client("s3")
    try:
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_CACHE)
        return pd.read_csv(LOCAL_CACHE)
    except s3.exceptions.ClientError:
        return None
    except Exception as e:
        log.warning(f"S3 download failed: {e}")
        return None


def _upload_to_s3(df: pd.DataFrame) -> None:
    df.to_csv(LOCAL_CACHE, index=False)
    boto3.client("s3").upload_file(LOCAL_CACHE, S3_BUCKET, S3_KEY)
    log.info(f"Uploaded {len(df)} rows to s3://{S3_BUCKET}/{S3_KEY}")


def _calc_units(prob: float, ml: float | None, won: bool) -> float | None:
    """P&L in units for a 1U bet."""
    if ml is None:
        return None
    if won:
        return round(100 / abs(ml) if ml < 0 else ml / 100, 3)
    return -1.0


def fetch_results(run_date: str) -> list[dict]:
    """Pull final game scores from MLB Stats API for a given date."""
    resp = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": run_date, "hydrate": "team,linescore"},
        timeout=15,
    )
    resp.raise_for_status()

    rows = []
    for date_obj in resp.json().get("dates", []):
        for game in date_obj.get("games", []):
            status = game.get("status", {})
            if (
                status.get("abstractGameState") != "Final"
                or status.get("detailedState") == "Postponed"
            ):
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            home_score = home.get("score", 0)
            away_score = away.get("score", 0)
            rows.append(
                {
                    "game_pk": game["gamePk"],
                    "team_h_abbr": home["team"]["abbreviation"],
                    "team_v_abbr": away["team"]["abbreviation"],
                    "team_h_full": home["team"]["name"],
                    "team_v_full": away["team"]["name"],
                    "home_score": home_score,
                    "away_score": away_score,
                    "actual_home_victory": int(home_score > away_score),
                    "actual_run_diff": home_score - away_score,
                }
            )
    return rows


def load_predictions(run_date: str) -> pd.DataFrame | None:
    """Load predictions CSV for the given date (must be YYYY-MM-DD)."""
    path = f"data/results/{run_date}_home_victory_preds.csv"
    if not os.path.exists(path):
        log.warning(f"No predictions file found: {path}")
        return None
    return pd.read_csv(path)


def update_results(run_date: str) -> None:
    """Fetch results for run_date, join with predictions, append to master CSV."""
    log.info(f"Fetching results for {run_date} …")

    actual = fetch_results(run_date)
    if not actual:
        log.info("No final games found — try again later")
        return

    actual_df = pd.DataFrame(actual)
    log.info(f"  {len(actual_df)} final games found")

    preds_df = load_predictions(run_date)
    if preds_df is None:
        log.info("No predictions file for this date — skipping S3 update")
        return

    # ── join predictions to actual results ────────────────────────────────────
    merged = preds_df.merge(actual_df, on=["team_h_full", "team_v_full"], how="inner")
    log.info(f"  {len(merged)} games matched to predictions")

    # ── derive tracking columns ───────────────────────────────────────────────
    merged["run_date"] = run_date
    merged["model_home_victory"] = (merged["prob"] >= 0.5).astype(int)
    merged["model_correct"] = (
        merged["actual_home_victory"] == merged["model_home_victory"]
    ).astype(int)

    merged["model_pick"] = merged["prob"].apply(
        lambda p: "H" if float(p) >= 0.5 else "V"
    )
    merged["pick_correct"] = merged.apply(
        lambda r: (
            int(r["actual_home_victory"] == 1)
            if r["model_pick"] == "H"
            else int(r["actual_home_victory"] == 0)
        ),
        axis=1,
    )

    def row_units(r):
        prob = float(r["prob"])
        if prob >= 0.5:
            ml = float(r["moneyline_h"]) if pd.notna(r.get("moneyline_h")) else None
            won = r["actual_home_victory"] == 1
        else:
            ml = float(r["moneyline_v"]) if pd.notna(r.get("moneyline_v")) else None
            won = r["actual_home_victory"] == 0
        return _calc_units(prob, ml, won)

    merged["edge"] = merged.apply(
        lambda r: r["edge_v"] if r["model_pick"] == "V" else r["edge_h"],
        axis=1,
    )
    merged["moneyline"] = merged.apply(
        lambda r: r["moneyline_v"] if r["model_pick"] == "V" else r["moneyline_h"],
        axis=1,
    )
    merged["units_profit"] = merged.apply(row_units, axis=1)

    # ── only track games where we'd actually bet (edge >= 4%) ─────────────────
    edge_numeric = pd.to_numeric(
        merged["edge"].astype(str).str.replace("%", ""), errors="coerce"
    )
    merged = merged[edge_numeric >= 4.0]
    log.info(f"  {len(merged)} games with edge >= 4% (trackable bets)")

    if merged.empty:
        log.info("No qualifying bets for this date — skipping S3 update")
        return

    # ── select output columns ─────────────────────────────────────────────────
    out_cols = [
        "run_date",
        "game_pk",
        "team_h_abbr",
        "team_v_abbr",
        "team_h_full",
        "team_v_full",
        "starting_pitcher_name_h",
        "starting_pitcher_name_v",
        "prob",
        "model_pick",
        "moneyline",
        "edge",
        "actual_home_victory",
        "actual_run_diff",
        "home_score",
        "away_score",
        "model_correct",
        "pick_correct",
        "units_profit",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols]

    _append_to_master(out)

    # ── summary ───────────────────────────────────────────────────────────────
    n = len(out)
    correct = out["pick_correct"].sum()
    units = out["units_profit"].sum()
    log.info(f"  {correct}/{n} correct ({correct/n:.1%}) | {units:+.2f}U")


def _append_to_master(df: pd.DataFrame) -> None:
    """Append rows to master results CSV, deduplicating on game_pk."""
    existing = _download_from_s3()
    if existing is not None:
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
    else:
        combined = df
    _upload_to_s3(combined)


def print_summary() -> None:
    """Print overall model performance summary from master CSV."""
    df = _download_from_s3()
    if df is None or df.empty:
        log.info("No results file yet")
        return

    if "pick_correct" not in df.columns or df["pick_correct"].isna().all():
        log.info("No matched results yet")
        return

    tracked = df[df["pick_correct"].notna()].copy()
    n = len(tracked)
    correct = tracked["pick_correct"].sum()
    units = tracked["units_profit"].sum()

    log.info(f"\n{'='*40}")
    log.info(f"Model Performance Summary")
    log.info(f"  Games tracked : {n}")
    log.info(f"  Win rate      : {correct/n:.1%} ({correct}/{n})")
    log.info(f"  Total P&L     : {units:+.2f}U")
    log.info(f"  ROI           : {units/n:.3f}U per game")

    if "edge" in tracked.columns:
        tracked["edge_val"] = pd.to_numeric(
            tracked["edge"].astype(str).str.replace("%", ""), errors="coerce"
        )

        log.info(f"\n  By edge bucket (model's pick side):")
        for lo, hi in [(4, 8), (8, 15), (15, 100)]:
            bucket = tracked[(tracked["edge_val"] >= lo) & (tracked["edge_val"] < hi)]
            if len(bucket):
                log.info(
                    f"    {lo}-{hi}%: {bucket['pick_correct'].sum()}/{len(bucket)} "
                    f"({bucket['pick_correct'].mean():.1%}) | "
                    f"{bucket['units_profit'].sum():+.2f}U"
                )

    if "model_pick" in tracked.columns:
        log.info(f"\n  By pick side (Home vs Visitor):")
        for side, label in [("H", "Home"), ("V", "Visitor")]:
            side_df = tracked[tracked["model_pick"] == side]
            if len(side_df):
                log.info(
                    f"    {label:8s}: {side_df['pick_correct'].sum()}/{len(side_df)} "
                    f"({side_df['pick_correct'].mean():.1%}) | "
                    f"{side_df['units_profit'].sum():+.2f}U"
                )

        if "edge_val" in tracked.columns:
            log.info(f"\n  By pick side x edge bucket:")
            for side, label in [("H", "Home"), ("V", "Visitor")]:
                side_df = tracked[tracked["model_pick"] == side]
                for lo, hi in [(4, 8), (8, 15), (15, 100)]:
                    bucket = side_df[
                        (side_df["edge_val"] >= lo) & (side_df["edge_val"] < hi)
                    ]
                    if len(bucket):
                        log.info(
                            f"    {label:8s} {lo}-{hi}%: {bucket['pick_correct'].sum()}/{len(bucket)} "
                            f"({bucket['pick_correct'].mean():.1%}) | "
                            f"{bucket['units_profit'].sum():+.2f}U"
                        )

    log.info(f"{'='*40}")


if __name__ == "__main__":
    raw_date = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
    run_date = parse_date(raw_date)
    update_results(run_date)
    print_summary()
