"""
results_tracker.py
──────────────────
Pull actual game outcomes from MLB Stats API and append to a master
results CSV for model performance tracking.

Usage:
    python3.11 results_tracker.py              # today's games
    python3.11 results_tracker.py 2026-05-21   # specific date

Makefile:
    make results            # today
    make results DATE=2026-05-21
"""

import logging
import os
import sys
from datetime import date

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
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            home_score = home.get("score", 0)
            away_score = away.get("score", 0)
            rows.append(
                {
                    "game_pk": game["gamePk"],
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
    """Load today's win predictions CSV if it exists."""
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
        log.info("No predictions to match against — writing results only")
        actual_df["run_date"] = run_date
        _append_to_master(actual_df)
        return

    # ── join predictions to actual results on home team ───────────────────────
    merged = preds_df.merge(actual_df, on=["team_h_full", "team_v_full"], how="inner")

    # ── derive tracking columns ───────────────────────────────────────────────
    merged["run_date"] = run_date
    merged["model_home_victory"] = (merged["prob"] >= 0.5).astype(int)
    merged["model_correct"] = (
        merged["actual_home_victory"] == merged["model_home_victory"]
    ).astype(int)

    # which side did the model back?
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

    # units P&L
    def row_units(r):
        prob = float(r["prob"])
        if prob >= 0.5:
            ml = float(r["moneyline_h"]) if pd.notna(r.get("moneyline_h")) else None
            won = r["actual_home_victory"] == 1
        else:
            ml = float(r["moneyline_v"]) if pd.notna(r.get("moneyline_v")) else None
            won = r["actual_home_victory"] == 0
        return _calc_units(prob, ml, won)

    merged["units_profit"] = merged.apply(row_units, axis=1)

    # ── select output columns ─────────────────────────────────────────────────
    out_cols = [
        "run_date",
        "team_h",
        "team_v",
        "starting_pitcher_name_h",
        "starting_pitcher_name_v",
        "prob",  # model's home win probability
        "home_victory",  # model's predicted winner (1=H, 0=V)
        "model_pick",  # "H" or "V"
        "moneyline_h",
        "moneyline_v",
        "edge_h",
        "edge_v",
        "actual_home_victory",
        "actual_run_diff",
        "home_score",
        "away_score",
        "model_correct",  # did predicted side win?
        "pick_correct",  # did model's pick win?
        "units_profit",  # P&L if 1U bet on model's pick
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols]

    _append_to_master(out)

    # ── summary ───────────────────────────────────────────────────────────────
    n = len(out)
    if n == 0:
        log.info("No results to summarize yet")
        return
    correct = out["pick_correct"].sum()
    units = out["units_profit"].sum()
    log.info(f"  {correct}/{n} correct ({correct/n:.1%}) | {units:+.2f}U")


def _append_to_master(df: pd.DataFrame) -> None:
    """Append rows to master results CSV, skipping duplicates."""
    existing = _download_from_s3()
    if existing is not None:
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["run_date", "team_h_full", "team_v_full"], keep="last"
        )
    else:
        combined = df
    _upload_to_s3(combined)


def print_summary() -> None:
    """Print overall model performance summary from master CSV."""
    df = _download_from_s3()
    if df is None or df.empty:
        log.info("No results file yet")
        return

    n = len(df)
    if "pick_correct" not in df.columns or df["pick_correct"].isna().all():
        log.info("No matched results yet")
        return
    correct = df["pick_correct"].sum()
    units = df["units_profit"].sum()

    log.info(f"\n{'='*40}")
    log.info(f"Model Performance Summary")
    log.info(f"  Games tracked : {n}")
    log.info(f"  Win rate      : {correct/n:.1%} ({correct}/{n})")
    log.info(f"  Total P&L     : {units:+.2f}U")
    log.info(f"  ROI           : {units/n:.3f}U per game")

    # by edge bucket
    if "edge_h" in df.columns:
        log.info(f"\n  By edge bucket (model's pick side):")
        df["edge_pick"] = df.apply(
            lambda r: (
                float(str(r["edge_h"]).replace("%", ""))
                if r["model_pick"] == "H"
                else float(str(r["edge_v"]).replace("%", ""))
            ),
            axis=1,
        )
        for lo, hi in [(4, 8), (8, 15), (15, 100)]:
            bucket = df[(df["edge_pick"] >= lo) & (df["edge_pick"] < hi)]
            if len(bucket):
                log.info(
                    f"    {lo}-{hi}%: {bucket['pick_correct'].sum()}/{len(bucket)} "
                    f"({bucket['pick_correct'].mean():.1%}) | "
                    f"{bucket['units_profit'].sum():+.2f}U"
                )
    log.info(f"{'='*40}")


if __name__ == "__main__":
    run_date = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
    update_results(run_date)
    print_summary()
