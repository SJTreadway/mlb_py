"""
simulator_inference.py
───────────────────────
Daily-pipeline glue for the Markov simulator — the analogue of
ensemble_inference.py for the simulator instead of the XGBoost fleet.

Unlike ensemble_inference.py (which loads a saved strategy artifact and
combines pre-trained model outputs), there is NO saved artifact here — the
simulator has no trained parameters, so this module runs the full
PA-rate-building → matchup → Monte Carlo pipeline live, every call, from
whatever lineup/pitcher data the daily pipeline already assembled.

Lives at the repo root (alongside ensemble_inference.py) so pipeline.py can
import it directly. Imports from sim/ for the actual engine.

Design: reuses pipeline.py's EXISTING batter_data_dict / pitcher_data_dict
structure (keyed by str(mlbam_id) -> pd.Series of latest rolling stats,
exactly as homerun.py's process_homerun_data already consumes them) and the
lineup_w_pitching_batting_team_weather_df's batter{slot}_id_{hv} columns —
no new data assembly required, this is the same shape your HR pipeline
already builds for every game, every day.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sim"))

from pa_outcome_model import PARates, pa_outcome_distribution, LEAGUE_AVG_RATES
from vectorized_sim import run_monte_carlo_vectorized

log = logging.getLogger(__name__)

NON_HR_HIT_SPLIT = {"single": 0.72, "double": 0.23, "triple": 0.05}

# Real park factors — same map as helpers.get_park_factors_map(). Import
# directly from helpers instead of duplicating, so this module never drifts
# out of sync with the production source of truth.
try:
    from helpers import get_park_factors_map

    PARK_HR_FACTORS = get_park_factors_map()
except ImportError:
    log.warning(
        "Could not import helpers.get_park_factors_map() — using a "
        "local fallback copy. Verify this matches helpers.py."
    )
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
        "AZ": 104,
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


def park_factor_for_team(team_code: str) -> float:
    if team_code in PARK_HR_FACTORS:
        return float(PARK_HR_FACTORS[team_code])
    log.warning(f"No park factor for team code '{team_code}' — using neutral 100")
    return 100.0


def _safe_float(value, default):
    if value is None:
        return default
    try:
        if (
            hasattr(value, "__len__") is False and value != value
        ):  # NaN check without pandas import
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def batter_row_to_rates(brow, window: int = 162) -> PARates:
    """Build PARates from one row of batter_data_dict (a pd.Series of latest
    rolling stats, the exact same object homerun.py already looks up via
    batter_data_dict.get(b_id)).
    """
    obp = _safe_float(brow.get(f"obp_{window}"), 0.315)
    slg = _safe_float(brow.get(f"slg_{window}"), 0.400)
    hr_rate = _safe_float(brow.get(f"hr_per_pa_{window}"), LEAGUE_AVG_RATES["hr"])
    pa = _safe_float(brow.get("rollsum_ab_162"), 30)

    hit_rate = min(max(slg / 1.7, 0.0), 1 - LEAGUE_AVG_RATES["out"])
    bb_rate = max(obp - hit_rate, 0.01)
    non_hr_hit_rate = max(hit_rate - hr_rate, 0.0)
    single = non_hr_hit_rate * NON_HR_HIT_SPLIT["single"]
    double = non_hr_hit_rate * NON_HR_HIT_SPLIT["double"]
    triple = non_hr_hit_rate * NON_HR_HIT_SPLIT["triple"]
    out = max(1.0 - bb_rate - single - double - triple - hr_rate, 0.05)

    return PARates(
        rates={
            "out": out,
            "bb": bb_rate,
            "single": single,
            "double": double,
            "triple": triple,
            "hr": hr_rate,
        },
        pa=max(int(pa), 30),
    )


def pitcher_row_to_rates(prow, window: int = 75) -> PARates:
    """Build PARates from one row of pitcher_data_dict."""
    h_bb_perc = _safe_float(prow.get(f"h_bb_perc_{window}"), 0.37)
    hr_per_bf = _safe_float(prow.get(f"hr_per_bf_{window}"), LEAGUE_AVG_RATES["hr"])
    bb_rate = max(h_bb_perc * 0.30, 0.04)
    hit_rate = max(h_bb_perc - bb_rate, 0.0)
    hr_rate = hr_per_bf
    non_hr_hit_rate = max(hit_rate - hr_rate, 0.0)
    single = non_hr_hit_rate * NON_HR_HIT_SPLIT["single"]
    double = non_hr_hit_rate * NON_HR_HIT_SPLIT["double"]
    triple = non_hr_hit_rate * NON_HR_HIT_SPLIT["triple"]
    out = max(1.0 - bb_rate - single - double - triple - hr_rate, 0.05)

    return PARates(
        rates={
            "out": out,
            "bb": bb_rate,
            "single": single,
            "double": double,
            "triple": triple,
            "hr": hr_rate,
        },
        pa=200,
    )


def build_lineup_dists(
    game_row, hv, batter_data_dict, opposing_pitcher_rates, park_factor
):
    """Build the 9 PA-outcome distributions for one team's lineup in one game,
    using the SAME batter{slot}_id_{hv} column convention homerun.py already
    reads. Missing/inactive batters fall back to league-average rates (logged)
    rather than crashing or silently truncating the lineup to <9 — the
    simulator requires exactly 9 batting-order slots.

    Returns (dists, slot_to_mlbam_id):
        dists            : list of 9 outcome-distribution dicts
        slot_to_mlbam_id : dict, slot index (0-8) -> str(mlbam_id) or None.
                           Needed downstream to map predict_game()'s
                           per-slot HR output back to real batter IDs (the
                           slot order here is NOT necessarily the same as
                           df_hr's row order, which is filtered/reordered by
                           process_homerun_data — matching must go through
                           MLBAM_ID, not position).
    """
    dists = []
    slot_to_mlbam_id = {}
    for slot in range(1, 10):
        b_id_raw = game_row.get(f"batter{slot}_id_{hv}")
        brow = None
        b_id = None
        if b_id_raw is not None:
            try:
                b_id = str(int(b_id_raw))
                brow = batter_data_dict.get(b_id)
            except (TypeError, ValueError):
                pass

        slot_to_mlbam_id[slot - 1] = (
            b_id  # 0-indexed to match predict_game's hr_prob_by_slot keys
        )

        if brow is None:
            log.warning(
                f"No batter data for slot {slot} ({hv}) — using league-average "
                f"fallback. Lineup may not be fully confirmed yet."
            )
            b_rates = PARates(rates=dict(LEAGUE_AVG_RATES), pa=200)
        else:
            b_rates = batter_row_to_rates(brow)

        dist = pa_outcome_distribution(
            b_rates, opposing_pitcher_rates, park_hr_factor=park_factor
        )
        dists.append(dist)

    return dists, slot_to_mlbam_id


def predict_game(
    game_row, batter_data_dict, pitcher_data_dict, n_sims: int = 15000, seed=None
):
    """Run the simulator for ONE game. Returns a dict with home_win_prob,
    away_win_prob, and per-slot HR probabilities for both teams — the
    simulator's analogue of predict_winner()'s (pred, prob) return, extended
    with the joint HR output predict_winner() doesn't have.

    game_row must have: team_h_full/team_h (for park factor lookup),
    starting_pitcher_id_h, starting_pitcher_id_v, batter{1-9}_id_h/v
    (matching homerun.py's existing column conventions).

    Returns None if either starting pitcher is missing from pitcher_data_dict
    (mirrors the data-quality skip logic used throughout the sim validation
    scripts — a missing starter means this game can't be simulated reliably).
    """
    sp_id_h_raw = game_row.get("starting_pitcher_id_h")
    sp_id_v_raw = game_row.get("starting_pitcher_id_v")
    if sp_id_h_raw is None or sp_id_v_raw is None:
        log.warning(
            "Missing starting pitcher ID — skipping simulator prediction for this game"
        )
        return None

    try:
        sp_id_h = str(int(sp_id_h_raw))
        sp_id_v = str(int(sp_id_v_raw))
    except (TypeError, ValueError):
        log.warning("Could not parse starting pitcher IDs — skipping")
        return None

    home_pitcher_row = pitcher_data_dict.get(sp_id_h)
    away_pitcher_row = pitcher_data_dict.get(sp_id_v)
    if home_pitcher_row is None or away_pitcher_row is None:
        log.warning("Starting pitcher missing from pitcher_data_dict — skipping")
        return None

    home_pitcher_rates = pitcher_row_to_rates(home_pitcher_row)
    away_pitcher_rates = pitcher_row_to_rates(away_pitcher_row)

    team_h_code = game_row.get("team_h") or game_row.get("team_h_full", "")
    park_factor = park_factor_for_team(team_h_code)

    home_lineup_dists, home_slot_map = build_lineup_dists(
        game_row, "h", batter_data_dict, away_pitcher_rates, park_factor
    )
    away_lineup_dists, away_slot_map = build_lineup_dists(
        game_row, "v", batter_data_dict, home_pitcher_rates, park_factor
    )

    summary = run_monte_carlo_vectorized(
        home_lineup_dists,
        away_lineup_dists,
        n_sims=n_sims,
        seed=seed,
    )
    # attach slot->MLBAM_ID maps so callers (e.g. get_hr_probs_by_batter) can
    # translate home_hr_prob_by_slot / away_hr_prob_by_slot back to real
    # batter IDs without re-deriving lineup order themselves.
    summary["home_slot_to_mlbam_id"] = home_slot_map
    summary["away_slot_to_mlbam_id"] = away_slot_map
    return summary


def get_hr_probs_by_batter(
    game_row, batter_data_dict, pitcher_data_dict, n_sims: int = 15000, seed=None
):
    """Run the simulator for ONE game and return a flat {mlbam_id: hr_prob}
    dict covering both lineups — the per-batter HR output the daily pipeline
    needs to merge into df_hr (which is one row per batter, not per slot).

    Matching is done via MLBAM_ID (through home_slot_to_mlbam_id /
    away_slot_to_mlbam_id), not lineup position — df_hr's row order is NOT
    guaranteed to match slot order, since process_homerun_data filters out
    inactive/unconfirmed batters before building it.

    Returns an empty dict if the game couldn't be simulated (missing
    starter/lineup data) — callers should treat batters from a skipped game
    as having no simulator HR prediction, same as predict_winner_simulator's
    None return for win probability.
    """
    summary = predict_game(
        game_row, batter_data_dict, pitcher_data_dict, n_sims=n_sims, seed=seed
    )
    if summary is None:
        return {}

    hr_probs = {}
    for slot, mlbam_id in summary["home_slot_to_mlbam_id"].items():
        if mlbam_id is not None:
            hr_probs[mlbam_id] = summary["home_hr_prob_by_slot"][slot]
    for slot, mlbam_id in summary["away_slot_to_mlbam_id"].items():
        if mlbam_id is not None:
            hr_probs[mlbam_id] = summary["away_hr_prob_by_slot"][slot]

    return hr_probs


def predict_winner_simulator(
    game_row, batter_data_dict, pitcher_data_dict, n_sims: int = 15000
):
    """Convenience wrapper mirroring pipeline.predict_winner's contract:
    returns (pred, prob) for the HOME team, or (None, None) if the game
    couldn't be simulated (missing data).
    """
    summary = predict_game(game_row, batter_data_dict, pitcher_data_dict, n_sims=n_sims)
    if summary is None:
        return None, None
    prob = summary["home_win_prob"]
    pred = int(prob > 0.5)
    return pred, prob
