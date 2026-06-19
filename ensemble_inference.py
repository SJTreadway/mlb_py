"""
ensemble_inference.py
─────────────────────
Inference-time glue for the fleet/stacking system. Loads the winning-strategy
artifact (models/best_ensemble_strategy_*.pkl) plus the fleet artifact
(models/ensemble_*.pkl) and produces combined probabilities for live prediction.

Lives at the repo root so pipeline.py can `from ensemble_inference import …`.
It has NO dependency on train_ensemble_core / stack_models — it only unpickles
the artifacts (plain dicts wrapping sklearn/xgboost/lightgbm estimators) and
calls predict_proba, so it's safe to import from the prediction pipeline.

Strategy artifact (written by train/stack_models.py) contains:
    "strategy"     : str   e.g. "stacked_meta_diverse" / "avg_diverse_subset"
                            / a single base-model name
    "members"      : list[str]   base models to combine, IN ORDER
    "meta_model"   : LogisticRegression   (only present for stacked_* winners)
    "ensemble_path": str   path to the fleet artifact it was built from

Fleet artifact (written by train/<model>/ensemble_<model>.py):
    "base_models": { name: {"model": <calibrated est>, "features": [...], ...} }

Combination rule:
    stacked_*  → column-stack member predict_proba (in `members` order),
                 feed through meta_model.predict_proba(...)[:, 1]
    avg_*      → unweighted mean of member predict_proba
    single     → that one member's predict_proba
The presence of "meta_model" in the strategy artifact is the switch between the
stacked path and the mean path (avg_* and single both fall through to the mean,
which for a single member is just that member).
"""

import logging
import pickle

import numpy as np

log = logging.getLogger(__name__)

# Simple process-level cache so repeated calls in one run don't re-unpickle.
_CACHE = {}


def _load(path):
    if path not in _CACHE:
        with open(path, "rb") as f:
            _CACHE[path] = pickle.load(f)
    return _CACHE[path]


def clear_cache():
    """Drop cached artifacts (e.g. after retraining within a long-lived process)."""
    _CACHE.clear()


def _member_proba(fleet, member, X):
    """P(positive) from one fleet member, using that member's own feature list."""
    if member not in fleet["base_models"]:
        raise KeyError(
            f"Strategy references member '{member}' not found in the fleet "
            f"artifact. Available: {list(fleet['base_models'])[:8]}…"
        )
    entry = fleet["base_models"][member]
    feats = entry["features"]
    missing = [c for c in feats if c not in X.columns]
    if missing:
        raise KeyError(
            f"Ensemble member '{member}' needs {len(missing)} feature column(s) "
            f"absent from X (e.g. {missing[:5]}). Make sure X is the assembled "
            f"feature frame, not the raw lineup frame."
        )
    return entry["model"].predict_proba(X[feats])[:, 1]


def predict_with_strategy(X, strategy_path, fleet_path=None):
    """Combined P(positive class) for the rows of X, as a 1-D np.ndarray.

    Mirrors the `prob` output of the single-model predict_* functions, so the
    caller can derive pred = (prob > 0.5) and compute edges exactly as before.

    Parameters
    ----------
    X : pd.DataFrame
        Assembled feature frame containing (at least) every column each member
        model needs. For the win model that's the output of
        assemble_win_model_features(); members use subsets of those columns.
    strategy_path : str
        Path to best_ensemble_strategy_*.pkl.
    fleet_path : str, optional
        Path to the fleet artifact. Defaults to the `ensemble_path` recorded in
        the strategy artifact; pass explicitly to be CWD-independent.
    """
    strat = _load(strategy_path)
    resolved_fleet = fleet_path or strat.get("ensemble_path")
    if not resolved_fleet:
        raise ValueError(
            "No fleet path: strategy artifact has no 'ensemble_path' and none "
            "was passed. Pass fleet_path explicitly."
        )
    fleet = _load(resolved_fleet)

    members = strat["members"]
    member_preds = [_member_proba(fleet, m, X) for m in members]
    stacked = np.column_stack(member_preds)

    if "meta_model" in strat and strat["meta_model"] is not None:
        prob = strat["meta_model"].predict_proba(stacked)[:, 1]  # stacked_*
    else:
        prob = stacked.mean(axis=1)  # avg_* / single

    return prob


def predict_winner_ensemble(X, strategy_path, fleet_path=None):
    """Convenience wrapper returning (pred, prob) like pipeline.predict_winner."""
    prob = predict_with_strategy(X, strategy_path, fleet_path)
    pred = (prob > 0.5).astype(int)
    return pred, prob
