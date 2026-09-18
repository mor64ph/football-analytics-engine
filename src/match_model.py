"""
XGBoost 3-class match outcome model.

TARGET
    0 = home win, 1 = draw, 2 = away win

WHY THE NATIVE XGBOOST API
--------------------------
xgb.train() with DMatrix rather than the scikit-learn wrapper. The wrapper
tripped a `__sklearn_tags__` MRO incompatibility elsewhere in this project
(see TROUBLESHOOTING.md #4); the native API has no sklearn dependency in its
call path and is stable across versions.

WHY A SHALLOW MODEL
-------------------
max_depth 3-4, not the usual 6. Football outcomes are close to irreducibly
noisy - even a perfect model tops out near 60% accuracy - so a deep tree has
far more capacity than there is signal to find, and spends the surplus
memorising which specific clubs happened to win in the training seasons. The
regularisation here is deliberately heavier than a default template.

WHAT WE OPTIMISE
----------------
mlogloss, never accuracy. Section-by-section reasoning lives in
match_metrics.py; the short version is that accuracy cannot see draws and
cannot tell a confident model from a hedging one.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from match_features import all_features, FEATURE_GROUPS
from match_metrics import evaluate, log_loss_multi

# Conservative defaults. Tuned in tune_match_model.py against a validation
# split - not against test.
DEFAULT_PARAMS = {
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "max_depth":        3,
    "eta":              0.03,      # learning rate
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,        # high: refuses to split on a handful of matches
    "lambda":           2.0,       # L2
    "alpha":            0.5,       # L1
    "seed":             42,
    "nthread":          -1,
}

MAX_ROUNDS = 3000
EARLY_STOPPING = 100


# ===================================================================== #
# Splitting
# ===================================================================== #

def time_split(
    df: pd.DataFrame,
    train_seasons: list[int],
    val_seasons: list[int],
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by season, never randomly.

    Rolling features and Elo are both functions of every prior match, so a
    randomly held-out row from 2019 has already influenced the features of
    rows used to train. The model is handed the future and then graded on it.
    Splitting by season is the only arrangement that matches how the model
    would actually be used.
    """
    return (
        df[df["season"].isin(train_seasons)].copy(),
        df[df["season"].isin(val_seasons)].copy(),
        df[df["season"].isin(test_seasons)].copy(),
    )


# ===================================================================== #
# Training
# ===================================================================== #

def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str] | None = None,
    params: dict | None = None,
    verbose: bool = False,
) -> tuple[xgb.Booster, list[str], int]:
    """
    Fit with early stopping on the validation split.

    Returns (booster, feature list, best round).

    Missing values are passed straight through - XGBoost learns a default
    branch direction for NaN at each split, which is more informative than
    imputing a median and pretending a team with no history is average.
    """
    feats = features or all_features()
    feats = [f for f in feats if f in train_df.columns]
    p = {**DEFAULT_PARAMS, **(params or {})}

    dtrain = xgb.DMatrix(train_df[feats], label=train_df["result"],
                         feature_names=feats, missing=np.nan)
    dval = xgb.DMatrix(val_df[feats], label=val_df["result"],
                       feature_names=feats, missing=np.nan)

    booster = xgb.train(
        p, dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=50 if verbose else False,
    )
    return booster, feats, booster.best_iteration


def predict_proba(booster: xgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Predicted (P_home, P_draw, P_away), one row per match."""
    d = xgb.DMatrix(df[features], feature_names=features, missing=np.nan)
    probs = booster.predict(d, iteration_range=(0, booster.best_iteration + 1))
    return np.asarray(probs, dtype=float)


# ===================================================================== #
# Ablation
# ===================================================================== #

def ablate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    Drop one feature group at a time and measure what breaks.

    This is how a feature earns its place. Correlation with the outcome is not
    evidence - a feature can correlate strongly and still be redundant because
    another feature already carries the same information. The only question
    that matters is whether removing it makes the model worse.

    A group whose removal IMPROVES validation loss is actively harmful: it is
    feeding the model noise to overfit. Drop it.
    """
    full_booster, full_feats, _ = train(train_df, val_df, params=params)
    full_val = log_loss_multi(val_df["result"],
                              predict_proba(full_booster, val_df, full_feats))
    full_test = log_loss_multi(test_df["result"],
                               predict_proba(full_booster, test_df, full_feats))

    rows = [{
        "config":     "ALL FEATURES",
        "n_features": len(full_feats),
        "val_logloss":  round(full_val, 4),
        "test_logloss": round(full_test, 4),
        "val_delta":  0.0,
        "verdict":    "baseline",
    }]

    for group in FEATURE_GROUPS:
        feats = all_features(exclude=[group])
        feats = [f for f in feats if f in train_df.columns]
        if not feats:
            continue

        b, f, _ = train(train_df, val_df, features=feats, params=params)
        v = log_loss_multi(val_df["result"], predict_proba(b, val_df, f))
        t = log_loss_multi(test_df["result"], predict_proba(b, test_df, f))
        delta = v - full_val   # positive => removing it hurt => the group helps

        if delta > 0.002:
            verdict = "KEEP (clearly helps)"
        elif delta > 0.0:
            verdict = "keep (marginal)"
        elif delta > -0.002:
            verdict = "neutral"
        else:
            verdict = "DROP (actively harmful)"

        rows.append({
            "config":       f"without '{group}'",
            "n_features":   len(f),
            "val_logloss":  round(v, 4),
            "test_logloss": round(t, 4),
            "val_delta":    round(delta, 4),
            "verdict":      verdict,
        })

    return pd.DataFrame(rows)


def group_only(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    The mirror of ablate(): train on each group ALONE.

    Ablation can understate a group that duplicates another - drop either and
    the survivor covers for it, so both look worthless. Training on one group
    at a time shows the standalone signal and makes that redundancy visible.
    """
    rows = []
    for group, feats in FEATURE_GROUPS.items():
        f = [x for x in feats if x in train_df.columns]
        if not f:
            continue
        b, used, _ = train(train_df, val_df, features=f, params=params)
        rows.append({
            "group":        group,
            "n_features":   len(used),
            "val_logloss":  round(log_loss_multi(
                val_df["result"], predict_proba(b, val_df, used)), 4),
            "test_logloss": round(log_loss_multi(
                test_df["result"], predict_proba(b, test_df, used)), 4),
        })
    return pd.DataFrame(rows).sort_values("val_logloss").reset_index(drop=True)


# ===================================================================== #
# Inspection
# ===================================================================== #

def importance(booster: xgb.Booster, top: int = 25) -> pd.DataFrame:
    """
    Feature importance by average gain.

    'gain' is how much each split using the feature improved the objective,
    averaged over splits. Preferred over 'weight' (raw split count), which
    flatters high-cardinality continuous features simply because there are
    more places to cut them.
    """
    gain = booster.get_score(importance_type="gain")
    cover = booster.get_score(importance_type="weight")
    rows = [{"feature": k, "gain": round(v, 2), "splits": cover.get(k, 0)}
            for k, v in gain.items()]
    return (
        pd.DataFrame(rows)
        .sort_values("gain", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
