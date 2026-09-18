"""
XGBoost market value prediction model.

Fix 4: Position-group split — train one model per position group (GK / DEF / MID / ATT).
Value drivers are fundamentally different per position:
  - GK/DEF: minutes, club prestige, age, international caps (goals ≈ 0)
  - MID: goals + assists + prestige + age
  - ATT: goals_p90 is the dominant signal
Mixing all positions forces the model to average out signals that point in opposite
directions — a GK with 0 goals looks identical to a forward who never plays.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

LOG_TARGET      = True
POSITION_GROUPS = ["GK", "DEF", "MID", "ATT"]
MIN_SAMPLES     = 50   # skip a group if it has fewer than this many training rows

MICRO_POSITION_GROUPS = ["GK", "CB", "FB", "CDM", "CM", "CAM", "W", "ST"]
MIN_SAMPLES_MICRO     = 30  # lower threshold for 8-group model


def _xgb_params(overrides: dict = None) -> dict:
    p = {
        "n_estimators":     400,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
    }
    if overrides:
        p.update(overrides)
    return p


def train(
    X: pd.DataFrame,
    y: pd.Series,
    experiment_name: str = "transfer_market_predictor",
    run_name: str = "xgb",
    params: dict = None,
) -> tuple[XGBRegressor, dict]:
    """Train a single XGBoost model, log to MLflow. Returns (model, metrics)."""
    # Imported here rather than at module scope because only training needs it.
    # The API imports this module for predict_by_position() and load_models(),
    # and mlflow is a heavy dependency that a serving container has no use for -
    # a top-level import made the whole app fail to start with
    # ModuleNotFoundError as soon as mlflow was left out of the runtime install.
    import mlflow
    import mlflow.xgboost

    y_train = np.log1p(y) if LOG_TARGET else y
    model   = XGBRegressor(**_xgb_params(params))

    # On Databricks, experiment names must be absolute workspace paths.
    # Try setting the experiment; if it fails (e.g. relative name on Databricks),
    # fall back to the notebook's auto-assigned experiment.
    try:
        mlflow.set_experiment(experiment_name)
    except Exception:
        pass  # notebook experiment is used automatically

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_xgb_params(params))

        model.fit(X, y_train)

        y_pred_log = model.predict(X)
        y_pred     = np.expm1(y_pred_log) if LOG_TARGET else y_pred_log

        metrics = {
            "train_mae_eur": mean_absolute_error(y, y_pred),
            "train_mape":    mean_absolute_percentage_error(y, y_pred),
            "train_r2":      r2_score(y, y_pred),
            "n_samples":     len(X),
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

    return model, metrics


def train_by_position(
    X: pd.DataFrame,
    y: pd.Series,
    position_groups: pd.Series,
    experiment_name: str = "transfer_market_predictor",
) -> tuple[dict, dict]:
    """
    Fix 4: Train one model per position group.
    Returns (models dict, metrics dict) keyed by position group label.
    """
    models  = {}
    metrics = {}

    for group in POSITION_GROUPS:
        idx = position_groups[position_groups == group].index
        n   = len(idx)
        if n < MIN_SAMPLES:
            print(f"  {group}: {n} samples — skipping (below {MIN_SAMPLES} threshold)")
            continue

        X_g = X.loc[idx].reset_index(drop=True)
        y_g = y.loc[idx].reset_index(drop=True)

        model, m = train(X_g, y_g, experiment_name=experiment_name, run_name=f"xgb_{group}")
        models[group]  = model
        metrics[group] = m
        print(f"  {group}: n={n:,}  R²={m['train_r2']:.3f}  "
              f"MAE=€{m['train_mae_eur']:,.0f}  MAPE={m['train_mape']:.1%}")

    return models, metrics


def train_by_micro_position(
    X: pd.DataFrame,
    y: pd.Series,
    micro_position_groups: pd.Series,
    experiment_name: str = "transfer_market_predictor",
) -> tuple[dict, dict]:
    """
    Train one model per micro-position group (8-group model).
    Uses MICRO_POSITION_GROUPS and MIN_SAMPLES_MICRO threshold.
    Returns (models dict, metrics dict) keyed by micro-position group label.
    Falls back to broad-group models from train_by_position() for groups
    with insufficient data.
    """
    models  = {}
    metrics = {}

    for group in MICRO_POSITION_GROUPS:
        idx = micro_position_groups[micro_position_groups == group].index
        n   = len(idx)
        if n < MIN_SAMPLES_MICRO:
            print(f"  {group}: {n} samples — skipping (below {MIN_SAMPLES_MICRO} threshold)")
            continue

        X_g = X.loc[idx].reset_index(drop=True)
        y_g = y.loc[idx].reset_index(drop=True)

        model, m = train(
            X_g, y_g,
            experiment_name=experiment_name,
            run_name=f"xgb_micro_{group}",
        )
        models[group]  = model
        metrics[group] = m
        print(f"  {group}: n={n:,}  R²={m['train_r2']:.3f}  "
              f"MAE=€{m['train_mae_eur']:,.0f}  MAPE={m['train_mape']:.1%}")

    return models, metrics


def predict(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict(X)
    return np.expm1(raw) if LOG_TARGET else raw


def predict_by_position(
    models: dict,
    X: pd.DataFrame,
    position_groups: pd.Series,
) -> np.ndarray:
    """Route each player to their position-specific model."""
    y_pred   = np.zeros(len(X))
    fallback = models.get("ATT") or next(iter(models.values()))

    for group, model in models.items():
        idx = position_groups[position_groups == group].index
        if len(idx):
            y_pred[X.index.get_indexer(idx)] = predict(model, X.loc[idx])

    # any position group not covered by a trained model → fallback
    covered_idx = set()
    for group in models:
        covered_idx |= set(position_groups[position_groups == group].index)
    uncovered = [i for i in X.index if i not in covered_idx]
    if uncovered:
        y_pred[X.index.get_indexer(uncovered)] = predict(fallback, X.loc[uncovered])

    return y_pred


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    results = pd.DataFrame({
        "actual_market_value_eur":    y_true.values,
        "predicted_market_value_eur": y_pred,
        "difference_eur":             y_pred - y_true.values,
        "pct_error":                  (y_pred - y_true.values) / y_true.values * 100,
    })
    results["abs_pct_error"] = results["pct_error"].abs()
    results["accuracy_band"] = pd.cut(
        results["abs_pct_error"],
        bins=[0, 10, 25, 50, 100, np.inf],
        labels=["<10%", "10-25%", "25-50%", "50-100%", ">100%"],
    )
    return results


def save_models(models: dict, path: str):
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(models, path)
    print(f"  Models saved → {path}")


def load_models(path: str) -> dict:
    import joblib
    return joblib.load(path)


def feature_importance(models: dict, feature_names: list[str]) -> pd.DataFrame:
    """Average feature importance across all position models."""
    rows = []
    for group, model in models.items():
        for feat, score in zip(feature_names, model.feature_importances_):
            rows.append({"position_group": group, "feature": feat, "importance": score})
    df = pd.DataFrame(rows)
    avg = (
        df.groupby("feature")["importance"]
        .mean()
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return avg
