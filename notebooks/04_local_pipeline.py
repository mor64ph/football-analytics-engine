"""
Local end-to-end pipeline — no Databricks required.
Trains on Transfermarkt data, scores via football-data.org API,
outputs CSVs ready for the React/FastAPI dashboard.

Step 0 (run once first):
    python src/prepare_tm_data.py

Then run this:
    python notebooks/04_local_pipeline.py
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from football_api import FootballDataClient, COMPETITIONS, scoring_season
from feature_engineering import build_features, FEATURE_COLS
from model import (
    train_by_position, predict_by_position,
    predict, evaluate, feature_importance, save_models,
)

RAW              = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED        = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TM_CSV           = os.path.join(RAW, "tm_training_data.csv")
PRESTIGE_LOOKUP  = os.path.join(RAW, "player_prestige_lookup.csv")
CAPS_LOOKUP      = os.path.join(RAW, "player_caps_lookup.csv")


def run(api_key: str, seasons: list[int] = None, competitions: list[str] = None):
    os.makedirs(PROCESSED, exist_ok=True)

    # ── 1. Load TM training data ────────────────────────────────────────────────
    print("Step 1: Loading Transfermarkt training data...")
    if not os.path.exists(TM_CSV):
        raise FileNotFoundError(
            f"TM training data not found at {TM_CSV}\n"
            "Run:  python src/prepare_tm_data.py"
        )
    tm_df = pd.read_csv(TM_CSV)
    print(f"  → {len(tm_df):,} player-season rows | {tm_df['player_id'].nunique():,} unique players")
    print(f"  → Market value range: €{tm_df['market_value_eur'].min():,.0f} – €{tm_df['market_value_eur'].max():,.0f}\n")

    # ── 2. Build features ───────────────────────────────────────────────────────
    print("Step 2: Building features...")
    X = build_features(tm_df)
    y = tm_df["market_value_eur"].reset_index(drop=True)
    position_groups = tm_df["position_group"].reset_index(drop=True)
    print(f"  → {X.shape[1]} features for {len(X):,} samples\n")

    # ── 3. Train position-group models ──────────────────────────────────────────
    print("Step 3: Training XGBoost models per position group...")
    X_train, X_test, y_train, y_test, pg_train, pg_test = train_test_split(
        X, y, position_groups, test_size=0.2, random_state=42
    )
    models, train_metrics = train_by_position(
        X_train, y_train, pg_train,
        experiment_name="transfer_market_local",
    )

    # ── 4. Evaluate on hold-out test set ────────────────────────────────────────
    print("\nStep 4: Evaluating on hold-out test set...")
    y_pred_test  = predict_by_position(models, X_test, pg_test)
    test_results = evaluate(y_test, y_pred_test)
    test_mae     = test_results["difference_eur"].abs().mean()
    test_mape    = test_results["abs_pct_error"].mean()
    test_r2      = 1 - ((y_test.values - y_pred_test) ** 2).sum() / ((y_test.values - y_test.values.mean()) ** 2).sum()

    print(f"  Test R²:   {test_r2:.3f}")
    print(f"  Test MAE:  €{test_mae:,.0f}")
    print(f"  Test MAPE: {test_mape:.1f}%")
    print(f"  Accuracy bands (test set):\n{test_results['accuracy_band'].value_counts().to_string()}\n")

    # ── 5. Pull current-season data from football-data.org ─────────────────────
    print("Step 5: Pulling current players from football-data.org API...")
    seasons      = seasons      or [scoring_season()]
    competitions = competitions or list(COMPETITIONS.keys())

    client = FootballDataClient(api_key=api_key)
    all_scorers = []
    failures = []
    for comp in competitions:
        for season in seasons:
            print(f"  {comp} {season}...", end=" ", flush=True)
            try:
                rows = client.get_scorers(competition=comp, season=season, limit=100)
                all_scorers.extend(rows)
                print(f"{len(rows)} players")
            except Exception as e:
                print(f"FAILED: {e}")
                failures.append(f"{comp} {season}: {e}")

    # A transient API error here used to be printed and then ignored, leaving a
    # dataset silently missing an entire league - one run dropped every
    # Bundesliga player and still wrote its output. Now that this runs in CI and
    # commits what it produces, a partial file would ship to the live site.
    # Fail instead; the previous day's committed data stays up and tomorrow's
    # run recovers.
    if failures:
        raise RuntimeError(
            "Scorer fetch failed for:\n  " + "\n  ".join(failures)
            + "\nRefusing to write a partial dataset."
        )

    api_df = pd.DataFrame(all_scorers)
    api_df.to_csv(os.path.join(PROCESSED, "bronze_scorers.csv"), index=False)
    print(f"  → {len(api_df)} total rows from API\n")

    # ── 6. Enrich API data with lookup tables, then score ───────────────────────
    print("Step 6: Enriching API players with prestige & caps lookups...")
    prestige_lk = pd.read_csv(PRESTIGE_LOOKUP)
    caps_lk     = pd.read_csv(CAPS_LOOKUP)

    def _norm(s: pd.Series) -> pd.Series:
        return s.str.lower().str.strip()

    api_df["_key"] = _norm(api_df["player_name"])
    prestige_lk["_key"] = _norm(prestige_lk["player_name"])
    caps_lk["_key"]     = _norm(caps_lk["player_name"])

    api_df = api_df.merge(prestige_lk[["_key", "club_prestige_eur"]], on="_key", how="left")
    api_df = api_df.merge(caps_lk[["_key", "international_caps"]],   on="_key", how="left")
    api_df.drop(columns=["_key"], inplace=True)

    # For players not found in the lookup (new/unlisted), use the competition median
    comp_medians = tm_df.groupby("competition")["club_prestige_eur"].median()
    for i, row in api_df.iterrows():
        if pd.isna(row["club_prestige_eur"]):
            api_df.at[i, "club_prestige_eur"] = comp_medians.get(row["competition"], 0)
    api_df["international_caps"] = api_df["international_caps"].fillna(0)

    matched = api_df["club_prestige_eur"].notna().sum()
    print(f"  → Prestige matched: {matched}/{len(api_df)} players")

    print("Step 6b: Predicting market values for current players...")
    X_api  = build_features(api_df)
    pg_api = X_api.apply(
        lambda r: "ATT" if r["pos_ATT"] else ("GK" if r["pos_GK"] else ("DEF" if r["pos_DEF"] else "MID")),
        axis=1,
    )
    y_api = predict_by_position(models, X_api, pg_api)

    score_df = api_df[[
        "player_name", "team_name", "competition", "season",
        "position", "goals", "assists", "played_matches",
    ]].copy().reset_index(drop=True)
    score_df["age"]                  = build_features(api_df)["age"].values
    score_df["predicted_value_eur"]  = y_api
    score_df.sort_values("predicted_value_eur", ascending=False, inplace=True)

    # ── 7. Historical evaluation output (for dashboard) ─────────────────────────
    print("Step 7: Generating actual vs predicted Gold layer...")
    X_all   = X.reset_index(drop=True)
    pg_all  = position_groups.reset_index(drop=True)
    y_all   = y.reset_index(drop=True)
    y_all_pred = predict_by_position(models, X_all, pg_all)
    eval_all   = evaluate(y_all, y_all_pred)

    gold_df = tm_df[[
        "player_name", "competition", "season", "position_group",
        "age", "goals", "assists", "minutes_played",
        "international_caps", "club_prestige_eur", "market_value_eur",
    ]].copy().reset_index(drop=True)
    gold_df["predicted_value_eur"] = y_all_pred
    gold_df["difference_eur"]      = eval_all["difference_eur"].values
    gold_df["pct_error"]           = eval_all["pct_error"].values
    gold_df["abs_pct_error"]       = eval_all["abs_pct_error"].values
    gold_df["accuracy_band"]       = eval_all["accuracy_band"].astype(str).values

    # ── 8. Feature importance ───────────────────────────────────────────────────
    fi_df = feature_importance(models, FEATURE_COLS)

    # ── 9. Save outputs ─────────────────────────────────────────────────────────
    gold_df.to_csv(os.path.join(PROCESSED, "gold_player_predictions.csv"), index=False)
    score_df.to_csv(os.path.join(PROCESSED, "gold_current_season_scores.csv"), index=False)
    fi_df.to_csv(os.path.join(PROCESSED, "gold_feature_importance.csv"), index=False)

    models_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "position_models.pkl")
    save_models(models, models_path)

    cols = ["player_name", "competition", "position_group", "market_value_eur", "predicted_value_eur", "pct_error"]
    print(f"\n  Top 10 TM-undervalued (model predicts HIGHER than TM — hidden gems):")
    print(gold_df.nlargest(10, "pct_error")[cols].to_string(index=False))

    print(f"\n  Top 10 TM-overvalued (TM price HIGHER than model — reputation premium):")
    print(gold_df.nsmallest(10, "pct_error")[cols].to_string(index=False))

    print(f"\n  Top 10 highest predicted current-season values:")
    print(score_df.head(10)[
        ["player_name", "competition", "position", "goals", "assists", "age", "predicted_value_eur"]
    ].to_string(index=False))

    print(f"\n  Feature importance (averaged across position models):")
    print(fi_df.to_string(index=False))

    print(f"\nOutputs written to: {os.path.abspath(PROCESSED)}/")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",      default=None)
    parser.add_argument("--seasons",      nargs="+", type=int, default=None)
    parser.add_argument("--competitions", nargs="+", default=None)
    args = parser.parse_args()

    run(
        api_key=args.api_key,
        seasons=args.seasons,
        competitions=args.competitions,
    )
