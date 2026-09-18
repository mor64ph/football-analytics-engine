"""
Train and evaluate the match outcome model.

Run:  python src/train_match_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from match_features import build_features, all_features
from match_model import (
    time_split, train, predict_proba, ablate, group_only, importance,
)
from match_metrics import (
    evaluate, compare, odds_to_probs, baseline_probs,
    calibration_table, log_loss_multi,
)
from elo import elo_to_probabilities, fit_draw_model

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(BASE, "data", "raw", "fdcouk_matches.csv")
FEATS = os.path.join(BASE, "data", "processed", "match_features.csv")
MODEL_OUT = os.path.join(BASE, "data", "models", "match_outcome.json")

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]
VAL_SEASONS   = [2022, 2023]
TEST_SEASONS  = [2024, 2025]

pd.set_option("display.width", 240)


def load() -> pd.DataFrame:
    """Use the cached feature matrix when present, otherwise rebuild it."""
    if os.path.exists(FEATS):
        df = pd.read_csv(FEATS, low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        print(f"Loaded cached features: {len(df):,} matches")
    else:
        raw = pd.read_csv(RAW)
        raw["date"] = pd.to_datetime(raw["date"])
        print(f"Building features from {len(raw):,} matches...")
        df = build_features(raw)
        os.makedirs(os.path.dirname(FEATS), exist_ok=True)
        df.to_csv(FEATS, index=False)

    # The feature file spans every division because the rating system needs
    # the second tiers. Modelling and scoring happen on the top tier only.
    if "tier" in df.columns:
        df = df[df["tier"] == 1].reset_index(drop=True)
        print(f"  top tier only: {len(df):,} matches")
    return df


def main():
    df = load()
    tr, va, te = time_split(df, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)
    print(f"  train {len(tr):,} ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]})   "
          f"val {len(va):,} ({VAL_SEASONS[0]}-{VAL_SEASONS[-1]})   "
          f"test {len(te):,} ({TEST_SEASONS[0]}-{TEST_SEASONS[-1]})\n")

    # ---------------------------------------------------------------- #
    print("=" * 78)
    print("STEP 1  Train on all features")
    print("=" * 78)
    booster, feats, best_round = train(tr, va, verbose=True)
    print(f"\n  Features: {len(feats)}   best round: {best_round}")

    p_val = predict_proba(booster, va, feats)
    p_test = predict_proba(booster, te, feats)
    print(f"  val  log loss: {log_loss_multi(va['result'], p_val):.4f}")
    print(f"  test log loss: {log_loss_multi(te['result'], p_test):.4f}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 2  Feature importance (average gain)")
    print("=" * 78)
    print(importance(booster, 20).to_string(index=False))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 3  Ablation — does each group earn its place?")
    print("=" * 78)
    print("  Positive val_delta = removing the group HURT = the group helps.\n")
    abl = ablate(tr, va, te)
    print(abl.to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 4  Each group ALONE — standalone signal")
    print("=" * 78)
    print(group_only(tr, va, te).to_string(index=False))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 5  Final scoreboard on the TEST seasons")
    print("=" * 78)

    y = te["result"].values
    results = [
        evaluate(y, np.tile([1/3, 1/3, 1/3], (len(te), 1)), "Uniform 1/3"),
        evaluate(y, baseline_probs(tr["result"].values, len(te)), "Class priors"),
    ]

    # Elo alone, calibrated - the previous best, for reference.
    dm, tau = fit_draw_model(
        pd.concat([tr, va])["elo_diff"].values,
        pd.concat([tr, va])["result"].values,
        home_advantage=65,
    )
    results.append(evaluate(
        y, elo_to_probabilities(te["elo_diff"].values, 65, dm, tau),
        "Elo alone (calibrated)"))

    results.append(evaluate(y, p_test, "XGBoost (all features)"))

    odds_cols = ["odds_close_home", "odds_close_draw", "odds_close_away"]
    has_odds = te[odds_cols].notna().all(axis=1)
    if has_odds.sum() > 100:
        sub = te[has_odds]
        book = odds_to_probs(sub["odds_close_home"], sub["odds_close_draw"],
                             sub["odds_close_away"])
        results.append(evaluate(sub["result"].values, book,
                                "BOOKMAKER (Pinnacle close)"))
        results.append(evaluate(
            sub["result"].values,
            predict_proba(booster, sub, feats),
            "XGBoost (odds subset)"))

    print()
    print(compare(results).to_string(index=False))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 6  Calibration of the final model")
    print("=" * 78)
    for idx, label in [(0, "HOME WIN"), (1, "DRAW"), (2, "AWAY WIN")]:
        print(f"\n  --- {label} ---")
        print(calibration_table(y, p_test, idx, 8).to_string(index=False))

    # ---------------------------------------------------------------- #
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    booster.save_model(MODEL_OUT)
    with open(MODEL_OUT.replace(".json", "_features.txt"), "w") as fh:
        fh.write("\n".join(feats))
    print(f"\n  Model saved -> {MODEL_OUT}")

    if has_odds.sum() > 100:
        rows = {r["model"]: r for r in results}
        gap = (rows["XGBoost (odds subset)"]["log_loss"]
               - rows["BOOKMAKER (Pinnacle close)"]["log_loss"])
        print(f"\n  Gap to the market: {gap:+.4f} log loss")


if __name__ == "__main__":
    main()
