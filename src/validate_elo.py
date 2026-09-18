"""
Validation harness for the Elo system.

Answers three questions, in order:

  1. Does converting Elo to real probabilities fix the calibration bias?
  2. Which (k_factor, home_advantage, regression) actually minimise log loss,
     now that the metric can see them?
  3. How far off bookmaker closing odds are we?

WHY THE SPLIT IS BY TIME, NOT RANDOM
------------------------------------
A random train/test split on time-series data leaks. Elo at any moment is a
function of every match before it, so a randomly held-out match from 2019 was
already baked into the ratings used to predict 2018 - the model is told the
future and then graded on it. Backtests look superb and live performance
collapses.

We therefore train on the earliest seasons and test on the latest, which is
the only arrangement matching how the model would actually be used: everything
known up to today, predicting a match that has not been played.

Run:  python src/validate_elo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from elo import EloRatingSystem, elo_to_probabilities, fit_draw_model
from match_metrics import (
    evaluate, compare, odds_to_probs, baseline_probs,
    calibration_table, log_loss_multi,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "raw", "fdcouk_matches.csv")

# Three-way split. Two would not be enough: if hyperparameters are chosen by
# minimising test-set loss, the test set has been used to make a decision and
# no longer measures generalisation. The reported number would be optimistic
# by an unknown amount.
#
#   train      fit Elo ratings and the draw model
#   validation choose k / hfa / regression
#   test       touched once, at the very end, for the number we report
TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]
VAL_SEASONS   = [2022, 2023]
TEST_SEASONS  = [2024, 2025]

pd.set_option("display.width", 220)


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "utc_date"})
    return df.sort_values("utc_date").reset_index(drop=True)


def rate(df: pd.DataFrame, k: float, hfa: float, reg: float) -> pd.DataFrame:
    """Run one Elo configuration across the whole history."""
    return EloRatingSystem(
        k_factor=k, home_advantage=hfa, season_regression=reg
    ).run(df)


def main():
    df = load()
    print(f"Loaded {len(df):,} matches  ({df['season'].min()}-{df['season'].max()})\n")

    # ---------------------------------------------------------------- #
    print("=" * 74)
    print("STEP 1  Raw Elo expectation vs calibrated probabilities")
    print("=" * 74)

    rated = rate(df, k=20, hfa=65, reg=0.75)
    train = rated[rated["season"].isin(TRAIN_SEASONS)]
    val   = rated[rated["season"].isin(VAL_SEASONS)]
    test  = rated[rated["season"].isin(TEST_SEASONS)]
    print(f"  train {len(train):,}   validation {len(val):,}   test {len(test):,}\n")

    # Naive reading: treat the Elo expectation as P(home) and split the
    # remainder evenly. This is the mistake we are correcting.
    e = test["elo_expected"].values
    naive = np.column_stack([e, np.full_like(e, 0.25), 1.0 - e - 0.25])
    naive = np.clip(naive, 1e-6, None)
    naive /= naive.sum(axis=1, keepdims=True)

    d_max, tau = fit_draw_model(
        train["elo_diff"].values, train["result"].values, home_advantage=65
    )
    print(f"  Fitted draw model on TRAIN only: d_max={d_max:.3f}  tau={tau:.0f}")
    print(f"    -> {d_max:.1%} draw rate between equals, decaying over ~{tau:.0f} Elo pts\n")

    calibrated = elo_to_probabilities(test["elo_diff"].values, 65, d_max, tau)

    print("  Calibration of HOME WIN probability (predicted vs actual):\n")
    print("  --- naive (raw Elo expectation) ---")
    print(calibration_table(test["result"].values, naive, 0, 8).to_string(index=False))
    print("\n  --- calibrated ---")
    print(calibration_table(test["result"].values, calibrated, 0, 8).to_string(index=False))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("STEP 2  Hyperparameter sweeps, scored by log loss")
    print("=" * 74)
    print("  Accuracy could not see home_advantage at all. Log loss can,")
    print("  because it grades the probabilities rather than the argmax.\n")

    def score(k, hfa, reg):
        """Log loss on the VALIDATION seasons. The test set is not touched here."""
        r = rate(df, k, hfa, reg)
        tr = r[r["season"].isin(TRAIN_SEASONS)]
        va = r[r["season"].isin(VAL_SEASONS)]
        dm, t = fit_draw_model(tr["elo_diff"].values, tr["result"].values, hfa)
        p = elo_to_probabilities(va["elo_diff"].values, hfa, dm, t)
        return log_loss_multi(va["result"].values, p)

    print("  k_factor:")
    ks = [10, 15, 20, 25, 30, 40, 50]
    k_scores = [(k, score(k, 65, 0.75)) for k in ks]
    for k, s in k_scores:
        print(f"    k={k:<4} log_loss={s:.4f}")
    best_k = min(k_scores, key=lambda x: x[1])[0]
    print(f"    -> best k = {best_k}")

    print("\n  home_advantage:")
    hs = [0, 30, 50, 65, 80, 100, 120]
    h_scores = [(h, score(best_k, h, 0.75)) for h in hs]
    for h, s in h_scores:
        print(f"    hfa={h:<4} log_loss={s:.4f}")
    best_h = min(h_scores, key=lambda x: x[1])[0]
    print(f"    -> best hfa = {best_h}   (accuracy showed no signal here at all)")

    print("\n  season_regression:")
    rs = [0.50, 0.65, 0.75, 0.85, 1.00]
    r_scores = [(r, score(best_k, best_h, r)) for r in rs]
    for r, s in r_scores:
        print(f"    reg={r:<5} log_loss={s:.4f}")
    best_r = min(r_scores, key=lambda x: x[1])[0]
    print(f"    -> best regression = {best_r}")

    print(f"\n  TUNED CONFIG: k={best_k}  hfa={best_h}  regression={best_r}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("STEP 3  Validation against reality, including the market")
    print("=" * 74)

    # Hyperparameters are now locked. For the final fit we can use train+val
    # together - more data for the draw model - because no further choices
    # are made from here. The test set is scored exactly once.
    tuned = rate(df, best_k, best_h, best_r)
    tr = tuned[tuned["season"].isin(TRAIN_SEASONS + VAL_SEASONS)]
    te = tuned[tuned["season"].isin(TEST_SEASONS)].copy()
    dm, t = fit_draw_model(tr["elo_diff"].values, tr["result"].values, best_h)

    y = te["result"].values
    results = [
        evaluate(y, np.tile([1/3, 1/3, 1/3], (len(te), 1)), "Uniform 1/3"),
        evaluate(y, baseline_probs(tr["result"].values, len(te)), "Class priors"),
        evaluate(y, naive, "Elo raw (uncalibrated)"),
        evaluate(y, elo_to_probabilities(te["elo_diff"].values, 65, d_max, tau),
                 "Elo calibrated (untuned)"),
        evaluate(y, elo_to_probabilities(te["elo_diff"].values, best_h, dm, t),
                 "Elo calibrated + tuned"),
    ]

    # Bookmaker benchmark - the number that actually matters.
    odds_cols = ["odds_close_home", "odds_close_draw", "odds_close_away"]
    has_odds = te[odds_cols].notna().all(axis=1)
    if has_odds.sum() > 100:
        sub = te[has_odds]
        book = odds_to_probs(sub["odds_close_home"], sub["odds_close_draw"],
                             sub["odds_close_away"])
        results.append(evaluate(sub["result"].values, book,
                                "BOOKMAKER (Pinnacle close)"))

        # Like-for-like on the identical subset, otherwise the comparison is unfair.
        results.append(evaluate(
            sub["result"].values,
            elo_to_probabilities(sub["elo_diff"].values, best_h, dm, t),
            "Elo tuned (odds subset)",
        ))

    print()
    print(compare(results).to_string(index=False))

    print("\n  Reading this table:")
    print("    log_loss - primary metric, lower is better")
    print("    ece      - mean calibration gap; under ~0.02 is trustworthy")
    print("    accuracy - reported for context only, never tuned on")

    if has_odds.sum() > 100:
        rows = {r["model"]: r for r in results}
        gap = rows["Elo tuned (odds subset)"]["log_loss"] - rows["BOOKMAKER (Pinnacle close)"]["log_loss"]
        print(f"\n  Gap to the market: {gap:+.4f} log loss")
        print("  Elo alone is one feature against a market pricing in team news,")
        print("  injuries, motivation and money. Closing that gap is the job of")
        print("  the remaining ~22 features plus the gradient-boosted model.")


if __name__ == "__main__":
    main()
