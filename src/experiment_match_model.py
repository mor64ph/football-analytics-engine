"""
Why did 42 features lose to a 2-parameter model, and what fixes it?

The full-feature model was beaten on test by calibrated Elo. Three candidate
explanations, each with a matching experiment:

  1. Too much capacity for the signal  -> shrink the feature set
  2. Too little regularisation         -> constrain the trees harder
  3. The wrong starting point          -> hand the model Elo's calibrated
                                          probabilities as features and let it
                                          learn corrections rather than
                                          rediscover the whole relationship

(3) is stacking: a strong structured model feeding a flexible one. Elo encodes
a logistic skill relationship and a draw structure directly. XGBoost has to
infer that shape from noisy data, and with football outcomes it cannot recover
it as cleanly as simply being told.

Run:  python src/experiment_match_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from match_model import time_split, train, predict_proba, DEFAULT_PARAMS
from match_metrics import (
    evaluate, compare, odds_to_probs, log_loss_multi, calibration_table,
)
from elo import elo_to_probabilities, fit_draw_model

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATS = os.path.join(BASE, "data", "processed", "match_features.csv")

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]
VAL_SEASONS   = [2022, 2023]
TEST_SEASONS  = [2024, 2025]

pd.set_option("display.width", 240)


def add_elo_probs(df: pd.DataFrame, d_max: float, tau: float,
                  hfa: float = 65.0) -> pd.DataFrame:
    """
    Attach the parametric Elo model's calibrated probabilities as features.

    The draw-model parameters must come from TRAINING data only - they are
    fitted quantities, and fitting them on all rows would leak.
    """
    out = df.copy()
    p = elo_to_probabilities(out["elo_diff"].values, hfa, d_max, tau)
    out["elo_p_home"] = p[:, 0]
    out["elo_p_draw"] = p[:, 1]
    out["elo_p_away"] = p[:, 2]
    # Log-odds of the home side. Trees split on thresholds, and a linear-ish
    # quantity is easier to cut usefully than a bounded probability.
    out["elo_logit"] = np.log(p[:, 0] / np.clip(p[:, 2], 1e-9, None))
    return out


def main():
    df = pd.read_csv(FEATS)
    df["date"] = pd.to_datetime(df["date"])
    tr, va, te = time_split(df, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)
    print(f"train {len(tr):,}   val {len(va):,}   test {len(te):,}\n")

    # Draw model fitted on train only.
    d_max, tau = fit_draw_model(tr["elo_diff"].values, tr["result"].values, 65)
    print(f"Draw model (train only): d_max={d_max:.3f} tau={tau:.0f}\n")

    tr_s = add_elo_probs(tr, d_max, tau)
    va_s = add_elo_probs(va, d_max, tau)
    te_s = add_elo_probs(te, d_max, tau)

    ELO_PROB_FEATS = ["elo_p_home", "elo_p_draw", "elo_p_away", "elo_logit"]

    experiments = {
        "A. Elo params only (3 raw)": (
            ["elo_diff", "elo_home_pre", "elo_away_pre"], None, False),

        "B. Minimal (elo_diff + 3 diffs)": (
            ["elo_diff", "form_ppg_diff", "proxy_xg_diff", "goals_for_diff"],
            None, False),

        "C. All 42, heavy regularisation": (
            None,
            {"max_depth": 2, "lambda": 10.0, "alpha": 2.0,
             "min_child_weight": 50, "eta": 0.02},
            False),

        "D. Stacked: Elo probs only": (
            ELO_PROB_FEATS, None, True),

        "E. Stacked + form/shot diffs": (
            ELO_PROB_FEATS + ["form_ppg_diff", "proxy_xg_diff",
                              "sot_diff", "goals_for_diff",
                              "home_venue_ppg", "away_venue_ppg"],
            None, True),

        "F. Stacked + all, regularised": (
            None,
            {"max_depth": 2, "lambda": 5.0, "min_child_weight": 40},
            True),
    }

    rows, preds = [], {}
    for name, (feats, params, stacked) in experiments.items():
        TR, VA, TE = (tr_s, va_s, te_s) if stacked else (tr, va, te)
        if feats is None:
            from match_features import all_features
            feats = [f for f in all_features() if f in TR.columns]
            if stacked:
                feats = feats + ELO_PROB_FEATS

        b, used, best = train(TR, VA, features=feats, params=params)
        pv = predict_proba(b, VA, used)
        pt = predict_proba(b, TE, used)
        preds[name] = pt

        rows.append({
            "experiment":   name,
            "n_feat":       len(used),
            "rounds":       best,
            "val_logloss":  round(log_loss_multi(VA["result"], pv), 4),
            "test_logloss": round(log_loss_multi(TE["result"], pt), 4),
        })
        print(f"  {name:<34} val={rows[-1]['val_logloss']:.4f}  "
              f"test={rows[-1]['test_logloss']:.4f}  ({len(used)} feat)")

    # Reference points -------------------------------------------------- #
    elo_only_test = elo_to_probabilities(te["elo_diff"].values, 65, d_max, tau)
    rows.append({
        "experiment":   "REF. Elo parametric (2 params)",
        "n_feat":       2,
        "rounds":       0,
        "val_logloss":  round(log_loss_multi(
            va["result"], elo_to_probabilities(va["elo_diff"].values, 65, d_max, tau)), 4),
        "test_logloss": round(log_loss_multi(te["result"], elo_only_test), 4),
    })

    print("\n" + "=" * 78)
    print("RESULTS (sorted by test log loss)")
    print("=" * 78)
    res = pd.DataFrame(rows).sort_values("test_logloss").reset_index(drop=True)
    print(res.to_string(index=False))

    # Against the market ------------------------------------------------- #
    print("\n" + "=" * 78)
    print("BEST MODEL vs THE MARKET (matches with closing odds)")
    print("=" * 78)

    best_name = res.iloc[0]["experiment"]
    odds_cols = ["odds_close_home", "odds_close_draw", "odds_close_away"]
    mask = te[odds_cols].notna().all(axis=1).values

    sub_y = te["result"].values[mask]
    book = odds_to_probs(*[te.loc[mask, c] for c in odds_cols])

    final = [evaluate(sub_y, book, "BOOKMAKER (Pinnacle close)")]
    if best_name.startswith("REF"):
        final.append(evaluate(sub_y, elo_only_test[mask], best_name))
    else:
        final.append(evaluate(sub_y, preds[best_name][mask], best_name))
    final.append(evaluate(sub_y, elo_only_test[mask], "Elo parametric"))

    print()
    print(compare(final).to_string(index=False))

    gap = final[1]["log_loss"] - final[0]["log_loss"]
    print(f"\n  Best model gap to market: {gap:+.4f}")

    print("\n" + "=" * 78)
    print("CALIBRATION OF THE WINNER — home win")
    print("=" * 78)
    wp = elo_only_test if best_name.startswith("REF") else preds[best_name]
    print(calibration_table(te["result"].values, wp, 0, 8).to_string(index=False))


if __name__ == "__main__":
    main()
