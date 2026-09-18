"""
Are the differences between these models real, or noise?

Six configurations landed between 0.9954 and 0.9980 test log loss. Before
declaring a winner we have to ask whether a gap that size is distinguishable
from chance on 2,280 matches. Reporting "model F wins" without checking is how
people convince themselves a pipeline improved when it did not.

METHOD: paired bootstrap
------------------------
Log loss is a mean over per-match losses, so it has a sampling distribution.
Resample matches with replacement, recompute both models' loss on the same
resampled set, and record the difference. Repeat many times to trace out the
distribution of that difference.

Pairing matters. Both models are scored on identical matches, so a shock
result hurts both and cancels in the difference. An unpaired test would drown
the comparison in variance that is common to both models.

If the 95% interval for (A - B) straddles zero, the models are indistinguishable.

Run:  python src/significance_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from match_model import time_split, train, predict_proba
from match_metrics import odds_to_probs, _clean
from match_features import all_features
from elo import elo_to_probabilities, fit_draw_model
from experiment_match_model import add_elo_probs

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATS = os.path.join(BASE, "data", "processed", "match_features.csv")

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]
VAL_SEASONS   = [2022, 2023]
TEST_SEASONS  = [2024, 2025]

N_BOOT = 4000
RNG = np.random.default_rng(42)

pd.set_option("display.width", 240)


def per_match_loss(y, probs) -> np.ndarray:
    """Log loss for each match individually, rather than the mean."""
    p = _clean(np.asarray(probs, dtype=float))
    y = np.asarray(y, dtype=int)
    return -np.log(p[np.arange(len(y)), y])


def bootstrap_diff(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    """Paired bootstrap of mean(loss_a) - mean(loss_b)."""
    n = len(loss_a)
    diff = loss_a - loss_b
    boot = np.array([
        diff[RNG.integers(0, n, n)].mean() for _ in range(N_BOOT)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "diff":        float(diff.mean()),
        "ci_low":      float(lo),
        "ci_high":     float(hi),
        # Two-sided: how often does the bootstrap cross zero?
        "p_two_sided": float(2 * min((boot > 0).mean(), (boot < 0).mean())),
        "significant": bool(lo > 0 or hi < 0),
    }


def main():
    df = pd.read_csv(FEATS, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    # Ratings are built across all divisions; only the top tier is scored.
    if "tier" in df.columns:
        df = df[df["tier"] == 1].reset_index(drop=True)
    tr, va, te = time_split(df, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)

    d_max, tau = fit_draw_model(tr["elo_diff"].values, tr["result"].values, 65)
    tr_s, va_s, te_s = (add_elo_probs(x, d_max, tau) for x in (tr, va, te))
    ELO_P = ["elo_p_home", "elo_p_draw", "elo_p_away", "elo_logit"]

    y = te["result"].values
    models: dict[str, np.ndarray] = {}

    # 1. parametric Elo
    models["Elo parametric (2 params)"] = elo_to_probabilities(
        te["elo_diff"].values, 65, d_max, tau)

    # 2. best stacked model from the experiments
    feats_f = [f for f in all_features() if f in tr_s.columns] + ELO_P
    bF, usedF, _ = train(tr_s, va_s, features=feats_f,
                         params={"max_depth": 2, "lambda": 5.0,
                                 "min_child_weight": 40})
    models["XGB stacked + all (46)"] = predict_proba(bF, te_s, usedF)

    # 3. minimal
    fmin = ["elo_diff", "form_ppg_diff", "proxy_xg_diff", "goals_for_diff"]
    bB, usedB, _ = train(tr, va, features=fmin)
    models["XGB minimal (4)"] = predict_proba(bB, te, usedB)

    # 4. full unregularised
    feats_all = [f for f in all_features() if f in tr.columns]
    bA, usedA, _ = train(tr, va, features=feats_all)
    models["XGB all features (42)"] = predict_proba(bA, te, usedA)

    print("=" * 78)
    print("TEST LOG LOSS")
    print("=" * 78)
    losses = {k: per_match_loss(y, v) for k, v in models.items()}
    for k, v in sorted(losses.items(), key=lambda x: x[1].mean()):
        print(f"  {k:<30} {v.mean():.4f}")

    se = np.std(list(losses.values())[0]) / np.sqrt(len(y))
    print(f"\n  Standard error of a single log loss estimate: {se:.4f}")
    print(f"  (n = {len(y)} matches)")
    print(f"  -> differences smaller than ~{2*se:.4f} cannot be resolved at all.")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("PAIRWISE COMPARISONS (paired bootstrap, 4000 resamples)")
    print("=" * 78)
    print("  Negative diff = first model is BETTER.\n")

    names = list(models)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = bootstrap_diff(losses[a], losses[b])
            rows.append({
                "comparison": f"{a}  vs  {b}",
                "diff":       round(r["diff"], 4),
                "95% CI":     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
                "p":          round(r["p_two_sided"], 3),
                "verdict":    "SIGNIFICANT" if r["significant"] else "indistinguishable",
            })
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("VERSUS THE MARKET")
    print("=" * 78)
    odds_cols = ["odds_close_home", "odds_close_draw", "odds_close_away"]
    mask = te[odds_cols].notna().all(axis=1).values
    ys = y[mask]
    book_loss = per_match_loss(ys, odds_to_probs(*[te.loc[mask, c] for c in odds_cols]))

    print(f"  Matches with closing odds: {mask.sum()}\n")
    rows = []
    for name, probs in models.items():
        ml = per_match_loss(ys, probs[mask])
        r = bootstrap_diff(ml, book_loss)
        rows.append({
            "model":    name,
            "logloss":  round(ml.mean(), 4),
            "vs_book":  round(r["diff"], 4),
            "95% CI":   f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
            "p":        round(r["p_two_sided"], 4),
            "verdict":  "market is better" if r["significant"] else "no clear difference",
        })
    print(f"  Bookmaker log loss: {book_loss.mean():.4f}\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
