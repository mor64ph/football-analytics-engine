"""
Fit and evaluate Dixon-Coles, then test whether combining it with Elo helps.

Run:  python src/evaluate_dixon_coles.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from dixon_coles import DixonColes, walk_forward
from match_metrics import evaluate, compare, odds_to_probs, log_loss_multi, _clean
from elo import elo_to_probabilities, fit_draw_model
from match_model import time_split

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATS = os.path.join(BASE, "data", "processed", "match_features.csv")

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]
VAL_SEASONS   = [2022, 2023]
TEST_SEASONS  = [2024, 2025]

pd.set_option("display.width", 240)
RNG = np.random.default_rng(42)


def per_match_loss(y, probs):
    p = _clean(np.asarray(probs, dtype=float))
    y = np.asarray(y, dtype=int)
    return -np.log(p[np.arange(len(y)), y])


def boot_diff(a, b, n_boot=4000):
    d = a - b
    n = len(d)
    bs = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi, (lo > 0 or hi < 0)


def main():
    df = pd.read_csv(FEATS, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    # Ratings are built across all divisions; only the top tier is scored.
    if "tier" in df.columns:
        df = df[df["tier"] == 1].reset_index(drop=True)
    tr, va, te = time_split(df, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)
    print(f"train {len(tr):,}   val {len(va):,}   test {len(te):,}\n")

    # ---------------------------------------------------------------- #
    print("=" * 78)
    print("STEP 1  Single fit on training data — does it look sane?")
    print("=" * 78)
    m = DixonColes(xi=0.0019).fit(tr)
    print(f"  matches used : {m._n_matches:,}")
    print(f"  home adv     : {m.home_adv:+.4f}  "
          f"(exp = {np.exp(m.home_adv):.3f}x goal rate at home)")
    print(f"  rho          : {m.rho:+.4f}  "
          f"({'inflates draws' if m.rho < 0 else 'suppresses draws'})")
    print("\n  Top 10 teams by attack + defence:")
    print(m.table(10).to_string(index=False))

    print("\n  Sample prediction — Man City vs Burnley:")
    p = m.predict("Man City", "Burnley")
    print(f"    home {p['p_home']:.1%}   draw {p['p_draw']:.1%}   away {p['p_away']:.1%}")
    print(f"    xG   {p['xg_home']:.2f} - {p['xg_away']:.2f}")
    print(f"    over 2.5 {p['over_2_5']:.1%}   BTTS {p['btts']:.1%}")
    print(f"    most likely scores: "
          + ", ".join(f"{s['score']} ({s['prob']:.1%})" for s in p["top_scores"][:4]))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 2  Tune the time-decay parameter xi on VALIDATION")
    print("=" * 78)
    print("  xi controls how fast old matches are discounted.")
    print("  xi=0 treats a 2016 match as equal to last week's.\n")

    val_mask = df["season"].isin(VAL_SEASONS)
    best_xi, best_loss = None, float("inf")
    for xi in [0.0, 0.0005, 0.001, 0.0019, 0.003, 0.005]:
        half_life = (np.log(2) / xi / 365) if xi > 0 else float("inf")
        p = walk_forward(df, val_mask, xi=xi, refit_days=45, verbose=False)
        loss = log_loss_multi(va["result"].values, p)
        hl = f"{half_life:.2f}y" if np.isfinite(half_life) else "never"
        print(f"    xi={xi:<7} half-life={hl:<7} val_logloss={loss:.4f}")
        if loss < best_loss:
            best_xi, best_loss = xi, loss
    print(f"    -> best xi = {best_xi}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 3  Walk-forward over the TEST seasons")
    print("=" * 78)
    test_mask = df["season"].isin(TEST_SEASONS)
    dc_probs = walk_forward(df, test_mask, xi=best_xi, refit_days=30, verbose=True)

    # Elo reference, fitted on train+val only.
    trva = pd.concat([tr, va])
    d_max, tau = fit_draw_model(trva["elo_diff"].values, trva["result"].values, 65)
    elo_probs = elo_to_probabilities(te["elo_diff"].values, 65, d_max, tau)

    # Blend. Two models built from the same results but through different
    # mechanisms - a rating difference vs a goal-scoring process - so their
    # errors are not identical. Averaging probabilities cancels some of the
    # independent error even when neither model is individually better.
    y = te["result"].values
    print("\n  Blend weight search (on validation):")
    val_dc = walk_forward(df, val_mask, xi=best_xi, refit_days=45, verbose=False)
    val_elo = elo_to_probabilities(va["elo_diff"].values, 65, d_max, tau)
    best_w, best_wl = 0.5, float("inf")
    for w in np.arange(0, 1.01, 0.1):
        bl = w * val_dc + (1 - w) * val_elo
        l = log_loss_multi(va["result"].values, bl)
        if l < best_wl:
            best_w, best_wl = float(w), l
        print(f"    w_dc={w:.1f}  val_logloss={l:.4f}")
    print(f"    -> best blend weight = {best_w:.1f} Dixon-Coles / "
          f"{1-best_w:.1f} Elo")

    blend = best_w * dc_probs + (1 - best_w) * elo_probs

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 4  Scoreboard")
    print("=" * 78)

    results = [
        evaluate(y, elo_probs, "Elo parametric"),
        evaluate(y, dc_probs, f"Dixon-Coles (xi={best_xi})"),
        evaluate(y, blend, f"Blend {best_w:.0%} DC / {1-best_w:.0%} Elo"),
    ]

    odds_cols = ["odds_close_home", "odds_close_draw", "odds_close_away"]
    mask = te[odds_cols].notna().all(axis=1).values
    book = odds_to_probs(*[te.loc[mask, c] for c in odds_cols])
    results.append(evaluate(te["result"].values[mask], book, "BOOKMAKER"))

    print()
    print(compare(results).to_string(index=False))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 5  Is any difference real? (paired bootstrap)")
    print("=" * 78)
    l_elo = per_match_loss(y, elo_probs)
    l_dc = per_match_loss(y, dc_probs)
    l_bl = per_match_loss(y, blend)

    for name, a, b in [
        ("Dixon-Coles vs Elo", l_dc, l_elo),
        ("Blend vs Elo",       l_bl, l_elo),
        ("Blend vs Dixon-Coles", l_bl, l_dc),
    ]:
        d, lo, hi, sig = boot_diff(a, b)
        print(f"  {name:<24} diff={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT' if sig else 'indistinguishable'}")

    ys = y[mask]
    d, lo, hi, sig = boot_diff(per_match_loss(ys, blend[mask]),
                               per_match_loss(ys, book))
    print(f"\n  {'Blend vs BOOKMAKER':<24} diff={d:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'market still better' if sig else 'no clear difference'}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 6  What Dixon-Coles gives that Elo cannot")
    print("=" * 78)
    final = DixonColes(xi=best_xi).fit(df[df["season"] <= 2025])
    for h, a in [("Man City", "Liverpool"), ("Arsenal", "Everton"),
                 ("Barcelona", "Real Madrid")]:
        if h in final.index and a in final.index:
            p = final.predict(h, a)
            print(f"\n  {h} vs {a}")
            print(f"    outcome : {p['p_home']:.1%} / {p['p_draw']:.1%} / {p['p_away']:.1%}")
            print(f"    xG      : {p['xg_home']:.2f} - {p['xg_away']:.2f}")
            print(f"    o2.5    : {p['over_2_5']:.1%}      BTTS: {p['btts']:.1%}")
            print(f"    scores  : "
                  + ", ".join(f"{s['score']} {s['prob']:.1%}" for s in p["top_scores"][:5]))


if __name__ == "__main__":
    main()
