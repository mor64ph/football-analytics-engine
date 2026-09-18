"""
Does our model add anything the market does not already know?

The optimisation run answered this with a linear blend and got a weight of
exactly zero - averaging our probabilities into the market's made the forecast
worse at every weight tested. That is a strong claim, so it is worth checking
whether the answer is real or an artefact of how the two were combined.

Linear pooling averages probabilities. It is the obvious method and it has a
known weakness: it is dragged toward whichever source is less confident, so a
well-sharpened forecast blended with a blunter one loses sharpness even when
the blunt one carries some independent signal.

Logarithmic pooling multiplies instead:

    p  proportional to  p_model^w  *  p_market^(1-w)

This is a weighted geometric mean, equivalent to averaging log-odds. It
preserves sharpness and is the standard choice when combining two calibrated
forecasts. If our model holds any information the market lacks, this is the
method most likely to find it.

A third option is stacking: fit a small model on both sets of log-odds and let
it learn the weighting, including any systematic bias correction. Fitted on
validation, scored on test.

Run:  python src/blend_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from elo_calibration import LeagueCalibrator, blend
from match_metrics import log_loss_multi, _clean
from optimise_accuracy import (
    elo_diffs, encode, odds_to_probs_power, per_match_loss, boot,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, ODDS, FEATS,
)

pd.set_option("display.width", 240)


def log_pool(a, b, w):
    """Weighted geometric mean of two probability sets, renormalised."""
    a = np.clip(np.asarray(a, float), 1e-9, None)
    b = np.clip(np.asarray(b, float), 1e-9, None)
    out = np.exp(w * np.log(a) + (1 - w) * np.log(b))
    return out / out.sum(axis=1, keepdims=True)


def best_weight(fn, a, b, y):
    best_w, best_l = 0.0, float("inf")
    curve = []
    for w in np.arange(0.0, 1.001, 0.05):
        l = log_loss_multi(y, fn(a, b, float(w)))
        curve.append((float(w), l))
        if l < best_l:
            best_w, best_l = float(w), l
    return best_w, best_l, curve


def main():
    df = pd.read_csv(FEATS, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    if "tier" not in df.columns:
        df["tier"] = 1
    df = df.sort_values("date").reset_index(drop=True)

    t1 = (df["tier"] == 1).to_numpy()
    is_tr = df["season"].isin(TRAIN_SEASONS).to_numpy()
    is_va = df["season"].isin(VAL_SEASONS).to_numpy()
    is_te = df["season"].isin(TEST_SEASONS).to_numpy()

    # Rebuild the shipped model exactly as optimise_accuracy selected it.
    d = elo_diffs(*encode(df), k=16.0, hfa=65.0, regression=0.85,
                  use_margin=True, regression_mode="division")

    fit_mask = is_tr & t1 & ~df["season"].isin([2019, 2020]).to_numpy()
    cal = LeagueCalibrator().fit(
        d[fit_mask], df.loc[fit_mask, "result"].to_numpy(int),
        df.loc[fit_mask, "competition"].to_numpy())

    def slice_for(mask):
        have = df.loc[mask, ODDS].notna().all(axis=1).to_numpy()
        y = df.loc[mask, "result"].to_numpy(int)[have]
        model = cal.predict(d[mask], df.loc[mask, "competition"].to_numpy())[have]
        o = [df.loc[mask, c].to_numpy()[have] for c in ODDS]
        return y, model, odds_to_probs_power(*o)

    y_va, m_va, b_va = slice_for(is_va & t1)
    y_te, m_te, b_te = slice_for(is_te & t1)

    print(f"validation: {len(y_va):,} matches with closing odds")
    print(f"test      : {len(y_te):,}")
    print(f"\n  VAL  model {log_loss_multi(y_va, m_va):.4f}   "
          f"market {log_loss_multi(y_va, b_va):.4f}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("WEIGHT SEARCH ON VALIDATION")
    print("=" * 74)
    lin_w, lin_l, lin_c = best_weight(blend, m_va, b_va, y_va)
    log_w, log_l, log_c = best_weight(log_pool, m_va, b_va, y_va)

    print("   w_model   linear   logarithmic")
    for (w, a), (_, b) in zip(lin_c, log_c):
        mark = ""
        if abs(w - lin_w) < 1e-9:
            mark += "  <- best linear"
        if abs(w - log_w) < 1e-9:
            mark += "  <- best log"
        print(f"    {w:4.2f}    {a:.4f}   {b:.4f}{mark}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("STACKING — let a model learn the combination")
    print("=" * 74)
    print("  Features are the log-odds of each source against the home-win")
    print("  baseline. Fitted on validation, so test remains untouched.\n")

    def logit_feats(model, book):
        lm = np.log(np.clip(model, 1e-9, None))
        lb = np.log(np.clip(book, 1e-9, None))
        return np.column_stack([lm, lb])

    try:
        from sklearn.linear_model import LogisticRegression
        stack = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        stack.fit(logit_feats(m_va, b_va), y_va)
        st_va = stack.predict_proba(logit_feats(m_va, b_va))
        st_te = stack.predict_proba(logit_feats(m_te, b_te))
        print(f"  VAL (in-sample, optimistic) : {log_loss_multi(y_va, st_va):.4f}")
        has_stack = True
    except Exception as e:
        print(f"  skipped: {e}")
        st_te, has_stack = None, False

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("TEST")
    print("=" * 74)
    rows = [
        ("Our model alone", m_te),
        ("Market alone", b_te),
        (f"Linear blend (w={lin_w:.2f})", blend(m_te, b_te, lin_w)),
        (f"Log pool (w={log_w:.2f})", log_pool(m_te, b_te, log_w)),
    ]
    if has_stack:
        rows.append(("Stacked (fitted on val)", st_te))

    print(f"  {'model':<28} {'log_loss':>9}  {'vs market':>10}")
    book_loss = per_match_loss(y_te, b_te)
    for name, p in rows:
        l = log_loss_multi(y_te, p)
        print(f"  {name:<28} {l:>9.4f}  {l - book_loss.mean():>+10.4f}")

    print("\n  Paired bootstrap against the market:")
    for name, p in rows:
        if name == "Market alone":
            continue
        dm, lo, hi, sig = boot(per_match_loss(y_te, p), book_loss)
        verdict = ("BEATS MARKET" if sig and dm < 0
                   else "worse than market" if sig else "indistinguishable")
        print(f"    {name:<28} diff={dm:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    print("\n" + "=" * 74)
    best = min(rows, key=lambda r: log_loss_multi(y_te, r[1]))
    print(f"  Best on test: {best[0]}  ({log_loss_multi(y_te, best[1]):.4f})")
    if lin_w == 0.0 and log_w == 0.0:
        print("\n  Both pooling methods put zero weight on our model. The market")
        print("  already contains everything it knows - which is what you would")
        print("  expect, since bookmakers run comparable models and additionally")
        print("  price in team news our features cannot see.")


if __name__ == "__main__":
    main()
