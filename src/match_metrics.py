"""
Evaluation metrics for 3-class match outcome prediction.

WHY NOT ACCURACY
----------------
Accuracy asks "was the single most likely outcome correct?" For football that
throws away almost everything we care about. A model saying 40/30/30 and a
model saying 90/5/5 produce the same prediction and the same accuracy, but one
is expressing near-certainty and the other a coin flip. If the 90% pick loses,
that model should be punished far harder.

Accuracy also cannot see draws. Draws are ~25% of matches but are almost never
the single most likely outcome, so a model optimised for accuracy learns to
never predict one - and is rewarded for it.

PROPER SCORING RULES
--------------------
A scoring rule is "proper" when it is minimised only by reporting your true
beliefs. You cannot game it by shading your numbers. Two are used here:

  Log loss  -mean(log p_true)  - the standard. Punishes confident errors
                                 brutally: p=0.01 on the true outcome costs
                                 4.6, while p=0.4 costs 0.92.

  Brier     mean sum (p - y)^2 - bounded and gentler on confident misses.
                                 Useful as a cross-check when a handful of
                                 shock results dominate log loss.

REFERENCE VALUES (3-class football)
-----------------------------------
  1.099  uniform 1/3 guess - ln(3), the "know nothing" ceiling
  ~1.06  always predicting the base rates
  ~1.00  decent model
  ~0.96  bookmaker closing odds  <- the number to beat
"""

import numpy as np
import pandas as pd

EPS = 1e-15
CLASS_NAMES = {0: "Home win", 1: "Draw", 2: "Away win"}


def _clean(probs: np.ndarray) -> np.ndarray:
    """Clip away exact zeros (log(0) is undefined) and renormalise to sum 1."""
    p = np.asarray(probs, dtype=float).copy()
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def log_loss_multi(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    -mean(log(probability assigned to the outcome that actually happened))

    Lower is better. This is the primary metric for everything downstream.
    """
    p = _clean(probs)
    y = np.asarray(y_true, dtype=int)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier_multi(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Mean squared error between the probability vector and a one-hot outcome.
    Range 0 (perfect) to 2 (maximally wrong). Lower is better.
    """
    p = _clean(probs)
    y = np.asarray(y_true, dtype=int)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Kept only for comparison against published numbers - not for tuning."""
    p = _clean(probs)
    return float(np.mean(np.argmax(p, axis=1) == np.asarray(y_true, dtype=int)))


def odds_to_probs(odds_home, odds_draw, odds_away) -> np.ndarray:
    """
    Convert decimal bookmaker odds to probabilities.

    1/odds is the implied probability, but the three sum to slightly more than
    1 - that excess is the bookmaker's margin (the "overround" or vig, usually
    3-8%). Dividing by the sum strips it out proportionally.

    Proportional removal slightly over-corrects heavy favourites; see
    odds_to_probs_power below, which measured 0.0007 log loss better on our
    test seasons and is what the benchmark now uses.
    """
    inv = np.column_stack([
        1.0 / np.asarray(odds_home, dtype=float),
        1.0 / np.asarray(odds_draw, dtype=float),
        1.0 / np.asarray(odds_away, dtype=float),
    ])
    return inv / inv.sum(axis=1, keepdims=True)


def odds_to_probs_power(odds_home, odds_draw, odds_away,
                        lo: float = 0.8, hi: float = 1.5) -> np.ndarray:
    """
    Strip the bookmaker margin with the power method rather than proportionally.

    Dividing every inverse odd by their sum assumes the margin is spread evenly
    across the three outcomes. It is not. Bookmakers load more of it onto
    longshots, because that is where recreational money goes - the
    favourite-longshot bias. The proportional shortcut therefore overstates
    outsiders and understates favourites.

    The power method instead finds the exponent p for which

        sum_i (1 / odds_i) ** p  =  1

    Raising to a power below 1 compresses large inverse odds more than small
    ones, removing margin where it was actually loaded. Solved by bisection:
    the left side decreases monotonically in p, so it cannot fail to converge.

    Getting this right matters because the market is our benchmark. A biased
    benchmark makes every claim measured against it biased too.
    """
    inv = np.column_stack([
        1.0 / np.asarray(odds_home, dtype=float),
        1.0 / np.asarray(odds_draw, dtype=float),
        1.0 / np.asarray(odds_away, dtype=float),
    ])
    out = np.zeros_like(inv)
    for i in range(len(inv)):
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if (inv[i] ** mid).sum() > 1.0:
                a = mid          # still over-round, push the exponent up
            else:
                b = mid
        row = inv[i] ** (0.5 * (a + b))
        out[i] = row / row.sum()
    return out


def calibration_table(y_true, probs, class_idx: int = 0, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability table for one class.

    Bucket predictions by confidence, then compare the mean predicted
    probability against how often the event actually occurred. A calibrated
    model has predicted ~= actual in every row.

    Discrimination and calibration are different properties. A model can rank
    matches perfectly and still be badly calibrated (raw Elo does exactly
    this), and a model can be perfectly calibrated while being useless (always
    predicting the base rate).
    """
    p = _clean(probs)[:, class_idx]
    y = (np.asarray(y_true, dtype=int) == class_idx).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin":       f"{bins[b]:.2f}-{bins[b+1]:.2f}",
            "n":         int(m.sum()),
            "predicted": round(float(p[m].mean()), 3),
            "actual":    round(float(y[m].mean()), 3),
            "gap":       round(float(p[m].mean() - y[m].mean()), 3),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    """
    Average |predicted - actual| across confidence bins, averaged over all
    three classes and weighted by bin population. One number for "how
    trustworthy are these probabilities". Lower is better; under ~0.02 is good.
    """
    total, n = 0.0, len(y_true)
    for c in range(probs.shape[1]):
        tbl = calibration_table(y_true, probs, class_idx=c, n_bins=n_bins)
        if not tbl.empty:
            total += float((tbl["gap"].abs() * tbl["n"]).sum() / n)
    return total / probs.shape[1]


def evaluate(y_true, probs, name: str = "model") -> dict:
    """All headline metrics for one set of predictions."""
    return {
        "model":     name,
        "log_loss":  round(log_loss_multi(y_true, probs), 4),
        "brier":     round(brier_multi(y_true, probs), 4),
        "accuracy":  round(accuracy(y_true, probs), 4),
        "ece":       round(expected_calibration_error(y_true, probs), 4),
        "n":         len(y_true),
    }


def baseline_probs(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """
    Class-prior baseline: predict the training base rates for every match.

    This is the real floor, not the uniform 1/3 guess. Any model that cannot
    beat it has learned nothing whatsoever from its features.
    """
    counts = np.bincount(np.asarray(y_train, dtype=int), minlength=3).astype(float)
    prior = counts / counts.sum()
    return np.tile(prior, (n_rows, 1))


def compare(results: list[dict]) -> pd.DataFrame:
    """Stack evaluate() outputs into a leaderboard, best log loss first."""
    return (
        pd.DataFrame(results)
        .sort_values("log_loss")
        .reset_index(drop=True)
    )
