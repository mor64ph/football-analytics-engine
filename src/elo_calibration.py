"""
Turning Elo ratings into well-calibrated probabilities, per league.

WHAT THIS FIXES
---------------
`elo.elo_to_probabilities` maps a rating difference to (home, draw, away)
using three numbers: a home advantage, and the two parameters of the draw
curve. Until now all three were global - one set shared by every division.

Leagues genuinely differ. Serie A draws more often than the Premier League.
Home advantage is not the same in the Bundesliga as in La Liga, and second
tiers behave differently again. A single global fit splits the difference and
is therefore slightly wrong everywhere.

THE OVERFITTING TRAP, AND THE FIX
---------------------------------
Fitting three free parameters independently per league is the obvious move
and the wrong one. A division with 300 matches in the training window will
produce parameters that fit its noise, and those will not transfer.

So each league's parameters are SHRUNK toward the global fit:

    weight = n / (n + PRIOR_STRENGTH)
    param  = weight * league_estimate + (1 - weight) * global_estimate

A league with many matches is trusted and keeps its own values. A thin one is
pulled most of the way back to the pooled estimate. PRIOR_STRENGTH sets how
much evidence is needed before a league is allowed to disagree with the pool;
at 1500 a full Premier League season is worth roughly a quarter of a vote.

This is empirical-Bayes shrinkage, and it is what makes per-group fitting
safe rather than a slow way to overfit.
"""

import numpy as np
import pandas as pd

from elo import elo_to_probabilities
from match_metrics import log_loss_multi

# Matches of evidence required before a league's own estimate outweighs the
# pooled one. Tuned on validation; the result is insensitive in 800-3000.
PRIOR_STRENGTH = 1500.0

HFA_GRID = np.arange(0.0, 145.0, 5.0)
DMAX_GRID = np.arange(0.16, 0.40, 0.01)
TAU_GRID = np.arange(100.0, 650.0, 25.0)


def fit_joint(elo_diff, y_true, hfa_grid=None, dmax_grid=None, tau_grid=None):
    """
    Fit (home_advantage, d_max, tau) together by minimising log loss.

    The original code tuned home advantage as an Elo hyperparameter and then
    fitted the draw curve on top of whatever it produced. That is a two-stage
    search over parameters that interact: widening the draw curve and shrinking
    home advantage both move probability mass toward the middle, so the best
    value of one depends on the other. Fitting them on one grid finds
    combinations the staged search cannot reach.

    Grid rather than gradient descent: the surface is smooth, only three
    dimensional, and a grid is deterministic and cannot fail to converge.

    MUST be called on training data only.
    """
    elo_diff = np.asarray(elo_diff, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    hfa_grid = HFA_GRID if hfa_grid is None else hfa_grid
    dmax_grid = DMAX_GRID if dmax_grid is None else dmax_grid
    tau_grid = TAU_GRID if tau_grid is None else tau_grid

    best, best_loss = (65.0, 0.28, 250.0), float("inf")
    for hfa in hfa_grid:
        # The home-advantage shift and the expected-score curve depend only on
        # hfa, so hoist them out of the inner two loops.
        delta = elo_diff + hfa
        expected_home = 1.0 / (1.0 + 10 ** (-delta / 400.0))
        for tau in tau_grid:
            bell = np.exp(-((delta / tau) ** 2))
            for d_max in dmax_grid:
                p_draw = d_max * bell
                probs = np.column_stack([
                    expected_home - 0.5 * p_draw,
                    p_draw,
                    1.0 - expected_home - 0.5 * p_draw,
                ])
                probs = np.clip(probs, 1e-6, None)
                probs /= probs.sum(axis=1, keepdims=True)
                loss = log_loss_multi(y_true, probs)
                if loss < best_loss:
                    best, best_loss = (float(hfa), float(d_max), float(tau)), loss

    return best


class LeagueCalibrator:
    """
    Per-league Elo -> probability mapping with shrinkage to a global fit.

    Usage:
        cal = LeagueCalibrator().fit(train["elo_diff"], train["result"],
                                     train["competition"])
        probs = cal.predict(test["elo_diff"], test["competition"])
    """

    def __init__(self, prior_strength: float = PRIOR_STRENGTH):
        self.prior_strength = prior_strength
        self.global_params: tuple[float, float, float] = (65.0, 0.28, 250.0)
        self.params: dict[str, tuple[float, float, float]] = {}
        self.counts: dict[str, int] = {}

    def fit(self, elo_diff, y_true, groups) -> "LeagueCalibrator":
        elo_diff = np.asarray(elo_diff, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        groups = np.asarray(groups)

        self.global_params = fit_joint(elo_diff, y_true)
        g_hfa, g_dmax, g_tau = self.global_params

        for grp in np.unique(groups):
            m = groups == grp
            n = int(m.sum())
            self.counts[grp] = n
            if n < 200:
                # Too thin to say anything of its own.
                self.params[grp] = self.global_params
                continue

            l_hfa, l_dmax, l_tau = fit_joint(elo_diff[m], y_true[m])
            w = n / (n + self.prior_strength)
            self.params[grp] = (
                w * l_hfa + (1 - w) * g_hfa,
                w * l_dmax + (1 - w) * g_dmax,
                w * l_tau + (1 - w) * g_tau,
            )
        return self

    def predict(self, elo_diff, groups) -> np.ndarray:
        elo_diff = np.asarray(elo_diff, dtype=float)
        groups = np.asarray(groups)
        out = np.zeros((len(elo_diff), 3))
        for grp in np.unique(groups):
            m = groups == grp
            hfa, d_max, tau = self.params.get(grp, self.global_params)
            out[m] = elo_to_probabilities(elo_diff[m], hfa, d_max, tau)
        return out

    def table(self) -> pd.DataFrame:
        """Fitted parameters per league - read this to sanity-check the fit."""
        g_hfa, g_dmax, g_tau = self.global_params
        rows = [{"group": "GLOBAL", "n": sum(self.counts.values()),
                 "hfa": round(g_hfa, 1), "d_max": round(g_dmax, 3),
                 "tau": round(g_tau, 0), "shrinkage_w": 1.0}]
        for grp, (hfa, d_max, tau) in sorted(self.params.items()):
            n = self.counts.get(grp, 0)
            rows.append({
                "group": grp, "n": n,
                "hfa": round(hfa, 1), "d_max": round(d_max, 3),
                "tau": round(tau, 0),
                "shrinkage_w": round(n / (n + self.prior_strength), 2),
            })
        return pd.DataFrame(rows)


def blend(probs_a: np.ndarray, probs_b: np.ndarray, w: float) -> np.ndarray:
    """
    Weighted average of two probability sets, renormalised.

    Averaging probabilities (a 'linear opinion pool') rather than averaging
    log-odds keeps the result inside the simplex and is more forgiving when
    one source is confidently wrong - which is the case we care about.
    """
    out = w * np.asarray(probs_a) + (1 - w) * np.asarray(probs_b)
    out = np.clip(out, 1e-6, None)
    return out / out.sum(axis=1, keepdims=True)


def best_blend_weight(probs_a, probs_b, y_true, grid=None) -> tuple[float, float]:
    """Weight on `probs_a` that minimises log loss. Fit on validation only."""
    grid = np.arange(0.0, 1.001, 0.02) if grid is None else grid
    best_w, best_loss = 0.0, float("inf")
    for w in grid:
        loss = log_loss_multi(y_true, blend(probs_a, probs_b, w))
        if loss < best_loss:
            best_w, best_loss = float(w), float(loss)
    return best_w, best_loss
