"""
Dixon-Coles model (Dixon & Coles, 1997).

A different shape of model to everything else here. Instead of classifying
W/D/L directly, it models the GOALS each side scores, then derives outcome
probabilities from the resulting score distribution.

    lambda (home goals) = exp(attack_home - defence_away + home_adv)
    mu     (away goals) = exp(attack_away - defence_home)

Every team carries an attack and a defence rating; home advantage is global.
Goals are Poisson, so the probability of an exact scoreline is:

    P(X=x, Y=y) = tau(x, y) * Poisson(x; lambda) * Poisson(y; mu)

THE tau CORRECTION - the paper's actual contribution
----------------------------------------------------
Treating the two teams' goals as independent Poisson draws underestimates low
scores, particularly 0-0 and 1-1. Real matches are not independent: at 0-0
late on, both sides often settle; at 1-1 the game frequently closes down.
Dixon and Coles added an adjustment applied only to the four scorelines where
the dependence actually shows up:

    tau(0,0) = 1 - lambda*mu*rho      tau(0,1) = 1 + lambda*rho
    tau(1,0) = 1 + mu*rho             tau(1,1) = 1 - rho
    tau(x,y) = 1                      everywhere else

rho is fitted. Negative rho inflates draws, which is exactly the direction
football needs.

TIME DECAY
----------
Squads change. A match from five years ago should not count as much as one
from last month, so each match is weighted exp(-xi * days_ago). xi is tuned
on validation; a half-life of roughly one season is typical.

WHY BUILD THIS AT ALL
---------------------
The significance testing already showed that outcome accuracy is capped by
the information in historical results - a 2-parameter Elo model matched a
46-feature ensemble. Dixon-Coles is unlikely to break that ceiling.

What it gives instead is a full distribution over SCORELINES. From one fit
you get exact-score probabilities, over/under any threshold, both-teams-to-
score, clean sheets and expected goals per side. Elo gives three numbers;
this gives a joint distribution. That is product value rather than accuracy
value, and it is worth being explicit about which one you are buying.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

MAX_GOALS = 10          # scoreline grid; P(11+ goals) is negligible
DEFAULT_XI = 0.0019     # time decay, ~1 year half-life: ln(2)/365
CENTRE_PENALTY = 1e-4   # keeps attack/defence from drifting (see below)


# ===================================================================== #
# Likelihood
# ===================================================================== #

def _tau(x, y, lam, mu, rho):
    """
    Dixon-Coles low-score correction.

    Vectorised over arrays of scorelines. Only the four cells where both
    sides scored 0 or 1 are adjusted; everything else passes through as 1.
    """
    t = np.ones_like(lam, dtype=float)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho

    # An extreme rho can push tau negative, which would make the log-likelihood
    # undefined. Flooring it keeps the optimiser inside a valid region instead
    # of failing outright.
    return np.clip(t, 1e-10, None)


def _nll_and_grad(params, home_idx, away_idx, hg, ag, weights,
                  log_fact_h, log_fact_a, n_teams):
    """
    Weighted negative log-likelihood AND its analytic gradient.

    The gradient is why this function exists. Without it, L-BFGS-B estimates
    derivatives by finite differences, which costs one extra likelihood
    evaluation per parameter - 199 of them here - for every single step. With
    199 teams that turned a fit into ~35 seconds. Supplying the gradient in
    closed form makes each fit roughly two orders of magnitude cheaper, which
    matters because walk-forward validation refits dozens of times.

    Derivation. With lambda = exp(a_h - d_a + gamma) and mu = exp(a_a - d_h),
    the Poisson part of the per-match log-likelihood differentiates to

        d/d(lambda) [x*log(lambda) - lambda] = x/lambda - 1

    and since d(lambda)/d(a_h) = lambda, the chain rule collapses neatly:

        dL/d(a_h) = (x - lambda)          dL/d(d_a) = -(x - lambda)
        dL/d(a_a) = (y - mu)              dL/d(d_h) = -(y - mu)

    The tau correction contributes only on the four low-score cells, handled
    separately below. Per-match gradients are scattered back onto team
    parameters with bincount.
    """
    attack   = params[:n_teams]
    defence  = params[n_teams:2 * n_teams]
    home_adv = params[-2]
    rho      = params[-1]

    log_lam = attack[home_idx] - defence[away_idx] + home_adv
    log_mu  = attack[away_idx] - defence[home_idx]
    lam = np.clip(np.exp(np.clip(log_lam, -20, 4)), 1e-10, 30.0)
    mu  = np.clip(np.exp(np.clip(log_mu,  -20, 4)), 1e-10, 30.0)

    # Poisson log-pmf computed directly. scipy.stats.poisson.logpmf performs
    # extensive validation on every call; the log-factorials here depend only
    # on the observed goals, so they are precomputed once by the caller.
    ll_pois = (hg * np.log(lam) - lam - log_fact_h
               + ag * np.log(mu) - mu - log_fact_a)

    tau = _tau(hg, ag, lam, mu, rho)
    ll = ll_pois + np.log(tau)

    # --- gradients of log tau ---------------------------------------- #
    dlt_dlam = np.zeros_like(lam)
    dlt_dmu  = np.zeros_like(mu)
    dlt_drho = np.zeros_like(lam)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    t00 = tau[m00]
    dlt_dlam[m00] = -mu[m00] * rho / t00
    dlt_dmu[m00]  = -lam[m00] * rho / t00
    dlt_drho[m00] = -lam[m00] * mu[m00] / t00

    t01 = tau[m01]
    dlt_dlam[m01] = rho / t01
    dlt_drho[m01] = lam[m01] / t01

    t10 = tau[m10]
    dlt_dmu[m10]  = rho / t10
    dlt_drho[m10] = mu[m10] / t10

    dlt_drho[m11] = -1.0 / tau[m11]

    # --- per-match gradients w.r.t. the rate parameters --------------- #
    # (x/lam - 1 + dlogtau/dlam) * dlam/d(param), with dlam/d(a_h) = lam
    g_lam = (hg - lam + dlt_dlam * lam) * weights
    g_mu  = (ag - mu  + dlt_dmu  * mu)  * weights

    grad = np.zeros_like(params)
    # attack: home side gains from g_lam, away side from g_mu
    grad[:n_teams] -= np.bincount(home_idx, g_lam, minlength=n_teams)
    grad[:n_teams] -= np.bincount(away_idx, g_mu,  minlength=n_teams)
    # defence: signs flip (conceding is the mirror of scoring)
    grad[n_teams:2 * n_teams] += np.bincount(away_idx, g_lam, minlength=n_teams)
    grad[n_teams:2 * n_teams] += np.bincount(home_idx, g_mu,  minlength=n_teams)
    grad[-2] = -np.sum(g_lam)
    grad[-1] = -np.sum(dlt_drho * weights)

    # Attack and defence are identified only up to a common shift, so the
    # optimiser could drift arbitrarily far without changing any prediction.
    # A small centring penalty pins the solution down without distorting it.
    mean_a, mean_d = attack.mean(), defence.mean()
    penalty = CENTRE_PENALTY * (mean_a ** 2 + mean_d ** 2)
    grad[:n_teams] += 2 * CENTRE_PENALTY * mean_a / n_teams
    grad[n_teams:2 * n_teams] += 2 * CENTRE_PENALTY * mean_d / n_teams

    return -np.sum(weights * ll) + penalty, grad


# ===================================================================== #
# Model
# ===================================================================== #

class DixonColes:
    def __init__(self, xi: float = DEFAULT_XI, max_goals: int = MAX_GOALS):
        self.xi = xi
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.index: dict[str, int] = {}
        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self.fitted = False

    # ------------------------------------------------------------------ #

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
            verbose: bool = False) -> "DixonColes":
        """
        Maximum-likelihood fit.

        `as_of` anchors the time decay and, critically, drops any match on or
        after that date. When walking forward through a season this is what
        keeps the fit honest - the model never sees a result it is about to
        predict.
        """
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        if as_of is not None:
            df = df[df["date"] < as_of]
        if df.empty:
            raise ValueError("no matches available to fit on")

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        home_idx = df["home_team"].map(self.index).values
        away_idx = df["away_team"].map(self.index).values
        hg = df["home_goals"].values.astype(int)
        ag = df["away_goals"].values.astype(int)

        anchor = as_of if as_of is not None else df["date"].max()
        days_ago = (anchor - df["date"]).dt.days.values.astype(float)
        weights = np.exp(-self.xi * np.clip(days_ago, 0, None))

        # Log-factorials depend only on the observed goals, so they are
        # constant across every optimiser iteration - hoist them out.
        log_fact_h = gammaln(hg + 1.0)
        log_fact_a = gammaln(ag + 1.0)

        # Sensible starting point: everyone average, mild home advantage,
        # no low-score correlation.
        x0 = np.concatenate([
            np.zeros(n),        # attack
            np.zeros(n),        # defence
            [0.25],             # home advantage
            [0.0],              # rho
        ])
        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-1, 1), (-0.4, 0.4)]

        res = minimize(
            _nll_and_grad, x0,
            args=(home_idx, away_idx, hg, ag, weights,
                  log_fact_h, log_fact_a, n),
            method="L-BFGS-B", bounds=bounds,
            jac=True,                       # gradient supplied analytically
            options={"maxiter": 500},
        )

        self.attack = res.x[:n]
        self.defence = res.x[n:2 * n]
        self.home_adv = float(res.x[-2])
        self.rho = float(res.x[-1])
        self.fitted = True
        self._nll = float(res.fun)
        self._n_matches = len(df)
        return self

    # ------------------------------------------------------------------ #

    def rates(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for each side. Unknown teams fall back to average."""
        hi = self.index.get(home)
        ai = self.index.get(away)
        a_h = self.attack[hi] if hi is not None else 0.0
        d_h = self.defence[hi] if hi is not None else 0.0
        a_a = self.attack[ai] if ai is not None else 0.0
        d_a = self.defence[ai] if ai is not None else 0.0

        lam = float(np.exp(a_h - d_a + self.home_adv))
        mu = float(np.exp(a_a - d_h))
        return float(np.clip(lam, 1e-6, 30)), float(np.clip(mu, 1e-6, 30))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """
        Joint probability over scorelines, indexed [home_goals, away_goals].

        Everything else the model reports is a sum over cells of this grid.
        """
        lam, mu = self.rates(home, away)
        g = np.arange(self.max_goals + 1)

        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))

        # Apply the low-score correction to the four affected cells.
        m[0, 0] *= 1.0 - lam * mu * self.rho
        m[0, 1] *= 1.0 + lam * self.rho
        m[1, 0] *= 1.0 + mu * self.rho
        m[1, 1] *= 1.0 - self.rho

        m = np.clip(m, 0, None)
        return m / m.sum()      # renormalise: the grid truncates at max_goals

    # ------------------------------------------------------------------ #

    def predict(self, home: str, away: str) -> dict:
        """
        Full prediction for one fixture.

        The outcome probabilities are the headline, but the scoreline grid is
        the reason this model exists - over/under, both-teams-to-score and
        exact scores all fall out of the same fit.
        """
        m = self.score_matrix(home, away)
        lam, mu = self.rates(home, away)

        p_home = float(np.tril(m, -1).sum())    # home goals > away goals
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())

        total = np.add.outer(np.arange(self.max_goals + 1),
                             np.arange(self.max_goals + 1))
        over25 = float(m[total > 2].sum())
        btts = float(m[1:, 1:].sum())

        flat = [((h, a), float(m[h, a]))
                for h in range(self.max_goals + 1)
                for a in range(self.max_goals + 1)]
        flat.sort(key=lambda x: -x[1])

        return {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "xg_home": lam, "xg_away": mu,
            "over_2_5": over25, "under_2_5": 1.0 - over25,
            "btts": btts,
            "top_scores": [
                {"score": f"{h}-{a}", "prob": round(p, 4)}
                for (h, a), p in flat[:6]
            ],
        }

    def predict_frame(self, matches: pd.DataFrame) -> np.ndarray:
        """(n, 3) outcome probabilities for a fixture list, for scoring."""
        out = np.zeros((len(matches), 3))
        for i, r in enumerate(matches.itertuples(index=False)):
            p = self.predict(r.home_team, r.away_team)
            out[i] = [p["p_home"], p["p_draw"], p["p_away"]]
        return out

    # ------------------------------------------------------------------ #

    def table(self, top: int = 20) -> pd.DataFrame:
        """Fitted attack and defence ratings - a sanity check on the fit."""
        df = pd.DataFrame({
            "team": self.teams,
            "attack": np.round(self.attack, 3),
            "defence": np.round(self.defence, 3),
        })
        df["overall"] = (df["attack"] + df["defence"]).round(3)
        return df.sort_values("overall", ascending=False).head(top).reset_index(drop=True)


# ===================================================================== #
# Walk-forward evaluation
# ===================================================================== #

def walk_forward(
    matches: pd.DataFrame,
    test_mask: pd.Series,
    xi: float = DEFAULT_XI,
    refit_days: int = 30,
    verbose: bool = True,
) -> np.ndarray:
    """
    Predict a held-out period, refitting periodically on everything prior.

    A single fit over the whole training set and then a static prediction for
    two seasons would let team ratings go stale - and would also quietly use
    the earliest test matches to inform none of the later ones, which is the
    opposite of how the model would run live. Refitting every `refit_days`
    mirrors production: retrain on history, predict the next block, repeat.

    Cost scales with 1/refit_days, so this is the main speed/realism dial.
    """
    df = matches.sort_values("date").reset_index(drop=True)
    test = df[test_mask.values].copy()

    preds = np.zeros((len(test), 3))
    model = None
    last_fit = None
    n_fits = 0

    for i, row in enumerate(test.itertuples(index=False)):
        if last_fit is None or (row.date - last_fit).days >= refit_days:
            model = DixonColes(xi=xi).fit(df, as_of=row.date)
            last_fit = row.date
            n_fits += 1
            if verbose:
                print(f"    refit {n_fits:>2} @ {str(row.date)[:10]}  "
                      f"({model._n_matches:,} matches, "
                      f"rho={model.rho:+.3f}, hfa={model.home_adv:.3f})")

        p = model.predict(row.home_team, row.away_team)
        preds[i] = [p["p_home"], p["p_draw"], p["p_away"]]

    if verbose:
        print(f"    {n_fits} fits for {len(test):,} matches")
    return preds
