"""
Elo rating system for football teams.

Elo compresses "how good is this team right now" into a single number that
updates after every match. It originated in chess; the football variant adds
three things chess does not need:

  1. Home advantage   - the home side is handed a rating bonus for the
                        expectation calculation only (not a permanent gain).
  2. Margin of victory - a 5-0 win says more than a 1-0 win. Chess has no
                        equivalent, so we scale the update by goal difference.
  3. Season regression - squads turn over, teams get promoted and relegated.
                        Carrying a rating forward untouched across seasons
                        overstates how much we know about a new-look side.

THE RULE THAT MATTERS MOST
--------------------------
Ratings are recorded as they stood BEFORE kick-off, then updated using the
result. A feature that contains any information from the match it is
predicting - or from any later match - produces a model that looks excellent
in backtesting and is worthless in production. That failure is called
temporal leakage, and it is the single most common mistake in sports ML.
`run()` below is structured so leakage is impossible by construction: the
pre-match snapshot is taken before `update()` is ever called.
"""

import pandas as pd
import numpy as np


DEFAULT_RATING = 1500.0


class EloRatingSystem:
    """
    Parameters
    ----------
    k_factor : float
        Learning rate. How far a rating moves after one surprising result.
        Low  (10-15) -> stable, slow to react, good for long seasons.
        High (40+)   -> reactive, noisy, overreacts to single upsets.
        20 is the usual football compromise and our starting point.

    home_advantage : float
        Rating points granted to the home side when computing the expected
        result. ~65 corresponds to the ~43% home-win rate we measured.
        This is NOT added to the stored rating - only to the expectation.

    season_regression : float
        Fraction of a team's distance from the mean that is retained when a
        new season starts. 0.75 means a team 100 points above average begins
        the next season 75 points above average. Models squad turnover.

    use_margin_multiplier : bool
        Whether to scale updates by goal difference.
    """

    # Defaults below are the validated configuration from src/validate_elo.py:
    # chosen on a held-out validation split (seasons 2022-23), then scored once
    # on test (2024-25). Tuning moved log loss by only 0.0005 - the priors were
    # already close to optimal.
    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 65.0,
        initial_rating: float = DEFAULT_RATING,
        season_regression: float = 0.85,
        use_margin_multiplier: bool = True,
        regression_mode: str = "division",
    ):
        self.k = k_factor
        self.hfa = home_advantage
        self.initial = initial_rating
        self.regression = season_regression
        self.use_margin = use_margin_multiplier
        self.regression_mode = regression_mode

        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []
        # Division each team has played in so far this season, used by
        # division-mode regression and reset at every season boundary.
        self.team_division: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Core maths
    # ------------------------------------------------------------------ #

    def get(self, team: str) -> float:
        """Current rating, initialising an unseen team at the base value."""
        return self.ratings.setdefault(team, self.initial)

    def expected_score(self, rating_home: float, rating_away: float) -> float:
        """
        Probability-like expectation that the home side "wins" the match,
        where a draw counts as half a win.

            E_home = 1 / (1 + 10 ^ ((R_away - R_home - HFA) / 400))

        The 400 is a scaling constant inherited from chess: a 400-point gap
        means the stronger side is expected to score 10x as often. It is a
        convention, not a law - changing it just rescales the ratings.

        Returns a value in (0, 1). Equal ratings with home advantage gives
        roughly 0.59, which lines up with real home-win rates.
        """
        diff = rating_away - rating_home - self.hfa
        return 1.0 / (1.0 + 10 ** (diff / 400.0))

    def margin_multiplier(self, goal_diff: int) -> float:
        """
        Scales the rating update by how emphatic the win was.

        Flat for 1-goal margins, then growing sub-linearly so a 6-0 does not
        move the rating six times as far as a 1-0. The diminishing return is
        deliberate: blowouts often say more about the loser collapsing than
        the winner improving.
        """
        if not self.use_margin:
            return 1.0
        g = abs(goal_diff)
        if g <= 1:
            return 1.0
        if g == 2:
            return 1.5
        return (11.0 + g) / 8.0

    # ------------------------------------------------------------------ #
    # Updating
    # ------------------------------------------------------------------ #

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> dict:
        """
        Apply one match result. Returns the pre-match snapshot so the caller
        can use it as a feature without ever touching post-match state.
        """
        r_home = self.get(home)
        r_away = self.get(away)

        expected_home = self.expected_score(r_home, r_away)

        # Actual score in Elo terms: 1 win / 0.5 draw / 0 loss.
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        mult = self.margin_multiplier(home_goals - away_goals)
        delta = self.k * mult * (actual_home - expected_home)

        # Zero-sum: whatever one side gains, the other loses. This keeps the
        # league-wide average rating constant, so ratings stay comparable
        # across seasons and competitions.
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta

        return {
            "elo_home_pre":  r_home,
            "elo_away_pre":  r_away,
            "elo_diff":      r_home - r_away,
            "elo_expected":  expected_home,
            "elo_delta":     delta,
        }

    def apply_season_regression(self):
        """
        Pull every rating toward a mean at a season boundary.

        Without this, a dominant side keeps a sky-high rating through a
        summer in which it sold its three best players, and a promoted club
        inherits nothing at all. Regression encodes "we are less certain
        about every team than we were in May".

        WHICH MEAN
        ----------
        With a single division the answer is obvious. Once second tiers are
        included it stops being obvious and the obvious answer is wrong.

        Divisions never play each other, so the only thing setting the level
        gap between two rating pools is teams moving between them. Regressing
        toward the mean of everyone overrides that every summer - second-tier
        sides get dragged up, top-tier sides pulled down - and ten seasons of
        it leaves promoted teams entering the top flight rated ABOVE its
        average, which is exactly backwards.

        Division mode regresses each team toward the mean of the division it
        has just played in. Regressing a group toward its own mean leaves that
        mean unchanged, so the gap between tiers survives the off-season.
        """
        if not self.ratings:
            return

        if self.regression_mode == "division" and self.team_division:
            groups: dict[str, list[str]] = {}
            for team, div in self.team_division.items():
                groups.setdefault(div, []).append(team)
            for teams in groups.values():
                mean = float(np.mean([self.ratings[t] for t in teams]))
                for t in teams:
                    self.ratings[t] = mean + (self.ratings[t] - mean) * self.regression
        else:
            mean = float(np.mean(list(self.ratings.values())))
            for team in self.ratings:
                self.ratings[team] = mean + (self.ratings[team] - mean) * self.regression

        self.team_division = {}

    # ------------------------------------------------------------------ #
    # Batch pass over a fixture list
    # ------------------------------------------------------------------ #

    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Walk a chronologically sorted fixture list, attaching pre-match Elo
        columns to every row.

        Expects: utc_date, season, home_team, away_team, home_goals, away_goals

        Returns a copy of `matches` with elo_home_pre, elo_away_pre, elo_diff,
        elo_expected and elo_delta added.
        """
        required = {"utc_date", "season", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"matches is missing required columns: {sorted(missing)}")

        df = matches.sort_values("utc_date").reset_index(drop=True)

        rows = []
        current_season = None
        has_comp = "competition" in df.columns

        for _, m in df.iterrows():
            # Season rollover - regress before this season's first match is scored.
            if current_season is not None and m["season"] != current_season:
                self.apply_season_regression()
            current_season = m["season"]

            if has_comp:
                self.team_division[m["home_team"]] = m["competition"]
                self.team_division[m["away_team"]] = m["competition"]

            # Snapshot first, update second. This ordering is the guarantee
            # against temporal leakage - do not reorder these two lines.
            snap = self.update(
                m["home_team"], m["away_team"],
                int(m["home_goals"]), int(m["away_goals"]),
            )
            rows.append(snap)

            self.history.append({
                "utc_date": m["utc_date"],
                "season":   m["season"],
                "home":     m["home_team"],
                "away":     m["away_team"],
                **snap,
            })

        return pd.concat([df, pd.DataFrame(rows)], axis=1)

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def table(self, top: int = 20) -> pd.DataFrame:
        """Current ratings, strongest first - a sanity check on the system."""
        return (
            pd.DataFrame(
                [{"team": t, "elo": round(r, 1)} for t, r in self.ratings.items()]
            )
            .sort_values("elo", ascending=False)
            .head(top)
            .reset_index(drop=True)
        )


# ====================================================================== #
# Turning Elo into real probabilities
# ====================================================================== #
#
# THE PROBLEM
# -----------
# Elo's expected score is not a win probability. It was designed for chess,
# where a draw scores half a point, so:
#
#       E_home = P(home win) + 0.5 * P(draw)
#
# Read E_home as P(home win) and you overshoot by roughly half the draw rate -
# about 12 points in football. That is the systematic bias visible in the raw
# calibration table, and it is a definitional mismatch rather than a bug.
#
# THE FIX
# -------
# E_home gives us one equation and probabilities sum to one gives a second.
# Three unknowns, so we need one more relationship - a model of P(draw).
#
# Draws peak when teams are evenly matched and fall away as the gap widens,
# which a bell curve over the effective rating gap captures well:
#
#       P(draw) = d_max * exp(-(delta / tau)^2)
#
# d_max is the draw rate between identical sides, tau sets how fast draws
# decay as one team pulls ahead. Both are fitted on training data by
# minimising log loss - never on the test set.

def elo_to_probabilities(
    elo_diff: np.ndarray,
    home_advantage: float = 65.0,
    d_max: float = 0.28,
    tau: float = 250.0,
) -> np.ndarray:
    """
    Map pre-match Elo differences to (P_home, P_draw, P_away).

    Returns an (n, 3) array of rows summing to 1, matching the project's
    class order: 0 = home win, 1 = draw, 2 = away win.
    """
    delta = np.asarray(elo_diff, dtype=float) + home_advantage

    expected_home = 1.0 / (1.0 + 10 ** (-delta / 400.0))
    p_draw = d_max * np.exp(-((delta / tau) ** 2))

    p_home = expected_home - 0.5 * p_draw
    p_away = 1.0 - expected_home - 0.5 * p_draw

    # For a lopsided tie the algebra can push a tail slightly negative.
    # Clip to a small floor and renormalise so every row is a valid
    # distribution - a genuine 0% would make log loss infinite anyway.
    probs = np.column_stack([p_home, p_draw, p_away])
    probs = np.clip(probs, 1e-6, None)
    return probs / probs.sum(axis=1, keepdims=True)


def fit_draw_model(
    elo_diff: np.ndarray,
    y_true: np.ndarray,
    home_advantage: float = 65.0,
) -> tuple[float, float]:
    """
    Fit (d_max, tau) by minimising log loss on the supplied data.

    Uses a coarse grid rather than gradient descent: the surface is smooth and
    only two-dimensional, so a grid is fast, deterministic, and cannot land in
    a bad local minimum or fail to converge.

    MUST be called on training data only. Fitting these on the test set is
    leakage - the scores would flatter a model that had already peeked.
    """
    from match_metrics import log_loss_multi

    best, best_loss = (0.28, 250.0), float("inf")

    for d_max in np.arange(0.18, 0.38, 0.01):
        for tau in np.arange(100.0, 600.0, 25.0):
            probs = elo_to_probabilities(elo_diff, home_advantage, d_max, tau)
            loss = log_loss_multi(y_true, probs)
            if loss < best_loss:
                best, best_loss = (float(d_max), float(tau)), loss

    return best


def evaluate_elo_alone(df: pd.DataFrame) -> dict:
    """
    How much predictive power does elo_diff carry on its own?

    This is a single-feature baseline, and it is worth measuring before
    building anything more complex. If a 25-feature gradient-boosted model
    cannot clear this by a useful margin, the extra features are not paying
    for the complexity they add.

    Uses a naive threshold rule - predict whichever side Elo favours, with a
    dead zone in the middle for draws - so it measures the signal in the
    feature rather than the skill of a classifier sitting on top of it.
    """
    sub = df.dropna(subset=["elo_diff", "result"]).copy()

    # Dead zone chosen so the predicted draw rate roughly matches the real one.
    DRAW_BAND = 25.0

    def predict(d):
        if d > DRAW_BAND:
            return 0     # home win
        if d < -DRAW_BAND:
            return 2     # away win
        return 1         # draw

    sub["pred"] = sub["elo_diff"].apply(predict)

    accuracy = float((sub["pred"] == sub["result"]).mean())
    baseline = float((sub["result"] == 0).mean())

    # Does the expectation line up with reality? Bucket matches by predicted
    # strength and compare against what actually happened. A well-behaved
    # rating system produces a monotonically increasing curve here.
    sub["bucket"] = pd.cut(sub["elo_expected"], bins=[0, .35, .45, .55, .65, 1.0])
    calib = (
        sub.groupby("bucket", observed=True)
           .agg(n=("result", "size"),
                predicted=("elo_expected", "mean"),
                actual_home_win=("result", lambda s: (s == 0).mean()))
           .round(3)
    )

    return {
        "accuracy":       round(accuracy, 4),
        "baseline":       round(baseline, 4),
        "lift":           round(accuracy - baseline, 4),
        "n_matches":      len(sub),
        "calibration":    calib,
    }
