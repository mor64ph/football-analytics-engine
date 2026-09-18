"""
The prediction service the API talks to.

Two models, each doing the job it is actually good at.

The evaluation work settled which is which. Elo with per-league calibration
won on outcome probabilities. Dixon-Coles lost that comparison badly - 1.1369
against 0.9950 log loss - but it is the only model here that produces a joint
distribution over scorelines, and every derived market (over/under, both teams
to score, correct score) falls out of that grid.

RECONCILING THE TWO
-------------------
Running them side by side naively produces a visible contradiction: the
headline says 55% home win while the listed scorelines add up to 48%. Users
notice that immediately and it destroys trust in both numbers.

The fix is to keep Dixon-Coles' shape and Elo's totals. Every cell of the
scoreline grid belongs to exactly one outcome, so the grid can be split into
three blocks - home wins below the diagonal, draws on it, away wins above -
and each block rescaled to hit Elo's probability for that outcome:

    cell *= P_elo(outcome of that cell) / P_dc(outcome of that cell)

Within a block nothing changes relative to anything else, so Dixon-Coles
still decides whether a home win is more likely to be 2-0 or 3-1. Across
blocks the totals are now exactly Elo's. We are using each model only for the
thing it demonstrably does better: Dixon-Coles for the conditional shape of a
scoreline, Elo for the marginal chance of each result.

Expected goals are reported from the raw Dixon-Coles fit, unscaled, because
they describe the scoring process rather than the outcome.
"""

import os
import pickle

import numpy as np
import pandas as pd

from dixon_coles import DixonColes
from elo import EloRatingSystem
from elo_calibration import LeagueCalibrator
from match_metrics import odds_to_probs_power

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATE = os.path.join(BASE, "data", "models", "match_predictor.pkl")

# Locked in by src/optimise_accuracy.py, chosen on validation seasons 2022-23
# and confirmed once on 2024-25 (0.9932 against 0.9989 for the previous
# configuration - significant at 95%). Changing any of these without re-running
# that script means the calibration no longer matches the ratings.
DEFAULT_CONFIG = {
    "k_factor": 16.0,
    "season_regression": 0.85,
    "use_margin_multiplier": True,
    "regression_mode": "division",
    "rating_hfa": 65.0,
    "dc_xi": 0.001,
    # Home advantage is a crowd effect and 2019-20 and 2020-21 were played
    # behind closed doors - home wins fell from ~45% to 39.9%. Those seasons
    # describe a world that no longer exists, so they are excluded from the
    # calibration fit while still counting toward the ratings themselves.
    "calibration_exclude_seasons": [2019, 2020],
}

# Spellings the fixture feed uses that our results source abbreviates or
# translates differently. Keyed on the normalised form, so accents and the
# usual FC/SV/VfB noise are already stripped before lookup. Only genuine
# mismatches belong here - anything containment can resolve should not.
NAME_ALIASES = {
    # Germany
    "bayern munchen":           "Bayern Munich",
    "borussia monchengladbach": "M'gladbach",
    "monchengladbach":          "M'gladbach",
    "eintracht frankfurt":      "Ein Frankfurt",
    # Spain
    "athletic":                 "Ath Bilbao",
    "athletic bilbao":          "Ath Bilbao",
    "atletico madrid":          "Ath Madrid",
    "espanyol barcelona":       "Espanol",
    "espanyol":                 "Espanol",
    "real racing santander":    "Santander",
    # France
    "paris saint germain":      "Paris SG",
    "saint etienne":            "St Etienne",
    # Italy
    "internazionale milano":    "Inter",
    "internazionale":           "Inter",
    # England
    "manchester city":          "Man City",
    "manchester united":        "Man United",
    "nottingham forest":        "Nott'm Forest",
    "sheffield united":         "Sheffield United",
    "west bromwich albion":     "West Brom",
}


class MatchPredictor:
    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.elo: EloRatingSystem | None = None
        self.calibrator: LeagueCalibrator | None = None
        self.dc: DixonColes | None = None
        self.team_competition: dict[str, str] = {}
        self.trained_to: pd.Timestamp | None = None
        self.n_matches: int = 0

    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, matches: pd.DataFrame, config: dict | None = None,
              verbose: bool = True) -> "MatchPredictor":
        """
        Fit everything on the full history.

        `matches` should contain every division, both tiers. The rating system
        needs the second tier so promoted sides arrive with a real rating;
        predictions are only ever requested for the top tier.
        """
        self = cls(config)
        df = matches.sort_values("date").reset_index(drop=True).copy()
        if "tier" not in df.columns:
            df["tier"] = 1

        cfg = self.config
        self.elo = EloRatingSystem(
            k_factor=cfg["k_factor"],
            home_advantage=cfg["rating_hfa"],
            season_regression=cfg["season_regression"],
            use_margin_multiplier=cfg["use_margin_multiplier"],
            regression_mode=cfg["regression_mode"],
        )
        rated = self.elo.run(df.rename(columns={"date": "utc_date"}))
        rated = rated.rename(columns={"utc_date": "date"})

        if verbose:
            print(f"  rated {len(rated):,} matches across "
                  f"{len(self.elo.ratings)} teams")

        # Calibration is fitted on top-tier matches only - that is the
        # population the product predicts, and second-tier scorelines have
        # their own draw behaviour that would bias the curve.
        top = rated[rated["tier"] == 1]
        fit_on = top[~top["season"].isin(cfg["calibration_exclude_seasons"])]
        self.calibrator = LeagueCalibrator().fit(
            fit_on["elo_diff"].to_numpy(),
            fit_on["result"].to_numpy(int),
            fit_on["competition"].to_numpy(),
        )
        if verbose:
            print(f"  calibrated on {len(fit_on):,} top-tier matches "
                  f"(excluded seasons {cfg['calibration_exclude_seasons']})")

        self.dc = DixonColes(xi=cfg["dc_xi"]).fit(df)
        if verbose:
            print(f"  Dixon-Coles: home_adv={self.dc.home_adv:+.3f}  "
                  f"rho={self.dc.rho:+.3f}")

        # Remember where each team last played so the right league calibration
        # is applied when the caller does not name a competition.
        last = (top.assign(t=top["home_team"])
                   .groupby("t")["competition"].last().to_dict())
        last.update((top.assign(t=top["away_team"])
                        .groupby("t")["competition"].last().to_dict()))
        self.team_competition = last

        self.trained_to = df["date"].max()
        self.n_matches = len(df)
        return self

    # ------------------------------------------------------------------ #

    def known(self, team: str) -> bool:
        return self.elo is not None and team in self.elo.ratings

    def resolve_name(self, name: str) -> str | None:
        """
        Map an external team name onto one this model knows.

        Matching runs over every rated team, not just those that have played in
        the top tier. That distinction matters most for exactly the teams we
        most want to price: a side promoted this summer has never appeared in
        the top flight, so a top-tier-only index would miss it - and the whole
        point of carrying second-tier history is that we already hold a real
        rating for them.

        Fuzzy matching is deliberately avoided. Edit distance happily pairs
        "Real Madrid" with "Real Sociedad", and a confidently wrong prediction
        is worse than an absent one. Instead: exact, then normalised, then an
        explicit alias table, then containment accepted only when exactly one
        candidate matches.
        """
        if self.elo is None or not name:
            return None
        if name in self.elo.ratings:
            return name

        from team_names import normalise

        if getattr(self, "_by_norm", None) is None:
            self._by_norm = {}
            for t in self.elo.ratings:
                self._by_norm.setdefault(normalise(t), t)

        n = normalise(name)
        if n in NAME_ALIASES:
            cand = NAME_ALIASES[n]
            return cand if cand in self.elo.ratings else None
        if n in self._by_norm:
            return self._by_norm[n]

        hits = {v for k, v in self._by_norm.items() if k and (k in n or n in k)}
        return hits.pop() if len(hits) == 1 else None

    def competition_for(self, home: str, away: str) -> str:
        return (self.team_competition.get(home)
                or self.team_competition.get(away)
                or "PL")

    def outcome_probabilities(self, home: str, away: str,
                              competition: str | None = None) -> np.ndarray:
        comp = competition or self.competition_for(home, away)
        diff = self.elo.get(home) - self.elo.get(away)
        return self.calibrator.predict(np.array([diff]), np.array([comp]))[0]

    # ------------------------------------------------------------------ #

    def predict(self, home: str, away: str, competition: str | None = None,
                market_odds: tuple[float, float, float] | None = None) -> dict:
        """
        Full prediction for one fixture.

        `market_odds` is optional and does NOT change the prediction.

        Blending the two was the obvious thing to try and it was measured:
        both linear and logarithmic pooling put a weight of exactly zero on
        this model when combined with closing odds. The market already
        contains everything we know - unsurprising, since bookmakers run
        comparable models and additionally price in team news that results-
        derived features cannot see. Mixing in our numbers only made the
        forecast worse, so the product does not pretend otherwise.

        What the odds are used for instead is comparison. Given a price, the
        response reports the market's implied probabilities alongside ours and
        the difference between them, which is the genuinely useful question:
        where do we disagree, and by how much?
        """
        comp = competition or self.competition_for(home, away)
        elo_probs = self.outcome_probabilities(home, away, comp)

        grid = self.dc.score_matrix(home, away)
        n = grid.shape[0]

        # Split the grid by outcome, then rescale each block onto Elo's total.
        tri_home = np.tril(np.ones((n, n), bool), -1)
        tri_draw = np.eye(n, dtype=bool)
        tri_away = np.triu(np.ones((n, n), bool), 1)

        scaled = grid.copy()
        for mask, target in zip((tri_home, tri_draw, tri_away), elo_probs):
            current = grid[mask].sum()
            scaled[mask] *= (target / current) if current > 1e-12 else 0.0
        scaled /= scaled.sum()

        total = np.add.outer(np.arange(n), np.arange(n))
        lam, mu = self.dc.rates(home, away)

        flat = sorted(
            (((h, a), float(scaled[h, a])) for h in range(n) for a in range(n)),
            key=lambda x: -x[1],
        )

        market = None
        if market_odds is not None and all(o and o > 1.0 for o in market_odds):
            book = odds_to_probs_power(*[[o] for o in market_odds])[0]
            market = {
                "home_win": round(float(book[0]), 4),
                "draw": round(float(book[1]), 4),
                "away_win": round(float(book[2]), 4),
                # How much margin the bookmaker built into this price.
                "overround": round(sum(1.0 / o for o in market_odds) - 1.0, 4),
                "edge": {
                    "home_win": round(float(elo_probs[0] - book[0]), 4),
                    "draw": round(float(elo_probs[1] - book[1]), 4),
                    "away_win": round(float(elo_probs[2] - book[2]), 4),
                },
            }

        return {
            "home_team": home,
            "away_team": away,
            "competition": comp,
            "known_teams": bool(self.known(home) and self.known(away)),
            "probabilities": {
                "home_win": round(float(elo_probs[0]), 4),
                "draw": round(float(elo_probs[1]), 4),
                "away_win": round(float(elo_probs[2]), 4),
            },
            "market": market,
            "elo": {
                "home": round(float(self.elo.get(home)), 1),
                "away": round(float(self.elo.get(away)), 1),
                "diff": round(float(self.elo.get(home) - self.elo.get(away)), 1),
            },
            "expected_goals": {
                "home": round(lam, 2),
                "away": round(mu, 2),
                "total": round(lam + mu, 2),
            },
            "goals_markets": {
                "over_1_5": round(float(scaled[total > 1].sum()), 4),
                "over_2_5": round(float(scaled[total > 2].sum()), 4),
                "over_3_5": round(float(scaled[total > 3].sum()), 4),
                "btts": round(float(scaled[1:, 1:].sum()), 4),
                "home_clean_sheet": round(float(scaled[:, 0].sum()), 4),
                "away_clean_sheet": round(float(scaled[0, :].sum()), 4),
            },
            "top_scores": [
                {"score": f"{h}-{a}", "probability": round(p, 4)}
                for (h, a), p in flat[:8]
            ],
            "scoreline_grid": [
                [round(float(scaled[h, a]), 5) for a in range(6)]
                for h in range(6)
            ],
        }

    # ------------------------------------------------------------------ #

    def ratings(self, competition: str | None = None, top: int = 30) -> list[dict]:
        rows = [
            {
                "team": t,
                "elo": round(float(r), 1),
                "competition": self.team_competition.get(t, "?"),
                "attack": (round(float(self.dc.attack[self.dc.index[t]]), 3)
                           if t in self.dc.index else None),
                "defence": (round(float(self.dc.defence[self.dc.index[t]]), 3)
                            if t in self.dc.index else None),
            }
            for t, r in self.elo.ratings.items()
        ]
        if competition:
            rows = [r for r in rows if r["competition"] == competition]
        rows.sort(key=lambda r: -r["elo"])
        return rows[:top]

    def teams(self, competition: str | None = None) -> list[str]:
        items = self.team_competition.items()
        if competition:
            items = [(t, c) for t, c in items if c == competition]
        return sorted(t for t, _ in items)

    # ------------------------------------------------------------------ #

    def save(self, path: str = STATE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # The Elo history list is a per-match audit log that is not needed at
        # prediction time and dominates the file size.
        history, self.elo.history = self.elo.history, []
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        finally:
            self.elo.history = history
        return path

    @staticmethod
    def load(path: str = STATE) -> "MatchPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)
