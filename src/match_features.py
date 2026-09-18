"""
Feature engineering for match outcome prediction.

THE ONE RULE
------------
Every feature describes what was knowable BEFORE kick-off. Nothing may touch
the match being predicted, or any match after it.

Rolling windows are where this goes wrong, because the obvious pandas idiom is
silently broken:

    df.groupby("team")["goals"].rolling(5).mean()        # WRONG
    df.groupby("team")["goals"].shift(1).rolling(5).mean()   # correct

The first includes the current row - the model is told how many goals a team
scored in the match it is being asked to predict. Backtests come out
spectacular and live performance collapses. `_rolling_prior()` below is the
only place a rolling mean is computed, so the shift cannot be forgotten in one
place and remembered in another.

STRUCTURE
---------
A match has two teams, so raw match rows are the wrong shape for per-team
history. We explode to a team-match log (two rows per match, one per side),
compute every rolling statistic there, then fold back to one row per match.

FEATURE GROUPS
--------------
Features are grouped so the model can be trained on subsets. That is what
makes an honest ablation possible: fit with and without a group, compare
validation log loss, and keep it only if it pays for itself. The proxy-xG
group exists specifically to be tested this way.
"""

import numpy as np
import pandas as pd

from elo import EloRatingSystem

# Shot-quality weights for proxy xG. A shot on target converts far more often
# than one off target; these are rough league-wide rates, not a real xG model
# (which would need shot location, the dominant factor, which we do not have).
# Deliberately crude - the point is to measure whether crude helps at all.
XG_WEIGHT_ON_TARGET = 0.33
XG_WEIGHT_OFF_TARGET = 0.05

DEFAULT_WINDOW = 5
H2H_LOOKBACK = 6

# Gaps longer than this are not rest, they are absence - a relegated side
# returning after several seasons produces a "rest" of hundreds of days. The
# fatigue effect saturates after about a fortnight anyway, so everything above
# the cap means the same thing: fully recovered.
MAX_REST_DAYS = 14


# ===================================================================== #
# Team-match log
# ===================================================================== #

def build_team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Explode matches into one row per team per match.

    Each side gets a row stated from its own perspective: goals_for /
    goals_against rather than home_goals / away_goals. Rolling history is
    then a simple groupby over `team`.
    """
    base = ["match_id", "date", "season", "competition"]

    home = pd.DataFrame({
        **{c: matches[c] for c in base},
        "team":            matches["home_team"],
        "opponent":        matches["away_team"],
        "is_home":         1,
        "goals_for":       matches["home_goals"],
        "goals_against":   matches["away_goals"],
        "shots_for":       matches.get("home_shots"),
        "shots_against":   matches.get("away_shots"),
        "sot_for":         matches.get("home_shots_target"),
        "sot_against":     matches.get("away_shots_target"),
        "corners_for":     matches.get("home_corners"),
        "corners_against": matches.get("away_corners"),
    })

    away = pd.DataFrame({
        **{c: matches[c] for c in base},
        "team":            matches["away_team"],
        "opponent":        matches["home_team"],
        "is_home":         0,
        "goals_for":       matches["away_goals"],
        "goals_against":   matches["home_goals"],
        "shots_for":       matches.get("away_shots"),
        "shots_against":   matches.get("home_shots"),
        "sot_for":         matches.get("away_shots_target"),
        "sot_against":     matches.get("home_shots_target"),
        "corners_for":     matches.get("away_corners"),
        "corners_against": matches.get("home_corners"),
    })

    log = pd.concat([home, away], ignore_index=True)

    log["points"] = np.select(
        [log["goals_for"] > log["goals_against"],
         log["goals_for"] == log["goals_against"]],
        [3, 1], default=0,
    )
    log["won"] = (log["goals_for"] > log["goals_against"]).astype(int)

    # Proxy xG, computed per match so it can be rolled like any other stat.
    on_target = log["sot_for"]
    off_target = log["shots_for"] - log["sot_for"]
    log["proxy_xg_for"] = (
        on_target * XG_WEIGHT_ON_TARGET + off_target * XG_WEIGHT_OFF_TARGET
    )
    on_target_a = log["sot_against"]
    off_target_a = log["shots_against"] - log["sot_against"]
    log["proxy_xg_against"] = (
        on_target_a * XG_WEIGHT_ON_TARGET + off_target_a * XG_WEIGHT_OFF_TARGET
    )

    return log.sort_values(["team", "date"]).reset_index(drop=True)


# ===================================================================== #
# Rolling history
# ===================================================================== #

def _prior_mean(s: pd.Series, window: int) -> pd.Series:
    """
    Mean of the previous `window` values, excluding the current row.

    shift(1) BEFORE rolling is what excludes the current row. Reversing the
    order silently leaks the outcome being predicted. Every rolling statistic
    in this module routes through here, so the shift cannot be applied in one
    place and forgotten in another.

    min_periods=1 lets a team's early matches use whatever history exists
    rather than dropping out entirely. Those estimates are noisy, so a
    companion `n_prior` count is emitted alongside.
    """
    return s.shift(1).rolling(window, min_periods=1).mean()


def add_rolling_features(log: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Attach rolling form, scoring, shot and proxy-xG statistics."""
    df = log.sort_values(["team", "date"]).copy()
    g = df.groupby("team")

    rolled = {
        "form_ppg":          "points",
        "goals_for_avg":     "goals_for",
        "goals_against_avg": "goals_against",
        "shots_for_avg":     "shots_for",
        "shots_against_avg": "shots_against",
        "sot_for_avg":       "sot_for",
        "sot_against_avg":   "sot_against",
        "corners_for_avg":   "corners_for",
        "proxy_xg_for_avg":     "proxy_xg_for",
        "proxy_xg_against_avg": "proxy_xg_against",
    }
    # transform() on a Series groupby keeps this vectorised and avoids
    # operating on grouping columns.
    for out, src in rolled.items():
        df[out] = g[src].transform(lambda s: _prior_mean(s, window))

    # Reliability indicator for the rolling means above, capped at the window.
    # Uncapped this would run to 379 and become a proxy for "this club has
    # never been relegated" - real information, but it invites the model to
    # memorise specific clubs rather than learn from form.
    df["n_prior"] = g.cumcount().clip(upper=window)

    # Venue-specific form. Teams behave very differently home and away and a
    # blended figure hides it, so each is rolled within its own partition -
    # only prior HOME matches feed a home figure.
    for venue, label in [(1, "home"), (0, "away")]:
        mask = df["is_home"] == venue
        df.loc[mask, f"venue_ppg_{label}"] = (
            df[mask].groupby("team")["points"]
                    .transform(lambda s: _prior_mean(s, window))
        )

    # Finishing efficiency: goals per shot on target. Kept separate from
    # chance creation on purpose - a team that creates a lot and finishes
    # badly is a different proposition from one that does the reverse, and
    # folding them together destroys that distinction.
    df["conversion_rate"] = np.where(
        df["sot_for_avg"] > 0, df["goals_for_avg"] / df["sot_for_avg"], np.nan
    )

    # Days since this team last played, capped (see MAX_REST_DAYS).
    df["rest_days"] = g["date"].diff().dt.days.clip(upper=MAX_REST_DAYS)

    # Season-to-date points per game, again strictly prior. Resets each season.
    df = df.sort_values(["team", "season", "date"])
    df["season_ppg"] = (
        df.groupby(["team", "season"])["points"]
          .transform(lambda s: s.shift(1).expanding().mean())
    )

    return df.sort_values(["team", "date"])


# ===================================================================== #
# Head to head
# ===================================================================== #

def add_h2h(matches: pd.DataFrame, lookback: int = H2H_LOOKBACK) -> pd.DataFrame:
    """
    Home side's record in recent meetings with this specific opponent.

    Keyed on an unordered pair so home and away legs share one history.
    Walked match by match in date order - each row sees only earlier meetings.
    """
    df = matches.sort_values("date").reset_index(drop=True).copy()

    seen: dict[tuple, list[tuple]] = {}
    rates, counts, goals = [], [], []

    for r in df.itertuples(index=False):
        key = tuple(sorted([r.home_team, r.away_team]))
        past = seen.get(key, [])[-lookback:]

        if past:
            wins = sum(1 for home, hg, ag in past
                       if (hg > ag if home == r.home_team else ag > hg))
            rates.append(wins / len(past))
            goals.append(float(np.mean([hg + ag for _, hg, ag in past])))
            counts.append(len(past))
        else:
            rates.append(np.nan)
            goals.append(np.nan)
            counts.append(0)

        seen.setdefault(key, []).append(
            (r.home_team, r.home_goals, r.away_goals)
        )

    df["h2h_home_win_rate"] = rates
    df["h2h_avg_goals"] = goals
    df["h2h_matches"] = counts
    return df


# ===================================================================== #
# Assembly
# ===================================================================== #

TEAM_FEATURES = [
    "form_ppg", "goals_for_avg", "goals_against_avg",
    "shots_for_avg", "shots_against_avg", "sot_for_avg", "sot_against_avg",
    "corners_for_avg", "proxy_xg_for_avg", "proxy_xg_against_avg",
    "conversion_rate", "rest_days", "season_ppg", "n_prior",
]

FEATURE_GROUPS = {
    "elo":      ["elo_home_pre", "elo_away_pre", "elo_diff"],
    "form": [
        "home_form_ppg", "away_form_ppg",
        "home_season_ppg", "away_season_ppg",
        "home_venue_ppg", "away_venue_ppg",
        "form_ppg_diff",
    ],
    "goals": [
        "home_goals_for_avg", "home_goals_against_avg",
        "away_goals_for_avg", "away_goals_against_avg",
        "goals_for_diff",
    ],
    "shots": [
        "home_shots_for_avg", "home_shots_against_avg",
        "away_shots_for_avg", "away_shots_against_avg",
        "home_sot_for_avg", "home_sot_against_avg",
        "away_sot_for_avg", "away_sot_against_avg",
        "home_corners_for_avg", "away_corners_for_avg",
        "home_conversion_rate", "away_conversion_rate",
        "sot_diff",
    ],
    "proxy_xg": [
        "home_proxy_xg_for_avg", "home_proxy_xg_against_avg",
        "away_proxy_xg_for_avg", "away_proxy_xg_against_avg",
        "proxy_xg_diff",
    ],
    "h2h":      ["h2h_home_win_rate", "h2h_avg_goals", "h2h_matches"],
    "context":  ["home_rest_days", "away_rest_days", "rest_diff",
                 "matchday_norm", "home_n_prior", "away_n_prior"],
}


def all_features(exclude: list[str] | None = None) -> list[str]:
    """Flatten FEATURE_GROUPS, optionally dropping named groups (for ablation)."""
    exclude = set(exclude or [])
    return [f for g, feats in FEATURE_GROUPS.items() if g not in exclude for f in feats]


def build_features(
    matches: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    elo_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Full feature matrix, one row per match.

    Input needs: match_id, date, season, competition, home_team, away_team,
    home_goals, away_goals, result, and ideally the shot columns.

    Returns the original columns plus every feature named in FEATURE_GROUPS.
    """
    df = matches.sort_values("date").reset_index(drop=True).copy()
    if "match_id" not in df.columns:
        df["match_id"] = np.arange(len(df))

    # --- Elo (own module; snapshot-before-update guarantees no leakage) ---
    elo = EloRatingSystem(**(elo_kwargs or {}))
    rated = elo.run(df.rename(columns={"date": "utc_date"}))
    rated = rated.rename(columns={"utc_date": "date"})
    df = rated

    # --- rolling team history ---
    log = add_rolling_features(build_team_match_log(df), window=window)

    home_log = log[log["is_home"] == 1].set_index("match_id")
    away_log = log[log["is_home"] == 0].set_index("match_id")

    for col in TEAM_FEATURES:
        if col in home_log.columns:
            df[f"home_{col}"] = df["match_id"].map(home_log[col])
            df[f"away_{col}"] = df["match_id"].map(away_log[col])

    df["home_venue_ppg"] = df["match_id"].map(home_log["venue_ppg_home"])
    df["away_venue_ppg"] = df["match_id"].map(away_log["venue_ppg_away"])

    # --- head to head ---
    h2h = add_h2h(df)
    for c in ["h2h_home_win_rate", "h2h_avg_goals", "h2h_matches"]:
        df[c] = df["match_id"].map(h2h.set_index("match_id")[c])

    # --- differentials -------------------------------------------------- #
    # Trees split on one feature at a time, so a gap the model would otherwise
    # have to reconstruct through several splits is worth handing over
    # directly. Elo showed how much signal lives in the difference.
    df["form_ppg_diff"]  = df["home_form_ppg"] - df["away_form_ppg"]
    df["goals_for_diff"] = df["home_goals_for_avg"] - df["away_goals_for_avg"]
    df["sot_diff"]       = df["home_sot_for_avg"] - df["away_sot_for_avg"]
    df["proxy_xg_diff"]  = df["home_proxy_xg_for_avg"] - df["away_proxy_xg_for_avg"]
    df["rest_diff"]      = df["home_rest_days"] - df["away_rest_days"]

    # Season progress. Early rounds carry thin history and behave differently;
    # this lets the model condition on that rather than treating every match
    # as equally well-informed.
    if "matchday" in df.columns:
        df["matchday_norm"] = df["matchday"] / 38.0
    else:
        # Divide by the season's EXPECTED fixture count, derived from how many
        # teams are in the division: a double round-robin is n*(n-1) matches.
        #
        # Two traps avoided here. A fixed divisor would misscale leagues of
        # different sizes - 306 matches in the Bundesliga, 552 in the
        # Championship - so the model would read "late season" as "played in
        # England". Dividing by matches actually present is worse still: for
        # the season in progress that count is however many have been played
        # so far, so the most recent fixture always scores 1.0 and every live
        # prediction claims to be the last day of the season.
        keys = ["season", "competition"]
        stacked = pd.concat([
            df[keys + ["home_team"]].rename(columns={"home_team": "team"}),
            df[keys + ["away_team"]].rename(columns={"away_team": "team"}),
        ])
        n_teams = stacked.groupby(keys)["team"].nunique()
        expected = (n_teams * (n_teams - 1)).rename("expected")

        df = df.merge(expected, left_on=keys, right_index=True, how="left")
        df["matchday_norm"] = (
            df.groupby(keys).cumcount() / df["expected"].clip(lower=1)
        ).clip(0, 1)
        df = df.drop(columns="expected")

    return df


def feature_report(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Null rate and spread per feature - a quick check that nothing is broken."""
    features = features or all_features()
    rows = []
    for f in features:
        if f not in df.columns:
            rows.append({"feature": f, "status": "MISSING"})
            continue
        s = pd.to_numeric(df[f], errors="coerce")
        rows.append({
            "feature":  f,
            "status":   "ok",
            "null_pct": round(float(s.isna().mean()), 4),
            "mean":     round(float(s.mean()), 3),
            "std":      round(float(s.std()), 3),
            "min":      round(float(s.min()), 2),
            "max":      round(float(s.max()), 2),
        })
    return pd.DataFrame(rows)
