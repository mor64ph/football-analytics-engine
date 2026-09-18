"""
Regression tests for the rules that stop the model seeing the future.

WHY THESE EXIST AND NOT OTHERS
------------------------------
Most bugs announce themselves. Temporal leakage does the opposite: reverse the
snapshot-then-update ordering in Elo, or drop the shift(1) before a rolling
mean, and every metric IMPROVES. Log loss falls, accuracy rises, the charts
look better, and the model is worthless in production because it has been shown
results it could not have known.

There is no failing assertion anywhere else in the pipeline that would catch
that. These tests are the only thing standing between a plausible-looking
refactor and a silently invalid model, which is why this file exists before any
test of the parts that fail loudly.

Run:  python -m pytest tests/ -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))

from elo import EloRatingSystem, elo_to_probabilities, fit_draw_model
from match_features import add_rolling_features, build_team_match_log, _prior_mean


def _fixtures(n_rounds: int = 6) -> pd.DataFrame:
    """
    A deterministic mini-league where one side always wins.

    Alpha beats every opponent; Omega loses to everyone. If any feature is
    computed from the match it describes, Alpha's pre-match numbers will be
    perfect from the very first game rather than building up over time.
    """
    teams = ["Alpha", "Beta", "Gamma", "Omega"]
    rows, day = [], pd.Timestamp("2024-08-01")

    for r in range(n_rounds):
        for home, away in [(teams[0], teams[1]), (teams[2], teams[3])]:
            hg, ag = (3, 0) if home == "Alpha" else (2, 1)
            rows.append({
                "match_id": len(rows), "date": day, "season": 2024,
                "competition": "PL", "tier": 1,
                "home_team": home, "away_team": away,
                "home_goals": hg, "away_goals": ag,
                "result": 0 if hg > ag else (1 if hg == ag else 2),
            })
        day += pd.Timedelta(days=7)

    return pd.DataFrame(rows)


# ===================================================================== #
# Elo: the snapshot must predate the update
# ===================================================================== #

def test_elo_snapshot_precedes_update():
    """
    The rating attached to a match must be the rating BEFORE it was played.

    update() returns the pre-match snapshot and mutates the stored rating. If
    those two ever swap order, the returned elo_diff already contains the
    result and the feature becomes an oracle.
    """
    elo = EloRatingSystem()
    snap = elo.update("Alpha", "Beta", 5, 0)

    # Both sides start level, so the snapshot must show no gap at all.
    assert snap["elo_diff"] == 0.0, (
        "elo_diff for the first match is non-zero, so the snapshot was taken "
        "after the update - the feature contains the result"
    )
    assert snap["elo_home_pre"] == 1500.0
    # And the stored rating must have moved.
    assert elo.get("Alpha") > 1500.0
    assert elo.get("Beta") < 1500.0


def test_elo_first_meeting_is_uninformed():
    """Every team's first match must be predicted from the default rating."""
    df = _fixtures()
    rated = EloRatingSystem().run(df.rename(columns={"date": "utc_date"}))

    first = rated.groupby("home_team").head(1)
    assert (first["elo_home_pre"] == 1500.0).all(), (
        "a team's first appearance already carries a non-default rating"
    )


def test_elo_is_zero_sum():
    """Points moved between sides, never created - otherwise ratings inflate."""
    elo = EloRatingSystem()
    for _ in range(25):
        elo.update("Alpha", "Beta", 3, 1)
    total = sum(elo.ratings.values())
    assert total == pytest.approx(3000.0, abs=1e-6)


def test_elo_ordering_is_not_accidentally_reversible():
    """
    A guard against the specific refactor that breaks this.

    Recording the rating after the update would make the home side's advantage
    appear before it was earned. Simulated here by comparing what update()
    returns against the stored value afterwards.
    """
    elo = EloRatingSystem()
    snap = elo.update("Alpha", "Beta", 4, 0)
    assert snap["elo_home_pre"] != elo.get("Alpha"), (
        "the returned snapshot equals the post-match rating"
    )


# ===================================================================== #
# Rolling features: shift before rolling
# ===================================================================== #

def test_prior_mean_excludes_the_current_row():
    """_prior_mean must average earlier values only."""
    s = pd.Series([10.0, 20.0, 30.0, 40.0])
    out = _prior_mean(s, window=3)

    assert pd.isna(out.iloc[0]), "the first row has no history and must be null"
    assert out.iloc[1] == 10.0
    assert out.iloc[2] == 15.0            # mean(10, 20)
    assert out.iloc[3] == 20.0            # mean(10, 20, 30)
    # If the shift were dropped, row 3 would be mean(20,30,40) = 30.
    assert out.iloc[3] != 30.0, "shift(1) is missing: the row sees its own value"


def test_rolling_features_start_null():
    """A team's first match can have no form, by construction."""
    log = add_rolling_features(build_team_match_log(_fixtures()), window=5)
    first = log.sort_values("date").groupby("team").head(1)

    for col in ("form_ppg", "goals_for_avg", "shots_for_avg"):
        assert first[col].isna().all(), (
            f"{col} is populated on a team's first match, so it was computed "
            f"from that match"
        )


def test_form_reflects_only_earlier_results():
    """
    Alpha wins every match, so its form should climb from null toward 3.0 -
    never start there.
    """
    log = add_rolling_features(build_team_match_log(_fixtures()), window=5)
    alpha = log[log["team"] == "Alpha"].sort_values("date")

    assert pd.isna(alpha["form_ppg"].iloc[0])
    assert alpha["form_ppg"].iloc[1] == pytest.approx(3.0)
    # Monotonic non-decreasing once it exists: a team that keeps winning cannot
    # have its average fall.
    seen = alpha["form_ppg"].dropna()
    assert (seen.diff().dropna() >= -1e-9).all()


def test_no_feature_correlates_suspiciously_with_the_outcome():
    """
    A blunt end-to-end check.

    Any feature correlating with the result far more strongly than the rating
    difference does is almost certainly carrying the result itself. This catches
    leakage introduced anywhere in the chain, not just in the two places tested
    above.
    """
    feats = os.path.join(BASE, "data", "processed", "match_features.csv")
    if not os.path.exists(feats):
        pytest.skip("feature matrix not built")

    df = pd.read_csv(feats, low_memory=False)
    if "tier" in df.columns:
        df = df[df["tier"] == 1]

    from match_features import all_features
    cols = [c for c in all_features() if c in df.columns]
    cors = (df[cols].apply(pd.to_numeric, errors="coerce")
            .corrwith(df["result"].astype(float)).abs())

    worst = cors.idxmax()
    assert cors.max() < 0.5, (
        f"{worst} correlates {cors.max():.3f} with the outcome; anything this "
        f"strong is carrying the result it is meant to predict"
    )


# ===================================================================== #
# Calibration must be fitted on training data only
# ===================================================================== #

def test_draw_model_returns_a_usable_distribution():
    diffs = np.linspace(-400, 400, 400)
    y = np.where(diffs > 80, 0, np.where(diffs < -80, 2, 1))

    d_max, tau = fit_draw_model(diffs, y, 65.0)
    probs = elo_to_probabilities(diffs, 65.0, d_max, tau)

    assert probs.shape == (400, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs > 0).all(), "a zero probability makes log loss infinite"


def test_stronger_team_is_favoured():
    """Sanity: the mapping must not be inverted."""
    probs = elo_to_probabilities(np.array([300.0]), 65.0, 0.26, 300.0)[0]
    assert probs[0] > probs[2], "the stronger side is not the favourite"
