"""
Transforms player stats into model-ready features.
All per-90 rates computed here so the model learns quality, not volume.

Two input modes:
  - TM mode (training): DataFrame from prepare_tm_data.py
    Has: age, goals, assists, minutes_played, competition, position_group,
         international_caps, club_prestige_eur
  - API mode (inference): DataFrame from FootballDataClient.get_scorers()
    Has: date_of_birth, season, goals, assists, played_matches, competition, position
    Missing: international_caps, club_prestige_eur → defaulted to 0
"""

import pandas as pd
import numpy as np
from datetime import date

# ── Micro-position constants ──────────────────────────────────────────────────
MICRO_POSITION_GROUPS = ["GK", "CB", "FB", "CDM", "CM", "CAM", "W", "ST"]

MICRO_POSITION_MAP = {
    # football-data.org strings
    "Goalkeeper":         "GK",
    "Centre-Back":        "CB",
    "Left-Back":          "FB",
    "Right-Back":         "FB",
    "Defensive Midfield": "CDM",
    "Central Midfield":   "CM",
    "Attacking Midfield": "CAM",
    "Left Midfield":      "CM",
    "Right Midfield":     "CM",
    "Left Winger":        "W",
    "Right Winger":       "W",
    "Second Striker":     "ST",
    "Centre-Forward":     "ST",
    "Forward":            "ST",
    # Broad group fallbacks
    "GK":  "GK",
    "DEF": "CB",
    "MID": "CM",
    "ATT": "ST",
}

# Broad-to-micro fallback (used when micro_position_group not in data)
_BROAD_TO_MICRO = {
    "GK":  "GK",
    "DEF": "CB",
    "MID": "CM",
    "ATT": "ST",
}

POSITION_MAP = {
    # football-data.org strings
    "Goalkeeper":         "GK",
    "Centre-Back":        "DEF",
    "Left-Back":          "DEF",
    "Right-Back":         "DEF",
    "Defensive Midfield": "MID",
    "Central Midfield":   "MID",
    "Attacking Midfield": "MID",
    "Left Midfield":      "MID",
    "Right Midfield":     "MID",
    "Left Winger":        "ATT",
    "Right Winger":       "ATT",
    "Centre-Forward":     "ATT",
    "Forward":            "ATT",
    # Transfermarkt broad group strings (already normalised by prepare_tm_data.py)
    "GK":  "GK",
    "DEF": "DEF",
    "MID": "MID",
    "ATT": "ATT",
}

# League strength, normalised to Serie A = 1.00 (the previous anchor, kept so
# the scale stays comparable).
#
# Only PL/PD/SA were mapped before, so Bundesliga and Ligue 1 fell through the
# .fillna(1.0) below and were silently priced as though they were Serie A —
# roughly a third of the player base carrying an invented coefficient on what
# feature importance ranks as the model's third-heaviest input. The three
# values that were present were guesses.
#
# Source is UEFA's five-year country coefficients (2025-26 edition), which
# measure how clubs from each country actually perform in European competition.
# Deliberately NOT derived from median market value per league: that is the
# training target, and feeding it back in as a feature would be leakage.
LEAGUE_DIFFICULTY = {
    "PL":  1.155,   # England  104.303
    "SA":  1.000,   # Italy     90.284  (anchor)
    "PD":  0.991,   # Spain     89.489
    "BL1": 0.959,   # Germany   86.624
    "FL1": 0.740,   # France    66.831
}

FEATURE_COLS = [
    "age",
    "goals_p90",
    "assists_p90",
    "gc_p90",
    "minutes_played",
    "league_difficulty",
    "age_factor",
    "international_caps",   # Fix 2: new
    "club_prestige_eur",    # Fix 3: new
    "pos_GK", "pos_DEF", "pos_MID", "pos_ATT",
]
# Alias so old code and new code both work
FEATURE_COLS_V1 = FEATURE_COLS

FEATURE_COLS_V2 = [
    "age", "goals_p90", "assists_p90", "gc_p90",
    "minutes_played", "league_difficulty", "age_factor",
    "international_caps", "club_prestige_eur",
    "contract_years_est", "form_score",
    "pos_GK", "pos_CB", "pos_FB", "pos_CDM", "pos_CM", "pos_CAM", "pos_W", "pos_ST",
]


def compute_age(dob_str: str, reference_date: date = None) -> float:
    if not dob_str or str(dob_str) in ("nan", "NaT", "None"):
        return np.nan
    ref = reference_date or date.today()
    try:
        return (ref - date.fromisoformat(str(dob_str)[:10])).days / 365.25
    except ValueError:
        return np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Age ────────────────────────────────────────────────────────────────────
    if "age" not in df.columns:
        df["age"] = df.apply(
            lambda r: compute_age(r["date_of_birth"], date(int(r["season"]), 8, 1)),
            axis=1,
        )

    # ── Position group + one-hot ───────────────────────────────────────────────
    if "position_group" not in df.columns:
        df["position_group"] = df["position"].map(POSITION_MAP).fillna("MID")

    pos_dummies = pd.get_dummies(df["position_group"], prefix="pos", dtype=int)
    for col in ["pos_GK", "pos_DEF", "pos_MID", "pos_ATT"]:
        if col not in pos_dummies:
            pos_dummies[col] = 0

    # ── Minutes played ─────────────────────────────────────────────────────────
    # TM data: real minutes_played column
    # API data: played_matches only → estimate 75 min/game
    if "minutes_played" not in df.columns:
        df["minutes_played"] = df.get("played_matches", pd.Series(0, index=df.index)) * 75

    minutes_90 = (df["minutes_played"] / 90).replace(0, np.nan)

    # ── Per-90 rates ───────────────────────────────────────────────────────────
    df["goals_p90"]   = df["goals"]   / minutes_90
    df["assists_p90"] = df["assists"] / minutes_90
    df["gc_p90"]      = df["goals_p90"].fillna(0) + df["assists_p90"].fillna(0)

    # ── League difficulty ──────────────────────────────────────────────────────
    df["league_difficulty"] = df["competition"].map(LEAGUE_DIFFICULTY).fillna(1.0)

    # ── Age factor ─────────────────────────────────────────────────────────────
    df["age_factor"] = df["age"].apply(_age_factor)

    # ── Fix 2: International caps ──────────────────────────────────────────────
    # API mode won't have this; default 0 (model still works, just loses this signal)
    if "international_caps" not in df.columns:
        df["international_caps"] = 0
    df["international_caps"] = pd.to_numeric(df["international_caps"], errors="coerce").fillna(0)

    # ── Fix 3: Club prestige ───────────────────────────────────────────────────
    # API mode won't have this; default 0
    if "club_prestige_eur" not in df.columns:
        df["club_prestige_eur"] = 0.0
    df["club_prestige_eur"] = pd.to_numeric(df["club_prestige_eur"], errors="coerce").fillna(0)

    scalar_cols = [
        "age", "goals_p90", "assists_p90", "gc_p90",
        "minutes_played", "league_difficulty", "age_factor",
        "international_caps", "club_prestige_eur",
    ]
    X = pd.concat([df[scalar_cols].fillna(0), pos_dummies], axis=1)
    X = X[FEATURE_COLS]
    X.index = df.index
    return X


def _age_factor(age: float) -> float:
    if pd.isna(age):
        return 1.0
    if age < 21:
        return 1.30
    if age <= 27:
        return 1.15
    if age <= 30:
        return 1.00
    if age <= 33:
        return 0.75
    return 0.50


# Alias so imports can use either name
build_features_v1 = build_features


def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    V2 feature set: 8 micro-position one-hots + contract_years_est + form_score.
    Like build_features() but uses micro_position_group when available, adds
    contract_years_est from age, and adds form_score (default 0).

    Output columns follow FEATURE_COLS_V2.
    """
    from form_engine import _estimate_contract_years  # local import avoids circular dep at module level

    df = df.copy()

    # ── Age ────────────────────────────────────────────────────────────────────
    if "age" not in df.columns:
        df["age"] = df.apply(
            lambda r: compute_age(r["date_of_birth"], date(int(r["season"]), 8, 1)),
            axis=1,
        )

    # ── Micro position group ───────────────────────────────────────────────────
    if "micro_position_group" not in df.columns:
        # Derive from sub_position or position if available, else fall back broad→micro
        if "sub_position" in df.columns:
            df["micro_position_group"] = df["sub_position"].map(MICRO_POSITION_MAP)
        if "micro_position_group" not in df.columns or df["micro_position_group"].isna().any():
            fallback_src = (
                df.get("position_group", df.get("position", pd.Series("MID", index=df.index)))
            )
            micro_fallback = fallback_src.map(MICRO_POSITION_MAP).map(_BROAD_TO_MICRO).fillna("CM")
            if "micro_position_group" not in df.columns:
                df["micro_position_group"] = micro_fallback
            else:
                df["micro_position_group"] = df["micro_position_group"].fillna(micro_fallback)

    df["micro_position_group"] = df["micro_position_group"].fillna("CM")

    # One-hot for all 8 micro positions
    pos_dummies = pd.get_dummies(df["micro_position_group"], prefix="pos", dtype=int)
    for col in [f"pos_{g}" for g in MICRO_POSITION_GROUPS]:
        if col not in pos_dummies:
            pos_dummies[col] = 0

    # ── Minutes played ─────────────────────────────────────────────────────────
    if "minutes_played" not in df.columns:
        df["minutes_played"] = df.get("played_matches", pd.Series(0, index=df.index)) * 75

    minutes_90 = (df["minutes_played"] / 90).replace(0, np.nan)

    # ── Per-90 rates ───────────────────────────────────────────────────────────
    df["goals_p90"]   = df["goals"]   / minutes_90
    df["assists_p90"] = df["assists"] / minutes_90
    df["gc_p90"]      = df["goals_p90"].fillna(0) + df["assists_p90"].fillna(0)

    # ── League difficulty ──────────────────────────────────────────────────────
    df["league_difficulty"] = df["competition"].map(LEAGUE_DIFFICULTY).fillna(1.0)

    # ── Age factor ─────────────────────────────────────────────────────────────
    df["age_factor"] = df["age"].apply(_age_factor)

    # ── International caps ─────────────────────────────────────────────────────
    if "international_caps" not in df.columns:
        df["international_caps"] = 0
    df["international_caps"] = pd.to_numeric(df["international_caps"], errors="coerce").fillna(0)

    # ── Club prestige ──────────────────────────────────────────────────────────
    if "club_prestige_eur" not in df.columns:
        df["club_prestige_eur"] = 0.0
    df["club_prestige_eur"] = pd.to_numeric(df["club_prestige_eur"], errors="coerce").fillna(0)

    # ── Contract years estimated from age ─────────────────────────────────────
    df["contract_years_est"] = df["age"].apply(
        lambda a: _estimate_contract_years(a) if not pd.isna(a) else 2.5
    )

    # ── Form score (default 0 if not present — e.g. training data) ────────────
    if "form_score" not in df.columns:
        df["form_score"] = 0.0
    df["form_score"] = pd.to_numeric(df["form_score"], errors="coerce").fillna(0.0)

    scalar_cols = [
        "age", "goals_p90", "assists_p90", "gc_p90",
        "minutes_played", "league_difficulty", "age_factor",
        "international_caps", "club_prestige_eur",
        "contract_years_est", "form_score",
    ]
    X = pd.concat([df[scalar_cols].fillna(0), pos_dummies], axis=1)
    X = X[FEATURE_COLS_V2]
    X.index = df.index
    return X
