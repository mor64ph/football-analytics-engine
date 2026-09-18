"""
Preprocesses the raw Transfermarkt Kaggle archive into a single training-ready CSV.
Run this once locally before training:
    python src/prepare_tm_data.py

Fixes applied (v2):
  1. Min 500 minutes filter — removes fringe players with misleading per-90 stats
  2. international_caps added — strong proxy for elite player status
  3. Club prestige score — median squad value per club-season (data-driven, not hardcoded)
  4. player_club_id threaded through for the prestige join
"""

import argparse
import os
import numpy as np
import pandas as pd


# All five divisions the match model covers. The value model previously stopped
# at three, so Bundesliga and Ligue 1 players appeared in fixture predictions
# but had no valuation anywhere in the product.
COMPETITION_LABEL = {
    "GB1": "PL",    # Premier League
    "ES1": "PD",    # La Liga
    "IT1": "SA",    # Serie A
    "L1":  "BL1",   # Bundesliga
    "FR1": "FL1",   # Ligue 1
}
TARGET_COMPETITIONS = set(COMPETITION_LABEL)
MIN_SEASON          = 2018
MIN_MINUTES         = 500   # Fix 1: was 90 — removes fringe-game noise


def extract_season(date_str: str) -> int:
    d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return None
    return d.year if d.month >= 8 else d.year - 1


def run(archive_dir: str, output_path: str):
    print("Loading CSVs...")

    # Fix 2: add international_caps to players load
    players_df = pd.read_csv(
        os.path.join(archive_dir, "players.csv"),
        usecols=["player_id", "name", "date_of_birth", "position", "sub_position", "international_caps"],
        dtype={"player_id": str},
    )
    players_df["date_of_birth"] = pd.to_datetime(
        players_df["date_of_birth"], errors="coerce"
    ).dt.date.astype(str)
    players_df["international_caps"] = pd.to_numeric(
        players_df["international_caps"], errors="coerce"
    ).fillna(0).astype(int)

    pos_map = {
        "Goalkeeper": "GK",
        "Defender":   "DEF",
        "Midfield":   "MID",
        "Attack":     "ATT",
    }
    players_df["position_group"] = players_df["position"].map(pos_map).fillna("MID")

    MICRO_POS_MAP = {
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
    }
    players_df["micro_position_group"] = players_df["sub_position"].map(MICRO_POS_MAP).fillna(
        players_df["position_group"]  # fallback to broad group if sub_position unknown
    )
    print(f"  players:     {len(players_df):,} rows")

    # ── Appearances ─────────────────────────────────────────────────────────────
    print("Loading appearances (large file, takes ~20s)...")
    # Fix 3: add player_club_id for prestige join
    appearances_df = pd.read_csv(
        os.path.join(archive_dir, "appearances.csv"),
        usecols=["player_id", "player_club_id", "date", "competition_id",
                 "goals", "assists", "minutes_played"],
        dtype={"player_id": str, "player_club_id": str, "competition_id": str},
    )
    appearances_df = appearances_df[
        appearances_df["competition_id"].isin(TARGET_COMPETITIONS)
    ].copy()
    appearances_df["season"] = appearances_df["date"].apply(extract_season)
    appearances_df = appearances_df[appearances_df["season"] >= MIN_SEASON]

    stats = (
        appearances_df
        .groupby(["player_id", "competition_id", "season"], as_index=False)
        .agg(
            goals          =("goals",          "sum"),
            assists        =("assists",         "sum"),
            minutes_played =("minutes_played",  "sum"),
            appearances    =("goals",           "count"),
            # mode of club_id across appearances — handles rare mid-season transfers
            player_club_id =("player_club_id",  lambda x: x.mode().iloc[0]),
        )
    )
    stats["competition"] = stats["competition_id"].map(COMPETITION_LABEL)
    print(f"  appearances: {len(stats):,} player-season rows (pre-minute filter)")

    # ── Market valuations ────────────────────────────────────────────────────────
    print("Loading player_valuations...")
    # Fix 3: also load current_club_id for prestige calculation
    vals_df = pd.read_csv(
        os.path.join(archive_dir, "player_valuations.csv"),
        usecols=["player_id", "date", "market_value_in_eur", "current_club_id"],
        dtype={"player_id": str, "current_club_id": str},
    )
    vals_df["season"] = vals_df["date"].apply(extract_season)
    vals_df = vals_df[vals_df["season"] >= MIN_SEASON]

    vals_df["val_date"]     = pd.to_datetime(vals_df["date"], errors="coerce")
    vals_df["season_start"] = pd.to_datetime(vals_df["season"].astype(str) + "-08-01")
    vals_df["days_from_start"] = (vals_df["val_date"] - vals_df["season_start"]).dt.days.abs()

    closest_val = (
        vals_df
        .sort_values("days_from_start")
        .groupby(["player_id", "season"], as_index=False)
        .first()[["player_id", "season", "market_value_in_eur"]]
    )
    print(f"  valuations:  {len(closest_val):,} player-season rows")

    # Fix 3: Club prestige = median squad market value per club-season
    # This is data-driven: a club where the median player is worth €20M is "bigger"
    # than one where the median is €2M — no hardcoded club lists needed
    print("Computing club prestige scores...")
    club_prestige = (
        vals_df
        .groupby(["current_club_id", "season"])["market_value_in_eur"]
        .median()
        .reset_index()
        .rename(columns={
            "current_club_id":    "player_club_id",
            "market_value_in_eur": "club_prestige_eur",
        })
    )
    stats = stats.merge(club_prestige, on=["player_club_id", "season"], how="left")
    prestige_median = stats["club_prestige_eur"].median()
    stats["club_prestige_eur"] = stats["club_prestige_eur"].fillna(prestige_median)
    print(f"  club prestige range: €{stats['club_prestige_eur'].min():,.0f} – €{stats['club_prestige_eur'].max():,.0f}")

    # ── Join everything ──────────────────────────────────────────────────────────
    print("Joining tables...")
    df = stats.merge(players_df, on="player_id", how="inner")
    df = df.merge(closest_val, on=["player_id", "season"], how="inner")

    df["dob"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["season_start_date"] = pd.to_datetime(df["season"].astype(str) + "-08-01")
    df["age"] = ((df["season_start_date"] - df["dob"]).dt.days / 365.25).round(1)

    df = df.dropna(subset=["age", "market_value_in_eur", "minutes_played"])
    # Fix 1: was 90 — now 500 minutes minimum
    df = df[df["minutes_played"] >= MIN_MINUTES]
    df = df[df["market_value_in_eur"] > 0]

    df = df.rename(columns={
        "name":                "player_name",
        "market_value_in_eur": "market_value_eur",
    })

    output_cols = [
        "player_id", "player_name", "competition", "season",
        "position", "position_group", "micro_position_group", "age", "date_of_birth",
        "goals", "assists", "minutes_played", "appearances",
        "international_caps", "club_prestige_eur",
        "market_value_eur",
    ]
    df = df[output_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # ── Lookup tables for API inference mode ─────────────────────────────────
    # These give inference the same club_prestige and international_caps signals
    # the model was trained on, even though football-data.org doesn't expose them.

    out_dir = os.path.dirname(output_path)

    # Club prestige lookup: most recent season's prestige per (team_name, competition)
    # team_name comes from players.csv via the join; we use player_name's club
    # Instead, use the raw stats table which has player_club_id + competition + season
    # and join back to clubs for a name.  Simplest: aggregate prestige per
    # (competition, season) keeping the max season per competition.
    club_prestige_lookup = (
        stats[["player_club_id", "competition", "season", "club_prestige_eur"]]
        .sort_values("season", ascending=False)
        .drop_duplicates(subset=["player_club_id", "competition"])
        [["player_club_id", "competition", "club_prestige_eur"]]
        .reset_index(drop=True)
    )
    # Also build a name-based lookup from players who appeared for each club
    # player_name → club_prestige (latest season they played)
    name_prestige = (
        df.sort_values("season", ascending=False)
        .drop_duplicates(subset=["player_name"])
        [["player_name", "club_prestige_eur"]]
        .reset_index(drop=True)
    )
    name_prestige.to_csv(os.path.join(out_dir, "player_prestige_lookup.csv"), index=False)

    # International caps lookup: one row per player (latest caps value)
    caps_lookup = (
        df.sort_values("season", ascending=False)
        .drop_duplicates(subset=["player_name"])
        [["player_name", "international_caps"]]
        .reset_index(drop=True)
    )
    caps_lookup.to_csv(os.path.join(out_dir, "player_caps_lookup.csv"), index=False)

    # Micro position lookup: one row per player (latest season), micro_position_group
    micro_lookup = (
        df.sort_values("season", ascending=False)
        .drop_duplicates(subset=["player_name"])
        [["player_name", "micro_position_group"]]
        .reset_index(drop=True)
    )
    micro_lookup.to_csv(os.path.join(out_dir, "player_micro_position_lookup.csv"), index=False)

    print(f"\nDone. {len(df):,} training rows written to: {output_path}")
    print(f"  Seasons:      {sorted(df['season'].unique())}")
    print(f"  Competitions: {sorted(df['competition'].unique())}")
    print(f"  Market value: €{df['market_value_eur'].min():,.0f} – €{df['market_value_eur'].max():,.0f}")
    print(f"  Players:      {df['player_id'].nunique():,} unique")
    print(f"  Intl caps range: {df['international_caps'].min()} – {df['international_caps'].max()}")
    print(f"  Lookup tables:  player_prestige_lookup.csv ({len(name_prestige):,} rows)")
    print(f"                  player_caps_lookup.csv     ({len(caps_lookup):,} rows)")
    print(f"                  player_micro_position_lookup.csv ({len(micro_lookup):,} rows)")


if __name__ == "__main__":
    # Paths resolve relative to the repository, not to one machine's home
    # directory. The previous defaults pointed at a Downloads folder, which
    # meant this step could only ever be run by hand on one laptop.
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-dir",
        default=os.path.join(BASE, "data", "external", "archive"),
        help="Directory holding players.csv, appearances.csv, "
             "player_valuations.csv (see src/fetch_tm_archive.py)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE, "data", "raw", "tm_training_data.csv"),
    )
    args = parser.parse_args()
    run(archive_dir=args.archive_dir, output_path=args.output)
