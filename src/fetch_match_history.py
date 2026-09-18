"""
Pulls historical match results for the match-outcome model.

Free tier gives us seasons 2023-2025 across PL / PD / SA -> ~3,420 matches.
Only results are available (no shots/possession/xG), so every feature the
model sees has to be derived from scorelines and scheduling.

Run:  python src/fetch_match_history.py
Out:  data/raw/match_history.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from football_api import FootballDataClient

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(BASE, "data", "raw", "match_history.csv")

# 2020-2022 return 403 on the free tier - these three are all we can reach.
MATCH_SEASONS = [2023, 2024, 2025]
MATCH_COMPS   = ["PL", "PD", "SA"]


def run():
    client = FootballDataClient()
    all_rows = []

    for comp in MATCH_COMPS:
        for season in MATCH_SEASONS:
            print(f"  {comp} {season}...", end=" ", flush=True)
            try:
                rows = client.get_matches(comp, season)
                all_rows.extend(rows)
                print(f"{len(rows)} matches")
            except Exception as e:
                print(f"FAILED: {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("\nNo matches retrieved - check the API key.")
        return

    # Drop anything without a final score (postponed / abandoned fixtures).
    before = len(df)
    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    if len(df) < before:
        print(f"  Dropped {before - len(df)} matches without a final score")

    for c in ["home_goals", "away_goals", "ht_home_goals", "ht_away_goals", "matchday"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["utc_date"] = pd.to_datetime(df["utc_date"])

    # Chronological order matters: every rolling feature and the Elo pass
    # depend on walking matches forward in time.
    df = df.sort_values("utc_date").reset_index(drop=True)

    # Target: 0 = home win, 1 = draw, 2 = away win
    df["result"] = df.apply(
        lambda r: 0 if r["home_goals"] > r["away_goals"]
        else (1 if r["home_goals"] == r["away_goals"] else 2),
        axis=1,
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\n  Saved {len(df):,} matches -> {OUT}")
    print(f"  Date range: {df['utc_date'].min().date()} to {df['utc_date'].max().date()}")
    print(f"  Teams: {pd.concat([df['home_team'], df['away_team']]).nunique()}")

    dist = df["result"].value_counts(normalize=True).sort_index()
    labels = {0: "Home win", 1: "Draw", 2: "Away win"}
    print("\n  Outcome distribution (this is our baseline to beat):")
    for k, v in dist.items():
        print(f"    {labels[k]:9s} {v:6.1%}")


if __name__ == "__main__":
    run()
