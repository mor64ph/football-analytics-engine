"""
Rebuild the match feature matrix from raw football-data.co.uk results.

THE TIER RULE
-------------
Ratings and rolling form are computed over EVERY division, top tier and
second tier together. The model is then trained and scored on the top tier
alone.

That split is deliberate. A side promoted into the Premier League used to
enter the model at the default 1500 rating - "perfectly average for this
league" - which is close to the worst possible prior, since promoted teams
are usually among the weakest in the division. Running the rating system
across both tiers means they arrive carrying a rating earned against real
opponents, and their recent form comes with them too.

Second-tier matches are never predicted or scored. They exist to give the
top tier a better memory.

Run:  python src/build_match_features.py
In :  data/raw/fdcouk_matches.csv
Out:  data/processed/match_features.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from match_features import build_features, all_features, feature_report

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(BASE, "data", "raw", "fdcouk_matches.csv")
OUT = os.path.join(BASE, "data", "processed", "match_features.csv")

pd.set_option("display.width", 240)


def main():
    raw = pd.read_csv(RAW)
    raw["date"] = pd.to_datetime(raw["date"])
    if "tier" not in raw.columns:
        raw["tier"] = 1

    print(f"Raw matches      : {len(raw):,}")
    print(f"  tier 1         : {(raw['tier'] == 1).sum():,}")
    print(f"  tier 2         : {(raw['tier'] == 2).sum():,}")
    print(f"  teams          : {pd.concat([raw['home_team'], raw['away_team']]).nunique():,}")
    print(f"  seasons        : {raw['season'].min()}-{raw['season'].max()}")

    print("\nBuilding features across ALL tiers...")
    df = build_features(raw)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows -> {OUT}")

    # ------------------------------------------------------------------ #
    top = df[df["tier"] == 1]
    print(f"\nModelling set (tier 1): {len(top):,} matches")
    for s in sorted(top["season"].unique()):
        n = (top["season"] == s).sum()
        print(f"    {s}  {n:>5,}")

    # ------------------------------------------------------------------ #
    # Did carrying second-tier history actually change anything? Compare the
    # Elo a promoted side walks in with against the 1500 default it used to
    # get. If these are all 1500 the cross-tier join silently failed.
    print("\n" + "=" * 78)
    print("PROMOTED-TEAM CHECK — do promoted sides arrive with real ratings?")
    print("=" * 78)
    rows = []
    for comp in sorted(top["competition"].unique()):
        c = top[top["competition"] == comp]
        for season in sorted(c["season"].unique())[1:]:
            prev = set(c[c["season"] == season - 1]["home_team"])
            cur = c[c["season"] == season]
            promoted = set(cur["home_team"]) - prev
            for t in promoted:
                first = cur[(cur["home_team"] == t) | (cur["away_team"] == t)].iloc[0]
                elo = (first["elo_home_pre"] if first["home_team"] == t
                       else first["elo_away_pre"])
                rows.append({"comp": comp, "season": season, "team": t,
                             "entry_elo": round(float(elo), 1)})
    pro = pd.DataFrame(rows)
    if len(pro):
        at_default = (pro["entry_elo"] == 1500.0).mean()
        print(f"  promoted sides found      : {len(pro)}")
        print(f"  still at the 1500 default : {at_default:.1%}   "
              f"(was 100% before second tiers were added)")
        print(f"  mean entry Elo            : {pro['entry_elo'].mean():.1f}")
        print(f"  mean top-tier Elo         : "
              f"{top['elo_home_pre'].mean():.1f}")
        print("\n  Sample:")
        print(pro.tail(10).to_string(index=False))

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("LEAKAGE CHECK — correlation of each feature with the outcome")
    print("=" * 78)
    print("  Anything above ~0.5 means a feature is carrying the result it is")
    print("  meant to predict. Elo should top out around 0.35-0.40.\n")
    feats = [f for f in all_features() if f in top.columns]
    cors = (top[feats].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .corrwith(top["result"].astype(float)).abs().sort_values(ascending=False))
    print(cors.head(8).round(4).to_string())
    worst = float(cors.max())
    print(f"\n  max |r| = {worst:.4f}  -> {'PASS' if worst < 0.5 else 'FAIL'}")

    print("\n" + "=" * 78)
    print("FEATURE HEALTH")
    print("=" * 78)
    rep = feature_report(top, feats)
    bad = rep[(rep["status"] == "MISSING") | (rep["null_pct"] > 0.25)]
    if len(bad):
        print("  Features missing or >25% null:")
        print(bad.to_string(index=False))
    else:
        print("  All features present, none above 25% null.")


if __name__ == "__main__":
    main()
