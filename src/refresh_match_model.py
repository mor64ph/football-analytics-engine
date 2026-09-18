"""
Bring the match model up to date with results played since it was last built.

WHY THIS HAS TO BE AUTOMATED
----------------------------
Elo ratings, Dixon-Coles attack/defence and every rolling form feature are
baked into a pickle at build time. Nothing about serving a prediction updates
them. So without a scheduled refresh the model keeps answering confidently
with ratings from whenever it was last built - and because the answers still
look plausible, nothing surfaces the staleness. That is exactly how this
project ended up shipping predictions for the 2026/27 season computed from
ratings frozen on 2026-05-31.

Only the season in progress is re-downloaded. Completed seasons are immutable,
so re-fetching all of them would be 110 requests to confirm that 100 files
have not changed.

Run:  python src/refresh_match_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from fetch_fdcouk import refresh_current_season, OUT as RAW
from match_features import build_features
from match_predictor import MatchPredictor, STATE

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATS = os.path.join(BASE, "data", "processed", "match_features.csv")

# Teams to report before/after, purely so a human can see the refresh did
# something. Any well-known side works; these just have to exist.
WATCH = ["Man City", "Arsenal", "Liverpool", "Barcelona", "Real Madrid",
         "Bayern Munich", "Inter", "Paris SG"]


def _ratings_snapshot() -> dict:
    if not os.path.exists(STATE):
        return {}
    try:
        m = MatchPredictor.load(STATE)
        return {t: round(float(m.elo.ratings[t]), 1)
                for t in WATCH if t in m.elo.ratings}
    except Exception:
        return {}


def main():
    before = _ratings_snapshot()

    print("Refreshing the current season...")
    n = refresh_current_season()

    raw = pd.read_csv(RAW, low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"])
    print(f"  data now covers {raw['date'].min().date()} .. {raw['date'].max().date()}"
          f"  ({len(raw):,} matches)")

    print("\nRebuilding features across all tiers...")
    feats = build_features(raw)
    os.makedirs(os.path.dirname(FEATS), exist_ok=True)
    feats.to_csv(FEATS, index=False)
    print(f"  {len(feats):,} rows -> {FEATS}")

    print("\nRefitting the match predictor...")
    m = MatchPredictor.build(raw, verbose=True)
    m.save(STATE)
    print(f"  saved -> {STATE}")

    # ------------------------------------------------------------------ #
    after = {t: round(float(m.elo.ratings[t]), 1)
             for t in WATCH if t in m.elo.ratings}
    if before:
        print("\nRating movement since the last build:")
        for t in WATCH:
            if t in before and t in after:
                d = after[t] - before[t]
                flag = "  (unchanged)" if abs(d) < 0.05 else ""
                print(f"    {t:<16} {before[t]:>7.1f} -> {after[t]:>7.1f}  "
                      f"{d:+6.1f}{flag}")
        if all(abs(after[t] - before[t]) < 0.05 for t in after if t in before):
            print("\n  Nothing moved. Either no new results have been played,")
            print("  or the current season is not being downloaded at all.")
    else:
        print("\n  No previous model to compare against.")

    print(f"\n  current-season matches held: {n:,}")
    print(f"  model trained to            : {str(m.trained_to)[:10]}")


if __name__ == "__main__":
    main()
