"""
Compute dynamic player values locally and append a snapshot to the history.

WHY THIS EXISTS
---------------
Form multipliers, dynamic values and the VALUE_HISTORY table were only ever
produced inside the Databricks notebook, so the Value Movers page went blank
the moment the warehouse was unavailable - which is now permanent, since the
workspace has been marked INACTIVE by Databricks' resource gatekeeper and no
compute of any kind can start.

None of that work actually needs a cluster. It is a few hundred rows, one API
call per competition, and some arithmetic. Running it locally makes the whole
product independent of a warehouse that may never come back, and the Databricks
path still takes precedence whenever it is reachable.

WHAT IT WRITES
--------------
  data/processed/gold_current_season_scores.csv   enriched in place
  data/processed/value_history.csv                one snapshot appended per run

Day-over-day and week-over-week movement need at least two snapshots on
different dates, so the first run produces a leaderboard with no deltas. That
is expected, not a failure - run it again tomorrow.

Run:  python src/build_local_dynamic_values.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import datetime as dt

import pandas as pd
from dotenv import load_dotenv

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE, ".env"))

from form_engine import FormEngine

PROCESSED = os.path.join(BASE, "data", "processed")
CURRENT = os.path.join(PROCESSED, "gold_current_season_scores.csv")
HISTORY = os.path.join(PROCESSED, "value_history.csv")

HISTORY_COLS = [
    "snapshot_date", "player_name", "team_name", "competition", "position",
    "age", "predicted_value_eur", "form_multiplier", "contract_modifier",
    "form_score", "goals_recent", "assists_recent", "matches_played_recent",
    "dynamic_value_eur", "value_delta_pct",
]


def main():
    if not os.path.exists(CURRENT):
        print(f"Missing {CURRENT}. Run the pipeline first.")
        return

    df = pd.read_csv(CURRENT)
    print(f"Loaded {len(df):,} current-season players")

    # Deliberately not pinned to the season in the CSV. That column reflects
    # when the value model was last trained, whereas form has to describe what
    # is happening now - letting the provider pick the current season is the
    # whole point of the feature.
    print("Fetching current-season scoring form...")

    engine = FormEngine()
    try:
        form = engine.fetch_contributions()
    except Exception as e:
        print(f"  form fetch failed ({e}); falling back to neutral form")
        form = {}

    enriched = engine.apply_dynamic_values(df, form)
    enriched.to_csv(CURRENT, index=False)
    print(f"Wrote {len(enriched):,} rows -> {CURRENT}")

    n_with_form = int((enriched["form_multiplier"] != 1.0).sum())
    print(f"  players with real form data : {n_with_form:,}")
    print(f"  form multiplier range       : "
          f"{enriched['form_multiplier'].min():.3f} - {enriched['form_multiplier'].max():.3f}")
    print(f"  dynamic vs base value       : "
          f"{enriched['value_delta_pct'].mean():+.2f}% mean")

    # ------------------------------------------------------------------ #
    today = dt.date.today().isoformat()
    snap = enriched.copy()
    snap["snapshot_date"] = today
    for c in HISTORY_COLS:
        if c not in snap.columns:
            snap[c] = None
    snap = snap[HISTORY_COLS]

    if os.path.exists(HISTORY):
        hist = pd.read_csv(HISTORY)
        # Re-running on the same day replaces that day rather than duplicating it.
        hist = hist[hist["snapshot_date"] != today]
        hist = pd.concat([hist, snap], ignore_index=True)
    else:
        hist = snap

    hist.to_csv(HISTORY, index=False)
    dates = sorted(hist["snapshot_date"].unique())
    print(f"\nHistory -> {HISTORY}")
    print(f"  {len(hist):,} rows across {len(dates)} snapshot date(s)")
    print(f"  {dates[0]} .. {dates[-1]}")
    if len(dates) < 2:
        print("\n  Only one snapshot so far, so Value Movers will show no")
        print("  deltas yet. Run this again on a later date to populate them.")
    else:
        print(f"\n  Movement available between {dates[-2]} and {dates[-1]}.")


if __name__ == "__main__":
    main()
