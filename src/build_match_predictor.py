"""
Fit the shipped fixture model and write it to disk for the API.

Reads the config that src/optimise_accuracy.py selected on validation, fits
Elo, the per-league calibration and Dixon-Coles on the full history, then
pickles the result. The API loads that pickle at startup; nothing is fitted
at request time.

Run:  python src/build_match_predictor.py
Out:  data/models/match_predictor.pkl
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from match_predictor import MatchPredictor, STATE

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(BASE, "data", "raw", "fdcouk_matches.csv")


def main():
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Fitting on {len(df):,} matches "
          f"({df['date'].min().date()} to {df['date'].max().date()})")

    m = MatchPredictor.build(df)
    path = m.save(STATE)
    size = os.path.getsize(path) / 1024
    print(f"\nSaved -> {path}  ({size:.0f} KB)")

    print("\n" + "=" * 70)
    print("Top rated teams")
    print("=" * 70)
    print(pd.DataFrame(m.ratings(top=12)).to_string(index=False))

    print("\n" + "=" * 70)
    print("Calibration actually in use")
    print("=" * 70)
    print(m.calibrator.table().to_string(index=False))

    print("\n" + "=" * 70)
    print("Sample predictions")
    print("=" * 70)
    for h, a in [("Man City", "Liverpool"),
                 ("Arsenal", "Everton"),
                 ("Barcelona", "Real Madrid"),
                 ("Bayern Munich", "Dortmund")]:
        if not (m.known(h) and m.known(a)):
            print(f"\n  {h} vs {a}: unknown team, skipped")
            continue
        p = m.predict(h, a)
        pr, xg, g = p["probabilities"], p["expected_goals"], p["goals_markets"]
        print(f"\n  {h} vs {a}  [{p['competition']}]")
        print(f"    {pr['home_win']:.1%} / {pr['draw']:.1%} / {pr['away_win']:.1%}"
              f"   xG {xg['home']:.2f}-{xg['away']:.2f}"
              f"   elo {p['elo']['home']:.0f} v {p['elo']['away']:.0f}")
        print(f"    o2.5 {g['over_2_5']:.1%}   BTTS {g['btts']:.1%}   "
              f"CS home {g['home_clean_sheet']:.1%}")
        print("    scores " + ", ".join(
            f"{s['score']} {s['probability']:.1%}" for s in p["top_scores"][:5]))

        # The rescaling in predict() is only correct if the grid still sums to
        # the headline probabilities. Check rather than assume.
        grid_home = sum(
            p["scoreline_grid"][h_][a_]
            for h_ in range(6) for a_ in range(6) if h_ > a_
        )
        print(f"    grid home-win mass {grid_home:.3f} vs headline "
              f"{pr['home_win']:.3f}  (truncated at 5 goals)")


if __name__ == "__main__":
    main()
