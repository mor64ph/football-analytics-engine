"""
Squad availability features from Transfermarkt lineups and injury spells.

WHAT THIS IS TRYING TO CAPTURE
------------------------------
Everything the model currently knows is derived from results. Results tell you
how good a team has been; they do not tell you that the team playing on
Saturday is missing its first-choice striker, centre-back and goalkeeper. That
is the information the market has and we do not, and it is the reason the
bookmaker sits 0.032 log loss ahead.

THE MEASURE
-----------
Not "how many players are injured" - that counts a fourth-choice full-back the
same as a captain. Instead each player carries an IMPORTANCE: the share of his
club's last ten matches that he started. A nailed-on starter approaches 1.0, a
squad player sits near 0.1. Availability is then the importance-weighted share
of the usual XI that is actually starting today.

A team missing two 0.9-importance players scores far worse than one missing
two 0.2-importance players, which is the distinction that matters and the one
a raw headcount throws away.

LEAKAGE, AND THE ONE THING TO BE CAREFUL ABOUT
----------------------------------------------
Importance is computed from matches STRICTLY BEFORE the one being predicted -
the rolling window is advanced only after a match has been used. Nothing about
a player's importance can reflect the match it is a feature for.

Today's starting XI is a different matter and deserves stating plainly. It is
not leakage - a teamsheet is published about an hour before kick-off and
contains no information about the result - but it is only available in that
final hour. Our benchmark is Pinnacle CLOSING odds, which are also set after
team news, so measuring against it is a fair comparison. It does mean a
forecast published days ahead cannot use these features, and would need
predicted lineups instead. That limit is real and is not fixed by anything
here.

Run:  python src/availability_features.py
Out:  data/processed/availability.csv
"""

import os
import sys
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXTERNAL = os.path.join(BASE, "data", "external")
OUT = os.path.join(BASE, "data", "processed", "availability.csv")

WINDOW = 10          # club matches used to establish who the regulars are
KEY_THRESHOLD = 0.6  # importance at which a player counts as a regular starter


def load():
    games = pd.read_csv(os.path.join(EXTERNAL, "tm_games.csv"))
    games["date"] = pd.to_datetime(games["date"], errors="coerce")

    lineups = pd.read_csv(os.path.join(EXTERNAL, "tm_lineups.csv"),
                          low_memory=False)
    cross = pd.read_csv(os.path.join(EXTERNAL, "club_crosswalk.csv"))
    injuries = pd.read_csv(os.path.join(EXTERNAL, "tm_injuries.csv"))
    injuries["from_date"] = pd.to_datetime(injuries["from_date"], errors="coerce")
    injuries["end_date"] = pd.to_datetime(injuries["end_date"], errors="coerce")
    return games, lineups, cross, injuries


def build() -> pd.DataFrame:
    games, lineups, cross, injuries = load()

    starters = lineups[lineups["type"] == "starting_lineup"][
        ["game_id", "club_id", "player_id"]]

    # One row per club per match, with its date, so each club can be walked
    # forward in time independently.
    home = games[["game_id", "date", "home_club_id"]].rename(
        columns={"home_club_id": "club_id"})
    away = games[["game_id", "date", "away_club_id"]].rename(
        columns={"away_club_id": "club_id"})
    club_games = (pd.concat([home, away], ignore_index=True)
                  .dropna(subset=["date"])
                  .sort_values(["club_id", "date"]))

    xi = (starters.groupby(["game_id", "club_id"])["player_id"]
          .apply(set).to_dict())

    rows = []
    for club_id, grp in club_games.groupby("club_id", sort=False):
        # Last WINDOW starting XIs for this club, oldest first.
        history: deque[set] = deque(maxlen=WINDOW)

        for r in grp.itertuples(index=False):
            today = xi.get((r.game_id, club_id), set())

            if history and today:
                counts: dict = defaultdict(int)
                for past in history:
                    for p in past:
                        counts[p] += 1
                n = len(history)
                importance = {p: c / n for p, c in counts.items()}

                # The XI we would expect: the eleven most established players.
                expected = sorted(importance.values(), reverse=True)[:11]
                expected_total = sum(expected) or 1.0

                present_total = sum(importance.get(p, 0.0) for p in today)
                regulars = {p for p, v in importance.items()
                            if v >= KEY_THRESHOLD}

                rows.append({
                    "game_id": r.game_id,
                    "club_id": club_id,
                    "date": r.date,
                    # Capped at 1: a side can field more established players
                    # than the top eleven when the window is unusually settled.
                    "availability": min(present_total / expected_total, 1.0),
                    "key_absent": len(regulars - today),
                    "n_regulars": len(regulars),
                    "xi_continuity": (len(today & history[-1]) / 11.0
                                      if history[-1] else np.nan),
                    "squad_known": 1,
                })
            else:
                rows.append({
                    "game_id": r.game_id, "club_id": club_id, "date": r.date,
                    "availability": np.nan, "key_absent": np.nan,
                    "n_regulars": np.nan, "xi_continuity": np.nan,
                    "squad_known": 0,
                })

            # Advance the window only AFTER this match has been described, so
            # importance never contains the match it is a feature for.
            if today:
                history.append(today)

    feat = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Injury load: how many of a club's recent starters were mid-injury on the
    # day. Independent of the teamsheet, so this part survives even when a
    # lineup is unavailable.
    print("  joining injury spells...")
    recent = (starters.merge(games[["game_id", "date"]], on="game_id")
              .sort_values("date"))
    squad = recent.groupby("club_id")["player_id"].apply(set).to_dict()

    inj = injuries.dropna(subset=["from_date"]).copy()
    inj["end_date"] = inj["end_date"].fillna(inj["from_date"] + pd.Timedelta(days=30))
    by_player = defaultdict(list)
    for r in inj.itertuples(index=False):
        by_player[r.player_id].append((r.from_date, r.end_date))

    counts = []
    for r in feat.itertuples(index=False):
        members = squad.get(r.club_id, set())
        if not members:
            counts.append(np.nan)
            continue
        d = r.date
        n = 0
        for p in members:
            for start, end in by_player.get(p, ()):
                if start <= d <= end:
                    n += 1
                    break
        counts.append(n)
    feat["injured_squad"] = counts

    return feat


def main():
    feat = build()
    cross = pd.read_csv(os.path.join(EXTERNAL, "club_crosswalk.csv"))
    games = pd.read_csv(os.path.join(EXTERNAL, "tm_games.csv"))
    games["date"] = pd.to_datetime(games["date"], errors="coerce")

    feat = feat.merge(cross[["club_id", "short_name"]], on="club_id", how="left")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    feat.to_csv(OUT, index=False)

    print(f"\n{len(feat):,} club-match rows -> {OUT}")
    print(f"  with a known squad : {int(feat['squad_known'].sum()):,} "
          f"({feat['squad_known'].mean():.1%})")
    print(f"  mapped to a name   : {int(feat['short_name'].notna().sum()):,}")

    print("\nFeature summary:")
    print(feat[["availability", "key_absent", "xi_continuity",
                "injured_squad"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
