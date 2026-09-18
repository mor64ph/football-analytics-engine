"""
Map Transfermarkt club ids onto football-data.co.uk short names.

THE PROBLEM WITH THE OBVIOUS APPROACH
-------------------------------------
The two sources name clubs completely differently - "Atletico de Madrid" vs
"Ath Madrid", "1. Fussballclub Heidenheim 1846" vs "Heidenheim". String
similarity gets most of them and confidently mismatches the rest; "Real Madrid"
against "Real Sociedad" is the classic. A wrong club mapping silently attaches
one team's lineup to another team's match, which is worse than having no
lineup data at all because nothing about it looks broken downstream.

MATCHING ON EVIDENCE INSTEAD OF SPELLING
----------------------------------------
Both sources describe the same fixtures. A match is identified by the date it
was played and the score it finished - and across a season, the combination of
(date, home goals, away goals) is nearly unique within a division.

So instead of comparing names we line the fixtures up and read the mapping off
them: whenever a Transfermarkt fixture and a football-data fixture agree on
date and scoreline, that is one vote that their home clubs are the same club
and their away clubs are the same club. Votes accumulate over ten seasons, so
a club is identified by dozens of fixtures and the occasional coincidental
score collision is outvoted rather than believed.

The output is checked, not assumed: a mapping only survives if it wins its
club by a clear margin, and coverage is reported so a silent shortfall is
visible.
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXTERNAL = os.path.join(BASE, "data", "external")
RAW = os.path.join(BASE, "data", "raw", "fdcouk_matches.csv")
OUT = os.path.join(EXTERNAL, "club_crosswalk.csv")

# A fixture can be listed a day apart by two sources (late kick-offs crossing
# midnight UTC, or a source recording the scheduled rather than played date).
DATE_SLACK = 1

# A mapping must win its club by at least this share of that club's votes.
MIN_SHARE = 0.60
MIN_VOTES = 5


def build() -> pd.DataFrame:
    tm = pd.read_csv(os.path.join(EXTERNAL, "tm_games.csv"))
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
    tm = tm.dropna(subset=["date", "home_club_goals", "away_club_goals"])

    fd = pd.read_csv(RAW, low_memory=False)
    fd["date"] = pd.to_datetime(fd["date"], errors="coerce")
    fd = fd[fd["tier"] == 1] if "tier" in fd.columns else fd
    fd = fd.dropna(subset=["date", "home_goals", "away_goals"])

    print(f"transfermarkt fixtures : {len(tm):,}")
    print(f"football-data fixtures : {len(fd):,}")

    # Only fixtures whose (date, competition, score) is unique on BOTH sides
    # are allowed to vote.
    #
    # Without this, three 2-1 home wins in the same division on the same day
    # produce nine merge rows, six of which pair unrelated clubs. Those wrong
    # pairs are spread across many club names while the right pair repeats, so
    # the winner survives - but the margin collapses, and genuinely
    # low-appearance clubs end up below any sensible acceptance threshold.
    # Discarding ambiguous keys costs a few fixtures and removes the noise
    # entirely.
    key_tm = ["date", "competition", "home_club_goals", "away_club_goals"]
    key_fd = ["date", "competition", "home_goals", "away_goals"]

    votes: dict[tuple, int] = defaultdict(int)
    matched = 0

    for offset in range(-DATE_SLACK, DATE_SLACK + 1):
        left = tm.copy()
        left["date"] = left["date"] + pd.Timedelta(days=offset)

        uniq_tm = left[~left.duplicated(key_tm, keep=False)]
        uniq_fd = fd[~fd.duplicated(key_fd, keep=False)]

        merged = uniq_tm.merge(uniq_fd, left_on=key_tm, right_on=key_fd,
                               suffixes=("_tm", "_fd"))
        for r in merged.itertuples(index=False):
            votes[(r.home_club_id, r.home_team)] += 1
            votes[(r.away_club_id, r.away_team)] += 1
        matched += len(merged)

    print(f"unambiguous agreements : {matched:,}")

    # Resolve each club id to the short name that wins it outright.
    by_club: dict[int, dict[str, int]] = defaultdict(dict)
    for (club_id, short), n in votes.items():
        by_club[club_id][short] = by_club[club_id].get(short, 0) + n

    rows = []
    for club_id, tally in by_club.items():
        total = sum(tally.values())
        short, n = max(tally.items(), key=lambda kv: kv[1])
        rows.append({
            "club_id": club_id,
            "short_name": short,
            "votes": n,
            "share": round(n / total, 3),
            "accepted": bool(n >= MIN_VOTES and n / total >= MIN_SHARE),
        })

    return pd.DataFrame(rows).sort_values("votes", ascending=False)


def main():
    cw = build()
    accepted = cw[cw["accepted"]]

    os.makedirs(EXTERNAL, exist_ok=True)
    accepted[["club_id", "short_name", "votes", "share"]].to_csv(OUT, index=False)

    print(f"\nclubs resolved : {len(accepted)} of {len(cw)}")
    print(f"saved -> {OUT}")

    # A short name claimed by two club ids means the vote failed somewhere.
    dupes = accepted["short_name"].value_counts()
    dupes = dupes[dupes > 1]
    if len(dupes):
        print(f"\nWARNING: {len(dupes)} short name(s) claimed by multiple clubs:")
        print(dupes.to_string())

    rejected = cw[~cw["accepted"]]
    if len(rejected):
        print(f"\nrejected (too few or too ambiguous): {len(rejected)}")
        print(rejected.head(10).to_string(index=False))

    print("\nSample:")
    print(accepted.head(12).to_string(index=False))

    # Does the crosswalk actually cover the teams we predict?
    fd = pd.read_csv(RAW, low_memory=False)
    fd = fd[fd["tier"] == 1] if "tier" in fd.columns else fd
    need = set(fd["home_team"]) | set(fd["away_team"])
    have = set(accepted["short_name"])
    missing = sorted(need - have)
    print(f"\ntop-tier teams covered: {len(need & have)} / {len(need)}")
    if missing:
        print(f"  unmapped: {missing[:20]}")


if __name__ == "__main__":
    main()
