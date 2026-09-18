"""
Download Transfermarkt lineups and injury histories, filtered to our leagues.

SOURCE AND LICENCE
------------------
dcaribou/transfermarkt-datasets, published CC0 to a public Cloudflare R2
bucket. No key, no Kaggle login, no scraping - we are downloading a dataset
that exists to be downloaded. Injury histories come from salimt/football-
datasets, also Transfermarkt-derived.

WHY THIS FILTERS WHILE IT READS
-------------------------------
game_lineups is 126MB gzipped and roughly 400MB expanded - about 3.2 million
rows covering every competition the source tracks, back to 2012. Loading that
to then discard 90% of it would need more memory than the machine should be
asked for, and far more than the 512MB the API host allows.

So games.csv is read first (small), reduced to the top-five-league fixtures we
actually model, and its game_id set is used to filter the lineup file chunk by
chunk. Peak memory stays in the low hundreds of MB and the result is a few
hundred thousand rows instead of millions.

KNOWN COVERAGE LIMIT
--------------------
Lineups run to 2026-05-24 and injuries stop at 2025-12-22. Neither covers the
2026/27 season now in progress. That is fine for measuring whether
availability helps historically, and it is NOT enough to serve live
predictions - doing that needs a current team-news feed, which is a paid
product. Worth knowing before building anything on top of this.

Run:  python src/fetch_lineups.py
Out:  data/external/tm_{games,lineups,injuries,players}.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import requests

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE, "data", "external")

R2 = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
INJURIES_URL = ("https://raw.githubusercontent.com/salimt/football-datasets/"
                "master/datalake/transfermarkt/player_injuries/player_injuries.csv")

# Transfermarkt competition ids for the divisions we model.
COMP_MAP = {
    "GB1": "PL",    # Premier League
    "ES1": "PD",    # La Liga
    "IT1": "SA",    # Serie A
    "L1":  "BL1",   # Bundesliga
    "FR1": "FL1",   # Ligue 1
}

FIRST_SEASON = 2016
CHUNK = 400_000


def _download(name: str) -> str:
    """Fetch a gzipped table to disk, skipping it if already present."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.csv.gz")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"  {name:<14} cached ({os.path.getsize(path) / 1e6:.1f} MB)")
        return path

    url = f"{R2}/{name}.csv.gz"
    print(f"  {name:<14} downloading...", end=" ", flush=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    print(f"{os.path.getsize(path) / 1e6:.1f} MB")
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Downloading source tables...")
    games_gz = _download("games")
    clubs_gz = _download("clubs")
    players_gz = _download("players")

    # ------------------------------------------------------------------ #
    print("\nFiltering fixtures to the top five leagues...")
    games = pd.read_csv(games_gz, compression="gzip", low_memory=False)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games[
        games["competition_id"].isin(COMP_MAP)
        & (games["season"] >= FIRST_SEASON)
    ].dropna(subset=["date"]).copy()
    games["competition"] = games["competition_id"].map(COMP_MAP)

    keep = ["game_id", "competition", "season", "date",
            "home_club_id", "away_club_id", "home_club_name", "away_club_name",
            "home_club_goals", "away_club_goals",
            "home_club_formation", "away_club_formation"]
    games = games[[c for c in keep if c in games.columns]]
    games.to_csv(os.path.join(OUT_DIR, "tm_games.csv"), index=False)

    print(f"  {len(games):,} fixtures  "
          f"{games['date'].min().date()} .. {games['date'].max().date()}")
    print(games.groupby("competition").size().to_string())

    wanted = set(games["game_id"])

    # ------------------------------------------------------------------ #
    print("\nStreaming lineups (large file, filtered per chunk)...")
    lineups_gz = _download("game_lineups")

    kept, seen = [], 0
    for chunk in pd.read_csv(lineups_gz, compression="gzip",
                             chunksize=CHUNK, low_memory=False):
        seen += len(chunk)
        sub = chunk[chunk["game_id"].isin(wanted)]
        if len(sub):
            kept.append(sub)
        print(f"    read {seen:>9,}  kept {sum(len(k) for k in kept):>8,}",
              end="\r", flush=True)

    lineups = pd.concat(kept, ignore_index=True)
    lineups.to_csv(os.path.join(OUT_DIR, "tm_lineups.csv"), index=False)
    print(f"\n  {len(lineups):,} lineup rows from {seen:,} scanned")
    print(lineups["type"].value_counts().to_string())

    # ------------------------------------------------------------------ #
    print("\nPlayers...")
    players = pd.read_csv(players_gz, compression="gzip", low_memory=False)
    pkeep = ["player_id", "name", "position", "sub_position",
             "date_of_birth", "market_value_in_eur",
             "highest_market_value_in_eur"]
    players[[c for c in pkeep if c in players.columns]].to_csv(
        os.path.join(OUT_DIR, "tm_players.csv"), index=False)
    print(f"  {len(players):,} players")

    # ------------------------------------------------------------------ #
    print("\nInjuries...")
    inj = pd.read_csv(INJURIES_URL, low_memory=False)
    inj["from_date"] = pd.to_datetime(inj["from_date"], errors="coerce")
    inj["end_date"] = pd.to_datetime(inj["end_date"], errors="coerce")
    inj = inj.dropna(subset=["from_date"])
    inj = inj[inj["from_date"] >= f"{FIRST_SEASON}-07-01"]
    inj.to_csv(os.path.join(OUT_DIR, "tm_injuries.csv"), index=False)
    print(f"  {len(inj):,} injury spells  "
          f"{inj['from_date'].min().date()} .. {inj['from_date'].max().date()}")
    print(f"  ongoing (no end date): {int(inj['end_date'].isna().sum()):,}")

    # ------------------------------------------------------------------ #
    clubs = pd.read_csv(clubs_gz, compression="gzip", low_memory=False)
    clubs[["club_id", "name", "domestic_competition_id"]].to_csv(
        os.path.join(OUT_DIR, "tm_clubs.csv"), index=False)
    print(f"\n  {len(clubs):,} clubs")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
