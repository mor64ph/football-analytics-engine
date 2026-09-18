"""
Downloads historical match data from football-data.co.uk.

These are free CSV files published for exactly this purpose - no scraping,
no auth, no rate limit. They give us three things the football-data.org free
tier does not:

  1. Ten seasons instead of three  (11,400 matches vs 3,420)
  2. Shot / corner / foul / card counts
  3. Bookmaker odds

A note on the odds columns, because it matters for how the model gets built:
they are a BENCHMARK, not a feature. Odds already encode everything the market
knows, so feeding them in produces a model that scores well and contributes
nothing - it has only learned to copy the bookmaker. We hold them out and use
them to answer the only question that matters: can our features beat the market?

Run:  python src/fetch_fdcouk.py
Out:  data/raw/fdcouk_matches.csv
"""

import io
import os
import sys
import time
from datetime import date

import requests
import urllib3
import pandas as pd
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
OUT = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "data", "raw", "fdcouk_matches.csv",
)

# football-data.co.uk division code -> (our competition code, tier, country).
# Codes match football-data.org's where one exists, so the sources can be joined.
#
# WHY THE SECOND TIERS ARE HERE
# -----------------------------
# A promoted side currently enters the model at the default 1500 rating, which
# says "exactly average for this league". That is badly wrong - promoted teams
# are usually among the weakest in the division, and the model spends half a
# season discovering it. Carrying second-tier history across the promotion
# fixes that: the team arrives with a rating earned against real opponents.
#
# The two tiers become comparable automatically. Promoted and relegated sides
# move between divisions every year carrying their ratings with them, and those
# exchanges are what calibrate the level gap - no explicit offset required.
LEAGUES = {
    "E0":  ("PL",  1, "England"),
    "E1":  ("ELC", 2, "England"),
    "SP1": ("PD",  1, "Spain"),
    "SP2": ("SD",  2, "Spain"),
    "I1":  ("SA",  1, "Italy"),
    "I2":  ("SB",  2, "Italy"),
    "D1":  ("BL1", 1, "Germany"),
    "D2":  ("BL2", 2, "Germany"),
    "F1":  ("FL1", 1, "France"),
    "F2":  ("FL2", 2, "France"),
}

# The product and every evaluation report cover the top tier only. Second-tier
# matches exist to feed the rating system, not to be predicted.
TOP_TIER = [v[0] for v in LEAGUES.values() if v[1] == 1]

FIRST_SEASON = 2016


def _current_season_start(today=None) -> int:
    """
    Calendar year in which the season now being played began.

    European leagues run August to May, so July onwards belongs to the season
    starting that year and anything earlier belongs to the one before.
    """
    d = today or date.today()
    return d.year if d.month >= 7 else d.year - 1


def _season_codes(today=None) -> list[str]:
    """
    Every season from FIRST_SEASON through the one in progress.

    Generated rather than hardcoded, and that is the entire point. A literal
    list stops growing the moment nobody remembers to edit it, and the failure
    is silent: ratings freeze at the previous May while the product carries on
    serving confident predictions built from them. This project shipped exactly
    that bug - the model spent the opening weeks of 2026/27 answering with
    ratings last updated on 2026-05-31. A list that extends itself cannot.
    """
    return [f"{str(y)[2:]}{str(y + 1)[2:]}"
            for y in range(FIRST_SEASON, _current_season_start(today) + 1)]


SEASONS = _season_codes()

# Raw column -> our name. Anything not listed is dropped; the source files
# carry 100+ columns, most of them alternative bookmakers we do not need.
COLUMN_MAP = {
    "Date":     "date",
    "Time":     "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG":     "home_goals",
    "FTAG":     "away_goals",
    "FTR":      "ft_result",
    "HTHG":     "ht_home_goals",
    "HTAG":     "ht_away_goals",
    "HTR":      "ht_result",
    "Referee":  "referee",
    "HS":       "home_shots",
    "AS":       "away_shots",
    "HST":      "home_shots_target",
    "AST":      "away_shots_target",
    "HC":       "home_corners",
    "AC":       "away_corners",
    "HF":       "home_fouls",
    "AF":       "away_fouls",
    "HY":       "home_yellows",
    "AY":       "away_yellows",
    "HR":       "home_reds",
    "AR":       "away_reds",
    # Odds: B365 = Bet365 opening, Avg = market average,
    # PSC = Pinnacle CLOSING (sharpest - closing lines absorb late team news).
    "B365H":    "odds_b365_home",
    "B365D":    "odds_b365_draw",
    "B365A":    "odds_b365_away",
    "AvgH":     "odds_avg_home",
    "AvgD":     "odds_avg_draw",
    "AvgA":     "odds_avg_away",
    "PSCH":     "odds_close_home",
    "PSCD":     "odds_close_draw",
    "PSCA":     "odds_close_away",
}

NUMERIC = [c for c in COLUMN_MAP.values()
           if c not in ("date", "time", "home_team", "away_team",
                        "ft_result", "ht_result", "referee")]


def _season_label(code: str) -> int:
    """'2324' -> 2023 (the calendar year the season started)."""
    return 2000 + int(code[:2])


def fetch_one(season_code: str, div: str, retries: int = 4) -> pd.DataFrame:
    url = f"{BASE_URL}/{season_code}/{div}.csv"

    # A hundred requests in a row is enough for a transient DNS or connection
    # reset to show up, and a silently missing season leaves a hole in the
    # rating history rather than an obvious error. Back off and try again.
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, verify=False)
            # A 404 is an answer, not a failure to get one: the division has
            # not kicked off yet. Retrying with backoff would just spend seven
            # seconds per division confirming it every run.
            if r.status_code == 404:
                raise FileNotFoundError(f"not published yet: {season_code}/{div}")
            r.raise_for_status()
            break
        except FileNotFoundError:
            raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    raw = pd.read_csv(
        io.StringIO(r.content.decode("utf-8", errors="replace")),
        on_bad_lines="skip",
    )

    keep = {k: v for k, v in COLUMN_MAP.items() if k in raw.columns}
    df = raw[list(keep)].rename(columns=keep).copy()

    comp, tier, country = LEAGUES[div]
    df["competition"] = comp
    df["tier"] = tier
    df["country"] = country
    df["season"] = _season_label(season_code)
    return df


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Parse, type and label raw rows. Shared by the full and incremental paths."""
    # Rows with no team name are trailing blanks in the source files. The
    # current season also carries scheduled-but-unplayed fixtures with empty
    # goal columns, and those must not become 0-0 draws.
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()

    # Source dates are DD/MM/YYYY (and DD/MM/YY in older files).
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()

    # Target encoding, consistent with the rest of the project:
    #   0 = home win, 1 = draw, 2 = away win
    df["result"] = np.select(
        [df["home_goals"] > df["away_goals"],
         df["home_goals"] == df["away_goals"]],
        [0, 1],
        default=2,
    )
    return df


def run():
    frames = []
    for div in LEAGUES:
        for code in SEASONS:
            label = f"{LEAGUES[div][0]} {_season_label(code)}"
            print(f"  {label:<10}", end=" ", flush=True)
            try:
                df = fetch_one(code, div)
                frames.append(df)
                print(f"{len(df):>4} matches")
            except FileNotFoundError:
                print("not published yet")
            except Exception as e:
                print(f"FAILED: {e}")
            time.sleep(0.3)   # be polite even though there is no stated limit

    if not frames:
        print("\nNothing downloaded.")
        return

    df = _clean_frame(pd.concat(frames, ignore_index=True))

    # Chronological order is load-bearing: Elo and every rolling feature walk
    # this frame forward in time and must never see a future row.
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    # ---------------------------------------------------------------- #
    print(f"\n  Saved {len(df):,} matches -> {OUT}")
    print(f"  Date range : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Seasons    : {df['season'].nunique()}  ({df['season'].min()}-{df['season'].max()})")
    print(f"  Teams      : {pd.concat([df['home_team'], df['away_team']]).nunique()}")

    top = df[df["tier"] == 1]
    print(f"  Tier 1     : {len(top):,} matches  (the product / evaluation set)")
    print(f"  Tier 2     : {len(df) - len(top):,} matches  (rating continuity only)")

    print("\n  Per competition:")
    per = (df.groupby(["tier", "competition"])
             .agg(matches=("result", "size"),
                  draws=("result", lambda s: (s == 1).mean()),
                  home_win=("result", lambda s: (s == 0).mean()))
             .reset_index())
    for r in per.itertuples(index=False):
        print(f"    T{r.tier} {r.competition:<4} {r.matches:>6,}   "
              f"draws {r.draws:5.1%}   home wins {r.home_win:5.1%}")

    print("\n  Outcome distribution:")
    labels = {0: "Home win", 1: "Draw", 2: "Away win"}
    for k, v in df["result"].value_counts(normalize=True).sort_index().items():
        print(f"    {labels[k]:9s} {v:6.1%}")

    print("\n  Column coverage (% non-null):")
    for c in ["home_shots", "home_shots_target", "home_corners", "home_fouls",
              "odds_b365_home", "odds_avg_home", "odds_close_home"]:
        if c in df.columns:
            print(f"    {c:<22s} {df[c].notna().mean():6.1%}")


def refresh_current_season(season_code: str | None = None) -> int:
    """
    Re-download only the season in progress and splice it into the saved file.

    Completed seasons never change, so a daily job has no reason to pull all
    ten of them - that is 110 requests to discover that 100 of them are
    byte-identical to yesterday. This fetches the current season across every
    division and replaces just those rows.

    Returns the number of rows now held for that season.
    """
    code = season_code or SEASONS[-1]
    season = _season_label(code)

    frames = []
    for div in LEAGUES:
        try:
            frames.append(fetch_one(code, div))
        except FileNotFoundError:
            print(f"  {LEAGUES[div][0]:<4} not published yet")
        except Exception as e:
            print(f"  {LEAGUES[div][0]:<4} FAILED: {str(e)[:70]}")
        time.sleep(0.3)

    if not frames:
        print(f"  Nothing available for {season} yet — leaving data unchanged.")
        return 0

    fresh = _clean_frame(pd.concat(frames, ignore_index=True))

    if os.path.exists(OUT):
        old = pd.read_csv(OUT, low_memory=False)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        old = old[old["season"] != season]
        combined = pd.concat([old, fresh], ignore_index=True)
    else:
        combined = fresh

    combined = combined.sort_values(["date", "home_team"]).reset_index(drop=True)
    combined.to_csv(OUT, index=False)

    n = int((combined["season"] == season).sum())
    print(f"  {season}: {n:,} matches "
          f"(total {len(combined):,}, latest {combined['date'].max().date()})")
    return n


if __name__ == "__main__":
    run()
