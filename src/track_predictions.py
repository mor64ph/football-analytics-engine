"""
Record predictions before kick-off, then score them once results land.

WHY NOT LOG AT REQUEST TIME
---------------------------
The obvious design is to append a row whenever the API serves a prediction.
It is also the wrong one here, for two reasons.

First, hosting: Render's free tier has an ephemeral filesystem, so anything
written during a request disappears on the next redeploy or idle restart. A
log that silently empties itself is worse than no log, because it looks like
the model was never wrong.

Second, sampling: request-time logging records whatever users happened to
click on, which is a biased sample of fixtures and cannot be compared against
a benchmark computed over all matches.

So this runs on a schedule instead. It snapshots a prediction for every
upcoming fixture, and on later runs fills in the actual result for anything
that has since been played. The output is a file the scheduled job commits, so
it survives redeploys and accumulates a genuine out-of-sample record.

THE POINT OF ALL THIS
---------------------
Backtesting said 0.9932 log loss on held-out seasons. That is a claim about the
future made from the past. This is the only thing that checks it against
matches the model had never seen at the time it spoke - the difference between
a validated model and a model that validated well once.

Run:  python src/track_predictions.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import datetime as dt

import numpy as np
import pandas as pd

from match_predictor import MatchPredictor, STATE
from match_metrics import log_loss_multi, brier_multi, accuracy

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(BASE, "data", "raw", "fdcouk_matches.csv")
LOG = os.path.join(BASE, "data", "processed", "prediction_log.csv")

# A fixture can be moved for TV or weather. When looking for the result of a
# prediction we allow this much drift before giving up on the match.
DATE_TOLERANCE_DAYS = 6

COLS = [
    "logged_at", "kickoff_date", "competition", "home_team", "away_team",
    "p_home", "p_draw", "p_away", "elo_home", "elo_away",
    "xg_home", "xg_away", "model_trained_to",
    "result", "home_goals", "away_goals", "scored_at",
]


def _load_log() -> pd.DataFrame:
    if os.path.exists(LOG):
        df = pd.read_csv(LOG)
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def fetch_upcoming(days: int = 10) -> list[dict]:
    """Scheduled fixtures from football-data.org, restricted to our leagues."""
    import requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, ".env"))

    token = os.getenv("FOOTBALL_DATA_API_KEY", "")
    if not token:
        print("  FOOTBALL_DATA_API_KEY not set — cannot log new fixtures.")
        return []

    m = MatchPredictor.load(STATE)
    codes = ",".join(sorted(m.calibrator.params))
    today = dt.date.today()
    try:
        r = requests.get(
            "https://api.football-data.org/v4/matches",
            headers={"X-Auth-Token": token},
            params={"dateFrom": today.isoformat(),
                    "dateTo": (today + dt.timedelta(days=days)).isoformat(),
                    "competitions": codes},
            timeout=45,
        )
        r.raise_for_status()
        return r.json().get("matches", [])
    except Exception as e:
        print(f"  fixture fetch failed: {str(e)[:120]}")
        return []


def log_upcoming(days: int = 10) -> int:
    m = MatchPredictor.load(STATE)
    log = _load_log()
    existing = set(zip(log["kickoff_date"].astype(str),
                       log["home_team"], log["away_team"]))

    rows = []
    for f in fetch_upcoming(days):
        home = m.resolve_name(f.get("homeTeam", {}).get("name", ""))
        away = m.resolve_name(f.get("awayTeam", {}).get("name", ""))
        if not (home and away):
            continue
        kickoff = str(f.get("utcDate", ""))[:10]
        if (kickoff, home, away) in existing:
            continue        # already recorded; never overwrite a prior forecast

        p = m.predict(home, away)
        rows.append({
            "logged_at": dt.date.today().isoformat(),
            "kickoff_date": kickoff,
            "competition": p["competition"],
            "home_team": home, "away_team": away,
            "p_home": p["probabilities"]["home_win"],
            "p_draw": p["probabilities"]["draw"],
            "p_away": p["probabilities"]["away_win"],
            "elo_home": p["elo"]["home"], "elo_away": p["elo"]["away"],
            "xg_home": p["expected_goals"]["home"],
            "xg_away": p["expected_goals"]["away"],
            "model_trained_to": str(m.trained_to)[:10],
            "result": np.nan, "home_goals": np.nan,
            "away_goals": np.nan, "scored_at": np.nan,
        })

    if rows:
        fresh = pd.DataFrame(rows)
        # Concatenating onto an all-NA frame makes pandas warn about future
        # dtype changes, so skip the join entirely when there is nothing to
        # join to.
        log = fresh if log.empty else pd.concat([log, fresh], ignore_index=True)
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        log[COLS].to_csv(LOG, index=False)
    print(f"  logged {len(rows)} new fixture(s)")
    return len(rows)


def score_pending() -> int:
    """Attach real results to predictions whose match has since been played."""
    log = _load_log()
    if log.empty:
        return 0

    pending = log["result"].isna()
    if not pending.any():
        print("  nothing pending to score")
        return 0

    if not os.path.exists(RAW):
        print("  no results file to score against")
        return 0

    res = pd.read_csv(RAW, low_memory=False)
    res["date"] = pd.to_datetime(res["date"], errors="coerce")
    res = res.dropna(subset=["date"])

    scored = 0
    for i in log.index[pending]:
        home, away = log.at[i, "home_team"], log.at[i, "away_team"]
        kickoff = pd.to_datetime(log.at[i, "kickoff_date"], errors="coerce")
        if pd.isna(kickoff):
            continue

        cand = res[(res["home_team"] == home) & (res["away_team"] == away)]
        if cand.empty:
            continue
        # Nearest fixture within tolerance, so a rescheduled match still scores
        # but a repeat of the same tie next season does not get mistaken for it.
        gap = (cand["date"] - kickoff).abs().dt.days
        near = cand[gap <= DATE_TOLERANCE_DAYS]
        if near.empty:
            continue
        row = near.iloc[int(gap[near.index].argmin())]

        log.at[i, "result"] = int(row["result"])
        log.at[i, "home_goals"] = int(row["home_goals"])
        log.at[i, "away_goals"] = int(row["away_goals"])
        log.at[i, "scored_at"] = dt.date.today().isoformat()
        scored += 1

    log[COLS].to_csv(LOG, index=False)
    print(f"  scored {scored} prediction(s)")
    return scored


def report() -> dict:
    """Live accuracy over everything scored so far."""
    log = _load_log()
    done = log.dropna(subset=["result"]) if not log.empty else log
    if done.empty:
        return {"scored": 0, "pending": int(log["result"].isna().sum())
                if not log.empty else 0}

    y = done["result"].astype(int).to_numpy()
    p = done[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

    out = {
        "scored": int(len(done)),
        "pending": int(log["result"].isna().sum()),
        "log_loss": round(log_loss_multi(y, p), 4),
        "brier": round(brier_multi(y, p), 4),
        "accuracy": round(accuracy(y, p), 4),
        "first": str(done["kickoff_date"].min()),
        "last": str(done["kickoff_date"].max()),
        # The number backtesting produced, so the two are always read together.
        "backtest_log_loss": 0.9932,
    }
    out["vs_backtest"] = round(out["log_loss"] - out["backtest_log_loss"], 4)
    return out


def main():
    print("Scoring predictions whose matches have been played...")
    score_pending()
    print("Logging upcoming fixtures...")
    log_upcoming()

    r = report()
    print("\n" + "=" * 60)
    print("LIVE ACCURACY")
    print("=" * 60)
    if r["scored"] == 0:
        print(f"  nothing scored yet ({r['pending']} awaiting results)")
        print("  Run again after these fixtures have been played.")
    else:
        print(f"  scored {r['scored']} matches  ({r['first']} .. {r['last']})")
        print(f"  log loss {r['log_loss']}   backtest was {r['backtest_log_loss']}"
              f"   ({r['vs_backtest']:+.4f})")
        print(f"  brier {r['brier']}   accuracy {r['accuracy']:.1%}")
        print(f"  {r['pending']} still awaiting results")
        if r["scored"] < 100:
            print("\n  Fewer than 100 matches: the interval on this is far wider")
            print("  than the gap being measured. Not yet a verdict.")


if __name__ == "__main__":
    main()
