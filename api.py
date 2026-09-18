"""
FastAPI backend for the Transfer Market Value Predictor dashboard.

Data source (priority order):
  1. Databricks SQL Warehouse  — when DATABRICKS_TOKEN is set and tables exist
  2. Local CSVs in data/processed/ — fallback for local dev / when Databricks unreachable

Run with: uvicorn api:app --reload --port 8000
"""

import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from feature_engineering import build_features, FEATURE_COLS
from model import predict_by_position, load_models

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED      = os.path.join(os.path.dirname(__file__), "data", "processed")
MODELS_PATH    = os.path.join(os.path.dirname(__file__), "data", "models", "position_models.pkl")
MATCH_MODEL    = os.path.join(os.path.dirname(__file__), "data", "models", "match_predictor.pkl")
HISTORY_CSV    = os.path.join(PROCESSED, "value_history.csv")
DB_HOST        = os.getenv("DATABRICKS_HOST", "").replace("https://", "")
DB_HTTP_PATH   = os.getenv("DATABRICKS_HTTP_PATH", "")
DB_TOKEN       = os.getenv("DATABRICKS_TOKEN", "")
DB_CATALOG     = os.getenv("DATABRICKS_CATALOG", "hive_metastore")
DB_SCHEMA      = os.getenv("DATABRICKS_SCHEMA", "transfer_market")
# The id is the last path segment of the HTTP path, so it does not need its own
# variable - but honour an explicit one if it is set.
DB_WAREHOUSE   = (os.getenv("DATABRICKS_WAREHOUSE_ID", "")
                  or DB_HTTP_PATH.rstrip("/").split("/")[-1])
USE_DATABRICKS = bool(DB_HOST and DB_HTTP_PATH and DB_TOKEN)

# A serverless warehouse auto-stops after ten minutes idle - that is the point
# of serverless, and on Free Edition it cannot be turned off. It restarts on
# demand, but a cold start takes longer than the connector is willing to wait,
# so the first query after a quiet spell fails and the app would fall back to
# stale CSVs for no good reason. These bound an explicit wake-and-wait.
WAREHOUSE_WAKE_TIMEOUT = 240     # seconds to wait for a cold start
WAREHOUSE_POLL = 6

_state: dict = {}


# ── Data loading ──────────────────────────────────────────────────────────────

def _warehouse_state() -> str | None:
    """Current warehouse state, or None if it cannot be read."""
    try:
        import requests
        r = requests.get(
            f"https://{DB_HOST}/api/2.0/sql/warehouses/{DB_WAREHOUSE}",
            headers={"Authorization": f"Bearer {DB_TOKEN}"}, timeout=30)
        return r.json().get("state") if r.status_code == 200 else None
    except Exception:
        return None


def _wake_warehouse() -> bool:
    """
    Start a stopped warehouse and block until it is serving.

    Returns False for conditions that waiting cannot fix - a deactivated
    workspace being the obvious one - so the caller can fall back immediately
    rather than sitting through the whole timeout.
    """
    import time
    import requests

    state = _warehouse_state()
    if state == "RUNNING":
        return True
    if state is None:
        return False

    try:
        r = requests.post(
            f"https://{DB_HOST}/api/2.0/sql/warehouses/{DB_WAREHOUSE}/start",
            headers={"Authorization": f"Bearer {DB_TOKEN}"}, timeout=30)
        if r.status_code >= 400 and "INACTIVE" in r.text:
            return False
    except Exception:
        return False

    print(f"  Warehouse is {state} — waiting for it to start...")
    waited = 0
    while waited < WAREHOUSE_WAKE_TIMEOUT:
        time.sleep(WAREHOUSE_POLL)
        waited += WAREHOUSE_POLL
        if _warehouse_state() == "RUNNING":
            print(f"  Warehouse running after {waited}s.")
            return True
    return False


def _query_databricks(sql: str, _retried: bool = False) -> pd.DataFrame:
    from databricks import sql as dbsql
    try:
        with dbsql.connect(
            server_hostname=DB_HOST,
            http_path=DB_HTTP_PATH,
            access_token=DB_TOKEN,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        # One retry, and only after the warehouse is confirmed up. Retrying
        # blindly against a deactivated workspace just multiplies the wait.
        if _retried or not _wake_warehouse():
            raise
        return _query_databricks(sql, _retried=True)


def _load_from_databricks():
    print("Loading data from Databricks SQL Warehouse...")
    _state["players"]    = _query_databricks(
        f"SELECT * FROM {DB_CATALOG}.{DB_SCHEMA}.GOLD_PLAYER_PREDICTIONS"
    )
    _state["current"]    = _query_databricks(
        f"SELECT * FROM {DB_CATALOG}.{DB_SCHEMA}.GOLD_CURRENT_SEASON_SCORES"
    )
    _state["importance"] = _query_databricks(
        f"SELECT * FROM {DB_CATALOG}.{DB_SCHEMA}.GOLD_FEATURE_IMPORTANCE"
    )
    print(f"  Players:   {len(_state['players']):,} rows")
    print(f"  Current:   {len(_state['current'])} rows")
    print(f"  Importance:{len(_state['importance'])} features")


def _load_from_csv():
    print("Loading data from local CSVs (fallback)...")
    _state["players"]    = pd.read_csv(os.path.join(PROCESSED, "gold_player_predictions.csv"))
    _state["current"]    = pd.read_csv(os.path.join(PROCESSED, "gold_current_season_scores.csv"))
    _state["importance"] = pd.read_csv(os.path.join(PROCESSED, "gold_feature_importance.csv"))
    print(f"  Players:   {len(_state['players']):,} rows (local CSV)")


def _load_data():
    if USE_DATABRICKS:
        # The connector logs the raw Thrift exception itself before raising,
        # which puts an alarming stack trace in the startup output for what is
        # a handled, recoverable condition. We report it ourselves instead.
        logging.getLogger("databricks.sql").setLevel(logging.CRITICAL)
        try:
            _load_from_databricks()
            _state["source"] = "databricks"
            return
        except Exception as e:
            msg = str(e)
            if "INACTIVE" in msg or "Cannot create the resource" in msg:
                print("  Databricks workspace is INACTIVE - no compute can start.")
                print("  (check trial status / billing in the account console)")
            else:
                print(f"  Databricks unavailable: {msg[:140]}")
            print("  Falling back to local CSVs - all features remain available.")
    _load_from_csv()
    _state["source"] = "local"


def _history_df() -> pd.DataFrame:
    """
    Local snapshots of dynamic player values.

    Written by src/build_local_dynamic_values.py. This exists so the Value
    Movers page does not depend on a warehouse - the computation is a few
    hundred rows of arithmetic and never needed a cluster.
    """
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_CSV)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df.dropna(subset=["snapshot_date"])


def _movers(hist: pd.DataFrame, days: int, limit: int, direction: str) -> dict | None:
    """
    Biggest value movements between the newest snapshot and one `days` earlier.

    Shared by the Databricks and local paths. Returns None when there is not
    enough history to compare, so the caller can fall through.

    Column names are normalised on the way in. Snapshots written before the
    form fields were renamed carry the old names, and a Delta table that has
    since evolved its schema carries BOTH - old rows populated, new rows null,
    or the reverse - so each pair is coalesced rather than blindly renamed.
    """
    if hist is None or hist.empty or hist["snapshot_date"].nunique() < 2:
        return None

    hist = hist.copy()

    # Drop snapshots where every player shares one multiplier. That is the
    # signature of a run where the form feed returned nothing, and for months
    # it returned nothing on every run. Differencing a working snapshot against
    # one of those reports the repair of the feed as player form movement -
    # a large, entirely fictional one-day swing. A broken baseline is worse
    # than no baseline, so those dates are not allowed to serve as one.
    if "form_multiplier" in hist.columns:
        spread = hist.groupby("snapshot_date")["form_multiplier"].nunique()
        usable = set(spread[spread > 1].index)
        if len(usable) >= 2:
            hist = hist[hist["snapshot_date"].isin(usable)]
        else:
            return None
    for old, new in (("goals_last5", "goals_recent"),
                     ("assists_last5", "assists_recent")):
        if old not in hist.columns:
            continue
        if new in hist.columns:
            hist[new] = hist[new].fillna(hist[old])
        else:
            hist = hist.rename(columns={old: new})

    dates = sorted(hist["snapshot_date"].unique())
    latest = dates[-1]
    target = latest - pd.Timedelta(days=days)
    # Nearest earlier snapshot to the requested lookback. With daily runs this
    # is exact; with gaps it degrades to the closest available, and the caller
    # is told the real gap via days_actual.
    prior = min(dates[:-1], key=lambda d: abs((d - target).days))

    cur = hist[hist["snapshot_date"] == latest].drop_duplicates("player_name")
    prev = (hist[hist["snapshot_date"] == prior]
            .drop_duplicates("player_name")[["player_name", "dynamic_value_eur"]]
            .rename(columns={"dynamic_value_eur": "prev_value"}))

    lb = cur.merge(prev, on="player_name", how="inner")
    lb["current_value"] = lb["dynamic_value_eur"]
    lb["delta_eur"] = lb["current_value"] - lb["prev_value"]
    lb["delta_pct"] = (lb["delta_eur"]
                       / lb["prev_value"].replace(0, np.nan) * 100).round(1)
    lb = lb.dropna(subset=["delta_pct"])
    if lb.empty:
        return None

    lb = (lb.nsmallest(limit, "delta_pct") if direction == "bottom"
          else lb.nlargest(limit, "delta_pct"))
    lb["current_date"] = latest.date().isoformat()
    lb["prev_date"] = prior.date().isoformat()
    lb["snapshot_date"] = lb["snapshot_date"].dt.date.astype(str)

    return {
        "data": lb.replace({np.nan: None}).to_dict(orient="records"),
        "days_actual": int((latest - prior).days),
    }


def _load_model():
    if os.path.exists(MODELS_PATH):
        _state["models"] = load_models(MODELS_PATH)
        print(f"  Model loaded from {MODELS_PATH}")
    else:
        _state["models"] = None
        print("  WARNING: Model file not found — /api/predict will be unavailable.")


# ── App ───────────────────────────────────────────────────────────────────────

def _load_match_predictor():
    """Elo + Dixon-Coles fixture model. Absent until build_match_predictor runs."""
    try:
        from match_predictor import MatchPredictor
        _state["match"] = MatchPredictor.load(MATCH_MODEL)
        m = _state["match"]
        print(f"  Match predictor loaded — {m.n_matches:,} matches, "
              f"trained to {str(m.trained_to)[:10]}")
    except Exception as e:
        _state["match"] = None
        print(f"  WARNING: match predictor unavailable ({e}) — "
              f"/api/match/* disabled.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_data()
    _load_model()
    _load_match_predictor()
    yield
    _state.clear()


app = FastAPI(title="Transfer Market Predictor API", lifespan=lifespan)

# Wide open locally, restricted once ALLOWED_ORIGINS is set. The deployed API
# is public and unauthenticated, so the browser-side restriction is not a
# security boundary - it just stops other sites from quietly using this backend
# as their own.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /api/players ──────────────────────────────────────────────────────────────

@app.get("/api/players")
def get_players(
    competition:  str | None = Query(None),
    position:     str | None = Query(None),
    season:       int | None = Query(None),
    search:       str | None = Query(None),
    latest_only:  bool       = Query(True),
    limit:        int        = Query(500),
    offset:       int        = Query(0),
):
    df = _state["players"].copy()
    if competition:
        df = df[df["competition"] == competition]
    if position:
        df = df[df["position_group"] == position]
    if season:
        df = df[df["season"] == season]
    if search:
        df = df[df["player_name"].str.contains(search, case=False, na=False)]

    # Default: one row per player (latest season they appear in)
    if latest_only and "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df = (
            df.sort_values("season", ascending=False)
              .drop_duplicates(subset=["player_name"], keep="first")
        )

    df = df.sort_values("abs_pct_error" if "abs_pct_error" in df.columns else df.columns[0])
    total = len(df)
    return {"data": df.iloc[offset:offset+limit].replace({np.nan: None}).to_dict(orient="records"), "total": total}


# ── /api/current ──────────────────────────────────────────────────────────────

@app.get("/api/current")
def get_current(
    competition: str | None = Query(None),
    limit:       int        = Query(100),
):
    df = _state["current"].copy()
    if competition:
        df = df[df["competition"] == competition]
    return {"data": df.head(limit).replace({np.nan: None}).to_dict(orient="records")}


# ── /api/leagues ──────────────────────────────────────────────────────────────

@app.get("/api/leagues")
def get_leagues():
    df  = _state["players"]
    agg = (
        df.groupby("competition")
        .agg(
            avg_actual_value    =("market_value_eur",    "mean"),
            avg_predicted_value =("predicted_value_eur", "mean"),
            avg_mape            =("abs_pct_error",       "mean"),
            total_players       =("player_name",         "count"),
            within_25_pct       =("abs_pct_error",       lambda x: (x <= 25).mean() * 100),
        )
        .reset_index()
        .round(2)
    )
    return {"data": agg.to_dict(orient="records")}


# ── /api/kpis ─────────────────────────────────────────────────────────────────

@app.get("/api/kpis")
def get_kpis():
    df = _state["players"]
    return {
        "avg_mape":        round(float(df["abs_pct_error"].mean()), 1),
        "within_10_pct":   round(float((df["abs_pct_error"] <= 10).mean() * 100), 1),
        "within_25_pct":   round(float((df["abs_pct_error"] <= 25).mean() * 100), 1),
        "total_players":   int(df["player_name"].nunique()),
        "total_records":   int(len(df)),
        "seasons_covered": sorted(df["season"].dropna().unique().astype(int).tolist()),
        "competitions":    sorted(df["competition"].unique().tolist()),
        "positions":       sorted(df["position_group"].unique().tolist()),
        "data_source":     _state.get("source", "unknown"),
        "last_refreshed":  str(df["refreshed_at"].max()) if "refreshed_at" in df.columns else "unknown",
    }


# ── /api/importance ───────────────────────────────────────────────────────────

@app.get("/api/importance")
def get_importance():
    return {"data": _state["importance"].replace({np.nan: None}).to_dict(orient="records")}


# ── /api/accuracy-bands ───────────────────────────────────────────────────────

@app.get("/api/accuracy-bands")
def get_accuracy_bands(competition: str | None = Query(None)):
    df = _state["players"].copy()
    if competition:
        df = df[df["competition"] == competition]
    bands = df["accuracy_band"].value_counts().reset_index()
    bands.columns = ["band", "count"]
    order = ["<10%", "10-25%", "25-50%", "50-100%", ">100%"]
    bands["order"] = bands["band"].map({b: i for i, b in enumerate(order)})
    return {"data": bands.sort_values("order").drop(columns="order").to_dict(orient="records")}


# ── /api/predict ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    age:                float = 24.0
    goals_p90:          float = 0.4
    assists_p90:        float = 0.2
    minutes_played:     float = 2500.0
    position:           str   = "ATT"
    competition:        str   = "PL"
    international_caps: int   = 20
    club_prestige_eur:  float = 10_000_000.0


@app.post("/api/predict")
def predict_value(body: PredictRequest):
    if _state.get("models") is None:
        raise HTTPException(503, "Model not loaded — run 04_local_pipeline.py first.")
    pos = body.position.upper()
    if pos not in ("GK", "DEF", "MID", "ATT"):
        raise HTTPException(400, "position must be GK | DEF | MID | ATT")

    row = {
        "age":                body.age,
        "goals":              body.goals_p90 * (body.minutes_played / 90),
        "assists":            body.assists_p90 * (body.minutes_played / 90),
        "minutes_played":     body.minutes_played,
        "competition":        body.competition,
        "position_group":     pos,
        "international_caps": body.international_caps,
        "club_prestige_eur":  body.club_prestige_eur,
    }
    X     = build_features(pd.DataFrame([row]))
    pg    = pd.Series([pos])
    value = predict_by_position(_state["models"], X, pg)[0]
    value_m = value / 1_000_000
    return {
        "predicted_value_eur": round(float(value), 0),
        "predicted_value_m":   round(float(value_m), 1),
        "formatted":           f"€{value_m:.1f}M",
    }


# ── /reload ───────────────────────────────────────────────────────────────────

@app.post("/reload")
def reload_data():
    """Re-fetches data from Databricks (or local CSVs). Called after pipeline refreshes."""
    try:
        _load_data()
        # The match predictor is a pickle rebuilt by the same scheduled run, so
        # reloading player data without it would leave the fixture predictions
        # running on the previous model.
        _load_match_predictor()
        m = _state.get("match")
        return {
            "status":  "reloaded",
            "source":  _state.get("source"),
            "players": len(_state["players"]),
            "current": len(_state["current"]),
            "match_model_trained_to": str(m.trained_to)[:10] if m else None,
        }
    except Exception as e:
        raise HTTPException(500, f"Reload failed: {e}")


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "data_source":  _state.get("source", "not loaded"),
        "model_loaded": _state.get("models") is not None,
        "players":      len(_state.get("players", [])),
    }


# ── /api/leaderboard ──────────────────────────────────────────────────────────
@app.get("/api/leaderboard")
def get_leaderboard(
    period: str = Query("wow", description="dod | wow | mom"),
    limit:  int = Query(50),
    direction: str = Query("top", description="top | bottom"),
):
    """Top movers from VALUE_HISTORY. Falls back to current snapshot if not enough history."""
    days = {"dod": 1, "wow": 7, "mom": 30}.get(period.lower(), 7)

    current = _state.get("current", pd.DataFrame())
    if current.empty:
        return {"data": [], "period": period, "days": days}

    # Two possible sources of snapshots, and the fresher one wins.
    #
    # Preferring Databricks unconditionally looks like the obvious rule and is
    # wrong: whichever pipeline ran most recently holds the better picture. The
    # Delta table currently lags the local file by days AND was written before
    # the form fix, so an unconditional preference serves a leaderboard where
    # every entry reads 0.0%. Comparing latest snapshot dates self-corrects in
    # both directions - when the notebook next runs, Databricks wins again.
    #
    # Both go through the same _movers(): the Delta table is ~1,500 rows, so
    # pulling it whole and comparing in pandas beats maintaining a second copy
    # of the logic in SQL, and it absorbs the two schemas having drifted apart.
    candidates: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []

    if USE_DATABRICKS and _state.get("source") == "databricks":
        try:
            hist = _query_databricks(
                f"SELECT * FROM {DB_CATALOG}.{DB_SCHEMA}.VALUE_HISTORY")
            hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"],
                                                   errors="coerce")
            hist = hist.dropna(subset=["snapshot_date"])
            if not hist.empty:
                candidates.append((hist["snapshot_date"].max(), "history", hist))
        except Exception as e:
            print(f"  Leaderboard Databricks query failed: {str(e)[:140]}")

    local = _history_df()
    if not local.empty:
        candidates.append((local["snapshot_date"].max(), "local_history", local))

    for _, label, hist in sorted(candidates, key=lambda c: c[0], reverse=True):
        out = _movers(hist, days, limit, direction)
        if out:
            return {**out, "period": period, "source": label}

    # Fallback: rank by form_multiplier from current data
    if "dynamic_value_eur" not in current.columns or "form_multiplier" not in current.columns:
        return {"data": [], "period": period,
                "note": "Run: python src/build_local_dynamic_values.py"}

    lb = current.copy()
    lb["delta_pct"] = ((lb["dynamic_value_eur"] - lb["predicted_value_eur"])
                        / lb["predicted_value_eur"].replace(0, np.nan) * 100).round(1)
    lb["delta_eur"] = lb["dynamic_value_eur"] - lb["predicted_value_eur"]

    if direction == "bottom":
        lb = lb.nsmallest(limit, "delta_pct")
    else:
        lb = lb.nlargest(limit, "delta_pct")

    return {"data": lb.replace({np.nan: None}).to_dict(orient="records"), "period": period, "source": "form_multiplier"}


@app.get("/api/history/{player_name}")
def get_player_history(player_name: str):
    """Value trend for one player, from local snapshots or VALUE_HISTORY."""
    hist = _history_df()
    if not hist.empty:
        sub = hist[hist["player_name"].str.lower() == player_name.lower()].copy()
        if not sub.empty:
            sub = sub.sort_values("snapshot_date")
            sub["snapshot_date"] = sub["snapshot_date"].dt.date.astype(str)
            cols = [c for c in ["snapshot_date", "dynamic_value_eur",
                                "form_multiplier", "form_score",
                                "predicted_value_eur"] if c in sub.columns]
            return {"data": sub[cols].replace({np.nan: None}).to_dict(orient="records"),
                    "player": player_name, "source": "local_history"}

    if not os.getenv("DATABRICKS_TOKEN") or _state.get("source") != "databricks":
        return {"data": [], "player": player_name}
    try:
        from databricks import sql as dbsql
        with dbsql.connect(
            server_hostname=os.getenv("DATABRICKS_HOST","").replace("https://",""),
            http_path=os.getenv("DATABRICKS_HTTP_PATH",""),
            access_token=os.getenv("DATABRICKS_TOKEN",""),
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT snapshot_date, dynamic_value_eur, form_multiplier, form_score
                    FROM main.transfer_market.VALUE_HISTORY
                    WHERE LOWER(player_name) = LOWER(?)
                    ORDER BY snapshot_date ASC
                """, [player_name])
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                df = pd.DataFrame(rows, columns=cols)
                return {"data": df.replace({np.nan: None}).to_dict(orient="records"), "player": player_name}
    except Exception as e:
        return {"data": [], "player": player_name, "error": str(e)}


# ── /api/market-insights ──────────────────────────────────────────────────────

# A player needs this much game time before a valuation gap says anything about
# the market rather than about a tiny, noisy sample.
MIN_MINUTES_FOR_INSIGHT = 900

# Formation for the undervalued XI: one keeper, four defenders, three midfielders,
# three forwards. position_group only distinguishes these four, which is exactly
# enough to fill a shape without pretending to know left-back from right-back.
XI_SHAPE = {"GK": 1, "DEF": 4, "MID": 3, "ATT": 3}


def _data_version() -> int:
    """
    Bumped whenever the in-memory data is replaced.

    Derived cheaply rather than tracked: a reload swaps the DataFrame object,
    so its identity changes. Caches keyed on this are invalidated by a /reload
    without anyone having to remember to clear them.
    """
    return id(_state.get("players"))


@app.get("/api/market-insights")
def market_insights():
    """
    Findings about the transfer market, as distinct from diagnostics about the
    model.

    The Overview page used to lead with MAPE and two restatements of it. Those
    describe how well the regression fits; they say nothing a scout would act
    on, and "58% mean error" reads as failure to anyone who does not know that
    market valuations are inherently noisy. Model diagnostics belong on Model
    Insights. This endpoint answers the question the front page actually poses:
    what does the model think the market has got wrong?

    difference_eur is predicted minus actual, so POSITIVE means the model rates
    a player above his market price - the market is underpricing him.
    """
    df = _state.get("players", pd.DataFrame())
    if df.empty:
        return {"available": False}

    # This aggregates 15,000+ rows into a handful of summary figures that change
    # once a day, and the front page calls it on every visit. Measured at
    # ~520ms per request before caching, which on a single free-tier worker is
    # time the process spends unable to serve anything else.
    cached = _state.get("insights_cache")
    if cached and cached["version"] == _data_version():
        return cached["payload"]

    df = df.copy()
    for c in ("market_value_eur", "predicted_value_eur", "difference_eur",
              "age", "minutes_played"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["market_value_eur", "predicted_value_eur", "age"])

    # Headline numbers describe the market as it stands, so they use each
    # player's most recent season rather than averaging over his career.
    latest_season = int(df["season"].max())
    latest = (df[df["season"] == latest_season]
              .sort_values("minutes_played", ascending=False)
              .drop_duplicates("player_name"))
    eligible = latest[latest["minutes_played"] >= MIN_MINUTES_FOR_INSIGHT]

    def _player(row) -> dict:
        return {
            "player_name": row["player_name"],
            "competition": row["competition"],
            "position": row["position_group"],
            "age": int(row["age"]),
            "market_value_eur": float(row["market_value_eur"]),
            "predicted_value_eur": float(row["predicted_value_eur"]),
            "gap_eur": float(row["difference_eur"]),
            "gap_pct": round(float(row["pct_error"]), 1),
        }

    under = over = None
    if not eligible.empty:
        under = _player(eligible.loc[eligible["difference_eur"].idxmax()])
        over = _player(eligible.loc[eligible["difference_eur"].idxmin()])

    # ------------------------------------------------------------------ #
    # Value curve by age. Median, not mean - a handful of superstars would drag
    # the mean around and hide the shape everyone actually wants to see.
    by_age = (df[df["age"].between(16, 38)]
              .groupby(df["age"].astype(int))
              .agg(median_value=("market_value_eur", "median"),
                   p75=("market_value_eur", lambda s: s.quantile(0.75)),
                   n=("market_value_eur", "size"))
              .reset_index().rename(columns={"age": "age"}))
    # Ages at the extremes have too few players for a median to mean anything.
    by_age = by_age[by_age["n"] >= 20]
    peak_age = int(by_age.loc[by_age["median_value"].idxmax(), "age"]) if len(by_age) else None

    age_curve = [
        {"age": int(r.age), "median_value_eur": float(r.median_value),
         "p75_value_eur": float(r.p75), "n": int(r.n)}
        for r in by_age.itertuples(index=False)
    ]

    # ------------------------------------------------------------------ #
    # The undervalued XI.
    xi = []
    for pos, count in XI_SHAPE.items():
        pool = eligible[eligible["position_group"] == pos]
        for _, row in pool.nlargest(count, "difference_eur").iterrows():
            xi.append({**_player(row), "slot": pos})

    # ------------------------------------------------------------------ #
    # Value spread per league. Quartiles rather than a mean, for the same
    # reason as the age curve.
    leagues = []
    for comp, g in latest.groupby("competition"):
        v = g["market_value_eur"]
        leagues.append({
            "competition": comp,
            "n": int(len(g)),
            "p25": float(v.quantile(0.25)),
            "median": float(v.median()),
            "p75": float(v.quantile(0.75)),
            "max": float(v.max()),
            "total_eur": float(v.sum()),
        })
    leagues.sort(key=lambda x: -x["median"])

    # ------------------------------------------------------------------ #
    # Where the model and the market disagree most, by age band. Presented as a
    # finding rather than an error: a systematic gap among the youngest players
    # is the market pricing potential that results-based features cannot see.
    bands = pd.cut(df["age"], bins=[15, 21, 24, 27, 30, 33, 45],
                   labels=["≤21", "22-24", "25-27", "28-30", "31-33", "34+"])
    mis = (df.assign(band=bands).dropna(subset=["band"])
             .groupby("band", observed=True)
             .agg(median_gap=("difference_eur", "median"),
                  n=("difference_eur", "size"))
             .reset_index())
    mispricing = [
        {"band": str(r.band), "median_gap_eur": float(r.median_gap), "n": int(r.n)}
        for r in mis.itertuples(index=False)
    ]

    payload = {
        "available": True,
        "season": latest_season,
        "total_value_eur": float(latest["market_value_eur"].sum()),
        "total_players": int(latest["player_name"].nunique()),
        "peak_age": peak_age,
        "most_undervalued": under,
        "most_overvalued": over,
        "age_curve": age_curve,
        "undervalued_xi": xi,
        "leagues": leagues,
        "mispricing_by_age": mispricing,
    }
    _state["insights_cache"] = {"version": _data_version(), "payload": payload}
    return payload


# ── Match outcome prediction ──────────────────────────────────────────────────

def _match_model():
    m = _state.get("match")
    if m is None:
        raise HTTPException(503, "Match predictor not loaded. "
                                 "Run: python src/build_match_predictor.py")
    return m


@app.get("/api/match/competitions")
def match_competitions():
    m = _match_model()
    names = {"PL": "Premier League", "PD": "La Liga", "SA": "Serie A",
             "BL1": "Bundesliga", "FL1": "Ligue 1"}
    codes = sorted(set(m.team_competition.values()))
    return {"data": [{"code": c, "name": names.get(c, c)} for c in codes]}


@app.get("/api/match/teams")
def match_teams(competition: str | None = Query(None)):
    return {"data": _match_model().teams(competition)}


@app.get("/api/match/ratings")
def match_ratings(competition: str | None = Query(None),
                  limit: int = Query(30)):
    """Current Elo alongside the Dixon-Coles attack and defence ratings."""
    return {"data": _match_model().ratings(competition, limit)}


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    competition: str | None = None
    # Supplying odds switches on the market blend. Omit them for the pure
    # model output - which is what any accuracy claim must be based on.
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None


@app.post("/api/match/predict")
def match_predict(body: MatchRequest):
    m = _match_model()
    home = m.resolve_name(body.home_team)
    away = m.resolve_name(body.away_team)
    if home is None:
        raise HTTPException(404, f"Unknown team: {body.home_team}")
    if away is None:
        raise HTTPException(404, f"Unknown team: {body.away_team}")

    odds = None
    if body.odds_home and body.odds_draw and body.odds_away:
        odds = (body.odds_home, body.odds_draw, body.odds_away)

    return m.predict(home, away, body.competition, market_odds=odds)


@app.get("/api/match/accuracy")
def match_accuracy():
    """
    How the model has actually performed on fixtures it predicted in advance.

    Distinct from the backtest figure: these are matches the model had never
    seen at the moment it committed to a probability. The backtest number is
    returned alongside so the two are always read together.
    """
    try:
        from track_predictions import report
        return report()
    except Exception as e:
        return {"scored": 0, "pending": 0, "error": str(e)[:160]}


@app.get("/api/match/fixtures")
def match_fixtures(days: int = Query(7)):
    """
    Upcoming fixtures from football-data.org, each with a prediction attached.

    Teams are named differently by the two sources, so anything that fails to
    resolve is returned with predicted=False rather than dropped - a fixture
    the user can see but we could not price is more honest than a gap.
    """
    import datetime as dt
    m = _match_model()
    token = os.getenv("FOOTBALL_DATA_API_KEY", "")
    if not token:
        return {"data": [], "error": "FOOTBALL_DATA_API_KEY not set"}

    try:
        import requests
        today = dt.date.today()
        # Restrict to the competitions the model is actually calibrated for.
        # Unfiltered, this returns every league the provider carries - Brazil,
        # the Championship - and second-tier fixtures would be priced with
        # top-tier calibration, which is worse than not pricing them at all.
        codes = ",".join(sorted(m.calibrator.params)) if m.calibrator else ""
        r = requests.get(
            "https://api.football-data.org/v4/matches",
            headers={"X-Auth-Token": token},
            params={"dateFrom": today.isoformat(),
                    "dateTo": (today + dt.timedelta(days=days)).isoformat(),
                    **({"competitions": codes} if codes else {})},
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json().get("matches", [])
    except Exception as e:
        return {"data": [], "error": str(e)}

    out = []
    for f in raw:
        h_raw = f.get("homeTeam", {}).get("name", "")
        a_raw = f.get("awayTeam", {}).get("name", "")
        h, a = m.resolve_name(h_raw), m.resolve_name(a_raw)
        item = {
            "utc_date": f.get("utcDate"),
            "competition": f.get("competition", {}).get("code"),
            "home_team": h_raw,
            "away_team": a_raw,
            "status": f.get("status"),
            "predicted": bool(h and a),
        }
        if h and a:
            item["prediction"] = m.predict(h, a)
        out.append(item)

    return {"data": out, "count": len(out),
            "predicted": sum(1 for o in out if o["predicted"])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
