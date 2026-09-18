# Databricks notebook source
# MAGIC %md
# MAGIC # Inference Pipeline — Scheduled every 24h via Databricks Jobs
# MAGIC Reads training data from a Delta table, re-trains the XGBoost model,
# MAGIC pulls fresh player stats from football-data.org, scores them, and writes
# MAGIC results as Delta tables.  No binary file uploads required.

# COMMAND ----------

# MAGIC %pip install xgboost scikit-learn joblib

# COMMAND ----------

import sys
import os

dbutils.widgets.text("WORKSPACE_SRC_PATH", "/Users/you@example.com/transfer_market_src")
dbutils.widgets.text("API_KEY", "")

WORKSPACE_SRC = dbutils.widgets.get("WORKSPACE_SRC_PATH").rstrip("/")
API_KEY       = dbutils.widgets.get("API_KEY") or os.getenv("FOOTBALL_DATA_API_KEY", "")

# Add workspace source path so we can import the helper modules
sys.path.insert(0, f"/Workspace{WORKSPACE_SRC}" if not WORKSPACE_SRC.startswith("/Workspace") else WORKSPACE_SRC)

import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

CATALOG = "main"
SCHEMA  = "transfer_market"

# COMMAND ----------

# ── 1. Read training data from Delta table ────────────────────────────────────
print("Reading training data from Delta table...")
tm_df = (
    spark.table(f"{CATALOG}.{SCHEMA}.TRAINING_DATA")
         .toPandas()
)

# Cast numeric columns that were stored as STRING back to float/int
NUM_COLS = [
    "age", "goals", "assists", "minutes_played",
    "goals_p90", "assists_p90", "gc_p90",
    "league_difficulty", "age_factor",
    "international_caps", "club_prestige_eur",
    "market_value_eur",
]
for c in NUM_COLS:
    if c in tm_df.columns:
        tm_df[c] = pd.to_numeric(tm_df[c], errors="coerce").fillna(0)

print(f"  Training rows: {len(tm_df):,}")

# COMMAND ----------

# ── 2. Re-train the XGBoost model ────────────────────────────────────────────
print("Training XGBoost models by position group...")

import mlflow

# Databricks requires experiment names to be absolute workspace paths.
# Patch mlflow.set_experiment so any call with a relative name is auto-fixed.
# This works even if model.py is cached from a previous import.
try:
    _notebook_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _user_email   = _notebook_ctx.userName().get()
except Exception:
    _user_email = "unknown"

_orig_set_experiment = mlflow.set_experiment
def _set_experiment_patched(name, **kw):
    if isinstance(name, str) and not name.startswith("/"):
        name = f"/Users/{_user_email}/{name}"
    return _orig_set_experiment(name, **kw)
mlflow.set_experiment = _set_experiment_patched
print(f"  MLflow experiment root: /Users/{_user_email}/")

from feature_engineering import build_features, FEATURE_COLS
from model import train_by_position, predict_by_position, evaluate

X_train = build_features(tm_df)
y_train = tm_df["market_value_eur"]
pos_grp = tm_df["position_group"]

models, metrics = train_by_position(X_train, y_train, pos_grp)
print(f"  Trained groups: {list(models.keys())}")

# Build feature importance table from all trained models (averaged)
fi_rows = []
for grp, mdl in models.items():
    for feat, imp in zip(FEATURE_COLS, mdl.feature_importances_):
        fi_rows.append({"position_group": grp, "feature": feat, "importance": float(imp)})
fi_df = pd.DataFrame(fi_rows)

# COMMAND ----------

# ── 3. Load lookup tables ─────────────────────────────────────────────────────
print("Loading lookup tables...")
prestige_lk = (
    spark.table(f"{CATALOG}.{SCHEMA}.PRESTIGE_LOOKUP").toPandas()
)
caps_lk = (
    spark.table(f"{CATALOG}.{SCHEMA}.CAPS_LOOKUP").toPandas()
)

for df in [prestige_lk, caps_lk]:
    for c in df.columns:
        if c != "player_name":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

def _norm(s): return s.str.lower().str.strip()
prestige_lk["_key"] = _norm(prestige_lk["player_name"])
caps_lk["_key"]     = _norm(caps_lk["player_name"])

print(f"  Prestige lookup: {len(prestige_lk):,} rows")
print(f"  Caps lookup:     {len(caps_lk):,} rows")

# COMMAND ----------

# ── 4. Pull fresh data from football-data.org ─────────────────────────────────
from football_api import FootballDataClient, COMPETITIONS, SEASONS

print(f"Pulling player stats from API (key present: {bool(API_KEY)})...")
client      = FootballDataClient(api_key=API_KEY or None)
all_scorers = []

for comp in COMPETITIONS:
    for season in [SEASONS[-1]]:
        print(f"  {comp} {season}...", end=" ", flush=True)
        try:
            rows = client.get_scorers(competition=comp, season=season, limit=100)
            all_scorers.extend(rows)
            print(f"{len(rows)} players")
        except Exception as e:
            print(f"FAILED: {e}")

api_df = pd.DataFrame(all_scorers)
print(f"  Total API players: {len(api_df):,}")

# COMMAND ----------

# ── 5. Enrich API data with lookups ──────────────────────────────────────────
api_df["_key"] = _norm(api_df["player_name"])
api_df = api_df.merge(prestige_lk[["_key", "club_prestige_eur"]], on="_key", how="left")
api_df = api_df.merge(caps_lk[["_key", "international_caps"]],   on="_key", how="left")
api_df.drop(columns=["_key"], inplace=True)

# Fill missing prestige with median for that competition
comp_medians = tm_df.groupby("competition")["club_prestige_eur"].median()
for i, row in api_df.iterrows():
    if pd.isna(row.get("club_prestige_eur")):
        api_df.at[i, "club_prestige_eur"] = comp_medians.get(row["competition"], 0)
api_df["international_caps"] = pd.to_numeric(
    api_df["international_caps"], errors="coerce"
).fillna(0)

# COMMAND ----------

# ── 6. Score current season players ──────────────────────────────────────────
X_api  = build_features(api_df)
pg_api = X_api.apply(
    lambda r: ("GK" if r["pos_GK"] else ("DEF" if r["pos_DEF"] else ("MID" if r["pos_MID"] else "ATT"))),
    axis=1,
)
y_pred_api = predict_by_position(models, X_api, pg_api)

current_df = api_df[[
    "player_name", "team_name", "competition", "season",
    "position", "goals", "assists", "played_matches",
]].copy().reset_index(drop=True)
current_df["age"]                 = X_api["age"].values
current_df["predicted_value_eur"] = y_pred_api
current_df["refreshed_at"]        = datetime.utcnow().isoformat()
current_df.sort_values("predicted_value_eur", ascending=False, inplace=True)

print(f"Current season predictions: {len(current_df):,} players")

# ── 7. Re-score historical training data (for comparison / accuracy charts) ───
y_tm_pred = predict_by_position(models, X_train, pos_grp)
eval_df   = evaluate(tm_df["market_value_eur"], y_tm_pred)

gold_df = tm_df[[
    "player_name", "competition", "season", "position_group",
    "age", "goals", "assists", "minutes_played",
    "international_caps", "club_prestige_eur", "market_value_eur",
]].copy().reset_index(drop=True)
gold_df["predicted_value_eur"] = y_tm_pred
gold_df["difference_eur"]      = eval_df["difference_eur"].values
gold_df["pct_error"]           = eval_df["pct_error"].values
gold_df["abs_pct_error"]       = eval_df["abs_pct_error"].values
gold_df["accuracy_band"]       = eval_df["accuracy_band"].astype(str).values
gold_df["refreshed_at"]        = datetime.utcnow().isoformat()

print(f"Historical predictions: {len(gold_df):,} rows")

# COMMAND ----------

# ── 8. Write Gold Delta tables ────────────────────────────────────────────────
def write_delta(pdf: pd.DataFrame, table: str):
    sdf = spark.createDataFrame(pdf)
    (
        sdf.write
           .format("delta")
           .mode("overwrite")
           .option("overwriteSchema", "true")
           .saveAsTable(f"{CATALOG}.{SCHEMA}.{table}")
    )
    print(f"  {CATALOG}.{SCHEMA}.{table}: {sdf.count():,} rows")

print("Writing Gold Delta tables...")
write_delta(gold_df,    "GOLD_PLAYER_PREDICTIONS")
write_delta(current_df, "GOLD_CURRENT_SEASON_SCORES")
write_delta(fi_df,      "GOLD_FEATURE_IMPORTANCE")

# ── VALUE_HISTORY: append today's snapshot ───────────────────────────────────
from form_engine import FormEngine

print("Computing form multipliers...")
fe = FormEngine(api_key=API_KEY)
# No season pinned: form has to describe the current season, and the provider
# already knows which one that is. Pinning it to 2024 silently measured a
# season that finished years ago.
form_data = fe.fetch_contributions()
current_df_dyn = fe.apply_dynamic_values(current_df, form_data)

n_real = int((current_df_dyn["form_multiplier"] != 1.0).sum())
print(f"  {n_real} of {len(current_df_dyn)} players have live form data")
if n_real == 0:
    print("  WARNING: every multiplier is 1.0 — the form feed returned nothing,")
    print("  so this snapshot will show no movement against the previous one.")

# Append to VALUE_HISTORY (create if not exists)
history_row = current_df_dyn[[
    "player_name", "competition", "predicted_value_eur",
    "form_multiplier", "contract_modifier", "dynamic_value_eur",
    "form_score", "goals_recent", "assists_recent",
]].copy()
history_row["snapshot_date"] = datetime.utcnow().date().isoformat()

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.VALUE_HISTORY (
        player_name STRING, competition STRING,
        predicted_value_eur DOUBLE, form_multiplier DOUBLE,
        contract_modifier DOUBLE, dynamic_value_eur DOUBLE,
        form_score DOUBLE, goals_recent INT, assists_recent INT,
        snapshot_date STRING
    ) USING DELTA
""")

hist_sdf = spark.createDataFrame(history_row)
# mergeSchema lets the new goals_recent/assists_recent columns land on a table
# that was created with the old names, instead of failing the append. Existing
# rows keep their old columns and read back as null for the new ones.
(hist_sdf.write.format("delta").mode("append")
         .option("mergeSchema", "true")
         .saveAsTable(f"{CATALOG}.{SCHEMA}.VALUE_HISTORY"))
print(f"  VALUE_HISTORY: {hist_sdf.count()} rows appended (date={datetime.utcnow().date()})")

# Also write dynamic current season scores
write_delta(current_df_dyn, "GOLD_CURRENT_SEASON_SCORES")

print(f"\nPipeline complete at {datetime.utcnow().isoformat()} UTC")
