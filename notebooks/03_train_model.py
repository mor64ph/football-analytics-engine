# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training — XGBoost with MLflow
# MAGIC Trains on silver features, logs model + metrics to MLflow,
# MAGIC then scores all players and writes predictions to Gold layer.

# COMMAND ----------

import sys
sys.path.insert(0, "/dbfs/FileStore/transfer_market/src")

import pandas as pd
import numpy as np
import mlflow
from pyspark.sql import SparkSession
from model import train, predict, evaluate, feature_importance
from sklearn.model_selection import train_test_split

spark = SparkSession.builder.getOrCreate()
mlflow.set_tracking_uri("databricks")

SILVER_PATH = "/mnt/transfer_market/silver"
GOLD_PATH   = "/mnt/transfer_market/gold"

# COMMAND ----------

# Load silver features
silver_df = spark.read.format("delta").load(f"{SILVER_PATH}/player_features").toPandas()

FEATURE_COLS = [
    "age", "goals_p90", "assists_p90", "gc_p90",
    "estimated_minutes", "league_difficulty", "age_factor",
    "pos_GK", "pos_DEF", "pos_MID", "pos_ATT",
]

X = silver_df[FEATURE_COLS].fillna(0)
y = silver_df["market_value_eur"]

print(f"Training set: {len(X)} players, {X.shape[1]} features")
print(f"Market value range: €{y.min():,.0f} – €{y.max():,.0f}")

# COMMAND ----------

# Train/test split — 80/20, stratified by league
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model, train_metrics = train(
    X_train, y_train,
    experiment_name="/Users/transfer_market/transfer_market_predictor",
    run_name="xgb_v1",
)

print("Train metrics:", train_metrics)

# COMMAND ----------

# Evaluate on hold-out test set
y_pred_test = predict(model, X_test)
test_results = evaluate(y_test, y_pred_test)
test_mae  = test_results["difference_eur"].abs().mean()
test_mape = test_results["abs_pct_error"].mean()

print(f"Test MAE:  €{test_mae:,.0f}")
print(f"Test MAPE: {test_mape:.1f}%")
print(test_results["accuracy_band"].value_counts())

# COMMAND ----------

# Score ALL players (for the PowerBI dashboard)
y_pred_all = predict(model, X)
eval_all   = evaluate(y, y_pred_all)

gold_df = silver_df[[
    "player_id", "player_name", "team_name", "competition",
    "season", "position_group", "age", "goals", "assists",
    "estimated_minutes", "market_value_eur",
]].copy()

gold_df["predicted_value_eur"]  = y_pred_all
gold_df["difference_eur"]       = eval_all["difference_eur"].values
gold_df["pct_error"]            = eval_all["pct_error"].values
gold_df["abs_pct_error"]        = eval_all["abs_pct_error"].values
gold_df["accuracy_band"]        = eval_all["accuracy_band"].astype(str).values

# COMMAND ----------

# Feature importance for PowerBI "model explainability" page
fi_df = feature_importance(model, FEATURE_COLS)
print(fi_df)

# COMMAND ----------

# Write Gold outputs
sdf_gold = spark.createDataFrame(gold_df)
(
    sdf_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_PATH}/player_predictions")
)

sdf_fi = spark.createDataFrame(fi_df)
(
    sdf_fi.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_PATH}/feature_importance")
)

print("Gold layer written.")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS transfer_market.GOLD_PLAYER_PREDICTIONS
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/transfer_market/gold/player_predictions';
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS transfer_market.GOLD_FEATURE_IMPORTANCE
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/transfer_market/gold/feature_importance'
