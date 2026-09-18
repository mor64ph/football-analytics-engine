# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer — Clean & Enrich
# MAGIC - Deduplicates players (a player may appear in multiple scorers pages)
# MAGIC - Computes age and position group
# MAGIC - Joins with Transfermarkt market value CSV (uploaded to DBFS)
# MAGIC - Writes to: /mnt/transfer_market/silver/

# COMMAND ----------

import sys
sys.path.insert(0, "/dbfs/FileStore/transfer_market/src")

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, datediff, lit, when, lower, trim
from feature_engineering import build_features, attach_market_values

spark = SparkSession.builder.getOrCreate()

BRONZE_PATH = "/mnt/transfer_market/bronze"
SILVER_PATH = "/mnt/transfer_market/silver"
TM_CSV_PATH = "/dbfs/FileStore/transfer_market/transfermarkt_values.csv"

# COMMAND ----------

# Load bronze scorers
bronze_df = spark.read.format("delta").load(f"{BRONZE_PATH}/scorers").toPandas()
print(f"Bronze rows: {len(bronze_df)}")

# COMMAND ----------

# Load Transfermarkt market values
# CSV download from: https://www.kaggle.com/datasets/davidcariboo/player-scores
# Required columns: player_name, market_value_eur, last_season
tm_df = pd.read_csv(TM_CSV_PATH)
print(f"TM rows: {len(tm_df)}")

# COMMAND ----------

# Attach market values — inner join (only players with known market values become training data)
merged = attach_market_values(bronze_df, tm_df)
print(f"Matched: {len(merged)} / {len(bronze_df)} rows")

# COMMAND ----------

# Build features
X = build_features(merged)
merged_with_features = pd.concat([merged.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

# COMMAND ----------

# Write to silver
sdf = spark.createDataFrame(merged_with_features)
(
    sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{SILVER_PATH}/player_features")
)
print(f"Written {sdf.count()} rows to {SILVER_PATH}/player_features")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS transfer_market.silver_player_features
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/transfer_market/silver/player_features'
