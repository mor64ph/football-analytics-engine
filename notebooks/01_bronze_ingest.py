# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — Raw Ingestion from football-data.org
# MAGIC Pulls scorers data for PL / La Liga / Serie A across the last 4 seasons.
# MAGIC Writes to Delta as-is (no transforms) at: /mnt/transfer_market/bronze/

# COMMAND ----------

import sys
sys.path.insert(0, "/dbfs/FileStore/transfer_market/src")

import json
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit
from football_api import FootballDataClient, COMPETITIONS, SEASONS

spark = SparkSession.builder.getOrCreate()

BRONZE_PATH = "/mnt/transfer_market/bronze"
API_KEY = dbutils.secrets.get(scope="transfer-market", key="football-data-api-key")

# COMMAND ----------

client = FootballDataClient(api_key=API_KEY)
all_scorers = []

for comp in COMPETITIONS:
    for season in SEASONS:
        print(f"Pulling {comp} season {season}...")
        try:
            scorers = client.get_scorers(competition=comp, season=season, limit=100)
            all_scorers.extend(scorers)
            print(f"  → {len(scorers)} scorers")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

# COMMAND ----------

pdf = pd.DataFrame(all_scorers)
pdf["ingested_at"] = datetime.utcnow().isoformat()

sdf = spark.createDataFrame(pdf)
(
    sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{BRONZE_PATH}/scorers")
)

print(f"Written {sdf.count()} rows to {BRONZE_PATH}/scorers")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS transfer_market.bronze_scorers
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/transfer_market/bronze/scorers'
