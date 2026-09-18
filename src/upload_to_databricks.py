"""
Free-tier compatible upload to Databricks.

Avoids the Files API (UC Volumes PUT = 403 on free tier) by using:
  - Workspace API  → Python source files (.py)
  - SQL connector  → training data and lookup tables as Delta tables
  - No binary upload → model is re-trained inside the notebook

Run from the project root after the local pipeline has completed:
    python src/upload_to_databricks.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from databricks_client import DatabricksClient

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SRC_FILES = [
    os.path.join(BASE, "src", "football_api.py"),
    os.path.join(BASE, "src", "feature_engineering.py"),
    os.path.join(BASE, "src", "model.py"),
    os.path.join(BASE, "src", "form_engine.py"),
]

NOTEBOOK_LOCAL = os.path.join(BASE, "notebooks", "05_databricks_inference.py")
# Must sit beside the modules it imports, and must be uploaded with
# upload_notebook() rather than upload_workspace_file(): the latter sends
# format=AUTO, which on an extensionless path creates a FILE. The import
# reports success either way, but a jobs notebook_task cannot run a FILE and
# fails with "Unable to access the notebook".
NOTEBOOK_REMOTE = "/Users/{user_email}/transfer_market_src/05_databricks_inference"

DATA_TABLES = [
    (os.path.join(BASE, "data", "raw", "tm_training_data.csv"),       "main.transfer_market.TRAINING_DATA"),
    (os.path.join(BASE, "data", "raw", "player_prestige_lookup.csv"), "main.transfer_market.PRESTIGE_LOOKUP"),
    (os.path.join(BASE, "data", "raw", "player_caps_lookup.csv"),     "main.transfer_market.CAPS_LOOKUP"),
]

# Feature importance CSV → Delta table (small, ~13 rows)
FI_CSV   = os.path.join(BASE, "data", "processed", "gold_feature_importance.csv")
FI_TABLE = "main.transfer_market.GOLD_FEATURE_IMPORTANCE"


def run():
    client = DatabricksClient()

    # ── Step 0: resolve workspace user email ─────────────────────────────────
    print("Resolving workspace user...")
    user_email = client.get_workspace_user()
    if not user_email:
        user_email = input("Enter your Databricks workspace email: ").strip()
    print(f"  User: {user_email}")

    workspace_src_folder = f"/Users/{user_email}/transfer_market_src"

    # ── Step 1: create schema (idempotent) ───────────────────────────────────
    print("\nStep 1: Ensuring Delta schema exists...")
    from databricks import sql as dbsql
    import urllib3; urllib3.disable_warnings()
    with dbsql.connect(
        server_hostname=client.host.replace("https://", ""),
        http_path=os.getenv("DATABRICKS_HTTP_PATH", ""),
        access_token=client.token,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS main.transfer_market")
    print("  main.transfer_market schema ready")

    # ── Step 2: upload Python source files via Workspace API ─────────────────
    print(f"\nStep 2: Uploading Python source files → {workspace_src_folder}/")
    client.create_workspace_folder(workspace_src_folder)
    for src_path in SRC_FILES:
        if not os.path.exists(src_path):
            print(f"  SKIP (not found): {src_path}")
            continue
        fname = os.path.basename(src_path)
        client.upload_workspace_file(src_path, f"{workspace_src_folder}/{fname}")

    # ── Step 2b: upload inference notebook ───────────────────────────────────
    notebook_remote = NOTEBOOK_REMOTE.format(user_email=user_email)
    print(f"\nStep 2b: Uploading inference notebook → {notebook_remote}")
    if os.path.exists(NOTEBOOK_LOCAL):
        client.upload_notebook(NOTEBOOK_LOCAL, notebook_remote)
    else:
        print(f"  SKIP (not found): {NOTEBOOK_LOCAL}")

    # ── Step 3: upload training + lookup data as Delta tables ─────────────────
    print("\nStep 3: Uploading training / lookup data as Delta tables...")
    for csv_path, table in DATA_TABLES:
        if not os.path.exists(csv_path):
            print(f"  SKIP (not found): {csv_path}")
            continue
        print(f"  {table} ...")
        df = pd.read_csv(csv_path)
        client.upload_dataframe(df, table)

    # ── Step 4: upload feature importance (if available) ─────────────────────
    if os.path.exists(FI_CSV):
        print(f"\nStep 4: Uploading feature importance → {FI_TABLE}")
        fi_df = pd.read_csv(FI_CSV)
        client.upload_dataframe(fi_df, FI_TABLE)
    else:
        print(f"\nStep 4: SKIP feature importance (run local pipeline first)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Upload complete. Next steps:")
    print(f"  1. In Databricks, open Workspace → {workspace_src_folder}/")
    print(f"     You should see football_api.py, feature_engineering.py, model.py")
    print(f"  2. Import notebooks/05_databricks_inference.py into your workspace")
    print(f"     (Workspace → Import → upload the .py file)")
    print(f"  3. Set WORKSPACE_SRC_PATH widget = {workspace_src_folder}")
    print(f"  4. Run the notebook to verify, then create the 24h job:")
    print(f"     python notebooks/06_create_job.py")
    print("="*60)


if __name__ == "__main__":
    run()
