"""
Creates a Databricks Job that runs the inference notebook every 24 hours.
Run this once from your local machine after importing the notebook to Databricks.

    python notebooks/06_create_job.py

Prerequisites:
  - 05_databricks_inference imported into your Databricks workspace
  - DATABRICKS_HOST and DATABRICKS_TOKEN set in .env
  - Set DATABRICKS_NOTEBOOK_PATH below (right-click notebook in workspace → Copy Path)
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HOST  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
TOKEN = os.getenv("DATABRICKS_TOKEN", "")
KEY   = os.getenv("FOOTBALL_DATA_API_KEY", "")

# ── Set this after importing the notebook ──────────────────────────────────────
NOTEBOOK_PATH = os.getenv(
    "DATABRICKS_NOTEBOOK_PATH",
    "/Users/you@example.com/transfer_market_src/05_databricks_inference",
)

HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def create_job():
    payload = {
        "name": "transfer_market_24h_refresh",
        "tasks": [
            {
                "task_key": "run_inference",
                "notebook_task": {
                    "notebook_path": NOTEBOOK_PATH,
                    "base_parameters": {
                        "API_KEY": KEY,
                    },
                },
                # No cluster spec = Databricks uses serverless compute automatically
            }
        ],
        "schedule": {
            # Every day at 03:00 UTC
            "quartz_cron_expression": "0 0 3 * * ?",
            "timezone_id": "UTC",
            "pause_status": "UNPAUSED",
        },
        "email_notifications": {},
        "tags": {"project": "transfer_market_predictor"},
    }

    r = requests.post(
        f"{HOST}/api/2.1/jobs/create",
        headers=HEADERS,
        json=payload,
        verify=False,
    )

    if r.status_code == 200:
        job_id = r.json()["job_id"]
        print(f"Job created successfully!")
        print(f"  Job ID:   {job_id}")
        print(f"  Schedule: daily at 03:00 UTC")
        print(f"  Notebook: {NOTEBOOK_PATH}")
        print(f"\nView it at: {HOST}/jobs/{job_id}")
        return job_id
    else:
        print(f"Failed to create job: {r.status_code}")
        print(r.text)
        return None


def trigger_now(job_id: int):
    """Optionally run the job immediately to test it."""
    r = requests.post(
        f"{HOST}/api/2.1/jobs/run-now",
        headers=HEADERS,
        json={"job_id": job_id},
        verify=False,
    )
    if r.status_code == 200:
        run_id = r.json()["run_id"]
        print(f"\nTriggered immediate run — run_id: {run_id}")
        print(f"Monitor at: {HOST}/jobs/{job_id}/runs/{run_id}")
    else:
        print(f"Failed to trigger run: {r.status_code} {r.text}")


if __name__ == "__main__":
    if not HOST or not TOKEN:
        print("Error: set DATABRICKS_HOST and DATABRICKS_TOKEN in .env")
        sys.exit(1)

    job_id = create_job()

    if job_id:
        ans = input("\nTrigger a test run right now? (y/n): ").strip().lower()
        if ans == "y":
            trigger_now(job_id)
