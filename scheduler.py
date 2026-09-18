"""
24-hour pipeline scheduler.

Preferred path: trigger the Databricks inference notebook, download the
refreshed Gold CSVs, and signal FastAPI to reload.

Fallback path: compute the same dynamic values locally. This matters more than
it sounds. Day-over-day and week-over-week movement can only exist if snapshots
are actually taken on separate days, so a scheduler that gives up whenever
Databricks is unreachable quietly guarantees the Value Movers page stays empty
forever. The local computation is a few hundred rows of arithmetic and one API
call per competition - it never needed a cluster.

Usage:
    python scheduler.py

Set in .env:
    DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
    DATABRICKS_TOKEN=your_pat_token
    DATABRICKS_CLUSTER_ID=your_cluster_id
    DATABRICKS_NOTEBOOK_PATH=/Users/your@email.com/05_databricks_inference
    FOOTBALL_DATA_API_KEY=your_key

Only FOOTBALL_DATA_API_KEY is required for the fallback path.
"""

import os
import time
import signal
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")

CLUSTER_ID     = os.getenv("DATABRICKS_CLUSTER_ID", "")
NOTEBOOK_PATH  = os.getenv("DATABRICKS_NOTEBOOK_PATH", "")
API_KEY        = os.getenv("FOOTBALL_DATA_API_KEY", "")
FASTAPI_URL    = os.getenv("FASTAPI_URL", "http://localhost:8000")
INTERVAL_HOURS = float(os.getenv("REFRESH_INTERVAL_HOURS", "24"))

_running = True

def handle_stop(sig, frame):
    global _running
    log.info("Shutdown signal received.")
    _running = False

signal.signal(signal.SIGINT,  handle_stop)
signal.signal(signal.SIGTERM, handle_stop)


def _notify_api():
    try:
        r = requests.post(f"{FASTAPI_URL}/reload", timeout=10, verify=False)
        if r.status_code == 200:
            log.info("FastAPI reloaded successfully.")
        else:
            log.warning(f"FastAPI reload returned {r.status_code}")
    except Exception:
        log.warning("FastAPI not reachable — data will load on next restart.")


def run_local_pipeline():
    """Compute dynamic values and append today's snapshot, without Databricks."""
    log.info("Running local value pipeline (no Databricks)...")
    try:
        import build_local_dynamic_values as local
        local.main()
    except Exception as e:
        log.error(f"Local value pipeline failed: {e}")
        return False
    return True


def run_value_pipeline():
    """Player valuations: Databricks notebook when available, else local."""
    if not NOTEBOOK_PATH:
        log.info("Databricks notebook not configured — using the local pipeline.")
        return run_local_pipeline()

    try:
        from databricks_client import DatabricksClient
        client = DatabricksClient()
        log.info(f"Submitting notebook: {NOTEBOOK_PATH}")
        # Widget names are case-sensitive. The notebook declares API_KEY; a
        # lowercase key silently leaves the widget on its empty default and the
        # run fails deep inside the notebook rather than at submission.
        run_id = client.run_notebook(NOTEBOOK_PATH, params={"API_KEY": API_KEY})

        log.info("Waiting for notebook to complete...")
        result = client.wait_for_run(run_id, timeout=1800)
        if result != "SUCCESS":
            log.error(f"Notebook finished with result: {result}")
            return run_local_pipeline()
    except Exception as e:
        log.error(f"Databricks run failed: {str(e)[:160]}")
        return run_local_pipeline()

    log.info("Notebook succeeded.")
    return True


def run_match_refresh():
    """
    Pull results played since the last build and refit the match model.

    Entirely local and independent of Databricks - the match model has never
    lived there. Skipping this is what let the model serve 2026/27 fixtures
    off ratings frozen at the end of the previous season.
    """
    log.info("Refreshing the match model...")
    try:
        import refresh_match_model
        refresh_match_model.main()
    except Exception as e:
        log.error(f"Match refresh failed: {str(e)[:160]}")
        return False
    return True


def run_tracking():
    """Score predictions whose matches have been played, then log new fixtures."""
    log.info("Tracking predictions...")
    try:
        import track_predictions
        track_predictions.score_pending()
        track_predictions.log_upcoming()
    except Exception as e:
        log.error(f"Prediction tracking failed: {str(e)[:160]}")
        return False
    return True


def run_pipeline():
    log.info("─── Pipeline run starting ───")

    ok_values = run_value_pipeline()
    ok_match = run_match_refresh()
    # After the refresh, so newly logged fixtures use the updated model and
    # anything played since the last run is scored against the results that
    # refresh just downloaded.
    run_tracking()

    # Reload once, after both halves, rather than after each.
    _notify_api()
    log.info(f"─── Pipeline run complete "
             f"(values={'ok' if ok_values else 'FAILED'}, "
             f"match={'ok' if ok_match else 'FAILED'}) ───\n")
    return ok_values and ok_match


def main():
    if not API_KEY:
        log.error("FOOTBALL_DATA_API_KEY not set in .env — nothing can run.")
        return

    mode = "Databricks" if (CLUSTER_ID and NOTEBOOK_PATH) else "local"
    log.info(f"Scheduler started. Pipeline runs every {INTERVAL_HOURS}h ({mode} mode).")
    if mode == "Databricks":
        log.info(f"  Cluster:  {CLUSTER_ID}")
        log.info(f"  Notebook: {NOTEBOOK_PATH}")
        log.info("  Falls back to the local pipeline if Databricks is unreachable.")

    next_run = datetime.now()

    while _running:
        now = datetime.now()
        if now >= next_run:
            success  = run_pipeline()
            next_run = datetime.now() + timedelta(hours=INTERVAL_HOURS)
            status   = "OK" if success else "FAILED"
            log.info(f"Next run scheduled at {next_run.strftime('%Y-%m-%d %H:%M')}  [{status}]")
        time.sleep(60)   # check every minute whether it's time to run


if __name__ == "__main__":
    main()
