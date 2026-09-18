"""
Databricks REST API client — free tier compatible.
- Python source files  → Workspace API (always permitted)
- Training data        → Delta table via SQL connector (always permitted)
- Model PKL            → NOT uploaded; model is re-trained inside the notebook
"""

import os
import base64
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CATALOG = os.getenv("DATABRICKS_CATALOG", "main")
SCHEMA  = os.getenv("DATABRICKS_SCHEMA",  "transfer_market")


class DatabricksClient:
    def __init__(self, host: str = None, token: str = None):
        self.host  = (host  or os.getenv("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.getenv("DATABRICKS_TOKEN", "")
        if not self.host or not self.token:
            raise ValueError("Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env")
        self._h = {"Authorization": f"Bearer {self.token}"}

    def _post(self, path: str, **kwargs):
        return requests.post(f"{self.host}{path}", headers=self._h, verify=False, **kwargs)

    def _get(self, path: str, **kwargs):
        return requests.get(f"{self.host}{path}", headers=self._h, verify=False, **kwargs)

    # ── Workspace file upload (Python source files) ───────────────────────────

    def upload_workspace_file(self, local_path: str, workspace_path: str):
        """
        Upload a text file (e.g. .py) to the Databricks Workspace.
        workspace_path example: /Users/you@email.com/transfer_market_src/football_api.py
        """
        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()

        r = self._post("/api/2.0/workspace/import", json={
            "path":      workspace_path,
            "format":    "AUTO",
            "content":   content_b64,
            "overwrite": True,
        })
        r.raise_for_status()
        print(f"  Uploaded {os.path.basename(local_path)} → {workspace_path}")

    def upload_notebook(self, local_path: str, workspace_path: str,
                        language: str = "PYTHON"):
        """
        Upload a .py file as a runnable NOTEBOOK object.

        Distinct from upload_workspace_file, and the distinction is not
        cosmetic. That method sends format=AUTO, which infers the object type
        from the path extension - so an extensionless notebook path produces a
        plain FILE. The import succeeds, the file is visibly there, and every
        later job fails with "Unable to access the notebook", because a jobs
        notebook_task cannot run a FILE. Passing SOURCE with an explicit
        language creates the notebook the job is actually looking for.
        """
        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()

        r = self._post("/api/2.0/workspace/import", json={
            "path":      workspace_path,
            "format":    "SOURCE",
            "language":  language,
            "content":   content_b64,
            "overwrite": True,
        })
        r.raise_for_status()
        print(f"  Uploaded notebook {os.path.basename(local_path)} → {workspace_path}")

    def workspace_status(self, path: str) -> dict:
        """Object type and language at a workspace path, for verification."""
        r = self._get("/api/2.0/workspace/get-status", params={"path": path})
        return r.json() if r.status_code == 200 else {"error": r.text[:200]}

    def get_workspace_user(self) -> str:
        """Returns the current user's email (used to build workspace paths)."""
        r = self._get("/api/2.0/preview/scim/v2/Me")
        r.raise_for_status()
        return r.json().get("userName", "")

    def create_workspace_folder(self, path: str):
        r = self._post("/api/2.0/workspace/mkdirs", json={"path": path})
        r.raise_for_status()

    # ── Delta table upload (training/lookup data via SQL connector) ───────────

    def upload_dataframe(self, df: pd.DataFrame, table: str,
                         chunk_size: int = 1000):
        """
        Write a pandas DataFrame to a Databricks Delta table using
        the SQL connector. Replaces the table on each run.
        table: fully qualified name e.g. 'main.transfer_market.TRAINING_DATA'
        """
        from databricks import sql as dbsql

        http_path = os.getenv("DATABRICKS_HTTP_PATH", "")
        cols     = list(df.columns)
        col_defs = ", ".join([f"`{c}` STRING" for c in cols])

        def _escape(v: str) -> str:
            return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"

        with dbsql.connect(
            server_hostname=self.host.replace("https://", ""),
            http_path=http_path,
            access_token=self.token,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
                cur.execute(f"DROP TABLE IF EXISTS {table}")
                cur.execute(f"CREATE TABLE {table} ({col_defs}) USING DELTA")

                rows = [
                    "(" + ", ".join(_escape(v) for v in row) + ")"
                    for row in df.itertuples(index=False)
                ]
                total = len(rows)
                for i in range(0, total, chunk_size):
                    batch = rows[i:i+chunk_size]
                    sql   = f"INSERT INTO {table} VALUES {', '.join(batch)}"
                    cur.execute(sql)
                    done = min(i + chunk_size, total)
                    print(f"    {done:,}/{total:,} rows", end="\r")
        print(f"  {table}: {len(df):,} rows written      ")

    # ── Jobs API ──────────────────────────────────────────────────────────────

    def run_notebook(self, notebook_path: str, params: dict = None) -> str:
        payload = {
            "run_name": "transfer_market_inference",
            "tasks": [{
                "task_key":      "run_inference",
                "notebook_task": {
                    "notebook_path":    notebook_path,
                    "base_parameters": params or {},
                },
            }],
        }
        r = self._post("/api/2.1/jobs/runs/submit", json=payload)
        r.raise_for_status()
        run_id = str(r.json()["run_id"])
        print(f"  Notebook run submitted: run_id={run_id}")
        return run_id

    def wait_for_run(self, run_id: str, timeout: int = 1800) -> str:
        print(f"  Polling run {run_id}...", end=" ", flush=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self._get("/api/2.1/jobs/runs/get", params={"run_id": run_id})
            r.raise_for_status()
            state = r.json()["state"]
            lc    = state["life_cycle_state"]
            if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                print(f" {lc}/{state.get('result_state', '?')}")
                return state.get("result_state", "UNKNOWN")
            print(".", end="", flush=True)
            time.sleep(20)
        raise TimeoutError(f"Run {run_id} did not finish within {timeout}s")
