# ScoutIQ — Troubleshooting Log

A record of every blocker hit during development and the exact fix applied.

---

## 1. Databricks Files API 403 on UC Volumes
**Error:** `PUT /api/2.0/fs/files/Volumes/...` returns 403  
**Cause:** Databricks free tier blocks binary uploads to Unity Catalog Volumes  
**Fix:** Switch entirely to Workspace API for Python files (`/api/2.0/workspace/import` with `format: AUTO`) and SQL connector for tabular data. Re-train the model inside the notebook instead of uploading a `.pkl`.

---

## 2. SQL Placeholder Syntax Error
**Error:** `PARSE_SYNTAX_ERROR at or near '%'`  
**Cause:** `databricks-sql-connector` uses `?` as the placeholder, not `%s`  
**Fix:** Replace all `%s` with `?` in parameterised queries, or switch to batched inline `VALUES` for bulk inserts.

---

## 3. MLflow Experiment Name Must Be Absolute Path
**Error:** `INVALID_PARAMETER_VALUE: experiment name must be absolute path`  
**Cause:** Databricks requires MLflow experiment names in the format `/Users/email/name`  
**Fix:** Monkey-patch `mlflow.set_experiment` in the notebook before importing `model.py`:
```python
_orig = mlflow.set_experiment
def _patched(name, **kw):
    if isinstance(name, str) and not name.startswith("/"):
        name = f"/Users/{_user_email}/{name}"
    return _orig(name, **kw)
mlflow.set_experiment = _patched
```

---

## 4. XGBoost / sklearn `__sklearn_tags__` Crash
**Error:** `AttributeError: 'super' object has no attribute '__sklearn_tags__'`  
**Cause:** XGBoost 2.0.3 + sklearn 1.6 MRO incompatibility triggered by `cross_val_score`  
**Fix:** Remove `cross_val_score` from `model.py` entirely — not needed in a production pipeline.

---

## 5. `python api.py` Silent Exit
**Error:** Running `python api.py` exits immediately with no output  
**Cause:** Missing `if __name__ == "__main__": uvicorn.run(...)` block  
**Fix:** Add to the bottom of `api.py`:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 6. `npm run dev` Blocked by Execution Policy
**Error:** PowerShell blocks `npm.ps1` script execution  
**Cause:** A corporate PowerShell execution policy blocking unsigned scripts  
**Fix:** Use `cmd /c "npm run dev"` instead of running npm directly in PowerShell.

---

## 7. Databricks Job — ResourceNotFound (Notebook Missing)
**Error:** `Unable to access the notebook "/Users/.../05_databricks_inference"`  
**Cause:** Notebook was never uploaded to the path the job expected, or was overwritten  
**Fix:** Manually import the notebook via **Workspace → Import** in the Databricks UI, or ensure `upload_to_databricks.py` uploads it to the correct path before running the job.

---

## 8. Notebook Uploaded to Wrong Folder
**Error:** Job still says ResourceNotFound even after upload  
**Cause:** `upload_to_databricks.py` placed the notebook inside `transfer_market_src/` but the job pointed one level up  
**Fix:** Update the job task's notebook path to match where the file actually landed:
```
/Users/<your-databricks-user>/transfer_market_src/05_databricks_inference
```

---

## 9. `%pip install xgboost==2.0.3` Fails on Serverless
**Error:** `Library installation failed — unable to find or download package`  
**Cause:** Databricks serverless compute blocks installs of specific pinned versions not available for its platform  
**Fix:** Remove the version pin — use `%pip install xgboost scikit-learn joblib` with no version constraints.

---

## 10. `ModuleNotFoundError: No module named 'xgboost'`
**Error:** Notebook fails at `import xgboost` after pip install cell was removed  
**Cause:** Serverless does not have xgboost pre-installed; removing the install cell entirely broke it  
**Fix:** Add back `%pip install xgboost scikit-learn joblib` (no version pins) — serverless can install the latest compatible version, just not a pinned old one.

---

## 11. API_KEY Widget Empty in Job Runs
**Error:** `ValueError: Set FOOTBALL_DATA_API_KEY in .env or pass api_key=`  
**Cause:** `.env` files are local only — Databricks does not load them. Job parameter was named `api_key` (lowercase) but widget expects `API_KEY` (uppercase)  
**Fix:** In the job task configuration, set parameter name to `API_KEY` (matching the widget name exactly). For interactive runs, paste the key directly into the widget field in the notebook UI.

---

## 12. f-string SyntaxError in form_engine.py
**Error:** `SyntaxError: f-string: expecting '=', or '!', or ':', or '}'`  
**Cause:** Python 3.11 (used by Databricks serverless) does not allow complex list comprehensions with nested `if` conditions inside f-strings  
**Fix:** Extract the expression into a variable before the f-string:
```python
# Before (broken on 3.11)
print(f"  {len([e for e in events if e['date'] >= x if condition])}")

# After
cutoff = matches[-1]["utcDate"][:10] if matches else ""
n_new = len([e for e in events if e["date"] >= cutoff]) if matches else 0
print(f"  {comp}: {len(matches)} matches, {n_new} events")
```

---

## 13. `from dotenv import load_dotenv` Fails on Databricks
**Error:** `ModuleNotFoundError: No module named 'dotenv'`  
**Cause:** `python-dotenv` is not available on Databricks serverless  
**Fix:** Remove the import from any file that runs on Databricks. API keys should come through notebook widgets or Databricks Secrets, not `.env` files.

---

## General Rules Learned

| Rule | Detail |
|---|---|
| Serverless pip installs | No version pins — use bare package names only |
| MLflow on Databricks | Always use absolute paths: `/Users/email/experiment` |
| Credentials on Databricks | Use widgets for interactive, job parameters for scheduled runs — never `.env` |
| f-strings on Python ≤3.11 | No complex expressions inside `{}` — extract to variables first |
| Workspace API upload | Uploads as a notebook if the `.py` file contains `# MAGIC` comments |
| npm on a locked-down Windows machine | Always use `cmd /c "npm run dev"` not PowerShell directly |
