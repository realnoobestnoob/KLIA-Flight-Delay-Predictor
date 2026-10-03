# KLIA Flight Delay Predictor (MLOps edition)

Predicts the chance a departure from Kuala Lumpur International Airport is delayed by 15 minutes or more.
Free to run: Neon free tier, open-source libraries, Open-Meteo weather, optional free hosting.

```
 Neon "departures" table
        │  only rows with id > watermark
        ▼
 ┌────────────── update job (daily, runs and exits) ──────────────┐
 │ validate → FeatureState → online model learns (test-then-train) │
 └───────────────────────────┬─────────────────────────────────────┘
                             │ saves one bundle (model + feature state + threshold)
                             ▼
                  Neon "model_registry" table
                             │ newest bundle is cached in RAM
                             ▼
   Docker container: FastAPI  POST /v1/predict   (never trains)
                             ▲
                  Streamlit page or curl
```

| Folder | What it does |
|---|---|
| `klia/etl/` | validation: parses mixed 12h/24h times, builds the delay label, rejects bad rows |
| `klia/features/state.py` | **FeatureState**: the one feature code path used by training and serving |
| `klia/model/` | incremental models (scikit-learn partial_fit) + ensembles, and the saved bundle |
| `klia/jobs/update.py` | the incremental update job |
| `klia/api/` | REST API and the in-memory model cache |
| `notebooks/` | EDA and model selection |
| `tests/` | 19 tests (`pytest`) |

---

## Setup, step by step

You need **Python 3.11 or newer** and **Git**. Docker is only needed for step 9.

### 1. Get the code
```
cd klia-mlops
python -m venv .venv
```
Activate it: Windows `.venv\Scripts\activate`, macOS/Linux `source .venv/bin/activate`. Then:
```
pip install -r requirements-train.txt
```

### 2. Check that it works (no database needed)
```
pytest -q
```
You should see `19 passed`.

### 3. Connect to Neon
1. Neon console → project **klia_flight_predictor** → **Connect** → copy the connection string.
2. Copy `.env.example` to `.env` and paste it as `DATABASE_URL=...`. (`.env` is git-ignored.)
3. Verify:
```
python -m klia.jobs.check
```
It must print `OK` for the connection, the table and the columns. The table is expected to be named `departures` with columns
`id, date, scheduled_departure, actual_departure, airline, destination` (plus optional `aircraft` and the weather columns).
If yours differ, change `data.table` in `config/config.yaml`, or tell me the real column names.

### 4. Pick the model (notebook)

Install the extra libraries first (mlflow, optuna, scikit-learn, river for drift detection):
```
pip install -r requirements-train.txt
```
Then open and run **Run -> Run All Cells**:
```
jupyter lab notebooks/01_model_selection_eda.ipynb
```

**What it does, in order:**
1. Loads your data (or synthetic demo data if `DATABASE_URL` is not set) and explores it
2. On a **subset** of the data (`eda.subset_fraction` in `config.yaml`, default 25%): compares
   single **incremental** models -- `sgd_log`, `sgd_hinge` (modified Huber), `gnb`, `mlp` -- each
   updated with `partial_fit` on one batch of rows at a time (never one row at a time, never
   retrained from scratch). The best one becomes the **benchmark**.
3. On the **same subset**: tries incremental **ensembles** -- bagging and random-subspace, built
   from the models above -- and compares them to the benchmark.
4. Picks the overall best model or ensemble, then **tunes its hyperparameters with Optuna**
   (`eda.n_optuna_trials` trials) -- still on the subset only.
5. **Trains the tuned model on the full dataset**, batch by batch, the same way production does it.
6. Saves `artifacts/pretrained_bundle.gz` + `artifacts/model_choice.json`, and **registers the
   model in the MLflow Model Registry** (`klia-flight-delay`, aliased `champion`).

Everything is logged to a local MLflow database (`mlflow.db`, git-ignored). Browse it any time:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000 -- you'll see the `klia-flight-delay` experiment (benchmark, ensembles,
tuning trials, final model) and the **Models** tab with the registered `champion` version.

With no `DATABASE_URL`, the notebook runs fully on synthetic data for you to see the workflow, but
does **NOT** save or register anything -- connect Neon and re-run for a real model.

### 5. Build the first model from your full history (once)
```
python -m klia.jobs.update --bootstrap
```
This loads `artifacts/pretrained_bundle.gz` from the notebook (if present) and replays your full
history through it in batches, so the first production model is exactly the one the notebook
picked and tuned -- it is not retrained from scratch. It also creates three small tables in Neon
(`etl_state`, `model_registry`, `etl_runs`); your `departures` table is never changed.

**Model size:** uncapped if stored only via MLflow's local artifact store. If/when a bundle is
pushed to Neon's `model_registry` table (every `python -m klia.jobs.update` run after bootstrap),
it is capped at `registry.max_mb_neon` (400 MB by default) -- raise it in `config/config.yaml` if
your ensemble is larger, there's no need to keep the model small otherwise.

### 6. Update as new flights arrive
```
python -m klia.jobs.update
```
Each run does **one `partial_fit` call per batch** of new rows (`data.batch_size` rows per batch) --
this is incremental learning, not per-row online learning. With no new rows it does nothing, so
it's safe to run as often as you like.

**Option A -- Task Scheduler / cron:** as before (see the original setup, step 6, in your task
history) -- schedule `python -m klia.jobs.update` daily.

**Option B -- GitHub Actions:** `.github/workflows/update.yml`, needs a repo secret `DATABASE_URL`.

**Option C -- Apache Airflow** (new): runs the same `klia.jobs.update.run()` function on a schedule,
with a connectivity check beforehand.
```
pip install apache-airflow==2.9.3        # in its own venv -- Airflow pins many dependencies
export AIRFLOW_HOME=~/airflow
airflow db init
```
Point Airflow at this project's DAG, either by editing `dags_folder` in `$AIRFLOW_HOME/airflow.cfg`
to `<project>/airflow/dags`, or by symlinking:
```
ln -s "$(pwd)/airflow/dags/klia_pipeline_dag.py" "$AIRFLOW_HOME/dags/klia_pipeline_dag.py"
```
Make sure `DATABASE_URL` is set in the environment Airflow's scheduler/webserver run in, then:
```
airflow webserver --port 8080 &
airflow scheduler
```
The DAG `klia_incremental_update` runs daily at 04:30 Malaysia time: `check_connection` (fails fast
if Neon is unreachable) then `run_update` (the incremental update; skips cleanly if there are no
new rows).

### 7. Run the API
```
uvicorn klia.api.app:app --port 8000
```
Open http://localhost:8000/docs, or:
```
curl -X POST http://localhost:8000/v1/predict -H "Content-Type: application/json" ^
  -d "{\"airline\":\"AirAsia\",\"destination\":\"Singapore\",\"scheduled_departure\":\"2026-10-03T18:30\"}"
```
(on macOS/Linux use `\` instead of `^`, and plain quotes.) Example reply:
```json
{"delay_probability":0.33,"predicted_delayed":true,"threshold":0.24,"risk":"high","model_version":2,
 "known_airline":true,"known_route":true,"weather_source":"open-meteo"}
```
Endpoints: `POST /v1/predict`, `GET /v1/options`, `GET /v1/model`, `GET /healthz`. Predictions use the cached model and never train. About every 30 minutes (`api.refresh_seconds`) a request triggers a background check and the API swaps in a newer model without a restart. If Neon is asleep or down it keeps serving the last model, also copied to disk.

### 8. Optional front-end
```
streamlit run streamlit_app/app.py
```
It only calls the API (`KLIA_API_URL`, default `http://localhost:8000`).

### 9. Docker
```
docker build -t klia-api .
docker run --rm -p 8000:8000 --env-file .env klia-api
```
The image holds only serving code (no training libraries), runs as a non-root user and has a health check. It runs a single worker on purpose, because the model lives in that process's memory.

**Free hosting (optional):** `render.yaml` deploys the image to Render's free plan. Set `DATABASE_URL` in the Render dashboard. Free instances sleep after about 15 idle minutes and take roughly a minute to wake. Check Render's current free-plan terms before relying on it. To protect a public API, set `API_KEY` and send it as the `X-API-Key` header.

---

## Design notes

- **Incremental ETL.** A watermark (`etl_state`) records the last processed `id`. The new bundle and the watermark are saved in one transaction, so a crashed run can simply be repeated.
- **No training/serving skew.** Every feature is a running statistic in `FeatureState`. Training calls `features()` then `update()` for each flight; the API calls `features()` on the saved state. A test proves the two produce identical numbers.
- **No label leakage.** A flight's own outcome is written to the state only after its features were read.
- **Incremental learning, not row-by-row online learning.** Models (`klia/model/incremental.py`) are scikit-learn `partial_fit` estimators -- `sgd_log`, `sgd_hinge` (modified Huber), `gnb`, `mlp` -- plus two incremental ensembles (bagging, random subspace). Each update job run calls `partial_fit` **once per batch** of new rows, never per-row and never a full refit. Each batch is scored before it is learned from (prequential), giving honest out-of-sample metrics. River's ADWIN detector watches the batch-level error and counts drift events; it is used only for drift detection, not for modeling. The decision threshold is re-fitted for F1 on the last 5,000 predictions.
- **Model selection and tuning (`notebooks/01_model_selection_eda.ipynb`).** Candidate models and ensembles are compared on a subset of the data, the winner is tuned with Optuna on the same subset, then retrained on the full dataset and logged to the **MLflow Model Registry**. See step 4 above.
- **Model bundle = one file** (model, feature state, threshold, metrics, library versions), gzip-pickled. Stored in Neon's `model_registry` table (newest 3 kept) when `DATABASE_URL` is set, capped at `registry.max_mb_neon` (400 MB) there; stored locally with no size cap otherwise. Bundles are also tracked in MLflow regardless of where the serving copy lives. Only load bundles you created yourself.
- **Good to know.** The KLIA weather forecast comes from Open-Meteo; if it is unreachable the API falls back to the training averages and says so in `weather_source`. Add new public holidays to `features.public_holidays` in `config/config.yaml` each year. Rows that fail validation are counted by reason in the job log, never silently fixed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `DATABASE_URL is not set` | create `.env` (step 3) or set the variable in your shell |
| `missing required columns` | column names differ from the expected ones; see step 3 |
| API returns 503 `no model available` | run `python -m klia.jobs.update --bootstrap` (step 5) |
| `another update job is already running` | wait for it; with the file store (CSV mode) delete `artifacts/store/update.lock` |
| bundle warns about a different `scikit-learn` version | install the same scikit-learn version as when the bundle was trained, or re-run the notebook and `--bootstrap` |
| model bundle exceeds the Neon cap (400 MB) | raise `registry.max_mb_neon` in `config/config.yaml`, shrink the ensemble (`n_estimators`), or store locally instead (no cap) |
| Neon connection is slow on the first request | the free database auto-suspends; the first wake-up takes a moment |
