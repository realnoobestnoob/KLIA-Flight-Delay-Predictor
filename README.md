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
| `klia/model/` | online models (River) and the saved model bundle |
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
```
jupyter lab notebooks/01_model_selection_eda.ipynb
```
Choose **Run → Run All Cells**. It explores your data, compares the online models in time order, and writes
`artifacts/model_choice.json`. Without `DATABASE_URL` it uses fake demo data and does **not** save a choice.

### 5. Build the first model from your full history (once)
```
python -m klia.jobs.update --bootstrap
```
This creates three small tables in Neon (`etl_state`, `model_registry`, `etl_runs`). It never changes your `departures` table.

### 6. Update as new flights arrive
```
python -m klia.jobs.update
```
It reads **only rows newer than the stored watermark**. With no new rows it does nothing, so it is safe to run as often as you like. Schedule it daily:
- **Windows:** Task Scheduler → Create Basic Task → Daily → Start a program: `<project>\.venv\Scripts\python.exe`, arguments `-m klia.jobs.update`, start in `<project>`.
- **Or free in the cloud:** push to GitHub, add a repository secret `DATABASE_URL`, and `.github/workflows/update.yml` runs it every day at 04:30 Malaysia time.

For the update job, prefer Neon's **direct** (non-pooled) connection string; it uses a database lock so two runs cannot overlap.

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
- **Online learning.** The model predicts each flight before learning from it, which gives honest rolling metrics. An ADWIN detector watches the error stream and counts drift events. The decision threshold is re-fitted for F1 on the last 5,000 predictions.
- **Model bundle = one file** (model, feature state, threshold, metrics, library versions), gzip-pickled and stored in Neon, with the newest 3 kept. River is pinned to an exact version because pickles must be loaded by the same version that wrote them. Only load bundles you created yourself.
- **Good to know.** The KLIA weather forecast comes from Open-Meteo; if it is unreachable the API falls back to the training averages and says so in `weather_source`. Add new public holidays to `features.public_holidays` in `config/config.yaml` each year. Rows that fail validation are counted by reason in the job log, never silently fixed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `DATABASE_URL is not set` | create `.env` (step 3) or set the variable in your shell |
| `missing required columns` | column names differ from the expected ones; see step 3 |
| API returns 503 `no model available` | run `python -m klia.jobs.update --bootstrap` (step 5) |
| `another update job is already running` | wait for it; with the file store (CSV mode) delete `artifacts/store/update.lock` |
| bundle warns about a different `river` version | install the pinned version from `requirements-serve.txt`, or re-run `--bootstrap` |
| Neon connection is slow on the first request | the free database auto-suspends; the first wake-up takes a moment |
