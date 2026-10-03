"""Generates notebooks/01_model_selection_eda.ipynb. Run: python build_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

M("""
# KLIA flight delays: EDA, model selection and tuning (incremental learning)

**Workflow:**
1. Load data, explore it
2. On a **subset** of the data: compare single incremental models -> pick a **benchmark**
3. On the **same subset**: try **ensembles** (bagging, random subspace) -> compare to the benchmark
4. Pick the overall best model/ensemble, then **tune its hyperparameters with Optuna** -- still on the subset only
5. Train the **tuned model on the FULL dataset** (chunked `partial_fit` calls, exactly like production)
6. Save the trained bundle and **promote it in the MLflow Model Registry**

All candidates use **incremental learning**: each is updated with `partial_fit` on a CHUNK of rows at a
time (never one row at a time, never refit from scratch). This matches exactly how the production
update job works: one `partial_fit` call per batch of new flights.

**Demo data warning:** if `DATABASE_URL` is not in `.env`, this runs on synthetic data and will NOT
save a model choice at the end. Connect your Neon database and re-run for a real result.
""")

C("""
import os, time, json, warnings, gzip, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.figsize": (10, 3.6), "axes.grid": True, "grid.alpha": .25})

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)

env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from klia.config import load_config, database_url, artifacts_dir
from klia.etl.validate import clean
from klia.features.state import FeatureState
from klia.pipeline import feature_stream
from klia.model import incremental as inc
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

cfg = load_config()
ECFG = cfg["eda"]

MLFLOW_DB = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("klia-flight-delay")
print(f"MLflow tracking: {MLFLOW_DB}")
print("Browse: mlflow ui --backend-store-uri sqlite:///mlflow.db   (http://127.0.0.1:5000)")
""")

M("## 1. Load data")
C("""
url = database_url()
if url:
    from sqlalchemy import create_engine
    raw = pd.read_sql(
        f"SELECT * FROM {cfg['data']['table']} WHERE actual_departure IS NOT NULL ORDER BY id",
        create_engine(url))
    SOURCE = "Neon"
else:
    from klia.demo import make
    raw = make(20000)
    SOURCE = "DEMO (synthetic)"

print(f"source: {SOURCE} | rows: {len(raw):,} | columns: {list(raw.columns)}")
raw.head()
""")

M("## 2. Data quality")
C("""
ok, bad = clean(raw, cfg)
print(f"accepted {len(ok):,} | rejected {len(bad):,} ({len(bad)/max(len(raw),1):.2%})")
if len(bad):
    display(bad["reason"].value_counts().to_frame("rows"))
print("date range:", ok.sched_dt.min(), "->", ok.sched_dt.max())
print(f"delay rate (>= {cfg['data']['delay_threshold_minutes']} min): {ok.is_delayed.mean():.1%}")
ok[["delay_min"]].describe().T
""")

M("## 3. Exploration")
C("""
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ok.delay_min.clip(-30, 180).plot.hist(bins=70, ax=ax[0], title="Delay minutes (clipped)")
ok.groupby(ok.sched_dt.dt.hour).is_delayed.mean().plot.bar(ax=ax[1], title="Delay rate by hour")
ok.groupby(ok.sched_dt.dt.dayofweek).is_delayed.mean().plot.bar(ax=ax[2], title="Delay rate by weekday (0=Mon)")
plt.tight_layout(); plt.show()

top = ok.groupby("airline").is_delayed.agg(["mean","count"]).query("count>=100").sort_values("mean")
top["mean"].plot.barh(figsize=(8, max(3,.3*len(top))), title="Delay rate by airline (>=100 flights)"); plt.show()
""")

M("""
## 4. Build the subset for model comparison and tuning
We only use **the first `subset_fraction` of the (time-ordered) data** for sections 5-7. This keeps
comparison and tuning fast; the winner is retrained on the full dataset in section 8.
""")
C("""
state_full = FeatureState(cfg)        # the REAL state, built once over all data, reused in section 8
X_full, y_full = feature_stream(ok, state_full)
y_full = np.array(y_full)
print(f"full dataset: {len(X_full):,} rows x {len(X_full[0])} features")

n_sub = max(2000, int(len(X_full) * ECFG["subset_fraction"]))
X_sub, y_sub = X_full[:n_sub], y_full[:n_sub]
BATCH = ECFG["batch_size"]
print(f"subset for comparison/tuning: {n_sub:,} rows ({ECFG['subset_fraction']:.0%} of data), "
     f"batch size {BATCH} ({n_sub // BATCH} batches)")
""")

M("""
## 5. Benchmark: compare single incremental models (subset only)
Each model is updated with `partial_fit` on one batch of `BATCH` rows at a time, scored on each
batch BEFORE learning from it (prequential), exactly like the production update job.
""")
C("""
SINGLE_MODELS = {
    "sgd_log":   {},
    "sgd_hinge": {},
    "gnb":       {},
    "mlp":       {"hidden": 32},
}
WARMUP_BATCHES = 1

def evaluate(name, params, X, y, batch_size=BATCH, log_run=True, run_name=None):
    m = inc.build(name, params)
    t0 = time.time()
    p = inc.batch_prequential(m, X, y, batch_size=batch_size, warmup_batches=WARMUP_BATCHES)
    secs = time.time() - t0
    cut = batch_size * WARMUP_BATCHES
    s, yy = p[cut:], np.asarray(y[cut:])
    auc = roc_auc_score(yy, s) if yy.min() != yy.max() else float("nan")
    ll  = log_loss(yy, np.clip(s, 1e-6, 1-1e-6))
    bs  = brier_score_loss(yy, s)
    thr, f1 = inc.best_threshold(s, yy)
    mb = len(gzip.compress(pickle.dumps(m))) / 1e6
    row = {"model": run_name or name, "AUC": auc, "logloss": ll, "brier": bs, "best_F1": f1,
          "bundle_MB": mb, "ms_per_row": 1000*secs/len(y), "spec": {"name": name, "params": params}}
    if log_run:
        with mlflow.start_run(run_name=row["model"], nested=True):
            mlflow.log_params({"name": name, "params": str(params), "batch_size": batch_size})
            mlflow.log_metrics({k: round(v,4) for k,v in row.items() if isinstance(v,(int,float))})
    return row, m, p

benchmark_results = []
with mlflow.start_run(run_name="1_benchmark_single_models"):
    mlflow.log_params({"rows": n_sub, "batch_size": BATCH, "source": SOURCE})
    for name, params in SINGLE_MODELS.items():
        row, _, _ = evaluate(name, params, X_sub, y_sub)
        benchmark_results.append(row)
        print(f"  {name:12s} AUC {row['AUC']:.4f}  bundle {row['bundle_MB']:.2f} MB  {row['ms_per_row']:.3f} ms/row")

bench_df = pd.DataFrame(benchmark_results)
BENCHMARK = bench_df.loc[bench_df.AUC.idxmax()]
print(f"\\nBENCHMARK: {BENCHMARK.model}  AUC {BENCHMARK.AUC:.4f}")
bench_df.drop(columns="spec").round(4).sort_values("AUC", ascending=False)
""")

M("""
## 6. Try ensembles (subset only)
Two incremental ensemble kinds, built from the base models above:
- **Bagging** -- each member trains on a bootstrap resample of every batch
- **Random subspace** -- each member trains on a random subset of features
""")
C("""
ENSEMBLE_CANDIDATES = {
    "bag_sgd_log":    {"n_estimators": 7},
    "bag_sgd_hinge":  {"n_estimators": 7},
    "rsub_sgd_log":   {"n_estimators": 7, "subspace_frac": 0.7},
    "bag_mlp":        {"n_estimators": 5},
}
ensemble_results = []
with mlflow.start_run(run_name="2_ensembles"):
    mlflow.log_params({"rows": n_sub, "batch_size": BATCH})
    for name, params in ENSEMBLE_CANDIDATES.items():
        row, _, _ = evaluate(name, params, X_sub, y_sub)
        ensemble_results.append(row)
        print(f"  {name:16s} AUC {row['AUC']:.4f}  bundle {row['bundle_MB']:.2f} MB  {row['ms_per_row']:.3f} ms/row")

ens_df = pd.DataFrame(ensemble_results)
all_df = pd.concat([bench_df, ens_df], ignore_index=True)
display(all_df.drop(columns="spec").round(4).sort_values("AUC", ascending=False).reset_index(drop=True))
""")

M("""
## 7. Choose the overall best model or ensemble
Model size is **not** a selection criterion here -- it only matters later if you store the bundle in
Neon (capped at `registry.max_mb_neon`, default 400 MB). Storing it locally via MLflow's own
artifact store has no cap at all. We simply pick the **highest AUC**.
""")
C("""
CHOSEN = all_df.loc[all_df.AUC.idxmax()]
print(f"CHOSEN: {CHOSEN.model}  (AUC {CHOSEN.AUC:.4f} on the {ECFG['subset_fraction']:.0%} subset)")
print(f"vs single-model benchmark {BENCHMARK.model}: {CHOSEN.AUC - BENCHMARK.AUC:+.4f} AUC")
CHOSEN_NAME, CHOSEN_PARAMS = CHOSEN.spec["name"], dict(CHOSEN.spec["params"])
""")

M("""
## 8. Tune hyperparameters with Optuna (subset only)
Still on the same subset used in sections 5-6 -- tuning on the full dataset would be slow and is
unnecessary, since the winning configuration is what gets retrained on the full data next.
""")
C("""
def suggest(trial: optuna.Trial, base_name: str) -> dict:
    is_ensemble = base_name.startswith(("bag_", "rsub_"))
    real_base = base_name.split("_", 1)[1] if is_ensemble else base_name
    p = {}
    if is_ensemble:
        p["n_estimators"] = trial.suggest_int("n_estimators", 3, 15)
        if base_name.startswith("rsub_"):
            p["subspace_frac"] = trial.suggest_float("subspace_frac", 0.4, 0.95)
    if real_base in ("sgd_log", "sgd_hinge"):
        p["alpha"] = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
        p["penalty"] = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    elif real_base == "gnb":
        p["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)
    elif real_base == "mlp":
        p["hidden"] = trial.suggest_categorical("hidden", [16, 32, 64])
        p["alpha"] = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        p["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    return p

N_TRIALS = ECFG["n_optuna_trials"]

with mlflow.start_run(run_name="3_optuna_tuning") as parent:
    mlflow.log_params({"model_type": CHOSEN_NAME, "n_trials": N_TRIALS, "rows": n_sub})

    def objective(trial):
        params = suggest(trial, CHOSEN_NAME)
        row, _, _ = evaluate(CHOSEN_NAME, params, X_sub, y_sub, log_run=False)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params({**params, "trial": trial.number})
            mlflow.log_metrics({"auc": round(row["AUC"], 4), "bundle_mb": round(row["bundle_MB"], 2)})
        return row["AUC"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    TUNED_PARAMS = study.best_params
    mlflow.log_params({f"best_{k}": v for k, v in TUNED_PARAMS.items()})
    mlflow.log_metric("best_auc", round(study.best_value, 4))

print(f"\\nbest AUC: {study.best_value:.4f}  (was {CHOSEN.AUC:.4f} with defaults)")
print(f"tuned params: {TUNED_PARAMS}")
""")

C("""
vals = [t.value for t in study.trials if t.value is not None]
best_so_far = np.maximum.accumulate(vals)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6))
a1.plot(best_so_far, marker="."); a1.set_xlabel("trial"); a1.set_ylabel("AUC"); a1.set_title("Best AUC over trials")
imp = optuna.importance.get_param_importances(study)
pd.Series(imp).sort_values().plot.barh(ax=a2, title="Hyperparameter importance")
plt.tight_layout(); plt.show()
""")

M("""
## 9. Train the tuned model on the FULL dataset
Same `partial_fit`-per-batch procedure as production, just run once over your entire history instead
of a daily slice of new rows. A fresh `FeatureState` is built alongside it (the one from section 4
already saw the full data in order, so we reuse it directly here instead of recomputing).
""")
C("""
from klia.model.bundle import Bundle

final_model = inc.build(CHOSEN_NAME, TUNED_PARAMS)
t0 = time.time()
p_full = inc.batch_prequential(final_model, X_full, y_full.tolist(), batch_size=BATCH, warmup_batches=1)
print(f"trained on {len(X_full):,} rows in {time.time()-t0:.1f}s "
     f"({len(X_full)//BATCH} batches of {BATCH})")

cut = BATCH
full_auc = roc_auc_score(y_full[cut:], p_full[cut:])
print(f"full-dataset prequential AUC: {full_auc:.4f}")

thr, f1 = inc.best_threshold(p_full[cut:], y_full[cut:])
final_bundle = Bundle(model=final_model, state=state_full, threshold=thr,
                      meta={"chosen_by": "notebooks/01_model_selection_eda.ipynb",
                            "subset_auc": round(float(CHOSEN.AUC), 4),
                            "full_auc": round(float(full_auc), 4),
                            "tuned_params": TUNED_PARAMS, "rows": int(len(X_full))})
final_bundle.remember(p_full[cut:], y_full[cut:].tolist(), cfg["model"]["calibration_window"])
final_bundle.refresh_threshold()
print(f"decision threshold: {final_bundle.threshold:.2f}")
""")

M("""
## 10. Save the final model and promote it in MLflow

- `artifacts/pretrained_bundle.gz` -- loaded automatically by `python -m klia.jobs.update --bootstrap`
- `artifacts/model_choice.json` -- a readable record of what was chosen
- Registered in the **MLflow Model Registry** (`klia-flight-delay`) and aliased **`champion`**
""")
C("""
if SOURCE.startswith("DEMO"):
    print("DEMO data -- NOT saving or registering. Connect DATABASE_URL and re-run for a real model.")
else:
    bundle_bytes = final_bundle.dumps()
    bundle_path = artifacts_dir() / "pretrained_bundle.gz"
    bundle_path.write_bytes(bundle_bytes)
    print(f"saved {bundle_path}  ({len(bundle_bytes)/1e6:.2f} MB)")

    cap = cfg["registry"]["max_mb_neon"]
    if len(bundle_bytes) / 1e6 > cap:
        print(f"NOTE: this bundle is over the Neon cap ({cap} MB). It will still be used for "
             f"--bootstrap locally; to also push it through Neon's model_registry table during "
             f"routine updates, shrink the ensemble or store it locally instead (no cap).")

    choice = {"name": CHOSEN_NAME, "params": TUNED_PARAMS, "subset_auc": round(float(CHOSEN.AUC), 4),
             "full_auc": round(float(full_auc), 4), "rows": int(len(X_full)), "n_trials": N_TRIALS,
             "chosen_by": "notebooks/01_model_selection_eda.ipynb"}
    (artifacts_dir() / "model_choice.json").write_text(json.dumps(choice, indent=2))
    print(json.dumps(choice, indent=2))

    from klia.model.mlflow_pyfunc import log_and_register
    sample_input = pd.DataFrame([X_full[0]])
    with mlflow.start_run(run_name="4_final_model"):
        mlflow.log_params({"name": CHOSEN_NAME, **{f"param_{k}": v for k, v in TUNED_PARAMS.items()}})
        mlflow.log_metrics({"subset_auc": round(float(CHOSEN.AUC), 4), "full_auc": round(float(full_auc), 4)})
        version = log_and_register(str(bundle_path), sample_input)
    print(f"\\nregistered 'klia-flight-delay' version {version}, aliased 'champion'")

print("\\n--- Next steps ---")
print("1. python -m klia.jobs.update --bootstrap   # loads pretrained_bundle.gz, replays full history")
print("2. uvicorn klia.api.app:app --port 8000")
print("3. mlflow ui --backend-store-uri sqlite:///mlflow.db")
""")

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/01_model_selection_eda.ipynb")
print("wrote notebooks/01_model_selection_eda.ipynb")
