"""Generates notebooks/01_model_selection_eda.ipynb. Run: python build_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

M("""
# KLIA flight delays: EDA and online-model selection

This notebook (1) checks your data, (2) explores it, and (3) compares models that can **learn one row at a time**,
then saves the winner to `artifacts/model_choice.json`. The update job reads that file.

**How to run:** put your Neon `DATABASE_URL` in `.env`, then *Run All*. It takes a few minutes for ~65k rows.
No `DATABASE_URL`? It falls back to synthetic demo data so you can see how it works, but **demo results mean nothing for your real flights**.

**How candidates are judged (prequential / test-then-train):** each flight is first predicted, then learned from. Every score is
therefore out-of-sample, in time order, exactly as in production. A batch LightGBM trained on the first 70% is shown as a reference only; it
cannot be updated cheaply and is not a candidate.
""")
C("""
import os, time, json, gzip, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
env = ROOT / ".env"
if env.exists():                                   # tiny .env loader, no extra dependency
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.figsize": (10, 3.6), "axes.grid": True, "grid.alpha": .25})

from klia.config import load_config, database_url, artifacts_dir
from klia.etl.validate import clean
from klia.features.state import FeatureState
from klia.pipeline import feature_stream
from klia.model.online import OnlineClassifier, prequential, best_threshold
from river import drift
cfg = load_config()
""")
M("## 1. Load data")
C("""
url = database_url()
if url:
    from sqlalchemy import create_engine
    raw = pd.read_sql(f"SELECT * FROM {cfg['data']['table']} WHERE actual_departure IS NOT NULL ORDER BY id", create_engine(url))
    SOURCE = "Neon"
else:
    from klia.demo import make
    raw = make(20000)
    SOURCE = "DEMO (synthetic)"
print(f"source: {SOURCE} | rows: {len(raw):,} | columns: {list(raw.columns)}")
raw.head()
""")
M("## 2. Data quality: what the validator accepts and rejects")
C("""
ok, bad = clean(raw, cfg)
print(f"accepted {len(ok):,} | rejected {len(bad):,} ({len(bad)/max(len(raw),1):.2%})")
if len(bad):
    display(bad["reason"].value_counts().to_frame("rows"))
    display(bad.head(5))
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

top = ok.groupby("airline").is_delayed.agg(["mean", "count"]).query("count >= 100").sort_values("mean")
top["mean"].plot.barh(figsize=(8, max(3, .3 * len(top))), title="Delay rate by airline (>= 100 flights)"); plt.show()

monthly = ok.set_index("sched_dt").is_delayed.resample("MS").agg(["mean", "count"])
fig, a = plt.subplots(figsize=(10, 3.4)); monthly["mean"].plot(ax=a, marker="o", title="Monthly delay rate (a moving target is why we use online learning)")
a.set_ylim(0, None); plt.show()
""")
C("""
wc = cfg["features"]["weather_cols"]
if ok[wc].notna().any().any():
    print("weather coverage:", (ok[wc].notna().mean() * 100).round(1).to_dict(), "(%)")
    d = ok.assign(gust_bin=pd.qcut(ok[wc[0]], 5, duplicates="drop"))
    d.groupby("gust_bin", observed=True).is_delayed.mean().plot.bar(title=f"Delay rate by {wc[0]} quintile"); plt.show()
else:
    print("no weather columns found, weather features default to 0 (add them to the table to use them)")
""")
M("""
## 4. Features, built once with the production code
`feature_stream` reads each row's features **before** that row's outcome is written into the running statistics. Nothing here can leak the label.
""")
C("""
state = FeatureState(cfg)
t0 = time.time(); X, y = feature_stream(ok, state); y = np.array(y)
print(f"{len(X):,} rows x {len(X[0])} features in {time.time()-t0:.1f}s | features: {list(X[0])}")
Xdf = pd.DataFrame(X)
corr = Xdf.assign(y=y).corr()["y"].drop("y").sort_values()
corr.plot.barh(figsize=(8, 6), title="Correlation of each feature with 'delayed'"); plt.show()
""")
M("""
## 5. Candidate online models
All candidates learn from one flight at a time and can be saved/loaded as a small file.
Settings are deliberately **size-capped** (tree depth, few trees): an unconstrained adaptive forest grows without limit and would outgrow the free tier.
""")
C("""
CANDIDATES = {
    "baseline: airline rate only": None,
    "logreg":                {"name": "logreg", "params": {}},
    "hat":                   {"name": "hat",    "params": {}},
    "arf (5 trees, depth 8)": {"name": "arf",   "params": {"n_models": 5,  "max_depth": 8, "grace_period": 300}},
    "arf (10 trees, depth 6)": {"name": "arf",  "params": {"n_models": 10, "max_depth": 6, "grace_period": 300}},
    "arf (10 trees, depth 8)": {"name": "arf",  "params": {"n_models": 10, "max_depth": 8, "grace_period": 300}},
}
WARM = min(1000, len(y) // 10)
results, probs = [], {}
for label, spec in CANDIDATES.items():
    if spec is None:                                       # a model-free yardstick
        p = Xdf["airline_rate"].values; secs = 0.0; mb = 0.0
    else:
        m = OnlineClassifier(spec["name"], spec["params"]); t0 = time.time()
        p, fired = prequential(m, X, y.tolist(), warmup=WARM, detector=drift.ADWIN(delta=0.002), fallback=state.base_rate)
        secs = time.time() - t0; mb = len(gzip.compress(pickle.dumps(m))) / 1e6
    probs[label] = p
    s, yy = p[WARM:], y[WARM:]
    thr, f1 = best_threshold(s, yy)
    results.append({"model": label, "AUC": roc_auc_score(yy, s), "logloss": log_loss(yy, np.clip(s, 1e-6, 1-1e-6)),
                    "brier": brier_score_loss(yy, s), "best_F1": f1, "ms_per_row": 1000 * secs / len(y), "bundle_MB": mb,
                    "spec": spec})
    print(f"done {label}")
res = pd.DataFrame(results)
res.drop(columns="spec").round(4).sort_values("AUC", ascending=False).reset_index(drop=True)
""")
C("""
# Reference only: batch LightGBM trained once on the first 70% (cannot learn incrementally)
try:
    import lightgbm as lgb
    k = int(len(y) * .7)
    g = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, verbose=-1).fit(Xdf[:k], y[:k])
    ref = roc_auc_score(y[k:], g.predict_proba(Xdf[k:])[:, 1])
    print(f"batch LightGBM on the last 30%: AUC {ref:.4f}")
    for label, p in probs.items():
        print(f"  {label:28s} same last 30%: AUC {roc_auc_score(y[k:], p[k:]):.4f}")
except ImportError:
    print("lightgbm not installed, skipping the reference point")
""")
C("""
# Rolling AUC over time: does a model keep up as the data changes?
fig, a = plt.subplots(figsize=(11, 4))
win = max(1500, len(y) // 12)
for label, p in probs.items():
    if label.startswith("baseline"): continue
    pts = []
    for i in range(WARM + win, len(y), win // 2):
        yy = y[i-win:i]
        if yy.min() != yy.max(): pts.append((ok.sched_dt.iloc[i-1], roc_auc_score(yy, p[i-win:i])))
    if pts: a.plot(*zip(*pts), label=label)
a.set_title(f"Rolling AUC (window {win:,} flights)"); a.legend(fontsize=8); plt.show()
""")
M("""
## 6. Choose
Rule: among models that fit the free tier (**bundle <= 15 MB** gzipped, **<= 3 ms** to learn a row), take the best AUC;
if several are within **0.005 AUC** of the best, prefer the smallest bundle, i.e. the simplest model. Change the limits if you like.
""")
C("""
MAX_MB, MAX_MS, TIE = 15, 3, 0.005
elig = res[res.spec.notna() & (res.bundle_MB <= MAX_MB) & (res.ms_per_row <= MAX_MS)]
best = elig.AUC.max()
pick = elig[elig.AUC >= best - TIE].sort_values("bundle_MB").iloc[0]
display(elig.drop(columns="spec").round(4).sort_values("AUC", ascending=False))
print(f"\\nCHOSEN: {pick.model}   AUC {pick.AUC:.4f}   bundle {pick.bundle_MB:.1f} MB   {pick.ms_per_row:.2f} ms/row")
base_auc = res[res.model.str.startswith("baseline")].AUC.iloc[0]
print(f"lift over the airline-rate baseline: {pick.AUC - base_auc:+.4f} AUC")
if SOURCE.startswith("DEMO"):
    print("\\nDEMO data: NOT saving a choice. Connect your Neon DATABASE_URL and re-run to pick a model for real.")
else:
    out = artifacts_dir() / "model_choice.json"
    out.write_text(json.dumps({"name": pick.spec["name"], "params": pick.spec["params"],
                               "auc": round(float(pick.AUC), 4), "rows": int(len(y)),
                               "chosen_by": "notebooks/01_model_selection_eda.ipynb", "label": pick.model}, indent=2))
    print("saved", out)
""")
M("""
## 7. What to do next
1. Commit `artifacts/model_choice.json` (it is tiny).
2. Run `python -m klia.jobs.update --bootstrap` to build the first model from your full history.
3. From then on, schedule `python -m klia.jobs.update` (daily is plenty). It reads **only new rows**.
""")
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/01_model_selection_eda.ipynb")
print("wrote notebook")
