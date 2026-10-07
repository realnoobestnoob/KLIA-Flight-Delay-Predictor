"""Optuna hyperparameter tuning for the KLIA flight delay predictor.

Tunes XGBoost params, ensemble settings, and feature top_k simultaneously.
Objective: maximise 0.4·AUC + 0.6·F1 while penalising train/val AUC gap (overfitting).
Slightly prioritises F1 (60%) over AUC (40%) for better precision-recall balance.

Usage:
    python -m klia.jobs.tune                        # fetch from Neon
    python -m klia.jobs.tune --csv data/departures.csv
    python -m klia.jobs.tune --trials 60            # override n_optuna_trials
    python -m klia.jobs.tune --sample 20000         # override sample row count

After completion, best params are written to config/config.yaml under
tuning.best_params, and printed with exact config.yaml lines to apply.
Apply them manually then run:
    python -m klia.jobs.update --bootstrap
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from klia.config import load_config
from klia.etl.validate import clean
from klia.features.state import FeatureState
from klia.features.selection import FeatureSelector
from klia.model.incremental import RandomSubspaceEnsemble, batch_prequential, best_threshold
from klia.store.base import open_store

log = logging.getLogger("klia.tune")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


# ─────────────────────────────── data ────────────────────────────────────────

def _fetch_all_features(cfg: dict, store, sample_rows: int) -> tuple[list[dict], list[int]]:
    """Fetch rows, validate, run feature engineering with all features (no selection yet).

    FeatureState is read-before-write per row — no leakage.
    Returns (X_all, y_all) with all 21 features present in every dict.
    """
    log.info("fetching up to %d rows for tuning...", sample_rows)
    raw = store.fetch_new_rows(0, sample_rows)
    if raw.empty:
        raise RuntimeError("no rows available in store")

    ok, bad = clean(raw, cfg)
    log.info("validation: %d ok, %d rejected", len(ok), len(bad))
    if ok.empty:
        raise RuntimeError("all rows rejected by validation")

    state = FeatureState(cfg)
    X_all: list[dict] = []
    y_all: list[int]  = []
    for rec in ok.to_dict("records"):
        label = int(rec["is_delayed"])
        feat  = state.features(rec)
        X_all.append(feat)
        y_all.append(label)
        state.update(rec, label)

    n_features = len(X_all[0]) if X_all else 0
    delay_rate = 100.0 * sum(y_all) / len(y_all) if y_all else 0.0
    log.info("feature engineering done: %d rows, %d features, %.1f%% delayed",
             len(y_all), n_features, delay_rate)
    return X_all, y_all


def _select_features(X_all: list[dict], y_all: list[int],
                     top_k: int, probe_rows: int) -> list[dict]:
    """Run SGD probe on the first probe_rows rows; return X filtered to top_k features.

    Uses a fresh FeatureSelector per top_k value (cached by the caller).
    Falls back to all features if the probe fails.
    """
    selector = FeatureSelector()
    ok = selector.fit_from_probe(X_all[:probe_rows], y_all[:probe_rows], top_k=top_k)
    if not ok:
        log.warning("probe failed for top_k=%d; using all features", top_k)
        return X_all
    return [selector.filter(x) for x in X_all]


# ─────────────────────────── model builder ───────────────────────────────────

def _build_model(params: dict, seed: int = 42) -> RandomSubspaceEnsemble:
    """Instantiate rsub_xgb from a flat params dict produced by the Optuna trial."""
    return RandomSubspaceEnsemble(
        base="xgb",
        n_estimators=params["n_estimators"],
        subspace_frac=params["subspace_frac"],
        params={
            "eta":              params["eta"],
            "max_depth":        params["max_depth"],
            "subsample":        params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"],
            "n_iter":           params["n_iter"],
        },
        seed=seed,
    )


# ─────────────────────────── evaluation ──────────────────────────────────────

def _evaluate(
    X: list[dict], y: list[int], params: dict,
    n_splits: int, batch_size: int,
) -> tuple[float, float, float, float]:
    """StratifiedKFold cross-validation that mirrors the production update loop.

    Each fold:
      - trains with batch_prequential (score-before-learn, no leakage)
      - evaluates on the held-out fold
      - tunes decision threshold on held-out fold (F1-maximizing)
      - measures train/val AUC gap as an overfitting proxy

    Returns (mean_val_auc, mean_f1, mean_gap, mean_threshold).
    A gap > 0.05 is a signal of overfitting; it is penalised in the objective.
    Threshold is tuned per fold on the hold-out set, then averaged (for audit).
    """
    y_arr = np.array(y)
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1s, gaps, thresholds = [], [], [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y_arr)):
        X_tr  = [X[i] for i in tr_idx]
        y_tr  = y_arr[tr_idx].tolist()
        X_val = [X[i] for i in val_idx]
        y_val = y_arr[val_idx]

        model    = _build_model(params, seed=42 + fold_idx)
        p_train  = batch_prequential(model, X_tr, y_tr,
                                     batch_size=batch_size, warmup_batches=1)

        if model.n_learned == 0 or y_val.min() == y_val.max():
            aucs.append(0.5); f1s.append(0.0); gaps.append(0.0); thresholds.append(0.5)
            continue

        p_val   = np.atleast_1d(model.predict_proba(X_val, fallback=0.3))
        val_auc = float(roc_auc_score(y_val, p_val))
        
        # ── Tune decision threshold on validation fold (F1-maximizing) ──────
        threshold, f1 = best_threshold(p_val, y_val, default=0.5)
        thresholds.append(threshold)

        # Train AUC on post-warmup portion (mirrors production: first batch is not scored)
        warmup_n   = batch_size
        scored_y   = np.array(y_tr[warmup_n:])
        scored_p   = p_train[warmup_n:]
        if len(scored_y) > 0 and scored_y.min() != scored_y.max():
            train_auc = float(roc_auc_score(scored_y, scored_p))
        else:
            train_auc = val_auc

        gap = max(0.0, train_auc - val_auc)
        log.debug("fold %d: val_auc=%.4f f1=%.4f threshold=%.2f gap=%.4f",
                  fold_idx, val_auc, f1, threshold, gap)

        aucs.append(val_auc)
        f1s.append(f1 if f1 > 0.0 else 0.0)
        gaps.append(gap)

    return float(np.mean(aucs)), float(np.mean(f1s)), float(np.mean(gaps)), float(np.mean(thresholds))


# ─────────────────────────── optuna objective ────────────────────────────────

def make_objective(X_all: list[dict], y_all: list[int], cfg: dict, tune_cfg: dict):
    """Return the Optuna objective closure.

    Caches filtered-X per top_k value so the SGD probe only runs once per
    distinct top_k value across all trials (not once per trial).
    """
    probe_rows  = int(cfg["feature_selection"].get("probe_rows", 5_000))
    n_splits    = int(tune_cfg.get("cv_folds",    3))
    batch_size  = int(tune_cfg.get("batch_size", 500))
    gap_penalty = float(tune_cfg.get("gap_penalty", 0.3))

    # top_k → filtered X  (avoid re-running probe for same top_k across trials)
    _cache: dict[int, list[dict]] = {}

    def objective(trial: optuna.Trial) -> float:
        # ── suggest hyperparameters ──────────────────────────────────────────
        top_k = trial.suggest_int("top_k",
                                   tune_cfg.get("top_k_min", 12),
                                   tune_cfg.get("top_k_max", 21))
        if top_k not in _cache:
            _cache[top_k] = _select_features(X_all, y_all, top_k, probe_rows)
        X = _cache[top_k]

        params = {
            "top_k":            top_k,
            "eta":              trial.suggest_float("eta",
                                    tune_cfg.get("eta_min", 0.01),
                                    tune_cfg.get("eta_max", 0.30), log=True),
            "max_depth":        trial.suggest_int("max_depth",
                                    tune_cfg.get("max_depth_min", 3),
                                    tune_cfg.get("max_depth_max", 7)),
            "subsample":        trial.suggest_float("subsample",
                                    tune_cfg.get("subsample_min", 0.60),
                                    tune_cfg.get("subsample_max", 1.00)),
            "colsample_bytree": trial.suggest_float("colsample_bytree",
                                    tune_cfg.get("colsample_min", 0.50),
                                    tune_cfg.get("colsample_max", 1.00)),
            "min_child_weight": trial.suggest_int("min_child_weight",
                                    tune_cfg.get("min_child_weight_min",  1),
                                    tune_cfg.get("min_child_weight_max", 20)),
            "n_iter":           trial.suggest_int("n_iter",
                                    tune_cfg.get("n_iter_min", 10),
                                    tune_cfg.get("n_iter_max", 50)),
            "n_estimators":     trial.suggest_int("n_estimators",
                                    tune_cfg.get("n_estimators_min",  3),
                                    tune_cfg.get("n_estimators_max", 10)),
            "subspace_frac":    trial.suggest_float("subspace_frac",
                                    tune_cfg.get("subspace_frac_min", 0.50),
                                    tune_cfg.get("subspace_frac_max", 0.95)),
        }

        # ── cross-validate (with threshold tuning per fold) ──────────────────
        auc, f1, gap, threshold = _evaluate(X, y_all, params, n_splits, batch_size)

        # ── composite objective ──────────────────────────────────────────────
        # 60% F1, 40% AUC — slightly prioritise F1 for better recall/precision balance.
        # gap penalty discourages overfitting.
        # gap_penalty=0.3 means a 0.05 gap costs 0.015 off the score.
        score = 0.4 * auc + 0.6 * f1 - gap_penalty * gap

        trial.set_user_attr("auc",       round(auc,       4))
        trial.set_user_attr("f1",        round(f1,        4))
        trial.set_user_attr("gap",       round(gap,       4))
        trial.set_user_attr("threshold", round(threshold, 2))
        log.info(
            "trial %3d | score=%.4f  auc=%.4f  f1=%.4f  gap=%.4f  threshold=%.2f | "
            "top_k=%d eta=%.4f depth=%d sub=%.2f col=%.2f mcw=%d "
            "n_iter=%d n_est=%d sfrac=%.2f",
            trial.number, score, auc, f1, gap, threshold,
            params["top_k"], params["eta"], params["max_depth"],
            params["subsample"], params["colsample_bytree"], params["min_child_weight"],
            params["n_iter"], params["n_estimators"], params["subspace_frac"],
        )
        return score

    return objective


# ─────────────────────── write results back ──────────────────────────────────

def _write_best_params(best_params: dict) -> None:
    """Write best_params into config.yaml under tuning.best_params (for audit)."""
    with open(_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    raw.setdefault("tuning", {})["best_params"] = best_params
    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    log.info("best params saved to config.yaml under tuning.best_params")


def _print_apply_instructions(best_params: dict) -> None:
    """Print the exact config.yaml stanza the user should paste to apply the results."""
    p = best_params
    print("""
──────────────────────────────────────────────────────
  Apply best params — paste into config/config.yaml
──────────────────────────────────────────────────────

model:
  name: rsub_xgb
  params:
    n_estimators: {n_estimators}
    subspace_frac: {subspace_frac:.2f}
    eta: {eta:.4f}
    max_depth: {max_depth}
    subsample: {subsample:.2f}
    colsample_bytree: {colsample_bytree:.2f}
    min_child_weight: {min_child_weight}
  boosting_n_iter: {n_iter}
  calibration_window: 5000              # update if needed
  decision_threshold: {threshold:.2f}   # NEW: tuned by Optuna; overrides dynamic threshold tuning

feature_selection:
  enabled: true
  top_k: {top_k}
  probe_rows: 5000   # keep as-is

──────────────────────────────────────────────────────
Then run:
  rm artifacts/pretrained_bundle.gz   # or del on Windows
  python -m klia.jobs.update --bootstrap

Note: decision_threshold is now tuned during Optuna runs and stored
in config.yaml. To go back to dynamic threshold tuning (per-run F1
maximisation), delete the decision_threshold line from config.yaml.
──────────────────────────────────────────────────────
""".format(**p))


# ──────────────────────────────── main ───────────────────────────────────────

def run(n_trials: int | None = None, sample_rows: int | None = None,
        csv: str | None = None) -> dict:
    cfg      = load_config()
    tune_cfg = cfg.get("tuning", {})
    trials   = n_trials   or cfg.get("eda",    {}).get("n_optuna_trials", 30)
    sample   = sample_rows or tune_cfg.get("sample_rows", 15_000)

    store = open_store(cfg, csv)
    store.ensure_schema()

    X_all, y_all = _fetch_all_features(cfg, store, sample)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    log.info("starting Optuna study: %d trials, sample=%d rows, cv_folds=%d",
             trials, sample, tune_cfg.get("cv_folds", 3))
    t0 = time.time()
    study.optimize(
        make_objective(X_all, y_all, cfg, tune_cfg),
        n_trials=trials,
        show_progress_bar=True,
    )

    best  = study.best_trial
    best_threshold = best.user_attrs.get("threshold", 0.5)
    result = {
        "best_score":      round(best.value, 4),
        "best_auc":        best.user_attrs.get("auc", 0.0),
        "best_f1":         best.user_attrs.get("f1",  0.0),
        "best_gap":        best.user_attrs.get("gap", 0.0),
        "best_threshold":  best_threshold,
        "best_params":     best.params,
        "n_trials":        trials,
        "seconds":         round(time.time() - t0, 1),
    }

    log.info(
        "best trial #%d: score=%.4f  auc=%.4f  f1=%.4f  gap=%.4f  threshold=%.2f",
        best.number, best.value,
        result["best_auc"], result["best_f1"], result["best_gap"], best_threshold,
    )
    # Store threshold in best_params for later retrieval
    best.params["threshold"] = best_threshold
    _write_best_params(best.params)
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trials", type=int,
                    help="number of Optuna trials (default: eda.n_optuna_trials from config.yaml)")
    ap.add_argument("--sample", type=int,
                    help="rows to fetch from Neon (default: tuning.sample_rows from config.yaml)")
    ap.add_argument("--csv",    help="read rows from this CSV instead of Neon")
    a = ap.parse_args()

    result = run(n_trials=a.trials, sample_rows=a.sample, csv=a.csv)

    print(f"\n── Tuning complete ({result['seconds']}s) ──")
    print(f"  Trials run : {result['n_trials']}")
    print(f"  Best score : {result['best_score']:.4f}  "
          f"(AUC={result['best_auc']:.4f}  F1={result['best_f1']:.4f}  "
          f"gap={result['best_gap']:.4f}  threshold={result['best_threshold']:.2f})")
    _print_apply_instructions(result["best_params"])


if __name__ == "__main__":
    main()