"""Core update logic, free of any database code so it can be tested and reused by the notebook.

The model is updated with ONE `partial_fit` call per batch of new rows (incremental learning),
not one call per row.  Within a single update-job run, each fetch from the store (up to
`data.batch_size` rows) is exactly one such batch.

Feature selection
-----------------
If bundle.selector is fitted (set during --bootstrap by the probe phase in update.py),
each feature dict is filtered to the selected subset AFTER state.features() computes it
but BEFORE the model sees it.  This means:
  - FeatureState always receives the full row and accumulates all running statistics
    (no information is lost from the state).
  - The model trains only on the selected features; model.feature_names locks to that
    subset on the very first partial_fit call.
  - Serving calls selector.filter() on the same dict → zero training/serving skew.

Drift detection uses Evidently AI (DataDriftPreset) to compare each update batch against the
reference feature distribution stored in bundle.meta["drift_reference"] by the notebook.
Results accumulate in bundle.meta["drift_log"] (capped at monitoring.drift_log_max entries).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from klia.features.state import FeatureState
from klia.model.bundle import Bundle


def feature_stream(ok: pd.DataFrame, state: FeatureState,
                   bundle: Bundle | None = None) -> tuple[list[dict], list[int]]:
    """Replay cleaned rows (already time-sorted) through the state: read features, then update.

    The label of a row is only written to the state AFTER its features were read, for every row,
    so there is no leakage regardless of how the rows are later grouped into batches.

    If bundle.selector is fitted, each feature dict is filtered to the selected subset
    before being appended to X — the model only ever sees the selected features.
    FeatureState is always updated with the full row regardless of selection.
    """
    selector = bundle.selector if bundle is not None else None
    X, y = [], []
    for rec in ok.to_dict("records"):
        label = int(rec["is_delayed"])
        feat  = state.features(rec)
        if selector is not None and selector.is_fitted:
            feat = selector.filter(feat)
        X.append(feat)
        y.append(label)
        state.update(rec, label)
    return X, y


def apply_rows(bundle: Bundle, ok: pd.DataFrame, cfg: dict) -> dict:
    """Learn from one batch of new, cleaned rows (in place).  Returns stats for the run log."""
    if ok.empty:
        return {"rows_learned": 0}

    mcfg   = cfg["model"]
    mon    = cfg.get("monitoring", {})
    X, y   = feature_stream(ok, bundle.state, bundle)   # selector applied inside

    # Score this batch BEFORE learning (out-of-sample), unless the model is still warming up.
    warm = bundle.model.n_learned < mcfg.get("warmup_rows", 500)
    base = bundle.state.base_rate
    p    = [base] * len(y) if warm else bundle.model.predict_proba(X, fallback=base)

    # ── Drift detection (Evidently AI) ────────────────────────────────────────
    # Skipped when: model is warming up, batch is too small, or no reference is stored yet.
    drift_fired  = 0
    drift_report: dict = {}
    min_batch    = mon.get("drift_min_batch", 50)

    if not warm and len(X) >= min_batch and bundle.meta.get("drift_reference"):
        try:
            from klia.monitoring.drift import run_drift_report
            drift_report = run_drift_report(X, bundle.meta)
            drift_fired  = 1 if drift_report.get("drift_detected", False) else 0
        except Exception as exc:
            drift_report = {
                "ts":             dt.datetime.utcnow().isoformat(timespec="seconds"),
                "drift_detected": False,
                "error":          str(exc),
            }

    # ── Learn ─────────────────────────────────────────────────────────────────
    bundle.model.partial_fit(X, y)
    if not warm:
        bundle.remember(p, y, mcfg["calibration_window"])
    bundle.drift_events += drift_fired

    # ── Persist drift log in bundle.meta ──────────────────────────────────────
    if drift_report:
        log = bundle.meta.setdefault("drift_log", [])
        log.append({
            "ts":             drift_report.get("ts", dt.datetime.utcnow().isoformat(timespec="seconds")),
            "rows":           len(X),
            "drift_detected": drift_report.get("drift_detected", False),
            "share_drifted":  drift_report.get("share_drifted",  0.0),
            "n_drifted":      drift_report.get("n_drifted",      0),
            "n_features":     drift_report.get("n_features",     0),
            "error":          drift_report.get("error"),
        })
        max_log = mon.get("drift_log_max", 100)
        bundle.meta["drift_log"]          = log[-max_log:]
        bundle.meta["last_drift_report"]  = drift_report

    # ── Stats for the run log ─────────────────────────────────────────────────
    stats = {
        "rows_learned": len(y),
        "drift_events": drift_fired,
        "scored_rows":  0 if warm else len(y),
    }
    bundle.meta.update({
        "n_rows":     bundle.model.n_learned,
        "last_sched": str(ok["sched_dt"].iloc[-1]),
        "model_name": bundle.model.name,
    })
    stats.update(bundle.refresh_threshold())
    bundle.meta["metrics"] = {
        k: v for k, v in stats.items()
        if k in ("roc_auc", "f1", "log_loss", "delay_rate", "n_scored")
    }
    return stats
