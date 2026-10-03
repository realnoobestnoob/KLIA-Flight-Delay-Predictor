"""Core update logic, free of any database code so it can be tested and reused by the notebook.

The model is updated with ONE `partial_fit` call per batch of new rows (incremental learning),
not one call per row. Within a single update-job run, each fetch from the store (up to
`data.batch_size` rows) is exactly one such batch.
"""
from __future__ import annotations

import pandas as pd
from river import drift

from klia.features.state import FeatureState
from klia.model.bundle import Bundle


def feature_stream(ok: pd.DataFrame, state: FeatureState) -> tuple[list[dict], list[int]]:
    """Replay cleaned rows (already time-sorted) through the state: read features, then update.

    The label of a row is only written to the state AFTER its features were read, for every row,
    so there is no leakage regardless of how the rows are later grouped into batches.
    """
    X, y = [], []
    for rec in ok.to_dict("records"):
        label = int(rec["is_delayed"])
        X.append(state.features(rec))
        y.append(label)
        state.update(rec, label)
    return X, y


def apply_rows(bundle: Bundle, ok: pd.DataFrame, cfg: dict) -> dict:
    """Learn from one batch of new, cleaned rows (in place). Returns stats for the run log."""
    if ok.empty:
        return {"rows_learned": 0}
    mcfg = cfg["model"]
    X, y = feature_stream(ok, bundle.state)

    # Score this batch BEFORE learning from it (out-of-sample), unless the model is still fresh.
    warm = bundle.model.n_learned < mcfg.get("warmup_rows", 500)
    base = bundle.state.base_rate
    if warm:
        p = [base] * len(y)
    else:
        p = bundle.model.predict_proba(X, fallback=base)

    # One drift check per BATCH (not per row): feed the batch's mean absolute error.
    import numpy as np
    det = drift.ADWIN(delta=0.002)
    err = float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
    det.update(err)
    drift_fired = 1 if det.drift_detected else 0

    bundle.model.partial_fit(X, y)
    if not warm:
        bundle.remember(p, y, mcfg["calibration_window"])
    bundle.drift_events += drift_fired

    stats = {"rows_learned": len(y), "drift_events": drift_fired, "scored_rows": 0 if warm else len(y)}
    bundle.meta.update({
        "n_rows": bundle.model.n_learned,
        "last_sched": str(ok["sched_dt"].iloc[-1]),
        "model_name": bundle.model.name,
    })
    stats.update(bundle.refresh_threshold())
    bundle.meta["metrics"] = {k: v for k, v in stats.items()
                              if k in ("roc_auc", "f1", "log_loss", "delay_rate", "n_scored")}
    return stats
