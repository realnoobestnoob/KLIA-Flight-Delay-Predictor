"""Core update logic, free of any database code so it can be tested and reused by the notebook."""
from __future__ import annotations

import pandas as pd
from river import drift

from klia.features.state import FeatureState
from klia.model.bundle import Bundle
from klia.model.online import prequential


def feature_stream(ok: pd.DataFrame, state: FeatureState) -> tuple[list[dict], list[int]]:
    """Replay cleaned rows (already time-sorted) through the state: read features, then update.

    The label of a row is only written to the state AFTER its features were read.
    """
    X, y = [], []
    for rec in ok.to_dict("records"):
        label = int(rec["is_delayed"])
        X.append(state.features(rec))
        y.append(label)
        state.update(rec, label)
    return X, y


def apply_rows(bundle: Bundle, ok: pd.DataFrame, cfg: dict) -> dict:
    """Learn from new, cleaned rows (in place). Returns a small stats dict for the run log."""
    if ok.empty:
        return {"rows_learned": 0}
    mcfg = cfg["model"]
    X, y = feature_stream(ok, bundle.state)
    # A fresh model is not scored while it is still warming up.
    warm = 0 if bundle.model.n_learned >= mcfg["warmup_rows"] else mcfg["warmup_rows"] - bundle.model.n_learned
    base = bundle.state.base_rate
    det = drift.ADWIN(delta=0.002)
    p, fired = prequential(bundle.model, X, y, warmup=warm, detector=det, fallback=base)
    bundle.remember(p[warm:], y[warm:], mcfg["calibration_window"])
    bundle.drift_events += len(fired)
    stats = {"rows_learned": len(y), "drift_events": len(fired), "scored_rows": max(len(y) - warm, 0)}
    bundle.meta.update({
        "n_rows": bundle.model.n_learned,
        "last_sched": str(ok["sched_dt"].iloc[-1]),
        "model_name": bundle.model.name,
    })
    stats.update(bundle.refresh_threshold())
    bundle.meta["metrics"] = {k: v for k, v in stats.items() if k in ("roc_auc", "f1", "log_loss", "delay_rate", "n_scored")}
    return stats
