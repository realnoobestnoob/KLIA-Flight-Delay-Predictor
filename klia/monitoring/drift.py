"""Evidently AI-based feature drift detection for the KLIA flight delay predictor.

Compares the current update batch against a reference sample of training feature
vectors stored in bundle.meta["drift_reference"] at notebook time.

Public API
----------
run_drift_report(X, bundle_meta)  -- called once per update job run in pipeline.py
set_drift_reference(meta, X, max_rows)  -- called at end of notebook section 9
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MIN_ROWS = 50  # minimum rows in both reference and current to run a report


# ──────────────────────────────────────────────── helpers ────────────────────

def _reference_df(bundle_meta: dict) -> pd.DataFrame | None:
    """Reconstruct the reference DataFrame from bundle.meta["drift_reference"]."""
    ref = bundle_meta.get("drift_reference")
    if not ref:
        return None
    cols: list[str] = ref.get("cols", [])
    data: list[list[float]] = ref.get("data", [])
    if not cols or not data:
        return None
    return pd.DataFrame(data, columns=cols)


def _current_df(X: list[dict]) -> pd.DataFrame:
    if not X:
        return pd.DataFrame()
    cols = list(X[0].keys())
    return pd.DataFrame([[r.get(c, 0.0) for c in cols] for r in X], columns=cols)


def _parse_evidently(result: dict, n_common: int) -> dict[str, Any]:
    """Extract a compact summary from an Evidently report.as_dict() result.

    Compatible with Evidently 0.4.x and 0.5.x (key names differ slightly
    between versions; we probe both).
    """
    metrics = result.get("metrics", [])

    # Dataset-level metric
    ds_entry = next(
        (m for m in metrics if "DatasetDriftMetric" in m.get("metric", "")), None
    )
    # Column-level table
    tbl_entry = next(
        (m for m in metrics if "DataDriftTable" in m.get("metric", "")), None
    )

    drift_detected = False
    share_drifted = 0.0
    n_features = n_common
    n_drifted = 0
    features: dict[str, Any] = {}

    if ds_entry:
        r = ds_entry.get("result", {})
        drift_detected = bool(r.get("dataset_drift", False))
        # Evidently 0.4 → "share_of_drifted_columns"; 0.5 → "drift_share"
        share_drifted = float(
            r.get("share_of_drifted_columns", r.get("drift_share", 0.0))
        )
        n_drifted = int(r.get("number_of_drifted_columns", 0))
        n_features = int(r.get("number_of_columns", n_common))

    if tbl_entry:
        for col, info in tbl_entry.get("result", {}).get("drift_by_columns", {}).items():
            features[col] = {
                "drifted": bool(info.get("drift_detected", False)),
                "score": round(float(info.get("drift_score", info.get("statistic", 0.0))), 4),
                "test": info.get("stattest_name", ""),
            }

    return {
        "drift_detected": drift_detected,
        "share_drifted": round(share_drifted, 4),
        "n_features": n_features,
        "n_drifted": n_drifted,
        "features": features,
    }


# ─────────────────────────────────────────────── public API ──────────────────

def run_drift_report(X: list[dict], bundle_meta: dict) -> dict:
    """Run an Evidently DataDriftPreset report comparing ``X`` to the stored reference.

    Parameters
    ----------
    X:
        Feature dicts for the current update batch (output of ``feature_stream``).
    bundle_meta:
        ``bundle.meta`` dict.  Must contain ``drift_reference`` (written by the
        notebook via :func:`set_drift_reference`).

    Returns
    -------
    dict with keys: ts, drift_detected, share_drifted, n_features, n_drifted,
    features (per-column detail), and optionally error.
    """
    ts = dt.datetime.utcnow().isoformat(timespec="seconds")
    base: dict[str, Any] = {
        "ts": ts,
        "drift_detected": False,
        "share_drifted": 0.0,
        "n_features": 0,
        "n_drifted": 0,
        "features": {},
    }

    ref_df = _reference_df(bundle_meta)
    cur_df = _current_df(X)

    if ref_df is None or len(ref_df) < _MIN_ROWS or len(cur_df) < _MIN_ROWS:
        base["error"] = (
            f"too few rows for drift report "
            f"(reference={0 if ref_df is None else len(ref_df)}, "
            f"current={len(cur_df)}, need >= {_MIN_ROWS} each)"
        )
        return base

    # Align to columns present in both datasets
    common = [c for c in ref_df.columns if c in cur_df.columns]
    if not common:
        base["error"] = "no common columns between reference and current"
        return base
    ref_df, cur_df = ref_df[common], cur_df[common]

    try:
        # evidently 0.4.x path (evidently.report.Report)
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError:
            # evidently 0.5+ moved the Report class; try the new location.
            # If this also fails the outer except returns a graceful error dict.
            from evidently import Report  # type: ignore[no-redef]
            from evidently.metric_preset import DataDriftPreset  # type: ignore[no-redef]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        parsed = _parse_evidently(report.as_dict(), len(common))
        return {**base, **parsed}

    except Exception as exc:
        # Log without exc_info so a version mismatch doesn't print a full traceback
        # into the bootstrap / update-job output.  The error is stored in the drift log.
        log.warning("Evidently drift report skipped: %s", exc)
        return {**base, "error": str(exc)}


def set_drift_reference(
    bundle_meta: dict, X: list[dict], max_rows: int = 1000
) -> None:
    """Store a random sample of ``X`` as the drift reference in ``bundle.meta``.

    Call this once at the end of notebook section 9, right after
    ``final_bundle.refresh_threshold()``, before saving the bundle.

    Parameters
    ----------
    bundle_meta:
        ``bundle.meta`` dict (mutated in place).
    X:
        Full training feature dicts from ``feature_stream``.
    max_rows:
        Maximum rows to keep.  1 000 is sufficient for drift detection and
        keeps the bundle size overhead under ~1 MB for typical feature counts.
    """
    if not X:
        log.warning("set_drift_reference called with empty X — skipping")
        return
    cols = list(X[0].keys())
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(max_rows, len(X)), replace=False)
    data = [[X[int(i)].get(c, 0.0) for c in cols] for i in idx]
    bundle_meta["drift_reference"] = {"cols": cols, "data": data}
    log.info("Drift reference set: %d rows × %d features", len(data), len(cols))