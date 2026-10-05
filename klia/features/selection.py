"""Feature importance extraction and importance-based feature selection.

Why importance selection and not PCA
--------------------------------------
PCA was evaluated on this project and dropped AUC from 0.72 to 0.68. The core
signal comes from interpretable rate features (airline_rate, route_rate, etc.)
whose individual meaning matters — PCA mixes them into opaque components and
loses the smoothed-prior structure that makes them work.  Importance-based
selection keeps the best features intact and merely discards the noisy tail.

Lifecycle
---------
Bootstrap (--bootstrap flag in update.py):
  1. Probe phase: feed first `probe_rows` rows through a lightweight SGD → rank
     all features by |coef_| → fit FeatureSelector with top-k names.
  2. Full training: every feature dict from pipeline.feature_stream() is filtered
     through bundle.selector BEFORE the real model sees it → model.feature_names
     locks to the selected subset on its very first partial_fit call.
  3. Selector is stored inside the Bundle so serving always matches training.

Incremental updates (normal daily runs):
  - bundle.selector is already fitted; filter is applied automatically in
    pipeline.feature_stream() → no action needed.
  - After enough new data has accumulated the caller may re-fit the selector
    via fit_from_bundle(), but this requires a --bootstrap to take effect
    (model.feature_names is already locked for the current bundle).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("klia.features.selection")

_DEFAULT_TOP_K = 20     # sensible default: keeps ~74% of the 27 engineered features
_PROBE_ROWS    = 5_000  # enough rows for stable coefficient estimates from SGD


# ─────────────────────────── importance extractors ───────────────────────────

def _imp_sgd(model, names: list[str]) -> dict[str, float]:
    coef = np.abs(np.atleast_2d(model.coef_)).mean(axis=0)
    return dict(zip(names, coef.tolist()))


def _imp_gnb(model, names: list[str]) -> dict[str, float]:
    # discriminative power = |difference in per-class means|
    imp = np.abs(model.theta_[1] - model.theta_[0])
    return dict(zip(names, imp.tolist()))


def _imp_mlp(model, names: list[str]) -> dict[str, float]:
    # L1 norm of first-layer weights per input feature
    imp = np.abs(model.coefs_[0]).sum(axis=1)
    return dict(zip(names, imp.tolist()))


def _imp_xgb(booster, names: list[str]) -> dict[str, float]:
    scores = booster.get_fscore()
    if not scores:
        return {}
    # XGBoost uses positional 'f0','f1',… when feature names were not set
    if any(k.startswith("f") and k[1:].isdigit() for k in scores):
        return {names[int(k[1:])]: float(v)
                for k, v in scores.items() if int(k[1:]) < len(names)}
    return {k: float(v) for k, v in scores.items()}


def _imp_lgbm(booster, names: list[str]) -> dict[str, float]:
    imp = booster.feature_importance(importance_type="gain")
    return dict(zip(names, imp.astype(float).tolist()))


def _imp_catboost(model, names: list[str]) -> dict[str, float]:
    imp = model.get_feature_importance()
    return dict(zip(names, imp.astype(float).tolist()))


def _member_importances(member) -> dict[str, float] | None:
    """Extract importance scores from one model member. Returns None if unsupported."""
    names: list[str] = list(getattr(member, "feature_names", None) or [])
    if not names:
        return None

    sk  = getattr(member, "model",    None)   # IncrementalModel / CatBoost wrapper
    bst = getattr(member, "booster",  None)   # XGB / LGBM
    is_cb = hasattr(member, "catboost_params")

    # ── sklearn linear (SGD) ──
    if sk is not None and not is_cb and hasattr(sk, "coef_") and sk.coef_ is not None:
        try:
            return _imp_sgd(sk, names)
        except Exception:
            pass

    # ── Gaussian Naïve Bayes ──
    if sk is not None and not is_cb and hasattr(sk, "theta_"):
        try:
            return _imp_gnb(sk, names)
        except Exception:
            pass

    # ── MLP ──
    if sk is not None and not is_cb and hasattr(sk, "coefs_") and sk.coefs_:
        try:
            return _imp_mlp(sk, names)
        except Exception:
            pass

    # ── XGBoost ──
    if bst is not None and hasattr(bst, "get_fscore"):
        try:
            return _imp_xgb(bst, names)
        except Exception:
            pass

    # ── LightGBM ──
    if bst is not None and hasattr(bst, "feature_importance"):
        try:
            return _imp_lgbm(bst, names)
        except Exception:
            pass

    # ── CatBoost ──
    cb = sk if is_cb else None
    if cb is not None and hasattr(cb, "get_feature_importance"):
        try:
            return _imp_catboost(cb, names)
        except Exception:
            pass

    return None


def feature_importances(bundle) -> dict[str, float] | None:
    """Return {feature_name: normalised_score} sorted descending, or None if unsupported.

    For ensembles, scores are averaged across members before normalisation.
    """
    model   = bundle.model
    members = getattr(model, "members", None)

    if members:
        accum: dict[str, list[float]] = {}
        for mem in members:
            imp = _member_importances(mem)
            if imp:
                for k, v in imp.items():
                    accum.setdefault(k, []).append(float(v))
        if not accum:
            return None
        raw = {k: float(np.mean(v)) for k, v in accum.items()}
    else:
        raw = _member_importances(model)
        if raw is None:
            return None

    total = sum(raw.values())
    if total <= 0:
        return None
    normalised = {k: v / total for k, v in raw.items()}
    return dict(sorted(normalised.items(), key=lambda kv: kv[1], reverse=True))


# ──────────────────────────── FeatureSelector ────────────────────────────────

class FeatureSelector:
    """Filters feature dicts to a fixed subset ranked by importance.

    Lives inside the Bundle so training and serving always use the same features.
    Before fitting, acts as a passthrough (all features forwarded unchanged).

    Fitting strategies
    ------------------
    fit_from_probe(X, y)    — lightweight SGD on a sample; use BEFORE the real
                              model trains so model.feature_names locks to the
                              selected subset on its first partial_fit call.
    fit_from_bundle(bundle) — extract importances from an already-trained bundle;
                              use for inspection / logging. Requires --bootstrap
                              to take effect (model.feature_names already locked).
    """

    def __init__(self, selected: list[str] | None = None,
                 importances: dict[str, float] | None = None):
        self.selected:    list[str] | None    = selected       # None = passthrough
        self.importances: dict[str, float]    = importances or {}

    @property
    def is_fitted(self) -> bool:
        return self.selected is not None

    # ---------------------------------------------------------------- apply
    def filter(self, x: dict) -> dict:
        """Return x filtered to selected features. Passthrough when not fitted."""
        if self.selected is None:
            return x
        return {k: x.get(k, 0.0) for k in self.selected}

    # ---------------------------------------------------------------- fit
    def fit_from_probe(self, X: list[dict], y: list[int],
                       top_k: int = _DEFAULT_TOP_K) -> bool:
        """Rank features by |coef_| from a fast SGD probe and select the top-k.

        Call this BEFORE the real model trains so that model.feature_names
        locks to the selected subset on its very first partial_fit call.
        Returns True on success, False if the sample was too small or single-class.
        """
        if len(X) < 50 or not y:
            log.warning("probe sample too small (%d rows); skipping feature selection", len(X))
            return False
        from sklearn.linear_model import SGDClassifier
        y_arr = np.asarray(y)
        if y_arr.min() == y_arr.max():
            log.warning("probe sample is single-class; skipping feature selection")
            return False

        cols = sorted(X[0].keys())
        arr  = np.array([[r.get(c, 0.0) for c in cols] for r in X], dtype=float)

        probe = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=200,
                              class_weight="balanced", random_state=42)
        probe.fit(arr, y_arr)

        coef  = np.abs(np.atleast_2d(probe.coef_)).mean(axis=0)
        raw   = dict(zip(cols, coef.tolist()))
        total = sum(raw.values())
        if total <= 0:
            return False
        imp   = dict(sorted({k: v / total for k, v in raw.items()}.items(),
                             key=lambda kv: kv[1], reverse=True))

        k = min(top_k, len(imp))
        self.selected    = list(imp.keys())[:k]
        self.importances = imp

        dropped = len(imp) - k
        log.info("probe: kept %d / %d features (dropped %d low-importance); top 5: %s",
                 k, len(imp), dropped,
                 ", ".join(f"{n}={v:.3f}" for n, v in list(imp.items())[:5]))
        return True

    def fit_from_bundle(self, bundle, top_k: int = _DEFAULT_TOP_K) -> bool:
        """Fit from a trained bundle's model importances (for inspection / logging).

        Note: to apply this selection to training you must re-run --bootstrap;
        the current bundle's model.feature_names is already locked.
        """
        imp = feature_importances(bundle)
        if imp is None:
            log.warning("importances not extractable from '%s'; keeping all features",
                        getattr(bundle.model, "name", "?"))
            return False
        k = min(top_k, len(imp))
        self.selected    = list(imp.keys())[:k]
        self.importances = imp
        log.info("post-hoc selection: %d / %d features; top 5: %s",
                 k, len(imp),
                 ", ".join(f"{n}={v:.3f}" for n, v in list(imp.items())[:5]))
        return True
