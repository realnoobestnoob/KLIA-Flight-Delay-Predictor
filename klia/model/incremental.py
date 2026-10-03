"""Incremental (mini-batch) classifiers: updated with `partial_fit` on a CHUNK of new rows at a
time, never retrained from scratch and never fit one row at a time.

Sklearn models (sgd_log, sgd_hinge, gnb, mlp) call sklearn's native partial_fit.

Boosting models (xgb, lgbm, catboost) use each library's warm-start mechanism to ADD trees on
every partial_fit call — XGBoost via xgb_model=, LightGBM via init_model=, CatBoost via
init_model=.  After k update-job runs, the model holds k × n_iter trees total.  Monitor bundle
size via GET /v1/model; raise config.yaml → model.boosting_n_iter if updates are too slow, or
lower it if the bundle grows too large.

Includes two incremental ensembles (BaggingEnsemble, RandomSubspaceEnsemble) that work with any
base model, including the three boosting wrappers.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

CLASSES = np.array([0, 1])

BASE_MODELS = {
    "sgd_log":   "Linear model, logistic loss (partial_fit)",
    "sgd_hinge": "Linear SVM-like, modified Huber loss (partial_fit, supports predict_proba)",
    "gnb":       "Gaussian Naive Bayes (partial_fit)",
    "mlp":       "Small neural net, 1 hidden layer (partial_fit)",
    "xgb":       "XGBoost — adds n_iter trees per partial_fit call (warm-start via xgb_model=)",
    "lgbm":      "LightGBM — adds n_iter trees per partial_fit call (warm-start via init_model=)",
    "catboost":  "CatBoost — adds n_iter trees per partial_fit call (warm-start via init_model=)",
}
ENSEMBLES = {
    "bag":  "Bagging: each member trained on a bootstrap resample of every batch",
    "rsub": "Random subspace: each member trained on a random subset of features",
}


# ─────────────────────────────────── sklearn helpers ─────────────────────────

def _sklearn_model(name: str, params: dict, seed: int):
    p = dict(params or {})
    if name == "sgd_log":
        return SGDClassifier(
            loss="log_loss", alpha=p.get("alpha", 1e-4),
            penalty=p.get("penalty", "l2"),
            learning_rate="optimal", random_state=seed,
        )
    if name == "sgd_hinge":
        # "hinge" has no predict_proba; modified_huber is the closest SVM-like loss that does.
        return SGDClassifier(
            loss="modified_huber", alpha=p.get("alpha", 1e-4),
            penalty=p.get("penalty", "l2"),
            learning_rate="optimal", random_state=seed,
        )
    if name == "gnb":
        return GaussianNB(var_smoothing=p.get("var_smoothing", 1e-9))
    if name == "mlp":
        # warm_start conflicts with partial_fit on single-class batches; omitted on purpose.
        return MLPClassifier(
            hidden_layer_sizes=(p.get("hidden", 32),),
            alpha=p.get("alpha", 1e-4),
            learning_rate_init=p.get("lr", 1e-3),
            random_state=seed,
        )
    raise ValueError(f"unknown sklearn base '{name}', choose from {list(BASE_MODELS)}")


# ────────────────────────────── sklearn wrapper ───────────────────────────────

class IncrementalModel:
    """One sklearn partial_fit estimator, with feature-column order fixed on first use."""

    def __init__(
        self,
        base: str,
        params: dict | None = None,
        feature_names: list[str] | None = None,
        seed: int = 42,
    ):
        self.name = base
        self.base = base
        self.params = dict(params or {})
        self.feature_names = feature_names
        self.model = _sklearn_model(base, self.params, seed)
        self.n_learned = 0

    def _array(self, rows: list[dict]) -> np.ndarray:
        cols = self.feature_names or sorted(rows[0])
        if self.feature_names is None:
            self.feature_names = cols
        return np.array([[r.get(c, 0.0) for c in cols] for r in rows], dtype=float)

    def partial_fit(self, X: list[dict], y: list[int], classes=None) -> None:
        # Always pass classes: MLP requires it on every call; others ignore it after the first.
        self.model.partial_fit(self._array(X), np.asarray(y), classes=CLASSES)
        self.n_learned += len(y)

    def predict_proba(self, X, fallback: float = 0.3):
        single = isinstance(X, dict)
        rows = [X] if single else X
        if self.n_learned == 0:
            out = np.full(len(rows), fallback)
        else:
            proba = self.model.predict_proba(self._array(rows))
            classes = list(self.model.classes_)
            out = (
                proba[:, classes.index(1)]
                if 1 in classes
                else np.full(len(rows), fallback)
            )
        return float(out[0]) if single else out


# ──────────────────────────── boosting wrappers ──────────────────────────────

class XGBIncremental:
    """XGBoost incremental classifier.

    Each partial_fit call passes the existing Booster as xgb_model= to xgb.train(),
    adding n_iter new trees.  Total trees after k calls = k × n_iter.
    Lazy-imports xgboost so the module loads even when xgboost is not installed.
    """

    def __init__(
        self,
        params: dict | None = None,
        n_iter: int = 20,
        feature_names: list[str] | None = None,
        seed: int = 42,
    ):
        p = dict(params or {})
        self.xgb_params = {
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "eta":              p.get("eta", 0.1),
            "max_depth":        int(p.get("max_depth", 4)),
            "subsample":        p.get("subsample", 0.8),
            "colsample_bytree": p.get("colsample_bytree", 0.8),
            "min_child_weight": int(p.get("min_child_weight", 1)),
            "seed":             seed,
            "verbosity":        0,
            "nthread":          1,
        }
        self.n_iter        = int(n_iter)
        self.feature_names = feature_names
        self.booster       = None
        self.n_learned     = 0
        self.name          = "xgb"
        self.base          = "xgb"
        self.params        = p

    def _array(self, rows: list[dict]) -> np.ndarray:
        cols = self.feature_names or sorted(rows[0])
        if self.feature_names is None:
            self.feature_names = cols
        return np.array([[r.get(c, 0.0) for c in cols] for r in rows], dtype=float)

    def partial_fit(self, X: list[dict], y: list[int], classes=None) -> None:
        import xgboost as xgb
        y_arr = np.asarray(y, dtype=float)
        # Skip single-class batches when no booster exists yet (would produce degenerate splits).
        if len(np.unique(y_arr)) < 2 and self.booster is None:
            return
        dtrain = xgb.DMatrix(self._array(X), label=y_arr)
        self.booster = xgb.train(
            self.xgb_params, dtrain,
            num_boost_round=self.n_iter,
            xgb_model=self.booster,   # None on first call → fresh model; Booster after → warm start
        )
        self.n_learned += len(y)

    def predict_proba(self, X, fallback: float = 0.3):
        import xgboost as xgb
        single = isinstance(X, dict)
        rows   = [X] if single else X
        if self.booster is None or self.n_learned == 0:
            out = np.full(len(rows), fallback)
        else:
            out = self.booster.predict(xgb.DMatrix(self._array(rows)))
        return float(out[0]) if single else out


class LGBMIncremental:
    """LightGBM incremental classifier.

    Each partial_fit call passes the existing Booster as init_model= to lgb.train(),
    adding n_iter new trees.  Total trees after k calls = k × n_iter.
    """

    def __init__(
        self,
        params: dict | None = None,
        n_iter: int = 20,
        feature_names: list[str] | None = None,
        seed: int = 42,
    ):
        p = dict(params or {})
        self.lgbm_params = {
            "objective":        "binary",
            "metric":           "binary_logloss",
            "learning_rate":    p.get("learning_rate", 0.1),
            "num_leaves":       int(p.get("num_leaves", 31)),
            "max_depth":        int(p.get("max_depth", -1)),
            "min_child_samples":int(p.get("min_child_samples", 20)),
            "subsample":        p.get("subsample", 0.8),
            "colsample_bytree": p.get("colsample_bytree", 0.8),
            "seed":             seed,
            "verbose":          -1,
            "verbosity":        -1,   # LightGBM 4.x key
            "num_threads":      1,
        }
        self.n_iter        = int(n_iter)
        self.feature_names = feature_names
        self.booster       = None
        self.n_learned     = 0
        self.name          = "lgbm"
        self.base          = "lgbm"
        self.params        = p

    def _array(self, rows: list[dict]) -> np.ndarray:
        cols = self.feature_names or sorted(rows[0])
        if self.feature_names is None:
            self.feature_names = cols
        return np.array([[r.get(c, 0.0) for c in cols] for r in rows], dtype=float)

    def partial_fit(self, X: list[dict], y: list[int], classes=None) -> None:
        import lightgbm as lgb
        y_arr = np.asarray(y, dtype=float)
        if len(np.unique(y_arr)) < 2 and self.booster is None:
            return
        train_data = lgb.Dataset(self._array(X), label=y_arr)
        self.booster = lgb.train(
            self.lgbm_params, train_data,
            num_boost_round=self.n_iter,
            init_model=self.booster,  # None on first call; Booster after → warm start
        )
        self.n_learned += len(y)

    def predict_proba(self, X, fallback: float = 0.3):
        single = isinstance(X, dict)
        rows   = [X] if single else X
        if self.booster is None or self.n_learned == 0:
            out = np.full(len(rows), fallback)
        else:
            out = self.booster.predict(self._array(rows))
        return float(out[0]) if single else out


class CatBoostIncremental:
    """CatBoost incremental classifier.

    Each partial_fit call fits a new CatBoostClassifier with n_iter iterations,
    starting from the previous model via init_model=.  Total trees after k calls = k × n_iter.
    Single-class batches are always skipped (CatBoost raises on them regardless of init_model).
    """

    def __init__(
        self,
        params: dict | None = None,
        n_iter: int = 20,
        feature_names: list[str] | None = None,
        seed: int = 42,
    ):
        p = dict(params or {})
        self.catboost_params = {
            "iterations":          int(n_iter),
            "learning_rate":       p.get("learning_rate", 0.1),
            "depth":               int(p.get("depth", 4)),
            "l2_leaf_reg":         p.get("l2_leaf_reg", 3.0),
            "random_seed":         seed,
            "verbose":             0,
            "allow_writing_files": False,
            "thread_count":        1,
            "task_type":           "CPU",
        }
        self.n_iter        = int(n_iter)
        self.feature_names = feature_names
        self.model         = None
        self.n_learned     = 0
        self.name          = "catboost"
        self.base          = "catboost"
        self.params        = p

    def _array(self, rows: list[dict]) -> np.ndarray:
        cols = self.feature_names or sorted(rows[0])
        if self.feature_names is None:
            self.feature_names = cols
        return np.array([[r.get(c, 0.0) for c in cols] for r in rows], dtype=float)

    def partial_fit(self, X: list[dict], y: list[int], classes=None) -> None:
        from catboost import CatBoostClassifier
        y_arr = np.asarray(y)
        # CatBoost always raises on single-class batches regardless of init_model.
        if len(np.unique(y_arr)) < 2:
            return
        new_model = CatBoostClassifier(**self.catboost_params)
        new_model.fit(self._array(X), y_arr, init_model=self.model, verbose=0)
        self.model = new_model
        self.n_learned += len(y)

    def predict_proba(self, X, fallback: float = 0.3):
        single = isinstance(X, dict)
        rows   = [X] if single else X
        if self.model is None or self.n_learned == 0:
            out = np.full(len(rows), fallback)
        else:
            out = self.model.predict_proba(self._array(rows))[:, 1]
        return float(out[0]) if single else out


# ─────────────────────── factory: right class for any base name ──────────────

def _make_member(
    base_name: str,
    params: dict,
    feature_names: list[str] | None,
    seed: int,
):
    """Return an incremental model of the correct class for base_name."""
    p = dict(params or {})
    n_iter = p.pop("n_iter", 20)
    if base_name == "xgb":
        return XGBIncremental(p, n_iter=n_iter, feature_names=feature_names, seed=seed)
    if base_name == "lgbm":
        return LGBMIncremental(p, n_iter=n_iter, feature_names=feature_names, seed=seed)
    if base_name == "catboost":
        return CatBoostIncremental(p, n_iter=n_iter, feature_names=feature_names, seed=seed)
    return IncrementalModel(base_name, p, feature_names, seed)


# ────────────────────────────────── ensembles ─────────────────────────────────

class _Ensemble:
    """Shared bookkeeping for BaggingEnsemble and RandomSubspaceEnsemble."""
    n_learned = 0  # class default; shadowed by instance attribute after first partial_fit

    def predict_proba(self, X, fallback: float = 0.3):
        single = isinstance(X, dict)
        preds = np.stack(
            [np.atleast_1d(m.predict_proba(X, fallback)) for m in self.members]
        )
        out = preds.mean(axis=0)
        return float(out[0]) if single else out


class BaggingEnsemble(_Ensemble):
    def __init__(
        self,
        base: str = "sgd_log",
        n_estimators: int = 7,
        params: dict | None = None,
        feature_names: list[str] | None = None,
        seed: int = 42,
    ):
        self.name = f"bag_{base}"
        self.base, self.n_estimators = base, n_estimators
        self.params = dict(params or {})
        self.rng = np.random.default_rng(seed)
        self.members = [
            _make_member(base, self.params, feature_names, seed + i)
            for i in range(n_estimators)
        ]

    def partial_fit(self, X: list[dict], y: list[int]) -> None:
        n = len(y)
        for m in self.members:
            idx = self.rng.integers(0, n, size=n)
            m.partial_fit([X[i] for i in idx], [y[i] for i in idx])
        self.n_learned += n


class RandomSubspaceEnsemble(_Ensemble):
    def __init__(
        self,
        base: str = "sgd_log",
        n_estimators: int = 7,
        subspace_frac: float = 0.7,
        params: dict | None = None,
        feature_names: list[str] | None = None,
        seed: int = 42,
    ):
        self.name = f"rsub_{base}"
        self.base, self.n_estimators, self.subspace_frac = base, n_estimators, subspace_frac
        self.params = dict(params or {})
        self.rng = np.random.default_rng(seed)
        self.feature_names = feature_names
        self.subspaces: list[list[str]] | None = None
        self.members = [
            _make_member(base, self.params, None, seed + i)
            for i in range(n_estimators)
        ]

    def _ensure(self, rows: list[dict]) -> None:
        if self.subspaces is not None:
            return
        cols = self.feature_names or sorted(rows[0])
        k = max(2, int(len(cols) * self.subspace_frac))
        self.subspaces = [
            list(self.rng.choice(cols, size=k, replace=False)) for _ in self.members
        ]
        for m, sub in zip(self.members, self.subspaces):
            m.feature_names = sub

    def partial_fit(self, X: list[dict], y: list[int]) -> None:
        self._ensure(X)
        for m in self.members:
            m.partial_fit(X, y)
        self.n_learned += len(y)

    def predict_proba(self, X, fallback: float = 0.3):
        rows = [X] if isinstance(X, dict) else X
        self._ensure(rows)
        return super().predict_proba(X, fallback)


# ─────────────────────────────────── public API ───────────────────────────────

def build(
    name: str,
    params: dict | None = None,
    feature_names: list[str] | None = None,
    seed: int = 42,
):
    """Build any incremental model or ensemble by name.

    name: base model key ('sgd_log', 'gnb', 'xgb', 'lgbm', 'catboost', …)
          or ensemble key 'bag_<base>' / 'rsub_<base>'.
    params: hyperparameters.  Special keys consumed here:
        n_estimators  — ensemble size (default 7)
        subspace_frac — feature fraction for rsub (default 0.7)
        n_iter        — trees per partial_fit for xgb/lgbm/catboost (default 20)
    """
    params = dict(params or {})
    n_estimators = params.pop("n_estimators", 7)
    subspace_frac = params.pop("subspace_frac", 0.7)

    if name.startswith("bag_"):
        return BaggingEnsemble(name[4:], n_estimators, params, feature_names, seed)
    if name.startswith("rsub_"):
        return RandomSubspaceEnsemble(name[5:], n_estimators, subspace_frac, params, feature_names, seed)

    # Single model — pop n_iter before passing to boosting constructors
    n_iter = params.pop("n_iter", 20)
    if name == "xgb":
        return XGBIncremental(params, n_iter=n_iter, feature_names=feature_names, seed=seed)
    if name == "lgbm":
        return LGBMIncremental(params, n_iter=n_iter, feature_names=feature_names, seed=seed)
    if name == "catboost":
        return CatBoostIncremental(params, n_iter=n_iter, feature_names=feature_names, seed=seed)
    return IncrementalModel(name, params, feature_names, seed)


def best_threshold(
    p: np.ndarray, y: np.ndarray, default: float = 0.5
) -> tuple[float, float]:
    p, y = np.asarray(p), np.asarray(y)
    if len(p) < 200 or y.min() == y.max():
        return default, float("nan")
    best_t, best_f = default, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = p >= t
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum(~pred & (y == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f:
            best_t, best_f = float(round(t, 2)), f1
    return best_t, best_f


def batch_prequential(
    model,
    X: list[dict],
    y: list[int],
    batch_size: int = 500,
    warmup_batches: int = 1,
):
    """Split (X, y) into chunks; score each BEFORE learning (no leakage), then partial_fit.
    Mirrors exactly how the production update job processes a batch of new rows.
    """
    n = len(y)
    p = np.empty(n)
    for i in range(0, n, batch_size):
        xb, yb = X[i : i + batch_size], y[i : i + batch_size]
        batch_idx = i // batch_size
        scored = batch_idx >= warmup_batches and model.n_learned
        p[i : i + len(yb)] = model.predict_proba(xb, fallback=0.3) if scored else 0.3
        model.partial_fit(xb, yb)
    return p
