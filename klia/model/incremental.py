"""Incremental (mini-batch) classifiers: updated with `partial_fit` on a CHUNK of new rows at a
time, never retrained from scratch and never fit one row at a time. This is the "incremental
learning" family (scikit-learn's partial_fit estimators), as opposed to River's row-by-row
"online learning" (learn_one). Includes two incremental ensembles built from these base learners.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

CLASSES = np.array([0, 1])

BASE_MODELS = {
    "sgd_log": "Linear model, logistic loss (partial_fit)",
    "sgd_hinge": "Linear SVM-like, modified Huber loss (partial_fit, supports predict_proba)",
    "gnb": "Gaussian Naive Bayes (partial_fit)",
    "mlp": "Small neural net, 1 hidden layer (partial_fit)",
}
ENSEMBLES = {
    "bag": "Bagging: each member trained on a bootstrap resample of every batch",
    "rsub": "Random subspace: each member trained on a random subset of features",
}


def _base(name: str, params: dict, seed: int):
    p = dict(params or {})
    if name == "sgd_log":
        return SGDClassifier(loss="log_loss", alpha=p.get("alpha", 1e-4), penalty=p.get("penalty", "l2"),
                             learning_rate="optimal", random_state=seed)
    if name == "sgd_hinge":
        # "hinge" has no predict_proba; modified_huber is the closest SVM-like loss that supports it.
        return SGDClassifier(loss="modified_huber", alpha=p.get("alpha", 1e-4), penalty=p.get("penalty", "l2"),
                             learning_rate="optimal", random_state=seed)
    if name == "gnb":
        return GaussianNB(var_smoothing=p.get("var_smoothing", 1e-9))
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(p.get("hidden", 32),), alpha=p.get("alpha", 1e-4),
                             learning_rate_init=p.get("lr", 1e-3), max_iter=1, warm_start=True, random_state=seed)
    raise ValueError(f"unknown base model '{name}', choose from {list(BASE_MODELS)}")


class IncrementalModel:
    """One sklearn partial_fit estimator, with a feature-column order fixed on first use."""

    def __init__(self, base: str, params: dict | None = None, feature_names: list[str] | None = None, seed: int = 42):
        self.name = base
        self.base = base
        self.params = dict(params or {})
        self.feature_names = feature_names
        self.model = _base(base, self.params, seed)
        self.n_learned = 0

    def _array(self, rows: list[dict]) -> np.ndarray:
        cols = self.feature_names or sorted(rows[0])
        if self.feature_names is None:
            self.feature_names = cols
        return np.array([[r.get(c, 0.0) for c in cols] for r in rows], dtype=float)

    def partial_fit(self, X: list[dict], y: list[int]) -> None:
        self.model.partial_fit(self._array(X), np.asarray(y), classes=None if self.n_learned else CLASSES)
        self.n_learned += len(y)

    def predict_proba(self, X, fallback: float = 0.3):
        single = isinstance(X, dict)
        rows = [X] if single else X
        if self.n_learned == 0:
            out = np.full(len(rows), fallback)
        else:
            proba = self.model.predict_proba(self._array(rows))
            classes = list(self.model.classes_)
            out = proba[:, classes.index(1)] if 1 in classes else np.full(len(rows), fallback)
        return float(out[0]) if single else out


class _Ensemble:
    """Shared bookkeeping for the two ensemble kinds below."""
    n_learned = 0

    def predict_proba(self, X, fallback: float = 0.3):
        single = isinstance(X, dict)
        preds = np.stack([np.atleast_1d(m.predict_proba(X, fallback)) for m in self.members])
        out = preds.mean(axis=0)
        return float(out[0]) if single else out


class BaggingEnsemble(_Ensemble):
    def __init__(self, base: str = "sgd_log", n_estimators: int = 7, params: dict | None = None,
                feature_names: list[str] | None = None, seed: int = 42):
        self.name = f"bag_{base}"
        self.base, self.n_estimators = base, n_estimators
        self.params = dict(params or {})
        self.rng = np.random.default_rng(seed)
        self.members = [IncrementalModel(base, self.params, feature_names, seed + i) for i in range(n_estimators)]

    def partial_fit(self, X: list[dict], y: list[int]) -> None:
        n = len(y)
        for m in self.members:
            idx = self.rng.integers(0, n, size=n)
            m.partial_fit([X[i] for i in idx], [y[i] for i in idx])
        self.n_learned += n


class RandomSubspaceEnsemble(_Ensemble):
    def __init__(self, base: str = "sgd_log", n_estimators: int = 7, subspace_frac: float = 0.7,
                params: dict | None = None, feature_names: list[str] | None = None, seed: int = 42):
        self.name = f"rsub_{base}"
        self.base, self.n_estimators, self.subspace_frac = base, n_estimators, subspace_frac
        self.params = dict(params or {})
        self.rng = np.random.default_rng(seed)
        self.feature_names = feature_names
        self.subspaces: list[list[str]] | None = None
        self.members = [IncrementalModel(base, self.params, None, seed + i) for i in range(n_estimators)]

    def _ensure(self, rows: list[dict]) -> None:
        if self.subspaces is not None:
            return
        cols = self.feature_names or sorted(rows[0])
        k = max(2, int(len(cols) * self.subspace_frac))
        self.subspaces = [list(self.rng.choice(cols, size=k, replace=False)) for _ in self.members]
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


def build(name: str, params: dict | None = None, feature_names: list[str] | None = None, seed: int = 42):
    """name is a base model key ('sgd_log', 'gnb', ...) or 'bag_<base>' / 'rsub_<base>' for an ensemble."""
    params = dict(params or {})
    n_estimators = params.pop("n_estimators", 7)
    subspace_frac = params.pop("subspace_frac", 0.7)
    if name.startswith("bag_"):
        return BaggingEnsemble(name[4:], n_estimators, params, feature_names, seed)
    if name.startswith("rsub_"):
        return RandomSubspaceEnsemble(name[5:], n_estimators, subspace_frac, params, feature_names, seed)
    return IncrementalModel(name, params, feature_names, seed)


def best_threshold(p: np.ndarray, y: np.ndarray, default: float = 0.5) -> tuple[float, float]:
    p, y = np.asarray(p), np.asarray(y)
    if len(p) < 200 or y.min() == y.max():
        return default, float("nan")
    best_t, best_f = default, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = p >= t
        tp = float(np.sum(pred & (y == 1))); fp = float(np.sum(pred & (y == 0))); fn = float(np.sum(~pred & (y == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f:
            best_t, best_f = float(round(t, 2)), f1
    return best_t, best_f


def batch_prequential(model, X: list[dict], y: list[int], batch_size: int = 500, warmup_batches: int = 1):
    """Split (X, y) into chunks, predict each chunk BEFORE learning from it (no leakage),
    then partial_fit on that chunk. Mimics exactly how the production update job behaves,
    one job run (= one batch of new rows) at a time."""
    n = len(y)
    p = np.empty(n)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        batch_idx = i // batch_size
        p[i:i + len(yb)] = model.predict_proba(xb, fallback=0.3) if batch_idx >= warmup_batches and model.n_learned else 0.3
        model.partial_fit(xb, yb)
    return p
