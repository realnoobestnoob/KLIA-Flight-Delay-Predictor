"""Online (incrementally trained) classifiers and the test-then-train loop."""
from __future__ import annotations

import numpy as np
from river import compose, drift, forest, linear_model, optim, preprocessing, tree

CANDIDATES = {
    "logreg": "Logistic regression (scaled, SGD)",
    "hat": "Hoeffding adaptive tree",
    "arf": "Adaptive random forest (10 trees)",
}


def _build(name: str, params: dict | None):
    p = dict(params or {})
    if name == "logreg":
        lr = p.pop("lr", 0.01)
        return preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(lr), l2=p.pop("l2", 0.0))
    if name == "hat":
        p.setdefault("grace_period", 200)
        p.setdefault("leaf_prediction", "nb")
        p.setdefault("seed", 42)
        return tree.HoeffdingAdaptiveTreeClassifier(**p)
    if name == "arf":
        p.setdefault("n_models", 10)
        p.setdefault("grace_period", 100)
        p.setdefault("seed", 42)
        return forest.ARFClassifier(**p)
    raise ValueError(f"unknown model '{name}', choose from {list(CANDIDATES)}")


class OnlineClassifier:
    """Thin wrapper: always returns a float probability of 'delayed'."""

    def __init__(self, name: str, params: dict | None = None):
        self.name = name
        self.params = dict(params or {})
        self.model = _build(name, self.params)
        self.n_learned = 0

    def predict_proba(self, x: dict, fallback: float = 0.3) -> float:
        if self.n_learned == 0:
            return fallback
        out = self.model.predict_proba_one(x)
        p = out.get(1, out.get(True))
        return fallback if p is None else float(min(max(p, 1e-6), 1 - 1e-6))

    def learn(self, x: dict, y: int) -> None:
        self.model.learn_one(x, int(y))
        self.n_learned += 1


def prequential(model: OnlineClassifier, X: list[dict], y: list[int], *,
                warmup: int = 0, detector: drift.ADWIN | None = None,
                fallback: float = 0.3) -> tuple[np.ndarray, list[int]]:
    """Test-then-train: predict each row BEFORE learning from it.

    Returns (probabilities, indices where the drift detector fired). Probabilities for the first
    `warmup` rows are still returned, so callers decide what to score.
    """
    p = np.empty(len(X))
    fired: list[int] = []
    for i, (x, yi) in enumerate(zip(X, y)):
        p[i] = model.predict_proba(x, fallback)
        if detector is not None and i >= warmup:
            detector.update(abs(yi - p[i]))
            if detector.drift_detected:
                fired.append(i)
        model.learn(x, yi)
    return p, fired


def best_threshold(p: np.ndarray, y: np.ndarray, default: float = 0.5) -> tuple[float, float]:
    """Threshold that maximises F1 on (p, y). Returns (threshold, f1)."""
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
