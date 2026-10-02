"""A Bundle is everything inference needs, saved and loaded as one artifact:
the online model, the feature state, the decision threshold and the metrics that go with them."""
from __future__ import annotations

import datetime as dt
import gzip
import pickle
import platform
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import river

from klia.features.state import FeatureState
from klia.model.online import OnlineClassifier, best_threshold


def _versions() -> dict:
    import pandas
    return {"python": platform.python_version(), "river": river.__version__,
            "numpy": np.__version__, "pandas": pandas.__version__}


@dataclass
class Bundle:
    model: OnlineClassifier
    state: FeatureState
    threshold: float = 0.5
    recent_p: list = field(default_factory=list)   # recent prequential probabilities
    recent_y: list = field(default_factory=list)   # and their outcomes
    meta: dict = field(default_factory=dict)
    drift_events: int = 0

    def remember(self, p, y, window: int) -> None:
        self.recent_p = (self.recent_p + [float(v) for v in p])[-window:]
        self.recent_y = (self.recent_y + [int(v) for v in y])[-window:]

    def refresh_threshold(self) -> dict:
        from sklearn.metrics import f1_score, log_loss, roc_auc_score
        p, y = np.array(self.recent_p), np.array(self.recent_y)
        m = {"n_scored": int(len(p))}
        if len(p) >= 200 and y.min() != y.max():
            self.threshold, _ = best_threshold(p, y, self.threshold)
            m["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
            m["f1"] = round(float(f1_score(y, p >= self.threshold)), 4)
            m["log_loss"] = round(float(log_loss(y, p)), 4)
            m["delay_rate"] = round(float(y.mean()), 4)
        m["threshold"] = self.threshold
        return m

    def dumps(self) -> bytes:
        self.meta["versions"] = _versions()
        return gzip.compress(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=6)

    @staticmethod
    def loads(blob: bytes) -> "Bundle":
        b = pickle.loads(gzip.decompress(blob))   # only ever load bundles you wrote yourself
        saved = b.meta.get("versions", {})
        if saved and saved.get("river") != river.__version__:
            warnings.warn(f"bundle trained with river {saved.get('river')}, running {river.__version__}")
        return b

    def describe(self) -> dict:
        return {**{k: v for k, v in self.meta.items() if k != "versions"},
                "model": self.model.name, "threshold": self.threshold,
                "n_learned": self.model.n_learned, "drift_events": self.drift_events}


def new_bundle(cfg: dict) -> Bundle:
    m = cfg["model"]
    return Bundle(model=OnlineClassifier(m["name"], m.get("params")), state=FeatureState(cfg),
                  meta={"created": dt.datetime.utcnow().isoformat() + "Z"})
