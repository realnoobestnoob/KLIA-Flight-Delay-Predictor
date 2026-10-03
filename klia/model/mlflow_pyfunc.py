"""Lets a Bundle be logged and promoted through the MLflow Model Registry as a normal pyfunc model."""
from __future__ import annotations

import gzip
import pickle

import mlflow.pyfunc
import pandas as pd


class BundlePyFunc(mlflow.pyfunc.PythonModel):
    """Wraps a gzip-pickled Bundle so MLflow can load it as a standard pyfunc model."""

    def load_context(self, context):
        with open(context.artifacts["bundle"], "rb") as f:
            self.bundle = pickle.loads(gzip.decompress(f.read()))

    def predict(self, context, model_input: pd.DataFrame):
        rows = model_input.to_dict("records")
        return [self.bundle.model.predict_proba(r, fallback=self.bundle.state.base_rate) for r in rows]


def log_and_register(bundle_path: str, sample_input: pd.DataFrame,
                     registered_name: str = "klia-flight-delay", alias: str = "champion") -> int:
    """Must be called inside an active `mlflow.start_run()` block. Logs the bundle as a pyfunc
    model, registers it under `registered_name`, and points `alias` at the new version.
    Returns the registry version number."""
    import mlflow
    from mlflow.tracking import MlflowClient

    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=BundlePyFunc(),
        artifacts={"bundle": bundle_path},
        input_example=sample_input,
        registered_model_name=registered_name,
    )
    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(registered_name, alias, version)
    return int(version)
