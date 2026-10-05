"""REST API. `uvicorn klia.api.app:app`.  Inference only: this process never trains."""
from __future__ import annotations

import datetime as dt
import hmac
import os
import time
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from klia.api.cache import ModelCache
from klia.config import artifacts_dir, load_config
from klia.etl.validate import norm
from klia.store.base import open_store


class PredictRequest(BaseModel):
    airline: str = Field(min_length=2, max_length=60, examples=["AirAsia"])
    destination: str = Field(min_length=2, max_length=60, examples=["Singapore"])
    scheduled_departure: dt.datetime = Field(
        description="Local KLIA time, e.g. 2026-10-02T08:30",
        examples=["2026-10-02T08:30"],
    )
    aircraft: str | None = Field(default=None, max_length=40, examples=["A320"])

    @field_validator("airline", "destination")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class PredictResponse(BaseModel):
    delay_probability: float
    predicted_delayed: bool
    threshold: float
    risk: str
    model_version: int
    known_airline: bool
    known_route: bool


def create_app(
    store=None,
    cache: ModelCache | None = None,
) -> FastAPI:
    cfg = load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.cache = cache or ModelCache(
            store or open_store(cfg),
            os.environ.get("KLIA_CACHE_DIR", artifacts_dir() / "cache"),
            cfg["api"]["refresh_seconds"],
        )
        if cache is None:
            app.state.cache.startup()
        yield

    app = FastAPI(title="KLIA Flight Delay API", version="2.0", lifespan=lifespan)
    tz      = ZoneInfo(cfg["api"]["timezone"])
    api_key = os.environ.get("API_KEY", "")

    def auth(x_api_key: str | None = Header(default=None)):
        if api_key and not hmac.compare_digest(x_api_key or "", api_key):
            raise HTTPException(401, "invalid or missing X-API-Key")

    def bundle_or_503():
        b = app.state.cache.get()
        if b is None:
            raise HTTPException(
                503,
                "no model available yet: run `python -m klia.jobs.update --bootstrap`",
            )
        return b

    # ── health / metadata ────────────────────────────────────────────────────

    @app.get("/healthz")
    def healthz():
        c = app.state.cache
        return {"status": "ok" if c.bundle is not None else "no_model", "model_version": c.version}

    @app.get("/v1/model", dependencies=[Depends(auth)])
    def model_info():
        b = bundle_or_503()
        c = app.state.cache
        return {
            "version": c.version,
            "loaded_age_seconds": int(time.time() - c.loaded_at),
            **b.describe(),
        }

    @app.get("/v1/options", dependencies=[Depends(auth)])
    def options():
        return bundle_or_503().state.options()

    # ── drift monitoring ─────────────────────────────────────────────────────

    @app.get("/v1/drift", dependencies=[Depends(auth)])
    def drift_status():
        """Return Evidently drift monitoring summary from the last update run.

        Response fields:
            drift_events_total  — cumulative count of batches where drift was detected
            reference_rows      — size of the stored reference sample (set by the notebook)
            latest_report       — full Evidently result from the most recent update batch
            drift_history       — last 20 per-batch summaries (ts, drift_detected, share_drifted, …)

        drift_history is empty until the first update job run after --bootstrap.
        latest_report is null until then too.
        """
        b   = bundle_or_503()
        log = b.meta.get("drift_log", [])
        last = b.meta.get("last_drift_report", {})
        ref  = b.meta.get("drift_reference") or {}

        return {
            "drift_events_total": b.drift_events,
            "reference_rows":     len(ref.get("data", [])),
            "latest_report": {
                "ts":             last.get("ts"),
                "drift_detected": last.get("drift_detected", False),
                "share_drifted":  last.get("share_drifted", 0.0),
                "n_features":     last.get("n_features", 0),
                "n_drifted":      last.get("n_drifted", 0),
                "features":       last.get("features", {}),
            } if last else None,
            "drift_history": [
                {k: v for k, v in entry.items() if k not in ("error",)}
                for entry in log[-20:]
            ],
        }

    # ── prediction ───────────────────────────────────────────────────────────

    @app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(auth)])
    def predict(req: PredictRequest):
        b    = bundle_or_503()
        when = req.scheduled_departure
        if when.tzinfo is not None:
            when = when.astimezone(tz).replace(tzinfo=None)
        a, d = norm(req.airline), norm(req.destination)
        row = {
            "sched_dt":   when,
            "airline":    a,
            "destination":d,
            "aircraft":   norm(req.aircraft) or "UNKNOWN",
        }
        p  = b.model.predict_proba(b.state.features(row), fallback=b.state.base_rate)
        t  = b.threshold
        risk = "high" if p >= t * 1.25 else "elevated" if p >= t else "low"
        return PredictResponse(
            delay_probability=round(p, 4),
            predicted_delayed=p >= t,
            threshold=t,
            risk=risk,
            model_version=app.state.cache.version or 0,
            known_airline=b.state.known_airline(a),
            known_route=b.state.known_route(a, d),
        )

    return app


app = create_app()
