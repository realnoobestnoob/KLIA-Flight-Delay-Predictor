"""Public (no-auth) prediction endpoint for direct HTTP access.

Exposes POST /v1/public/predict — no API key required.
Rate-limited to 30 requests per minute per IP using a simple in-memory
token bucket (no extra dependencies).

Mount in klia/api/app.py with TWO lines:

    from klia.api.public import router as public_router
    app.include_router(public_router)

The local import inside the endpoint function avoids the circular-import
that would occur if this module imported get_bundle at module level.

Field names in the feature dict (rec) must match the column names your
cleaned DataFrame produces in klia/etl/validate.py. Adjust the rec dict
in public_predict() if your column names differ (e.g. dest vs destination).
"""
from __future__ import annotations

import collections
import threading
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/v1/public", tags=["public"])


# ── In-memory rate limiter ────────────────────────────────────────────────────

_RATE_LIMIT  = 30   # max requests per window per IP
_RATE_WINDOW = 60   # window in seconds
_ip_log: dict[str, collections.deque] = {}
_lock   = threading.Lock()


def _check_rate_limit(ip: str) -> None:
    now = time.monotonic()
    with _lock:
        dq = _ip_log.setdefault(ip, collections.deque())
        while dq and now - dq[0] > _RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {_RATE_LIMIT} requests per {_RATE_WINDOW}s per IP.",
            )
        dq.append(now)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PublicPredictRequest(BaseModel):
    airline: str
    destination: str
    scheduled_departure: datetime
    aircraft: Optional[str] = None


class PublicPredictResponse(BaseModel):
    delay_probability: float
    predicted_delayed: bool
    threshold: float
    risk: str
    model_version: int
    known_airline: bool
    known_route: bool


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PublicPredictResponse,
    summary="Predict flight delay (no API key required)",
)
async def public_predict(body: PublicPredictRequest, request: Request) -> PublicPredictResponse:
    """Predict the probability that a KLIA departure is delayed ≥15 minutes.

    No API key required. Rate-limited to 30 requests per minute per IP.
    Predictions use the same live model bundle as the authenticated /v1/predict endpoint.
    If the model has not seen this airline or route before, the estimate falls back
    to the base delay rate and is less reliable (flagged in the response).
    """
    _check_rate_limit(request.client.host)

    # Local import avoids circular import at module level.
    from klia.api.app import get_bundle  # adjust if your bundle accessor has a different name
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not available yet. Try again shortly.")

    # Build the feature record. Keys must match the column names produced by
    # klia/etl/validate.py. Adjust if your cleaned DataFrame uses different names.
    rec: dict = {
        "airline":  body.airline,
        "dest":     body.destination,   # rename to "destination" if that's your column name
        "sched_dt": body.scheduled_departure,
        "aircraft": body.aircraft or "",
    }

    # Check whether the airline and route are in the model's history.
    # Uses the same FeatureState that was accumulated during training.
    state         = bundle.state
    known_airline = bool(getattr(state, "airline_counts", {}).get(body.airline, 0))
    known_route   = bool(
        getattr(state, "route_counts", {}).get((body.airline, body.destination), 0)
    )

    # Compute features and apply selector (same path as training and serving).
    base = float(getattr(state, "base_rate", 0.3))
    try:
        feat = state.features(rec)
    except Exception:
        # If FeatureState requires fields not in the public request, fall back to base rate.
        return PublicPredictResponse(
            delay_probability=round(base, 4),
            predicted_delayed=False,
            threshold=base,
            risk="low",
            model_version=bundle.meta.get("version", -1),
            known_airline=known_airline,
            known_route=known_route,
        )

    if bundle.selector.is_fitted:
        feat = bundle.selector.filter(feat)

    prob      = float(bundle.model.predict_proba([feat], fallback=base)[0])
    threshold = float(
        bundle.meta.get("metrics", {}).get("threshold")
        or bundle.meta.get("threshold")
        or base
    )

    # Risk bucketing — mirrors the authenticated /v1/predict response.
    if prob >= threshold:
        risk = "high"
    elif prob >= threshold * 0.65:
        risk = "elevated"
    else:
        risk = "low"

    return PublicPredictResponse(
        delay_probability=round(prob, 4),
        predicted_delayed=prob >= threshold,
        threshold=round(threshold, 2),
        risk=risk,
        model_version=bundle.meta.get("version", -1),
        known_airline=known_airline,
        known_route=known_route,
    )
