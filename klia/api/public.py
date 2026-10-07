"""Public (no-auth) prediction endpoint for direct HTTP access.

Exposes POST /v1/public/predict — no API key required.
Rate-limited to 30 requests per minute per IP using an async token bucket.

Mount in klia/api/app.py with TWO lines:

    from klia.api.public import router as public_router, set_bundle_accessor
    app.include_router(public_router)

The _bundle_accessor variable is injected by app.py via set_bundle_accessor().
This avoids circular import issues.
"""
from __future__ import annotations

import asyncio
import collections
import time
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from klia.config import load_config
from klia.etl.validate import norm

router = APIRouter(prefix="/v1/public", tags=["public"])

# ── Bundle accessor injection ─────────────────────────────────────────────────
_bundle_accessor: Optional[Callable] = None


def set_bundle_accessor(accessor: Callable) -> None:
    """Inject the get_bundle function from app.py to avoid circular imports."""
    global _bundle_accessor
    _bundle_accessor = accessor


def _get_bundle():
    if _bundle_accessor is None:
        return None
    try:
        return _bundle_accessor()
    except Exception:
        return None


# ── Timezone (loaded once at import, not per-request) ────────────────────────
_cfg = load_config()
_tz  = ZoneInfo(_cfg["api"]["timezone"])


# ── Async rate limiter ────────────────────────────────────────────────────────
_RATE_LIMIT  = 30   # max requests per window per IP
_RATE_WINDOW = 60   # window in seconds
_ip_log: dict[str, collections.deque] = {}
_lock = asyncio.Lock()


async def _check_rate_limit(ip: str) -> None:
    now = time.monotonic()
    async with _lock:
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
    await _check_rate_limit(request.client.host)

    b = _get_bundle()
    if b is None:
        raise HTTPException(status_code=503, detail="Model not available yet. Try again shortly.")

    when = body.scheduled_departure
    if when.tzinfo is not None:
        when = when.astimezone(_tz).replace(tzinfo=None)

    a = norm(body.airline)
    d = norm(body.destination)

    row = {
        "sched_dt":    when,
        "airline":     a,
        "destination": d,
        "aircraft":    norm(body.aircraft) or "UNKNOWN",
    }

    feat = b.state.features(row)
    p    = b.model.predict_proba(feat, fallback=b.state.base_rate)
    t    = b.threshold
    risk = "high" if p >= t * 1.25 else "elevated" if p >= t else "low"

    return PublicPredictResponse(
        delay_probability=round(p, 4),
        predicted_delayed=p >= t,
        threshold=t,
        risk=risk,
        model_version=int(b.meta.get("version", b.meta.get("model_version", -1))),
        known_airline=b.state.known_airline(a),
        known_route=b.state.known_route(a, d),
    )