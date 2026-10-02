"""Free Open-Meteo forecast with an in-memory cache. On any failure the caller falls back to running means."""
from __future__ import annotations

import datetime as dt
import threading
import time

import httpx

HOURLY = ["wind_gusts_10m", "precipitation", "cloud_cover_mid"]


class Weather:
    def __init__(self, cfg: dict):
        a = cfg["api"]
        self.lat, self.lon, self.tz = a["klia_lat"], a["klia_lon"], a["timezone"]
        self.ttl = a["weather_ttl_seconds"]
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _fetch_day(self, day: dt.date) -> dict | None:
        try:
            r = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=5.0, params={
                "latitude": self.lat, "longitude": self.lon, "hourly": ",".join(HOURLY),
                "start_date": day.isoformat(), "end_date": day.isoformat(), "timezone": self.tz})
            r.raise_for_status()
            h = r.json().get("hourly") or {}
            return h if "time" in h else None
        except Exception:
            return None

    def get(self, when: dt.datetime) -> tuple[dict | None, str]:
        """Return ({col: value}, source) for the hour of `when`, or (None, 'fallback')."""
        key = when.date().isoformat()
        with self._lock:
            hit = self._cache.get(key)
        if hit is None or time.monotonic() - hit[0] > self.ttl:
            data = self._fetch_day(when.date())
            if data is not None:
                with self._lock:
                    self._cache[key] = (time.monotonic(), data)
                    if len(self._cache) > 64:
                        self._cache.pop(next(iter(self._cache)))
            elif hit is not None:
                data = hit[1]
        else:
            data = hit[1]
        if not data:
            return None, "fallback"
        try:
            idx = next(i for i, t in enumerate(data["time"]) if dt.datetime.fromisoformat(t).hour == when.hour)
            out = {c: data[c][idx] for c in HOURLY}
            return (out, "open-meteo") if all(v is not None for v in out.values()) else (None, "fallback")
        except Exception:
            return None, "fallback"
