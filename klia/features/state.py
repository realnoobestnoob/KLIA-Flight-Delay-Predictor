"""FeatureState: the single source of truth for features, used by training AND serving.

Every feature is a smoothed running statistic that is kept up to date one flight at a time:

    x = state.features(flight)     # READ  : never mutates the state
    state.update(flight, label)    # WRITE : call after the outcome is known

Because training replays flights in time order through exactly this code, and the API calls
`features()` on the saved state, there is no training/serving skew and no look-ahead leakage.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import deque

from klia.etl.validate import norm

AIRCRAFT_UNKNOWN = "UNKNOWN"


class Rate:
    """Running delay rate for one key (a count and a sum)."""
    __slots__ = ("n", "s")

    def __init__(self):
        self.n = 0
        self.s = 0

    def smoothed(self, prior: float, m: float) -> float:
        return (self.s + m * prior) / (self.n + m)


def _rate(d: dict, key, prior: float, m: float) -> float:
    r = d.get(key)
    return prior if r is None else r.smoothed(prior, m)


def _bump(d: dict, key, y: int) -> None:
    r = d.get(key)
    if r is None:
        r = d[key] = Rate()
    r.n += 1
    r.s += y


class FeatureState:
    def __init__(self, cfg: dict):
        f = cfg["features"]
        self.m = float(f["prior_strength"])
        self.alpha = 2.0 / (float(f["ewm_span"]) + 1.0)
        self.windows = list(f["route_windows_days"])
        self.default_fph = float(f["default_flights_per_hour"])
        hol = {dt.date.fromisoformat(str(d)) for d in f.get("public_holidays", [])}
        self.holidays = {d.toordinal() for d in hol}
        self.holiday_eves = {d.toordinal() - 1 for d in hol}
        # Configurable with sensible KLIA defaults: morning rush (06-09), evening rush (17-21)
        self.peak_hours = frozenset(f.get("peak_hours", [6, 7, 8, 17, 18, 19, 20]))
        # Red-eye: late night and early morning where demand patterns differ sharply
        self.red_eye_hours = frozenset(f.get("red_eye_hours", [0, 1, 2, 3, 4, 22, 23]))

        self.n = 0
        self.s = 0
        self.airline: dict[str, Rate] = {}
        self.destination: dict[str, Rate] = {}
        self.aircraft: dict[str, Rate] = {}
        self.route: dict[tuple, Rate] = {}
        self.airline_hour: dict[tuple, Rate] = {}
        self.route_hour: dict[tuple, Rate] = {}
        self.airline_ewm: dict[str, float] = {}
        self.airline_last3: dict[str, deque] = {}
        self.route_hist: dict[tuple, deque] = {}
        self.hour_days: dict[int, dict[int, int]] = {}
        self.last_date_ord = 0

        # Cascade delay propagation: last observed outcome per airline and route.
        # Pipeline.py's feature_stream guarantees features() is read BEFORE update()
        # is called, so there is no leakage — each flight sees only past outcomes.
        self.airline_last: dict[str, int] = {}
        self.route_last: dict[tuple, int] = {}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("peak_hours", frozenset([6, 7, 8, 17, 18, 19, 20]))
        self.__dict__.setdefault("red_eye_hours", frozenset([0, 1, 2, 3, 4, 22, 23]))
        self.__dict__.setdefault("airline_last", {})
        self.__dict__.setdefault("route_last", {})

    # ---------------------------------------------------------------- READ
    @property
    def base_rate(self) -> float:
        return (self.s + 1.0) / (self.n + 3.0)   # ~0.33 before any data, converges to the true rate

    def features(self, row: dict) -> dict:
        """row keys: sched_dt (datetime), airline, destination, aircraft (opt)."""
        t: dt.datetime = row["sched_dt"]
        a, dest = row["airline"], row["destination"]
        ac = row.get("aircraft") or AIRCRAFT_UNKNOWN
        hour, dow, d_ord = t.hour, t.weekday(), t.toordinal()
        m, g = self.m, self.base_rate
        route = (a, dest)

        r_air   = _rate(self.airline,      a,           g,     m)
        r_dest  = _rate(self.destination,  dest,        g,     m)
        r_route = _rate(self.route,        route,       r_air, m)
        r_air_h = _rate(self.airline_hour, (a, hour),   r_air, m)
        r_route_h = _rate(self.route_hour, (route, hour), r_route, m)
        r_ac    = g if ac == AIRCRAFT_UNKNOWN else _rate(self.aircraft, ac, g, m)

        ewm   = self.airline_ewm.get(a)
        last3 = self.airline_last3.get(a)
        hist  = self.route_hist.get(route)

        # ── Time-of-day categories ─────────────────────────────────────────────
        # Explicit flags complement sinusoidal encoding: sin/cos encode smoothly
        # across midnight but cannot cleanly separate a peak band for linear models.
        is_peak    = float(hour in self.peak_hours)
        is_red_eye = float(hour in self.red_eye_hours)

        # ── Cascade delay propagation ──────────────────────────────────────────
        # ~30-40% of real-world delays propagate from the previous flight on the
        # same airline or route. Default to the smoothed rate when no history exists.
        airline_prev1 = self.airline_last.get(a, g)
        route_prev1   = self.route_last.get(route, r_route)

        # ── Route maturity ─────────────────────────────────────────────────────
        # log1p of flights seen on this route: tells the model how reliable
        # route_rate is. Low count → rate shrinks to prior; high count → trust it.
        r_obj      = self.route.get(route)
        route_log_n = math.log1p(r_obj.n if r_obj is not None else 0)

        x = {
            "hour_sin":            math.sin(2 * math.pi * hour / 24),
            "hour_cos":            math.cos(2 * math.pi * hour / 24),
            "dow_sin":             math.sin(2 * math.pi * dow / 7),
            "dow_cos":             math.cos(2 * math.pi * dow / 7),
            "is_weekend":          float(dow >= 5),
            "is_public_holiday":   float(d_ord in self.holidays),
            "is_holiday_eve":      float(d_ord in self.holiday_eves),
            "is_peak_hour":        is_peak,
            "is_red_eye":          is_red_eye,
            "airline_rate":        r_air,
            "airline_rate_ewm":    r_air if ewm is None else float(ewm),
            "airline_prev3":       r_air if not last3 else sum(last3) / len(last3),
            "airline_prev1":       airline_prev1,
            "airline_hour_rate":   r_air_h,
            "destination_rate":    r_dest,
            "route_rate":          r_route,
            "route_hour_rate":     r_route_h,
            "route_prev1":         route_prev1,
            "route_log_n":         route_log_n,
            "aircraft_rate":       r_ac,
        }

        for w in self.windows:
            if hist:
                lo = d_ord - w
                ys = [y for (d, y) in hist if lo <= d < d_ord]
                x[f"route_rate_{w}d"] = sum(ys) / len(ys) if ys else r_route
            else:
                x[f"route_rate_{w}d"] = r_route

        days = self.hour_days.get(hour)
        fph = self.default_fph
        if days:
            cnts = [c for d, c in days.items() if d_ord - 30 <= d < d_ord]
            if cnts:
                fph = sum(cnts) / len(cnts)
        x["flights_per_hour"]          = fph
        x["congestion_x_airline_rate"] = fph * r_air

        # ── Interaction features ───────────────────────────────────────────────
        # Linear / SGD models (the default bag_sgd_log base) cannot discover
        # multiplicative relationships on their own. This explicit product gives
        # the model a direct signal for the strongest combined effect.
        x["peak_x_airline_rate"] = is_peak * r_air

        return x

    # --------------------------------------------------------------- WRITE
    def update(self, row: dict, y: int) -> None:
        t: dt.datetime = row["sched_dt"]
        a, dest = row["airline"], row["destination"]
        ac = row.get("aircraft") or AIRCRAFT_UNKNOWN
        hour, d_ord = t.hour, t.toordinal()
        route = (a, dest)

        self.n += 1
        self.s += y
        _bump(self.airline,      a,           y)
        _bump(self.destination,  dest,        y)
        if ac != AIRCRAFT_UNKNOWN:
            _bump(self.aircraft, ac,          y)
        _bump(self.route,        route,       y)
        _bump(self.airline_hour, (a, hour),   y)
        _bump(self.route_hour,   (route, hour), y)

        prev = self.airline_ewm.get(a)
        self.airline_ewm[a] = float(y) if prev is None else prev + self.alpha * (y - prev)
        self.airline_last3.setdefault(a, deque(maxlen=3)).append(y)
        self.route_hist.setdefault(route, deque(maxlen=max(self.windows) * 12)).append((d_ord, y))

        days = self.hour_days.setdefault(hour, {})
        days[d_ord] = days.get(d_ord, 0) + 1
        if len(days) > 120:                       # keep memory bounded
            for k in [k for k in days if k < d_ord - 60]:
                del days[k]

        self.last_date_ord = max(self.last_date_ord, d_ord)

        # ── Cascade delay propagation ──────────────────────────────────────────
        # Written AFTER features() is read (pipeline.py's feature_stream guarantees
        # read → update order), so there is no leakage.
        self.airline_last[a]    = y
        self.route_last[route]  = y

    # --------------------------------------------------------------- misc
    def known_airline(self, a: str) -> bool:
        return a in self.airline

    def known_route(self, a: str, d: str) -> bool:
        return (a, d) in self.route

    def options(self) -> dict:
        routes: dict[str, list] = {}
        for (a, d) in self.route:
            routes.setdefault(a, []).append(d)
        return {"airlines": sorted(self.airline), "destinations": sorted(self.destination),
                "routes": {k: sorted(v) for k, v in sorted(routes.items())}}


def row_from_record(rec: dict) -> dict:
    """Make a feature-ready row from a cleaned record (normalises text keys)."""
    r = dict(rec)
    r["airline"]     = norm(r["airline"])
    r["destination"] = norm(r["destination"])
    r["aircraft"]    = norm(r.get("aircraft")) or AIRCRAFT_UNKNOWN
    return r
