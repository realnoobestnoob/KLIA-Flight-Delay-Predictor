"""Synthetic 'departures' data shaped like your Neon table. For tests, CI and trying the pipeline
without a database. NOT real flights, never use it to judge model quality.

    python -m klia.demo --rows 20000 --out data/demo_departures.csv
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

AIRLINES = ["AIRASIA", "MALAYSIA AIRLINES", "BATIK AIR", "SCOOT", "FIREFLY", "INDIGO", "CATHAY PACIFIC",
            "SINGAPORE AIRLINES", "EMIRATES", "QATAR AIRWAYS", "JETSTAR", "THAI LION AIR"]
DESTS = ["SINGAPORE", "BANGKOK", "JAKARTA", "BALI", "HONG KONG", "TOKYO", "LONDON", "DUBAI", "DOHA", "SYDNEY",
         "PENANG", "KOTA KINABALU", "KUCHING", "LANGKAWI", "SEOUL", "MANILA", "HO CHI MINH CITY", "DELHI"]
AIRCRAFT = ["A320", "A321", "A330", "A350", "B737", "B787", "B777", "ATR72"]


def make(rows: int = 20000, seed: int = 7, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    air_eff = dict(zip(AIRLINES, rng.normal(0, 0.5, len(AIRLINES))))
    dest_eff = dict(zip(DESTS, rng.normal(0, 0.35, len(DESTS))))
    air_dests = {a: list(rng.choice(DESTS, size=rng.integers(4, 10), replace=False)) for a in AIRLINES}
    ac_eff = dict(zip(AIRCRAFT, rng.normal(0, 0.15, len(AIRCRAFT))))
    per_day = 60
    days = rows // per_day + 1
    recs, rid = [], 0
    for d in range(days):
        day = dt.date.fromisoformat(start) + dt.timedelta(days=d)
        storm = rng.random() < 0.12
        day_recs = []
        for _ in range(per_day):
            a = AIRLINES[rng.integers(len(AIRLINES))]
            dest = air_dests[a][rng.integers(len(air_dests[a]))]
            hour = int(np.clip(rng.normal(13, 5), 0, 23))
            minute = int(rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))
            ac = AIRCRAFT[rng.integers(len(AIRCRAFT))]
            gust = float(max(0, rng.normal(25 if storm else 14, 6)))
            rain = float(max(0, rng.normal(4 if storm else 0.3, 2)))
            drift = 0.5 if d > days * 0.6 and a in AIRLINES[:3] else 0.0   # concept drift late in the data
            z = (-1.0 + air_eff[a] + dest_eff[dest] + ac_eff[ac] + drift + 0.55 * (16 <= hour <= 20)
                 + 0.03 * (gust - 15) + 0.08 * rain + 0.3 * (day.weekday() >= 5) + rng.normal(0, 0.6))
            delay = float(rng.normal(34, 22) if rng.random() < 1 / (1 + np.exp(-z)) else rng.normal(2, 7))
            sched = dt.datetime.combine(day, dt.time(hour, minute))
            act = sched + dt.timedelta(minutes=delay)
            fmt12 = rng.random() < 0.15                          # mixed 12h/24h strings, like the real data
            f = (lambda t: t.strftime("%I:%M %p").lstrip("0")) if fmt12 else (lambda t: t.strftime("%H:%M"))
            day_recs.append({"id": 0, "flight_number": f"{a[:2]}{rng.integers(100, 999)}", "date": day.isoformat(),
                         "scheduled_departure": f(sched), "actual_departure": f(act), "airline": a,
                         "destination": dest, "aircraft": ac, "wind_gusts_10m": round(gust, 1),
                         "precipitation": round(rain, 2), "cloud_cover_mid": float(rng.integers(0, 100)), "_sched": sched})
        # ids follow scheduled time within a day, like rows appended as flights are recorded
        day_recs.sort(key=lambda r: r["_sched"])
        for r in day_recs:
            rid += 1
            r["id"] = rid
            del r["_sched"]
        recs.extend(day_recs)
    return pd.DataFrame(recs).head(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--out", default="data/demo_departures.csv")
    a = ap.parse_args()
    make(a.rows).to_csv(a.out, index=False)
    print(f"wrote {a.rows} rows to {a.out}")
