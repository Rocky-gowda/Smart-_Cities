"""Synthetic GPS trajectories for end-to-end system runs.

A small Bangalore city-centre map is used: a planned walking route through
tourist spots and a few geofenced hazard zones. Scenarios produce realistic
GPS fix streams (one fix per minute, ~8 m noise) that exercise the Stage-1 gate.
All coordinates are illustrative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from .geofence import CircleZone, GpsFix, PolygonZone

# Planned route: MG Road metro -> Cubbon Park -> Vidhana Soudha -> Museum Road
PLANNED_ROUTE = [
    (12.9755, 77.6066), (12.9763, 77.6008), (12.9779, 77.5951),
    (12.9796, 77.5906), (12.9771, 77.5871), (12.9740, 77.5960), (12.9718, 77.6040),
]

ZONES = [
    CircleZone("Lake periphery (unlit)", 12.9862, 77.5960, 250, hazard=True),
    CircleZone("Construction site", 12.9690, 77.5915, 180, hazard=True),
    PolygonZone("Railway yard (restricted)", [(12.9800, 77.5700), (12.9840, 77.5700),
                                              (12.9840, 77.5780), (12.9800, 77.5780)], hazard=True),
    CircleZone("Cubbon Park (safe zone)", 12.9763, 77.5929, 300, hazard=False),
]

M_PER_DEG_LAT = 111_320.0


def offset(lat: float, lon: float, north_m: float, east_m: float):
    return (lat + north_m / M_PER_DEG_LAT,
            lon + east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat))))


@dataclass
class Scenario:
    name: str
    description: str
    start_hour: float
    stop_at: Optional[tuple]        # (lat, lon) where the tourist stops, None = keeps walking
    stop_minutes: int
    context_z: float = 0.0          # cloud-only context (e.g. recent reports nearby)
    incident: int = 0               # cloud-only nearby incident flag


SCENARIOS = [
    Scenario("normal_walk", "Walks the planned route without stopping", 11.0, None, 0),
    Scenario("cafe_break", "Stops 20 min at a cafe on the route, midday", 13.0, PLANNED_ROUTE[2], 20),
    Scenario("lost_offroute", "Wanders ~400 m off route and stays 25 min, evening", 18.5,
             offset(*PLANNED_ROUTE[4], 350, -200), 25, context_z=0.3),
    Scenario("hazard_night", "Stops 40 min inside the unlit lake periphery at 01:00", 1.0,
             (12.9860, 77.5958), 40, context_z=1.0, incident=1),
    Scenario("restricted_area", "Idle 30 min inside the railway yard at 23:00", 23.0,
             (12.9820, 77.5740), 30, context_z=0.5, incident=1),
]


def walk(route, start: datetime, rng, speed_mps=1.2, gps_sd_m=8.0, every_s=60):
    """Fixes along a polyline at walking speed."""
    fixes, t = [], start
    for (a_lat, a_lon), (b_lat, b_lon) in zip(route[:-1], route[1:]):
        d_n = (b_lat - a_lat) * M_PER_DEG_LAT
        d_e = (b_lon - a_lon) * M_PER_DEG_LAT * math.cos(math.radians(a_lat))
        dist = math.hypot(d_n, d_e)
        steps = max(1, int(dist / (speed_mps * every_s)))
        for k in range(steps):
            f = k / steps
            lat, lon = offset(a_lat, a_lon, f * d_n + rng.normal(0, gps_sd_m), f * d_e + rng.normal(0, gps_sd_m))
            fixes.append(GpsFix(t, lat, lon))
            t += timedelta(seconds=every_s)
    return fixes, t


def make_trace(sc: Scenario, seed: int = 0, day=datetime(2026, 12, 16)) -> List[GpsFix]:
    rng = np.random.default_rng(seed)
    start = day + timedelta(hours=sc.start_hour)
    if sc.stop_at is None:
        return walk(PLANNED_ROUTE, start, rng)[0]
    # walk to the nearest route vertex, detour to the stop point, then stay still
    fixes, t = walk(PLANNED_ROUTE[:3] + [sc.stop_at], start, rng)
    for _ in range(sc.stop_minutes + 1):
        lat, lon = offset(*sc.stop_at, rng.normal(0, 6), rng.normal(0, 6))
        fixes.append(GpsFix(t, lat, lon))
        t += timedelta(minutes=1)
    return fixes
