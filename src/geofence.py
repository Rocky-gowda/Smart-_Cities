"""Stage 1: geofencing and the deterministic event gate.

Implements the pieces of the STSMIRS Stage-1 detector that produce the inputs
for the on-device risk model:
  * Haversine great-circle distance,
  * circular and polygon geofence zones (hazard / restricted areas),
  * distance from the tourist's planned route (route deviation),
  * idle (stationary) detection over a stream of GPS fixes,
  * a gate that raises a candidate event when the tourist has been idle for
    at least 15 minutes, and turns it into the five model features.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
LatLon = Tuple[float, float]


# --------------------------------------------------------------------------- geometry
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Equirectangular projection (metres) around a reference point; accurate to
    well under 1% over a few kilometres, which is all a city geofence needs."""
    x = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def point_segment_distance_m(p: LatLon, a: LatLon, b: LatLon) -> float:
    """Shortest distance from point p to segment a-b, in metres."""
    px, py = _to_local_xy(*p, *a)
    bx, by = _to_local_xy(*b, *a)
    seg2 = bx * bx + by * by
    t = 0.0 if seg2 == 0 else max(0.0, min(1.0, (px * bx + py * by) / seg2))
    return math.hypot(px - t * bx, py - t * by)


def distance_to_route_m(p: LatLon, route: Sequence[LatLon]) -> float:
    """Distance from a point to a planned route (polyline)."""
    if len(route) == 1:
        return haversine_m(*p, *route[0])
    return min(point_segment_distance_m(p, a, b) for a, b in zip(route[:-1], route[1:]))


def point_in_polygon(p: LatLon, poly: Sequence[LatLon]) -> bool:
    """Ray-casting test; polygon given as (lat, lon) vertices."""
    lat, lon = p
    inside = False
    n = len(poly)
    for i in range(n):
        (y1, x1), (y2, x2) = poly[i], poly[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


# --------------------------------------------------------------------------- zones
@dataclass
class CircleZone:
    name: str
    lat: float
    lon: float
    radius_m: float
    hazard: bool = True

    def contains(self, p: LatLon) -> bool:
        return haversine_m(self.lat, self.lon, *p) <= self.radius_m


@dataclass
class PolygonZone:
    name: str
    vertices: List[LatLon]
    hazard: bool = True

    def contains(self, p: LatLon) -> bool:
        return point_in_polygon(p, self.vertices)


# --------------------------------------------------------------------------- GPS fixes and events
@dataclass
class GpsFix:
    t: datetime
    lat: float
    lon: float

    @property
    def point(self) -> LatLon:
        return (self.lat, self.lon)


@dataclass
class CandidateEvent:
    """Output of the Stage-1 gate: one event the on-device model must classify."""
    tourist_id: str
    t: datetime
    lat: float
    lon: float
    displacement_m: float          # distance from planned route
    idle_min: float
    hour: float
    hazard_flag: int               # from the cached (possibly stale) gazetteer
    zones: List[str] = field(default_factory=list)

    def features(self, gps_floor_m: float = 20.0) -> np.ndarray:
        """The five on-device features, in the order the model expects."""
        return np.array([[math.log(max(self.displacement_m, gps_floor_m)),
                          math.log(max(self.idle_min, 1e-3)),
                          math.sin(2 * math.pi * self.hour / 24),
                          math.cos(2 * math.pi * self.hour / 24),
                          float(self.hazard_flag)]], dtype=np.float32)


# --------------------------------------------------------------------------- the gate
class GeofenceGate:
    """Stream GPS fixes for one tourist; emit a CandidateEvent when the tourist has
    stayed within `stationary_radius_m` for at least `idle_threshold_min` minutes.

    Only one event is emitted per idle episode; a new episode starts when the
    tourist moves away from the anchor point.
    """

    def __init__(self, tourist_id: str, planned_route: Sequence[LatLon], zones: Iterable = (),
                 idle_threshold_min: float = 15.0, stationary_radius_m: float = 30.0):
        self.tourist_id = tourist_id
        self.route = list(planned_route)
        self.zones = list(zones)
        self.idle_threshold_min = idle_threshold_min
        self.stationary_radius_m = stationary_radius_m
        self._anchor: Optional[GpsFix] = None
        self._fired = False

    def zones_at(self, p: LatLon) -> List[str]:
        return [z.name for z in self.zones if z.contains(p)]

    def hazard_at(self, p: LatLon) -> int:
        return int(any(z.hazard and z.contains(p) for z in self.zones))

    def update(self, fix: GpsFix) -> Optional[CandidateEvent]:
        if self._anchor is None or haversine_m(*self._anchor.point, *fix.point) > self.stationary_radius_m:
            self._anchor, self._fired = fix, False
            return None
        idle = (fix.t - self._anchor.t).total_seconds() / 60.0
        if idle >= self.idle_threshold_min and not self._fired:
            self._fired = True
            return CandidateEvent(
                tourist_id=self.tourist_id, t=fix.t, lat=fix.lat, lon=fix.lon,
                displacement_m=distance_to_route_m(fix.point, self.route), idle_min=idle,
                hour=fix.t.hour + fix.t.minute / 60.0, hazard_flag=self.hazard_at(fix.point),
                zones=self.zones_at(fix.point))
        return None

    def process(self, fixes: Iterable[GpsFix]) -> List[CandidateEvent]:
        return [e for e in (self.update(f) for f in fixes) if e is not None]
