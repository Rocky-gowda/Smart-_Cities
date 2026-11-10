"""Unit tests: geofencing and the Stage-1 gate."""
from datetime import datetime, timedelta

import pytest

from src.geofence import (CandidateEvent, CircleZone, GeofenceGate, GpsFix, PolygonZone,
                          distance_to_route_m, haversine_m, point_in_polygon)
from src.trajectory import PLANNED_ROUTE, SCENARIOS, ZONES, make_trace, offset

pytestmark = pytest.mark.unit


def test_haversine_zero_distance():
    assert haversine_m(12.97, 77.59, 12.97, 77.59) == 0


def test_haversine_one_degree_latitude():
    # one degree of latitude is ~111.2 km
    assert haversine_m(12.0, 77.0, 13.0, 77.0) == pytest.approx(111_195, rel=1e-3)


def test_haversine_known_city_pair():
    # Bangalore (12.9716, 77.5946) to Chennai (13.0827, 80.2707): ~290 km
    assert haversine_m(12.9716, 77.5946, 13.0827, 80.2707) / 1000 == pytest.approx(290, abs=5)


def test_haversine_symmetric():
    a, b = (12.97, 77.59), (12.99, 77.62)
    assert haversine_m(*a, *b) == pytest.approx(haversine_m(*b, *a))


def test_circle_zone_inside_and_outside():
    z = CircleZone("z", 12.97, 77.59, 200)
    assert z.contains(offset(12.97, 77.59, 150, 0))
    assert not z.contains(offset(12.97, 77.59, 250, 0))


def test_polygon_zone():
    sq = [(12.0, 77.0), (12.01, 77.0), (12.01, 77.01), (12.0, 77.01)]
    assert point_in_polygon((12.005, 77.005), sq)
    assert not point_in_polygon((12.02, 77.005), sq)
    assert PolygonZone("p", sq).contains((12.001, 77.009))


def test_distance_to_route_on_route_is_small():
    mid = ((PLANNED_ROUTE[0][0] + PLANNED_ROUTE[1][0]) / 2, (PLANNED_ROUTE[0][1] + PLANNED_ROUTE[1][1]) / 2)
    assert distance_to_route_m(mid, PLANNED_ROUTE) < 5


def test_distance_to_route_offset_point():
    p = offset(*PLANNED_ROUTE[0], 300, 0)   # 300 m north of the first waypoint
    assert 200 < distance_to_route_m(p, PLANNED_ROUTE) <= 301


def _still_fixes(minutes, lat=12.97, lon=77.59, start=datetime(2026, 1, 1, 22, 0)):
    return [GpsFix(start + timedelta(minutes=m), lat, lon) for m in range(minutes + 1)]


def test_gate_no_event_before_threshold():
    gate = GeofenceGate("t", PLANNED_ROUTE)
    assert gate.process(_still_fixes(14)) == []


def test_gate_fires_once_at_threshold():
    gate = GeofenceGate("t", PLANNED_ROUTE)
    events = gate.process(_still_fixes(40))
    assert len(events) == 1
    assert events[0].idle_min == pytest.approx(15)


def test_gate_resets_after_moving():
    gate = GeofenceGate("t", PLANNED_ROUTE)
    fixes = _still_fixes(20)
    t = fixes[-1].t
    lat, lon = offset(12.97, 77.59, 500, 0)
    fixes += [GpsFix(t + timedelta(minutes=1 + m), lat, lon) for m in range(20)]
    assert len(gate.process(fixes)) == 2


def test_gate_sets_hazard_flag_and_zone():
    z = CircleZone("danger", 12.97, 77.59, 100, hazard=True)
    gate = GeofenceGate("t", PLANNED_ROUTE, [z])
    ev = gate.process(_still_fixes(16))[0]
    assert ev.hazard_flag == 1 and ev.zones == ["danger"]


def test_safe_zone_does_not_set_hazard():
    z = CircleZone("park", 12.97, 77.59, 100, hazard=False)
    ev = GeofenceGate("t", PLANNED_ROUTE, [z]).process(_still_fixes(16))[0]
    assert ev.hazard_flag == 0 and ev.zones == ["park"]


def test_event_features_shape_and_values():
    ev = CandidateEvent("t", datetime(2026, 1, 1, 6), 0, 0, displacement_m=5, idle_min=30, hour=6, hazard_flag=1)
    x = ev.features()
    assert x.shape == (1, 5)
    assert x[0, 0] == pytest.approx(2.9957, abs=1e-3)   # log(20 m GPS floor)
    assert x[0, 2] == pytest.approx(1.0, abs=1e-6)       # sin(6h) = 1
    assert x[0, 4] == 1.0


def test_walking_trace_has_no_event():
    sc = next(s for s in SCENARIOS if s.name == "normal_walk")
    gate = GeofenceGate("t", PLANNED_ROUTE, ZONES)
    assert gate.process(make_trace(sc, 0)) == []


@pytest.mark.parametrize("name", ["cafe_break", "lost_offroute", "hazard_night", "restricted_area"])
def test_stopping_scenarios_raise_exactly_one_event(name):
    sc = next(s for s in SCENARIOS if s.name == name)
    for seed in range(5):
        assert len(GeofenceGate("t", PLANNED_ROUTE, ZONES).process(make_trace(sc, seed))) == 1


def test_hazard_scenarios_detect_hazard_zone():
    for name in ("hazard_night", "restricted_area"):
        sc = next(s for s in SCENARIOS if s.name == name)
        ev = GeofenceGate("t", PLANNED_ROUTE, ZONES).process(make_trace(sc, 1))[0]
        assert ev.hazard_flag == 1
