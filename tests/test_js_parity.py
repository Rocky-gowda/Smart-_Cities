"""The phone's JavaScript engine (web/static/geo.js) must give the same answers as the Python code."""
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from server.services import Model, features
from src.geofence import CircleZone, GeofenceGate, GpsFix, distance_to_route_m, haversine_m, point_in_polygon
from src.trajectory import PLANNED_ROUTE, SCENARIOS, ZONES, make_trace

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.unit, pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")]

ZONES_JSON = [{"name": z.name, "hazard": z.hazard, "kind": "circle",
               "geometry": {"lat": z.lat, "lon": z.lon, "radius_m": z.radius_m}} if isinstance(z, CircleZone) else
              {"name": z.name, "hazard": z.hazard, "kind": "polygon", "geometry": {"vertices": z.vertices}}
              for z in ZONES]


def run_js(script: str, data):
    code = f"const G = require({json.dumps(str(ROOT / 'web/static/geo.js'))});\nconst D = {json.dumps(data)};\n{script}"
    out = subprocess.run([shutil.which("node"), "-e", code], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "TZ": "UTC"})
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_haversine_route_polygon_parity():
    rng = np.random.default_rng(0)
    pts = [(12.95 + rng.random() * 0.05, 77.56 + rng.random() * 0.06) for _ in range(200)]
    poly = ZONES[2].vertices
    js = run_js("console.log(JSON.stringify(D.pts.map(p => [G.haversine(p[0], p[1], 12.9716, 77.5946), "
                "G.distanceToRoute(p, D.route), G.pointInPolygon(p, D.poly)])))",
                {"pts": pts, "route": PLANNED_ROUTE, "poly": poly})
    for (h, d, inside), p in zip(js, pts):
        assert h == pytest.approx(haversine_m(*p, 12.9716, 77.5946), rel=1e-9)
        assert d == pytest.approx(distance_to_route_m(p, PLANNED_ROUTE), rel=1e-9, abs=1e-6)
        assert inside == point_in_polygon(p, poly)


def test_model_parity():
    m = Model(ROOT / "models" / "deployed_model.json")
    rng = np.random.default_rng(1)
    events = [{"displacement_m": float(np.exp(rng.normal(5.7, 1))), "idle_min": float(15 + rng.gamma(2, 20)),
               "hour": float(rng.uniform(0, 24)), "hazard_flag": int(rng.random() < .3)} for _ in range(300)]
    js = run_js("console.log(JSON.stringify(D.ev.map(e => G.predict(D.model, G.features(e)).probabilities)))",
                {"ev": events, "model": m.js})
    for p_js, e in zip(js, events):
        p_py = m.proba(features(e["displacement_m"], e["idle_min"], e["hour"], e["hazard_flag"]))
        assert np.allclose(p_js, p_py, atol=1e-9)


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_geofence_gate_parity_on_scenarios(name):
    sc = next(s for s in SCENARIOS if s.name == name)
    fixes = make_trace(sc, 3)
    py = GeofenceGate("t", PLANNED_ROUTE, ZONES).process(fixes)
    js = run_js("const g = new G.GeofenceGate(D.route, D.zones, {idleMinutes: 15, timeScale: 1});"
                "console.log(JSON.stringify(D.fixes.map(f => g.update(f)).filter(e => e)))",
                {"route": PLANNED_ROUTE, "zones": ZONES_JSON,
                 "fixes": [{"t": f.t.isoformat() + "Z", "lat": f.lat, "lon": f.lon} for f in fixes]})
    assert len(js) == len(py)
    for a, b in zip(js, py):
        assert a["idle_min"] == pytest.approx(b.idle_min)
        assert a["displacement_m"] == pytest.approx(b.displacement_m, rel=1e-9)
        assert a["hazard_flag"] == b.hazard_flag and a["zones"] == b.zones
        assert a["hour"] == pytest.approx(b.hour)


def test_time_scale_accelerates_idle():
    t0 = datetime(2026, 1, 1, 12, 0)
    fixes = [{"t": (t0 + timedelta(seconds=s)).isoformat() + "Z", "lat": 12.97, "lon": 77.59} for s in range(0, 40, 5)]
    js = run_js("const g = new G.GeofenceGate(D.route, [], {idleMinutes: 15, timeScale: 60});"
                "console.log(JSON.stringify(D.fixes.map(f => g.update(f)).filter(e => e)))",
                {"route": PLANNED_ROUTE, "fixes": fixes})
    assert len(js) == 1 and js[0]["idle_min"] == pytest.approx(15)       # 15 real seconds x 60 = 15 minutes


def test_escalation_policy_parity():
    js = run_js("console.log(JSON.stringify([G.escalationDecision({confidence: .9, index: 0}, .725, 'safety').escalate,"
                "G.escalationDecision({confidence: .9, index: 3, label: 'CRITICAL'}, .725, 'safety').escalate,"
                "G.escalationDecision({confidence: .5, index: 0}, .725, 'conf').escalate]))", {})
    assert js == [False, True, True]
