"""Integration tests of the real application's server (REST API, auth, live feed, blockchain)."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from server.llm import CloudVerifier
from server.main import create_app
from server.settings import Settings

pytestmark = pytest.mark.integration

TOURIST = dict(name="Asha Verma", nationality="IN", doc_type="Aadhaar", doc_number="123412341234",
               phone="+919845000000", emergency_contact="+919886000000", itinerary=["MG Road", "Cubbon Park"],
               valid_from="2026-01-01", valid_to="2030-12-31")
LAKE = (12.9860, 77.5958)         # inside the seeded "Lake periphery (unlit)" hazard zone
ON_ROUTE = (12.9757, 77.6000)     # on the MG Road -> Cubbon Park route


def anthropic_mock(level, calls):
    def handler(req: httpx.Request):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"content": [{"type": "text",
                                                      "text": json.dumps({"risk_level": level, "reason": "test"})}]})
    return CloudVerifier("anthropic", "test-model", "k", transport=httpx.MockTransport(handler))


@pytest.fixture
def env(artifacts, tmp_path):
    def make(verifier=None, **kw):
        app = create_app(Settings(data_dir=tmp_path, **kw), verifier=verifier)
        return TestClient(app), app
    return make


@pytest.fixture
def client(env):
    return env()[0]


def register(c, **over):
    r = c.post("/api/tourists/register", json={**TOURIST, **over})
    assert r.status_code == 200, r.text
    return r.json()


def police(c):
    r = c.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": "Bearer " + r.json()["token"]}


def bearer(tok):
    return {"Authorization": "Bearer " + tok}


def event(cid, lat, lon, idle=20, hour=13.0, **kw):
    return {"client_event_id": cid, "t": "2026-11-01T13:00:00+05:30", "lat": lat, "lon": lon,
            "idle_min": idle, "hour": hour, **kw}


# ---------------------------------------------------------------- basics
def test_health_and_pages(client):
    assert client.get("/api/health").json()["status"] == "ok"
    for path in ("/", "/tourist/", "/police/", "/static/geo.js", "/tourist/manifest.json", "/tourist/sw.js", "/docs"):
        assert client.get(path).status_code == 200, path


def test_config_has_model_and_zones(client):
    cfg = client.get("/api/config").json()
    assert len(cfg["model"]["coef"]) == 4 and len(cfg["zones"]) >= 4 and cfg["theta"] > 0


# ---------------------------------------------------------------- auth
def test_police_login_and_wrong_password(client):
    assert client.post("/api/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    assert client.get("/api/me", headers=police(client)).json()["role"] == "admin"


def test_police_endpoints_require_login(client):
    for path in ("/api/incidents", "/api/tourists", "/api/chain", "/api/me"):
        assert client.get(path).status_code == 401
    t = register(client)
    assert client.get("/api/incidents", headers=bearer(t["token"])).status_code == 401   # tourist is not police


def test_tourist_endpoints_require_token(client):
    assert client.post("/api/track", json={"fixes": []}).status_code == 401
    assert client.post("/api/track", json={"fixes": []}, headers=bearer("forged.token")).status_code == 401


def test_admin_can_add_officer(client):
    h = police(client)
    assert client.post("/api/officers", json={"username": "officer1", "password": "secret99"}, headers=h).status_code == 200
    r = client.post("/api/auth/login", json={"username": "officer1", "password": "secret99"}).json()
    assert r["role"] == "police"
    assert client.post("/api/officers", json={"username": "x2", "password": "secret99"},
                       headers=bearer(r["token"])).status_code == 403


# ---------------------------------------------------------------- registration and digital ID
def test_registration_issues_blockchain_id(client):
    t = register(client)
    assert t["tourist_id"].startswith("TID-") and t["qr_png"] and t["block"] >= 1
    me = client.get("/api/tourists/me", headers=bearer(t["token"])).json()
    assert me["credential"]["tourist_id"] == t["tourist_id"]


def test_registration_rejects_bad_input(client):
    assert client.post("/api/tourists/register", json={**TOURIST, "itinerary": []}).status_code == 400
    assert client.post("/api/tourists/register", json={**TOURIST, "name": ""}).status_code == 422


def test_duplicate_registration_same_trip_blocked(client):
    register(client)
    assert client.post("/api/tourists/register", json=TOURIST).status_code == 400


def test_verify_valid_forged_and_revoked(client):
    t = register(client)
    h = police(client)
    cred = json.dumps(t["credential"])
    assert client.post("/api/verify", json={"credential": cred}, headers=h).json()["valid"]
    forged = json.dumps({**t["credential"], "name": "Impostor"})
    assert not client.post("/api/verify", json={"credential": forged}, headers=h).json()["valid"]
    assert client.post("/api/verify", json={"credential": "not json"}, headers=h).status_code == 400
    client.post(f"/api/tourists/{t['tourist_id']}/revoke", json={"reason": "test"}, headers=h)
    r = client.post("/api/verify", json={"credential": cred}, headers=h).json()
    assert not r["valid"] and "revoked" in r["reason"]


# ---------------------------------------------------------------- tracking and events
def test_tracking_stores_positions(client):
    t = register(client)
    fixes = [{"t": f"2026-11-01T10:0{i}:00Z", "lat": 12.975 + i * 1e-4, "lon": 77.60} for i in range(5)]
    assert client.post("/api/track", json={"fixes": fixes}, headers=bearer(t["token"])).json()["stored"] == 5
    tr = client.get(f"/api/tourists/{t['tourist_id']}/track", headers=police(client)).json()
    assert len(tr) == 5
    row = client.get("/api/tourists", headers=police(client)).json()[0]
    assert row["lat"] == pytest.approx(12.9754)


def test_low_risk_event_decided_on_device(client):
    t = register(client)
    r = client.post("/api/events", json={"events": [event("e1", *ON_ROUTE, idle=16, hour=13)]},
                    headers=bearer(t["token"])).json()["decisions"][0]
    assert r["decided_by"] == "on-device" and not r["escalated"] and r["final_class"] in ("LOW", "MODERATE")


def test_hazard_event_escalated_to_llm(env):
    calls = []
    c, _ = env(verifier=anthropic_mock("CRITICAL", calls))
    t = register(c)
    r = c.post("/api/events", json={"events": [event("e2", *LAKE, idle=40, hour=1)]},
               headers=bearer(t["token"])).json()["decisions"][0]
    assert r["escalated"] and r["final_class"] == "CRITICAL" and r["decided_by"].startswith("anthropic")
    prompt = calls[0]["messages"][0]["content"]
    assert "Lake periphery" in prompt and "hazard zone: yes" in prompt


def test_llm_failure_falls_back_to_rules(env):
    def boom(req):
        raise httpx.ConnectTimeout("no network")
    c, _ = env(verifier=CloudVerifier("openai", "m", "k", transport=httpx.MockTransport(boom)))
    t = register(c)
    r = c.post("/api/events", json={"events": [event("e3", *LAKE, idle=40, hour=1)]},
               headers=bearer(t["token"])).json()["decisions"][0]
    assert r["decided_by"] == "rules engine (fallback)" and r["final_class"] in ("HIGH", "CRITICAL")


def test_server_recomputes_features(client):
    """The server does not trust the phone's displacement / hazard values."""
    t = register(client)
    r = client.post("/api/events", json={"events": [event("e4", *LAKE, idle=40, hour=1, displacement_m=1.0,
                                                          hazard_flag=0)]}, headers=bearer(t["token"])).json()
    inc = client.get("/api/incidents", headers=police(client)).json()[0]
    assert inc["features"]["hazard_flag"] == 1 and inc["features"]["displacement_m"] > 500
    assert r["decisions"][0]["final_class"] in ("HIGH", "CRITICAL")


def test_offline_events_synced_later_and_idempotent(client):
    t = register(client)
    batch = {"events": [event("off1", *LAKE, idle=30, hour=23, offline=True),
                        event("off2", *ON_ROUTE, idle=16, hour=12, offline=True)]}
    first = client.post("/api/events", json=batch, headers=bearer(t["token"])).json()["decisions"]
    again = client.post("/api/events", json=batch, headers=bearer(t["token"])).json()["decisions"]
    assert len(first) == 2 and all(d.get("duplicate") for d in again)
    incs = client.get("/api/incidents", headers=police(client)).json()
    assert len(incs) == 2 and all(i["synced_late"] for i in incs)


def test_sos_is_critical(client):
    t = register(client)
    r = client.post("/api/sos", json={"client_event_id": "s1", "t": "2026-11-01T22:00:00Z", "lat": 12.97, "lon": 77.59},
                    headers=bearer(t["token"])).json()
    assert r["final_class"] == "CRITICAL" and r["decided_by"] == "tourist SOS"


def test_incident_workflow(client):
    t = register(client)
    client.post("/api/sos", json={"t": "2026-11-01T22:00:00Z", "lat": 12.97, "lon": 77.59}, headers=bearer(t["token"]))
    h = police(client)
    iid = client.get("/api/incidents", headers=h).json()[0]["id"]
    assert client.post(f"/api/incidents/{iid}/ack", json={}, headers=h).json()["status"] == "acknowledged"
    row = client.post(f"/api/incidents/{iid}/resolve", json={"notes": "tourist safe"}, headers=h).json()
    assert row["status"] == "resolved" and row["handled_by"] == "admin"
    assert client.get("/api/incidents?status=open", headers=h).json() == []
    assert client.post(f"/api/incidents/{iid}/explode", json={}, headers=h).status_code == 404


def test_every_incident_anchored_on_blockchain(client):
    t = register(client)
    for i in range(3):
        client.post("/api/sos", json={"t": "2026-11-01T22:00:00Z", "lat": 12.97, "lon": 77.59}, headers=bearer(t["token"]))
    ch = client.get("/api/chain", headers=police(client)).json()
    kinds = [tx["kind"] for b in ch["blocks"] for tx in b["transactions"]]
    assert ch["intact"] and kinds.count("INCIDENT") == 3 and kinds.count("REGISTER_ID") == 1
    assert all(i["chain_tx"] for i in client.get("/api/incidents", headers=police(client)).json())


def test_nearby_incidents_raise_context(env):
    calls = []
    c, _ = env(verifier=anthropic_mock("HIGH", calls))
    a, b = register(c), register(c, doc_number="P9999999", name="Ben Ray", doc_type="Passport", nationality="GB")
    for i in range(2):
        c.post("/api/sos", json={"t": "2026-11-01T22:00:00Z", "lat": LAKE[0], "lon": LAKE[1]}, headers=bearer(a["token"]))
    c.post("/api/events", json={"events": [event("n1", *LAKE, idle=40, hour=1)]}, headers=bearer(b["token"]))
    assert "within 500 m in the last 24 h: 2" in calls[-1]["messages"][0]["content"]


# ---------------------------------------------------------------- zones
def test_police_adds_zone_and_it_affects_risk(client):
    t = register(client)
    h = police(client)
    spot = (12.9690, 77.6100)      # far from the seeded zones
    assert client.post("/api/zones", json={"name": "New underpass", "lat": spot[0], "lon": spot[1], "radius_m": 150},
                       headers=h).status_code == 200
    assert any(z["name"] == "New underpass" for z in client.get("/api/zones").json())
    client.post("/api/events", json={"events": [event("z1", *spot, idle=30, hour=23)]}, headers=bearer(t["token"]))
    inc = client.get("/api/incidents", headers=h).json()[0]
    assert inc["zones"] == ["New underpass"] and inc["features"]["hazard_flag"] == 1
    assert client.post("/api/zones", json={"name": "bad", "kind": "circle"}, headers=h).status_code == 400


# ---------------------------------------------------------------- live feed
def test_websocket_pushes_incidents(client):
    t = register(client)
    tok = police(client)["Authorization"].split()[1]
    with client.websocket_connect(f"/ws?token={tok}") as ws:
        client.post("/api/sos", json={"t": "2026-11-01T22:00:00Z", "lat": 12.97, "lon": 77.59}, headers=bearer(t["token"]))
        msg = ws.receive_json()
        assert msg["type"] == "incident" and msg["incident"]["final_class"] == "CRITICAL"


def test_websocket_rejects_without_login(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws?token=bad") as ws:
            ws.receive_text()


def test_data_survives_restart(env, tmp_path):
    c, _ = env()
    t = register(c)
    c2, _ = env()                                   # new server process, same data folder
    me = c2.get("/api/tourists/me", headers=bearer(t["token"])).json()
    assert me["tourist_id"] == t["tourist_id"]
    assert c2.get("/api/chain", headers=police(c2)).json()["intact"]
