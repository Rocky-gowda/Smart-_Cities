"""Tests of the tourist incident-report feature: tourist -> police dashboard (live), blockchain evidence,
warnings for other tourists, crowd-corroborated caution zones and police review."""
import base64
import hashlib
import io

import pytest
from PIL import Image

from tests.test_server import LAKE, ON_ROUTE, bearer, client, env, event, police, register  # noqa: F401

pytestmark = pytest.mark.integration


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def report(c, tok, lat, lon, cat="theft", cid=None, photo=None, **kw):
    body = {"t": "2026-11-10T11:00:00+05:30", "lat": lat, "lon": lon, "category": cat,
            "description": "phone snatched near bus stop", **kw}
    if cid:
        body["client_report_id"] = cid
    if photo:
        body["photo"] = "data:image/jpeg;base64," + base64.b64encode(photo).decode()
    return c.post("/api/reports", json=body, headers=bearer(tok))


def test_report_reaches_police_with_photo_and_blockchain_evidence(client):
    tok = register(client)["token"]
    img = jpeg()
    with client.websocket_connect("/ws?token=" + police(client)["Authorization"][7:]) as ws:
        r = report(client, tok, *ON_ROUTE, cid="r1", photo=img)
        assert r.status_code == 200, r.text
        msg = ws.receive_json()
    d = r.json()
    assert d["app_said_safe"] is True and d["severity"] == "MODERATE" and d["chain_tx"]
    assert d["photo_sha256"] == hashlib.sha256(img).hexdigest()
    assert msg["type"] == "report" and msg["incident"]["source"] == "report"
    reps = client.get("/api/reports", headers=police(client)).json()
    assert reps[0]["description"] == "phone snatched near bus stop"
    photo = client.get(f"/api/reports/{reps[0]['id']}/photo", headers=police(client))
    assert photo.status_code == 200 and photo.content == img
    chain = client.get("/api/chain", headers=police(client)).json()
    evid = [t["payload"].get("evidence_sha256") for b in chain["blocks"] for t in b["transactions"] if t["kind"] == "INCIDENT"]
    assert d["photo_sha256"] in evid and chain["intact"]
    inc = client.get("/api/incidents", headers=police(client)).json()[0]
    assert inc["source"] == "report" and inc["features"]["app_said_safe"] is True


def test_report_inside_hazard_zone_and_severity(client):
    tok = register(client)["token"]
    d = report(client, tok, *LAKE, cat="assault").json()
    assert d["app_said_safe"] is False and d["severity"] == "CRITICAL"


def test_duplicate_and_validation(client):
    tok = register(client)["token"]
    assert report(client, tok, *ON_ROUTE, cid="same").json()["duplicate"] is False
    assert report(client, tok, *ON_ROUTE, cid="same").json()["duplicate"] is True
    assert len(client.get("/api/reports", headers=police(client)).json()) == 1
    assert report(client, tok, *ON_ROUTE, cat="aliens").status_code == 400
    bad = base64.b64encode(b"MZ\x90\x00 not an image").decode()
    assert client.post("/api/reports", json={"t": "x", "lat": 12.97, "lon": 77.6, "category": "theft", "photo": bad},
                       headers=bearer(tok)).status_code == 400
    assert client.post("/api/reports", json={"t": "x", "lat": 12.97, "lon": 77.6, "category": "theft"}).status_code == 401
    assert client.get("/api/reports").status_code == 401


def test_other_tourists_see_anonymised_warning(client):
    a = register(client)["token"]
    b = register(client, name="Ben Clark", nationality="GB", doc_type="Passport", doc_number="P1234567")["token"]
    report(client, a, *ON_ROUTE)
    near = client.get("/api/reports/nearby", headers=bearer(b)).json()
    assert len(near) == 1 and near[0]["label"].startswith("Theft")
    assert not ({"name", "phone", "tourist_id", "photo_file"} & near[0].keys())


def test_two_tourists_make_a_caution_zone_that_phones_receive(client):
    a = register(client)["token"]
    b = register(client, name="Ben Clark", nationality="GB", doc_type="Passport", doc_number="P1234567")["token"]
    before = len(client.get("/api/zones").json())
    assert report(client, a, *ON_ROUTE).json()["caution_zone"] is False
    assert len(client.get("/api/zones").json()) == before          # one tourist alone: no zone yet
    d = report(client, b, ON_ROUTE[0] + 0.0008, ON_ROUTE[1]).json()  # ~90 m away, another tourist
    assert d["caution_zone"] is True and d["corroborations"] == 2
    zones = client.get("/api/config").json()["zones"]
    z = [z for z in zones if z.get("source") == "crowd"]
    assert len(z) == 1 and z[0]["hazard"] and z[0]["expires"]
    # the server's geofence now treats the place as hazardous for everyone
    tok_c = register(client, name="Chen Li", nationality="CN", doc_type="Passport", doc_number="E7654321")["token"]
    inc = client.post("/api/events", json={"events": [event("ev-z", *ON_ROUTE)]}, headers=bearer(tok_c))
    assert inc.status_code == 200
    row = client.get("/api/incidents", headers=police(client)).json()[0]
    assert row["features"]["hazard_flag"] == 1


def test_police_verify_and_reject(client):
    tok = register(client)["token"]
    rid = report(client, tok, *ON_ROUTE).json()["report_id"]
    h = police(client)
    v = client.post(f"/api/reports/{rid}/verify", headers=h).json()
    assert v["status"] == "verified" and v["zone_id"]
    assert any(z.get("source") == "verified" for z in client.get("/api/zones").json())
    rid2 = report(client, tok, 12.9900, 77.6200, cat="scam").json()["report_id"]
    r = client.post(f"/api/reports/{rid2}/reject", headers=h).json()
    assert r["status"] == "rejected"
    assert all(n["id"] != rid2 for n in client.get("/api/reports/nearby", headers=bearer(tok)).json())
    assert client.post(f"/api/reports/{rid}/explode", headers=h).status_code == 404


def test_rejecting_removes_crowd_zone(client):
    a = register(client)["token"]
    b = register(client, name="Ben Clark", nationality="GB", doc_type="Passport", doc_number="P1234567")["token"]
    report(client, a, *ON_ROUTE)
    rid = report(client, b, *ON_ROUTE).json()["report_id"]
    assert any(z.get("source") == "crowd" for z in client.get("/api/zones").json())
    client.post(f"/api/reports/{rid}/reject", headers=police(client))
    assert not any(z.get("source") == "crowd" for z in client.get("/api/zones").json())


def test_expired_zone_disappears(env):
    c, app = env()
    core = app.state.core
    core.db.add_zone("old", "circle", {"lat": 12.97, "lon": 77.6, "radius_m": 100}, True, "t",
                     expires="2000-01-01 00:00:00", source="crowd")
    assert all(z["name"] != "old" for z in c.get("/api/zones").json())
