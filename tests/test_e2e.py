"""End-to-end tests of the real application in a real browser (Chromium via Playwright).

A server is started on a free port. One browser context plays the tourist's phone (with a
simulated GPS position that the test moves around), another plays the police dashboard.
Screenshots are saved to results/e2e/ for the report.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "results" / "e2e"
pytestmark = pytest.mark.e2e
pw = pytest.importorskip("playwright.sync_api")

LAKE = (12.9860, 77.5958)


@pytest.fixture(scope="module")
def server(tmp_path_factory, artifacts):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "DATA_DIR": str(tmp_path_factory.mktemp("srv")), "TIME_SCALE": "60"}
    proc = subprocess.Popen([sys.executable, "-c", f"import uvicorn; from server.main import create_app; "
                             f"uvicorn.run(create_app(), host='127.0.0.1', port={port}, log_level='warning')"],
                            cwd=ROOT, env=env)
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if httpx.get(base + "/api/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("server did not start")
    yield base
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium not available: {e}")
        yield b
        b.close()


def phone(browser, lat=12.9755, lon=77.6066):
    ctx = browser.new_context(viewport={"width": 412, "height": 900}, is_mobile=True, has_touch=True,
                              geolocation={"latitude": lat, "longitude": lon, "accuracy": 10}, permissions=["geolocation"])
    return ctx, ctx.new_page()


def register(page, base, name="Asha Verma", doc="123412341234"):
    page.goto(base + "/tourist/")
    page.wait_for_selector("#regForm")
    page.fill("[name=name]", name)
    page.fill("[name=doc_number]", doc)
    page.fill("[name=phone]", "+919845000000")
    page.fill("[name=emergency_contact]", "+919886000000")
    page.click("#regBtn")
    page.wait_for_selector("#nav:not(.hidden)", timeout=20000)


def police_page(browser, base):
    ctx = browser.new_context(viewport={"width": 1400, "height": 850})
    page = ctx.new_page()
    page.goto(base + "/police/")
    page.fill("[name=username]", "admin")
    page.fill("[name=password]", "admin123")
    page.click("#loginForm button")
    page.wait_for_selector("#app:not(.hidden)")
    page.wait_for_function("document.querySelector('#live').textContent.includes('live')", timeout=10000)
    return ctx, page


def shot(page, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / name))


def test_landing_page(server, browser):
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto(server + "/")
    assert "Smart Tourist Safety" in page.title()
    page.wait_for_function("document.querySelector('#health').textContent.includes('Server OK')")
    ctx.close()


def test_real_gps_tracking_alert_reaches_police(server, browser):
    """Phone registers, tracks real (simulated) GPS, stops in a hazard zone; police see the alert live."""
    pctx, police = police_page(browser, server)
    tctx, t = phone(browser)
    register(t, server)
    shot(t, "tourist_registered.png")
    t.click("#trackBtn")
    for lat, lon in [(12.978, 77.600), (12.982, 77.598), LAKE]:
        tctx.set_geolocation({"latitude": lat, "longitude": lon, "accuracy": 8})
        t.wait_for_timeout(1200)
    t.wait_for_selector("#decision:not(.hidden)", timeout=60000)          # idle 15 min (x60 time scale)
    assert "hazard zone" in t.text_content("#zoneNote")
    t.wait_for_function("document.querySelector('#decision').textContent.includes('Verified')", timeout=20000)
    final = t.text_content("#decision .badge")
    assert final in ("HIGH", "CRITICAL")
    police.wait_for_selector(".inc", timeout=15000)
    card = police.text_content(".inc")
    assert "Asha Verma" in card and final in card and "Lake periphery" in card
    shot(t, "tourist_alert.png")
    shot(police, "police_alert.png")
    police.click(".inc button[data-act=ack]")
    police.wait_for_selector(".inc .pill:has-text('acknowledged')")
    tctx.close()
    pctx.close()


def test_offline_alert_is_queued_and_synced(server, browser):
    pctx, police = police_page(browser, server)
    tctx, t = phone(browser)
    register(t, server, name="Ravi Kumar", doc="999988887777")
    tctx.set_offline(True)
    t.click("details summary")
    t.click("#simBtn")
    t.wait_for_function("document.querySelector('#decision').textContent.includes('No network')", timeout=30000)
    assert t.text_content("#queue") != "0"
    shot(t, "tourist_offline.png")
    n_before = police.locator(".inc").count()
    tctx.set_offline(False)
    t.evaluate("window.dispatchEvent(new Event('online'))")
    police.wait_for_function(f"document.querySelectorAll('.inc').length > {n_before}", timeout=30000)
    assert "sent after reconnecting" in police.text_content(".inc")
    t.wait_for_function("document.querySelector('#queue').textContent === '0'", timeout=15000)
    tctx.close()
    pctx.close()


def test_sos_button(server, browser):
    pctx, police = police_page(browser, server)
    tctx, t = phone(browser, *LAKE)
    register(t, server, name="Meera Nair", doc="111122223333")
    n_before = police.locator(".inc").count()
    t.click("#sos")
    t.click("#sos", force=True)              # second tap confirms (button is pulsing, so skip the stability wait)
    t.wait_for_function("document.querySelector('#decision').textContent.includes('received your SOS')", timeout=15000)
    police.wait_for_function(f"document.querySelectorAll('.inc').length > {n_before}", timeout=15000)
    assert "SOS button" in police.text_content(".inc") and "CRITICAL" in police.text_content(".inc")
    tctx.close()
    pctx.close()


def test_police_verify_digital_id(server, browser):
    tctx, t = phone(browser)
    register(t, server, name="Kenji Tanaka", doc="TK1234567")
    t.click("#nav button[data-view=idview]")
    t.wait_for_selector("#qr[src^='data:image/png']")
    shot(t, "tourist_id.png")
    cred = t.evaluate("JSON.stringify(JSON.parse(localStorage.getItem('tsm.profile')).credential)")
    pctx, police = police_page(browser, server)
    police.click(".tabs button[data-tab=verify]")
    police.fill("#credText", cred)
    police.click("#verifyBtn")
    police.wait_for_selector("#verifyOut h3")
    assert "VALID" in police.text_content("#verifyOut h3") and "Kenji Tanaka" in police.text_content("#verifyOut")
    shot(police, "police_verify.png")
    police.fill("#credText", json.dumps({**json.loads(cred), "valid_to": "2099-12-31"}))
    police.click("#verifyBtn")
    police.wait_for_function("document.querySelector('#verifyOut h3').textContent.includes('REJECTED')")
    tctx.close()
    pctx.close()


def test_police_adds_hazard_zone_on_map(server, browser):
    pctx, police = police_page(browser, server)
    police.click(".tabs button[data-tab=zones]")
    police.click("#map", position={"x": 300, "y": 300})
    police.fill("#zName", "E2E test zone")
    police.click("#zSave")
    police.wait_for_selector("#zList td:has-text('E2E test zone')")
    assert any(z["name"] == "E2E test zone" for z in httpx.get(server + "/api/zones").json())
    police.click(".tabs button[data-tab=chain]")
    police.wait_for_selector("#chainOut .card")
    assert "Chain intact" in police.text_content("#chainOut")
    shot(police, "police_chain.png")
    pctx.close()


def test_tourist_app_is_installable(server, browser):
    tctx, t = phone(browser)
    t.goto(server + "/tourist/")
    manifest = httpx.get(server + "/tourist/manifest.json").json()
    assert manifest["display"] == "standalone" and manifest["start_url"] == "/tourist/"
    t.wait_for_function("navigator.serviceWorker.getRegistration().then(r => !!r)", timeout=15000)
    tctx.close()


def test_tourist_incident_report_reaches_police_and_warns_others(server, browser, tmp_path):
    """A tourist reports an incident with a photo while travelling; the police dashboard receives it live
    with the photo; a second tourist walking nearby gets a warning; police verify it -> caution zone."""
    from PIL import Image
    photo = tmp_path / "incident.jpg"
    Image.new("RGB", (800, 600), (180, 60, 40)).save(photo)
    spot = (12.9757, 77.6000)                                   # on the route, outside every hazard zone
    pctx, police = police_page(browser, server)
    police.click(".tabs button[data-tab=reports]")
    ctx, t = phone(browser, *spot)
    register(t, server, name="Priya Nair", doc="555566667777")
    t.click("#trackBtn")
    t.wait_for_timeout(1500)
    t.click("#reportBtn")
    t.wait_for_selector("#reportCard:not(.hidden)")
    t.select_option("#rCat", "theft")
    t.fill("#reportForm textarea", "Phone snatched by two men on a bike near the bus stop")
    t.set_input_files("#rPhoto", str(photo))
    t.wait_for_selector("#rPreview:not(.hidden)")
    t.wait_for_function("document.querySelector('#rLoc').textContent.includes('safe')")
    shot(t, "tourist_report_form.png")
    t.click("#rSend")
    t.wait_for_function("document.querySelector('#toast').textContent.includes('Report sent')", timeout=15000)
    police.wait_for_selector(".rep img[src^='blob:']", timeout=15000)
    assert "Phone snatched" in police.inner_text("#repList")
    assert "safe" in police.inner_text("#repList")
    shot(police, "police_report_received.png")
    # a second tourist walking nearby is warned
    ctx2, t2 = phone(browser, spot[0] + 0.0005, spot[1])
    register(t2, server, name="Tom Baker", doc="P7654321")
    t2.click("#trackBtn")
    t2.wait_for_selector("#nearWarn:not(.hidden)", timeout=15000)
    assert "Theft" in t2.inner_text("#nearWarn")
    t2.wait_for_selector(".rep-pin")
    shot(t2, "tourist_nearby_warning.png")
    # police verify -> caution zone appears
    police.click(".rep button[data-ract=verify]")
    police.wait_for_selector(".rep .pill:has-text('verified')")
    zones = httpx.get(server + "/api/zones").json()
    assert any(z.get("source") == "verified" for z in zones)
    shot(police, "police_report_verified.png")
    ctx.close(); ctx2.close(); pctx.close()
