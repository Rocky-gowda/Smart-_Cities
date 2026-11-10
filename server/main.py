"""FastAPI application: REST API, WebSocket live feed and the two web apps.

  /           landing page
  /tourist/   tourist mobile web app (installable PWA)
  /police/    police / authority dashboard
  /api/...    REST API          /ws   live feed for the dashboard
  /docs       interactive API documentation (Swagger UI)
"""
from __future__ import annotations

import asyncio
import json
from typing import List, Optional, Set

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .llm import CLASSES
from .services import REPORT_CATEGORIES, SPOTS, App
from .settings import ROOT, Settings

WEB = ROOT / "web"


# ------------------------------------------------------------------ request bodies
class LoginIn(BaseModel):
    username: str
    password: str


class Point(BaseModel):
    lat: float
    lon: float


class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    nationality: str = Field(min_length=2, max_length=3)
    doc_type: str
    doc_number: str = Field(min_length=4, max_length=20)
    phone: str = Field(min_length=6, max_length=20)
    emergency_contact: str = Field(min_length=6, max_length=20)
    itinerary: List[str] = []
    start: Optional[Point] = None
    valid_from: str
    valid_to: str


class Fix(BaseModel):
    t: str
    lat: float
    lon: float
    accuracy: Optional[float] = None


class TrackIn(BaseModel):
    fixes: List[Fix]


class EventIn(BaseModel):
    client_event_id: str
    t: str
    lat: float
    lon: float
    idle_min: float
    hour: float
    displacement_m: Optional[float] = None
    hazard_flag: Optional[int] = None
    zones: List[str] = []
    local_class: Optional[str] = None
    local_confidence: Optional[float] = None
    offline: bool = False


class EventsIn(BaseModel):
    events: List[EventIn]


class SosIn(BaseModel):
    client_event_id: Optional[str] = None
    t: str
    lat: float
    lon: float
    message: str = ""
    offline: bool = False


class ReportIn(BaseModel):
    client_report_id: Optional[str] = None
    t: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    category: str
    description: str = Field(default="", max_length=500)
    photo: Optional[str] = Field(default=None, max_length=6_000_000)   # data URL / base64 (≤ 4 MB image)
    offline: bool = False


class StatusIn(BaseModel):
    notes: str = ""


class ZoneIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    kind: str = "circle"
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_m: Optional[float] = Field(default=None, gt=5, le=20000)
    vertices: Optional[List[List[float]]] = None
    hazard: bool = True


class VerifyIn(BaseModel):
    credential: str


class RevokeIn(BaseModel):
    reason: str = "revoked by police"


class OfficerIn(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=6)
    role: str = "police"


# ------------------------------------------------------------------ live feed
class Hub:
    def __init__(self):
        self.clients: Set[WebSocket] = set()

    async def send(self, msg: dict):
        data = json.dumps(msg, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                self.clients.discard(ws)


def create_app(settings: Optional[Settings] = None, verifier=None) -> FastAPI:
    s = settings or Settings()
    core = App(s, verifier)
    hub = Hub()
    api = FastAPI(title="Smart Tourist Safety Monitoring & Incident Response System",
                  description="Geo-fencing · on-device AI with selective LLM escalation · blockchain digital ID",
                  version="1.0")
    api.state.core, api.state.hub = core, hub

    def bearer(authorization: str = Header(default="")) -> str:
        return authorization.removeprefix("Bearer ").strip()

    def police(token: str = Depends(bearer)):
        u = core.auth(token, ("police", "admin"))
        if not u:
            raise HTTPException(401, "police login required")
        return u

    def admin(token: str = Depends(bearer)):
        u = core.auth(token, ("admin",))
        if not u:
            raise HTTPException(403, "admin only")
        return u

    def tourist(token: str = Depends(bearer)):
        u = core.auth(token, ("tourist",))
        if not u:
            raise HTTPException(401, "tourist token required")
        return u

    # ------------------------------------------------------------ public
    @api.get("/api/health")
    def health():
        return {"status": "ok", "llm": core.verifier.name, "chain_height": len(core.chain.chain)}

    @api.get("/api/config")
    def config():
        return {"model": core.model.js, "theta": s.theta, "policy": s.policy, "idle_minutes": s.idle_minutes,
                "time_scale": s.time_scale, "classes": CLASSES, "spots": SPOTS, "zones": core.zones(),
                "report_categories": {k: v[0] for k, v in REPORT_CATEGORIES.items()},
                "llm": core.verifier.name}

    @api.get("/api/qr")
    def qr(text: str):
        import io
        import qrcode
        from fastapi.responses import Response
        buf = io.BytesIO()
        qrcode.make(text[:500]).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @api.get("/api/zones")
    def zones():
        return core.zones()

    @api.post("/api/auth/login")
    def login(body: LoginIn):
        r = core.login(body.username, body.password)
        if not r:
            raise HTTPException(401, "wrong username or password")
        return r

    # ------------------------------------------------------------ tourist
    @api.post("/api/tourists/register")
    async def register(body: RegisterIn):
        try:
            r = core.register(body.model_dump())
        except ValueError as e:
            raise HTTPException(400, str(e))
        await hub.send({"type": "tourist", "tourist_id": r["tourist_id"], "name": body.name})
        return r

    @api.get("/api/tourists/me")
    def me(u=Depends(tourist)):
        return {"tourist_id": u["sub"], **core.profile(u["sub"]),
                "incidents": core.db.incidents(tourist_id=u["sub"], limit=20),
                "reports": core.db.reports(tourist_id=u["sub"], limit=20)}

    @api.post("/api/track")
    async def track(body: TrackIn, u=Depends(tourist)):
        last = core.track(u["sub"], [f.model_dump() for f in body.fixes])
        if last:
            await hub.send({"type": "position", "tourist_id": u["sub"], **last})
        return {"stored": len(body.fixes)}

    @api.post("/api/events")
    async def events(body: EventsIn, u=Depends(tourist)):
        out = []
        for ev in body.events:
            r = await asyncio.to_thread(core.handle_event, u["sub"], ev.model_dump(), ev.offline)
            if not r.get("duplicate"):
                await hub.send({"type": "incident", "incident": r["incident"]})
            out.append({k: v for k, v in r.items() if k != "incident"})
        return {"decisions": out}

    @api.post("/api/sos")
    async def sos(body: SosIn, u=Depends(tourist)):
        r = await asyncio.to_thread(core.sos, u["sub"], body.model_dump())
        if not r.get("duplicate"):
            await hub.send({"type": "incident", "incident": r["incident"]})
        return {k: v for k, v in r.items() if k != "incident"}

    @api.post("/api/reports")
    async def report(body: ReportIn, u=Depends(tourist)):
        try:
            r = await asyncio.to_thread(core.report, u["sub"], body.model_dump())
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not r.get("duplicate"):
            await hub.send({"type": "report", "report": r["report"], "incident": r["incident"]})
            if r["zone_created"]:
                await hub.send({"type": "zones", "zones": core.zones()})
        rep = r["report"]
        return {"report_id": rep["id"], "status": rep["status"], "severity": rep.get("final_class"),
                "app_said_safe": bool(rep["app_said_safe"]), "corroborations": rep["corroborations"],
                "caution_zone": rep["zone_id"] is not None, "chain_tx": rep.get("chain_tx"),
                "photo_sha256": rep["photo_sha256"], "duplicate": bool(r.get("duplicate"))}

    @api.get("/api/reports/nearby")
    def nearby_reports(u=Depends(tourist)):
        return core.public_reports()

    # ------------------------------------------------------------ police
    @api.get("/api/me")
    def whoami(u=Depends(police)):
        return u

    @api.get("/api/tourists")
    def tourists(u=Depends(police)):
        return core.db.tourists_overview()

    @api.get("/api/tourists/{tid}/track")
    def tourist_track(tid: str, u=Depends(police)):
        return core.db.track(tid)

    @api.post("/api/tourists/{tid}/revoke")
    def revoke(tid: str, body: RevokeIn, u=Depends(police)):
        try:
            core.revoke(tid, body.reason)
        except KeyError:
            raise HTTPException(404, "unknown tourist ID")
        return {"revoked": tid}

    @api.get("/api/incidents")
    def incidents(status: Optional[str] = None, u=Depends(police)):
        return core.db.incidents(status=status)

    @api.post("/api/incidents/{iid}/{action}")
    async def incident_action(iid: int, action: str, body: StatusIn, u=Depends(police)):
        status = {"ack": "acknowledged", "resolve": "resolved", "reopen": "open"}.get(action)
        if not status or not core.db.incident(iid):
            raise HTTPException(404, "unknown incident or action")
        row = core.set_status(iid, status, u["sub"], body.notes)
        await hub.send({"type": "incident_update", "incident": row})
        return row

    @api.get("/api/reports")
    def reports(status: Optional[str] = None, u=Depends(police)):
        return core.db.reports(status=status)

    @api.get("/api/reports/{rid}/photo")
    def report_photo(rid: int, u=Depends(police)):
        rep = core.db.report(rid)
        if not rep or not rep["photo_file"]:
            raise HTTPException(404, "no photo for this report")
        return FileResponse(core.photo_dir / rep["photo_file"])

    @api.post("/api/reports/{rid}/{action}")
    async def review_report(rid: int, action: str, u=Depends(police)):
        if action not in ("verify", "reject") or not core.db.report(rid):
            raise HTTPException(404, "unknown report or action")
        rep = core.review_report(rid, action, u["sub"])
        await hub.send({"type": "report_update", "report": rep})
        if rep["incident_id"]:
            await hub.send({"type": "incident_update", "incident": core.db.incident(rep["incident_id"])})
        await hub.send({"type": "zones", "zones": core.zones()})
        return rep

    @api.post("/api/zones")
    async def add_zone(body: ZoneIn, u=Depends(police)):
        if body.kind == "circle":
            if body.lat is None or body.lon is None or body.radius_m is None:
                raise HTTPException(400, "circle needs lat, lon and radius_m")
            geom = {"lat": body.lat, "lon": body.lon, "radius_m": body.radius_m}
        elif body.kind == "polygon" and body.vertices and len(body.vertices) >= 3:
            geom = {"vertices": body.vertices}
        else:
            raise HTTPException(400, "polygon needs at least 3 vertices")
        zid = core.db.add_zone(body.name, body.kind, geom, body.hazard, u["sub"])
        await hub.send({"type": "zones", "zones": core.zones()})
        return {"id": zid}

    @api.delete("/api/zones/{zid}")
    async def delete_zone(zid: int, u=Depends(police)):
        core.db.delete_zone(zid)
        await hub.send({"type": "zones", "zones": core.zones()})
        return {"deleted": zid}

    @api.post("/api/verify")
    def verify(body: VerifyIn, u=Depends(police)):
        try:
            return core.verify(body.credential)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"unreadable credential: {type(e).__name__}")

    @api.get("/api/chain")
    def chain(u=Depends(police)):
        return core.chain_summary()

    @api.post("/api/officers")
    def add_officer(body: OfficerIn, u=Depends(admin)):
        core.add_officer(body.username, body.password, body.role if body.role in ("police", "admin") else "police")
        return {"created": body.username}

    # ------------------------------------------------------------ live feed
    @api.websocket("/ws")
    async def ws(socket: WebSocket, token: str = ""):
        if not core.auth(token, ("police", "admin")):
            await socket.close(code=4401)
            return
        await socket.accept()
        hub.clients.add(socket)
        try:
            while True:
                await socket.receive_text()   # keep-alive pings from the dashboard
        except WebSocketDisconnect:
            hub.clients.discard(socket)

    # ------------------------------------------------------------ web apps
    @api.get("/")
    def home():
        return FileResponse(WEB / "index.html")

    @api.get("/tourist")
    def tourist_redirect():
        return RedirectResponse("/tourist/")

    @api.get("/police")
    def police_redirect():
        return RedirectResponse("/police/")

    api.mount("/tourist", StaticFiles(directory=WEB / "tourist", html=True), name="tourist")
    api.mount("/police", StaticFiles(directory=WEB / "police", html=True), name="police")
    api.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
    return api


app = None


def get_app():
    global app
    if app is None:
        app = create_app()
    return app
