"""Business logic of the real application: registration, tracking, event handling, SOS,
digital-ID verification, hazard zones and the blockchain audit trail."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from src.blockchain import Authority, Blockchain, Credential, DigitalIDRegistry, credential_qr_png, new_network
from src.geofence import CircleZone, PolygonZone, distance_to_route_m, haversine_m
from src.system import SEVERITY_ACTION
from src.trajectory import ZONES as DEFAULT_ZONES

from . import security
from .db import DB
from .llm import CLASSES, CloudVerifier
from .settings import Settings

# Tourist spots offered at registration (approximate public coordinates)
SPOTS = {
    "MG Road": (12.9755, 77.6066), "Cubbon Park": (12.9763, 77.5929), "Vidhana Soudha": (12.9796, 77.5906),
    "Museum Road": (12.9718, 77.6040), "Commercial Street": (12.9822, 77.6083), "Lalbagh": (12.9507, 77.5848),
    "Bangalore Palace": (12.9987, 77.5920), "ISKCON Temple": (13.0098, 77.5511),
}


# Tourist incident reports: category -> (label shown in the app, initial severity)
REPORT_CATEGORIES = {
    "harassment": ("Harassment / stalking", "HIGH"),
    "assault": ("Assault / violence", "CRITICAL"),
    "theft": ("Theft / snatching / pickpocketing", "MODERATE"),
    "scam": ("Scam / overcharging / fake guide", "MODERATE"),
    "medical": ("Medical emergency", "HIGH"),
    "accident": ("Road accident", "HIGH"),
    "crowd": ("Overcrowding / stampede risk", "HIGH"),
    "natural": ("Flood / fire / fallen tree / landslide", "HIGH"),
    "unsafe_place": ("Unsafe place (dark, deserted, broken)", "MODERATE"),
    "other": ("Something else", "MODERATE"),
}
CORROBORATE_RADIUS_M = 300      # reports this close together ...
CORROBORATE_HOURS = 2           # ... within this time window ...
MIN_REPORTERS = 2               # ... from this many different tourists turn the place into a caution zone
CAUTION_RADIUS_M = 250
CAUTION_HOURS = 6               # crowd-reported caution zones expire on their own
VERIFIED_HOURS = 12             # zones confirmed by the police last longer
MAX_PHOTO_BYTES = 4 * 1024 * 1024
PHOTO_TYPES = {b"\xff\xd8\xff": "jpg", b"\x89PNG": "png", b"RIFF": "webp"}


def _utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def decode_photo(data: str) -> tuple[bytes, str]:
    """Photo sent by the phone as a data URL or plain base64; returns (bytes, extension)."""
    if "," in data[:100] and data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception:  # noqa: BLE001
        raise ValueError("photo is not valid base64")
    if len(raw) > MAX_PHOTO_BYTES:
        raise ValueError("photo is larger than 4 MB")
    for magic, ext in PHOTO_TYPES.items():
        if raw.startswith(magic) and (ext != "webp" or raw[8:12] == b"WEBP"):
            return raw, ext
    raise ValueError("photo must be a JPEG, PNG or WebP image")


class Model:
    """Python copy of the on-device model (same JSON the phone uses)."""

    def __init__(self, path):
        self.js = json.loads(open(path).read())
        self.mean, self.scale = np.array(self.js["scaler_mean"]), np.array(self.js["scaler_scale"])
        self.coef, self.b = np.array(self.js["coef"]), np.array(self.js["intercept"])

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.scale) @ self.coef.T + self.b
        e = np.exp(z - z.max())
        return e / e.sum()


def features(displacement_m: float, idle_min: float, hour: float, hazard: int) -> np.ndarray:
    return np.array([math.log(max(displacement_m, 20.0)), math.log(max(idle_min, 1e-3)),
                     math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24), float(hazard)])


def zone_objects(rows) -> list:
    out = []
    for z in rows:
        g = z["geometry"]
        if z["kind"] == "circle":
            out.append(CircleZone(z["name"], g["lat"], g["lon"], g["radius_m"], bool(z["hazard"])))
        else:
            out.append(PolygonZone(z["name"], [tuple(v) for v in g["vertices"]], bool(z["hazard"])))
    return out


class App:
    def __init__(self, settings: Settings, verifier: Optional[CloudVerifier] = None):
        self.s = settings
        self.db = DB(settings.db_path)
        self.model = Model(settings.model_json)
        self.verifier = verifier or CloudVerifier(settings.llm_provider, settings.llm_model,
                                                  settings.api_keys.get(settings.llm_provider, ""),
                                                  settings.llm_timeout)
        self.chain_lock = threading.Lock()
        self._load_chain()
        if not self.db.user(settings.admin_user):
            self.db.add_user(settings.admin_user, security.hash_password(settings.admin_password), "admin")
        if not self.db.zones():
            for z in DEFAULT_ZONES:
                if isinstance(z, CircleZone):
                    self.db.add_zone(z.name, "circle", {"lat": z.lat, "lon": z.lon, "radius_m": z.radius_m}, z.hazard, "seed")
                else:
                    self.db.add_zone(z.name, "polygon", {"vertices": z.vertices}, z.hazard, "seed")

    # ---------------------------------------------------------------- blockchain persistence
    @property
    def _chain_files(self):
        d = self.s.data_dir
        return d / "chain.json", d / "authority_keys.json", d / "kyc_vault.json"

    def _load_chain(self):
        c, k, v = self._chain_files
        if c.exists() and k.exists():
            self.chain = Blockchain.load(c, k)
            vault = json.loads(v.read_text()) if v.exists() else {}
            self.registry = DigitalIDRegistry(self.chain, vault)
        else:
            self.chain, self.registry = new_network()
            self._save_chain()

    def _save_chain(self):
        c, k, v = self._chain_files
        self.chain.save(c, k)
        v.write_text(json.dumps(self.registry.vault, indent=1, ensure_ascii=False))

    # ---------------------------------------------------------------- auth
    def login(self, username, password) -> Optional[Dict]:
        u = self.db.user(username)
        if not u or not security.check_password(password, u["password_hash"]):
            return None
        return {"token": security.make_token(self.s.secret_key, username, u["role"], 12 * 3600),
                "role": u["role"], "username": username}

    def add_officer(self, username, password, role="police"):
        self.db.add_user(username, security.hash_password(password), role)

    def auth(self, token: str, roles=("police", "admin")) -> Optional[Dict]:
        data = security.read_token(self.s.secret_key, token or "")
        if not data or data["role"] not in roles:
            return None
        if data["role"] == "tourist":
            t = self.db.tourist(data["sub"])
            if not t or t["token_hash"] != security.token_hash(token):
                return None
        return data

    # ---------------------------------------------------------------- zones
    def zones(self):
        return self.db.zones()

    def hazard_and_zones(self, lat, lon):
        objs = zone_objects(self.db.zones())
        inside = [z for z in objs if z.contains((lat, lon))]
        return int(any(z.hazard for z in inside)), [z.name for z in inside]

    # ---------------------------------------------------------------- registration
    def register(self, data: Dict) -> Dict:
        vf, vt = date.fromisoformat(data["valid_from"]), date.fromisoformat(data["valid_to"])
        route = []
        if data.get("start"):
            route.append([float(data["start"]["lat"]), float(data["start"]["lon"])])
        for spot in data.get("itinerary", []):
            if spot in SPOTS:
                route.append(list(SPOTS[spot]))
        if not route:
            raise ValueError("choose at least one itinerary spot or share your current location")
        with self.chain_lock:
            cred = self.registry.register_tourist(
                data["name"], data["nationality"], data["doc_type"], data["doc_number"], data["phone"],
                data["emergency_contact"], data.get("itinerary", []), vf, vt, "Tourism Department")
            self._save_chain()
        ttl = int((datetime.combine(vt, datetime.min.time()) + timedelta(days=2) - datetime.now()).total_seconds())
        token = security.make_token(self.s.secret_key, cred.tourist_id, "tourist", max(ttl, 3600))
        self.db.add_tourist(tourist_id=cred.tourist_id, name=data["name"], nationality=data["nationality"],
                            phone=data["phone"], emergency_contact=data["emergency_contact"],
                            credential=json.loads(cred.to_json()), route=route,
                            itinerary=data.get("itinerary", []), token_hash=security.token_hash(token),
                            valid_from=vf.isoformat(), valid_to=vt.isoformat())
        return {"tourist_id": cred.tourist_id, "token": token, "route": route, **self.profile(cred.tourist_id)}

    def profile(self, tourist_id) -> Dict:
        t = self.db.tourist(tourist_id)
        cred = Credential(**t["credential"])
        qr = base64.b64encode(credential_qr_png(cred)).decode()
        found = self.registry.lookup(tourist_id)
        return {"credential": t["credential"], "qr_png": qr, "route": t["route"], "name": t["name"],
                "itinerary": t["itinerary"], "block": found[0] if found else None,
                "revoked": self.registry.is_revoked(tourist_id)}

    # ---------------------------------------------------------------- tracking
    def track(self, tourist_id, fixes: List[Dict]):
        fixes = [f for f in fixes if -90 <= f["lat"] <= 90 and -180 <= f["lon"] <= 180]
        if fixes:
            self.db.add_positions(tourist_id, fixes)
        return fixes[-1] if fixes else None

    # ---------------------------------------------------------------- context for the cloud
    def context(self, tourist_id, lat, lon) -> Dict:
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        recent = self.db.all("SELECT lat, lon, tourist_id FROM incidents WHERE created >= ? AND final_class != 'LOW'",
                             (since,))
        nearby = sum(1 for r in recent if r["lat"] is not None and r["tourist_id"] != tourist_id
                     and haversine_m(lat, lon, r["lat"], r["lon"]) <= 500)
        t = self.db.tourist(tourist_id) or {}
        return {"nearby_incidents": nearby, "nationality": t.get("nationality"),
                "tourist_incidents": len(self.db.incidents(tourist_id=tourist_id))}

    # ---------------------------------------------------------------- events (Stage 2)
    def handle_event(self, tourist_id: str, ev: Dict, synced_late: bool = False) -> Dict:
        """Authoritative server-side processing of an event raised by the phone's geofence gate."""
        cid = ev.get("client_event_id")
        if cid and (old := self.db.incident_by_client_id(cid)):
            return {"incident_id": old["id"], "final_class": old["final_class"], "duplicate": True}
        t0 = time.perf_counter()
        tour = self.db.tourist(tourist_id)
        lat, lon = float(ev["lat"]), float(ev["lon"])
        disp = distance_to_route_m((lat, lon), [tuple(p) for p in tour["route"]])
        hazard, zones = self.hazard_and_zones(lat, lon)
        idle, hour = float(ev["idle_min"]), float(ev["hour"])
        p = self.model.proba(features(disp, idle, hour, hazard))
        local, conf = int(p.argmax()), float(p.max())
        reasons = []
        if conf < self.s.theta:
            reasons.append(f"confidence {conf:.2f} < θc {self.s.theta:.3f}")
        if self.s.policy == "safety" and local >= 2:
            reasons.append(f"on-device {CLASSES[local]} needs verification")
        escalate = bool(reasons)
        event = {"t": ev["t"], "hour": hour, "idle_min": idle, "displacement_m": disp, "zones": zones,
                 "hazard_flag": hazard, "local_class": CLASSES[local], "local_confidence": conf}
        timings = {"server_model_ms": (time.perf_counter() - t0) * 1000}
        final, decided_by, cloud_reason = local, "on-device", None
        if escalate:
            cd = self.verifier.verify(event, self.context(tourist_id, lat, lon))
            final, decided_by, cloud_reason = CLASSES.index(cd.risk_level), cd.provider, cd.reason
            timings["cloud_s"] = cd.latency_s
        inc = {"client_event_id": cid, "tourist_id": tourist_id, "source": "geofence", "t": ev["t"],
               "lat": lat, "lon": lon, "zones": zones,
               "features": {"displacement_m": disp, "idle_min": idle, "hour": hour, "hazard_flag": hazard,
                            "probabilities": [round(float(v), 4) for v in p],
                            "device_local_class": ev.get("local_class"), "device_offline": bool(ev.get("offline"))},
               "local_class": CLASSES[local], "local_confidence": conf, "escalated": int(escalate),
               "escalation_reason": "; ".join(reasons) or "confident, low severity",
               "final_class": CLASSES[final], "decided_by": decided_by, "cloud_reason": cloud_reason,
               "action": SEVERITY_ACTION[CLASSES[final]], "synced_late": int(synced_late), "timings": timings}
        return self._store(inc)

    def sos(self, tourist_id: str, data: Dict) -> Dict:
        lat, lon = float(data["lat"]), float(data["lon"])
        hazard, zones = self.hazard_and_zones(lat, lon)
        inc = {"client_event_id": data.get("client_event_id"), "tourist_id": tourist_id, "source": "sos",
               "t": data["t"], "lat": lat, "lon": lon, "zones": zones,
               "features": {"message": data.get("message", ""), "hazard_flag": hazard},
               "local_class": "CRITICAL", "local_confidence": 1.0, "escalated": 0,
               "escalation_reason": "SOS pressed by tourist", "final_class": "CRITICAL",
               "decided_by": "tourist SOS", "cloud_reason": None, "action": SEVERITY_ACTION["CRITICAL"],
               "synced_late": int(bool(data.get("offline"))), "timings": {}}
        cid = inc["client_event_id"]
        if cid and (old := self.db.incident_by_client_id(cid)):
            return {"incident_id": old["id"], "final_class": old["final_class"], "duplicate": True}
        return self._store(inc)

    def _store(self, inc: Dict) -> Dict:
        iid = self.db.add_incident(inc)
        with self.chain_lock:
            tx = self.registry.record_incident(inc["tourist_id"], {**inc, "id": iid}, "City Police")
            self.chain.seal_block()
            self._save_chain()
        self.db.set_chain_tx(iid, tx.tx_hash)
        row = self.db.incident(iid)
        return {"incident_id": iid, "final_class": row["final_class"], "decided_by": row["decided_by"],
                "action": row["action"], "escalated": bool(row["escalated"]), "reason": row["escalation_reason"],
                "cloud_reason": row["cloud_reason"], "chain_tx": tx.tx_hash, "incident": row}

    # ---------------------------------------------------------------- tourist incident reports (novel feature)
    @property
    def photo_dir(self):
        d = self.s.data_dir / "reports"
        d.mkdir(exist_ok=True)
        return d

    def report(self, tourist_id: str, data: Dict) -> Dict:
        """A tourist reports a real incident (photo + description + GPS), typically at a place the app
        still shows as safe. The report becomes a police incident anchored on the blockchain (with the
        photo's SHA-256 as tamper-evident evidence), and independent reports from several tourists near
        each other turn the place into a temporary caution zone that every phone picks up."""
        cid = data.get("client_report_id")
        if cid and (old := self.db.report_by_client_id(cid)):
            return {"report": old, "duplicate": True}
        if data["category"] not in REPORT_CATEGORIES:
            raise ValueError("unknown report category")
        label, severity = REPORT_CATEGORIES[data["category"]]
        lat, lon = float(data["lat"]), float(data["lon"])
        photo_file = sha = None
        if data.get("photo"):
            raw, ext = decode_photo(data["photo"])
            sha = hashlib.sha256(raw).hexdigest()
            photo_file = f"{sha}.{ext}"
            (self.photo_dir / photo_file).write_bytes(raw)
        hazard, zones = self.hazard_and_zones(lat, lon)
        rid = self.db.add_report({"client_report_id": cid, "tourist_id": tourist_id, "t": data["t"], "lat": lat,
                                  "lon": lon, "category": data["category"],
                                  "description": (data.get("description") or "").strip()[:500],
                                  "photo_file": photo_file, "photo_sha256": sha, "app_said_safe": int(not hazard),
                                  "zones": zones})
        inc = {"client_event_id": f"report-{cid or rid}", "tourist_id": tourist_id, "source": "report",
               "t": data["t"], "lat": lat, "lon": lon, "zones": zones,
               "features": {"report_id": rid, "category": data["category"], "label": label,
                            "description": (data.get("description") or "").strip()[:500], "photo_sha256": sha,
                            "app_said_safe": not hazard, "hazard_flag": hazard},
               "local_class": severity, "local_confidence": 1.0, "escalated": 0,
               "escalation_reason": f"tourist report: {label}" + (" (app showed this place as safe)" if not hazard else ""),
               "final_class": severity, "decided_by": "tourist report", "cloud_reason": None,
               "action": SEVERITY_ACTION[severity], "synced_late": int(bool(data.get("offline"))), "timings": {}}
        stored = self._store(inc)
        self.db.update_report(rid, incident_id=stored["incident_id"])
        zone_created = self._corroborate(rid)
        return {"report": self.db.report(rid), "incident": stored["incident"], "zone_created": zone_created,
                "chain_tx": stored["chain_tx"]}

    def public_reports(self, hours: int = 24) -> List[Dict]:
        """Recent reports shown to every tourist as warnings: no names, phones or photos."""
        since = _utc(datetime.now(timezone.utc) - timedelta(hours=hours))
        return [{"id": r["id"], "t": r["t"], "lat": round(r["lat"], 4), "lon": round(r["lon"], 4),
                 "category": r["category"], "label": REPORT_CATEGORIES[r["category"]][0],
                 "severity": r["final_class"], "status": r["status"], "corroborations": r["corroborations"],
                 "description": r["description"]}
                for r in self.db.reports(since=since, limit=300) if r["status"] != "rejected"]

    def _cluster(self, rep: Dict) -> List[Dict]:
        since = _utc(datetime.now(timezone.utc) - timedelta(hours=CORROBORATE_HOURS))
        return [r for r in self.db.reports(since=since, limit=1000)
                if r["status"] != "rejected"
                and haversine_m(rep["lat"], rep["lon"], r["lat"], r["lon"]) <= CORROBORATE_RADIUS_M]

    def _corroborate(self, rid: int) -> bool:
        rep = self.db.report(rid)
        cluster = self._cluster(rep)
        reporters = {r["tourist_id"] for r in cluster}
        for r in cluster:
            self.db.update_report(r["id"], corroborations=len(reporters))
        if len(reporters) < MIN_REPORTERS:
            return False
        lat = sum(r["lat"] for r in cluster) / len(cluster)
        lon = sum(r["lon"] for r in cluster) / len(cluster)
        zid, created = self._caution_zone(lat, lon, rep["category"], f"crowd ({len(reporters)} tourists)",
                                          CAUTION_HOURS, "crowd")
        for r in cluster:
            self.db.update_report(r["id"], zone_id=zid,
                                  **({"status": "corroborated"} if r["status"] == "pending" else {}))
        return created

    def _caution_zone(self, lat, lon, category, by, hours, source):
        for z in self.db.zones():
            if z.get("source") in ("crowd", "verified") and z["kind"] == "circle" and \
                    haversine_m(lat, lon, z["geometry"]["lat"], z["geometry"]["lon"]) <= z["geometry"]["radius_m"]:
                if source == "verified":       # police confirmation upgrades and extends the zone
                    self.db.execute("UPDATE zones SET source='verified', expires=? WHERE id=?",
                                    (_utc(datetime.now(timezone.utc) + timedelta(hours=hours)), z["id"]))
                return z["id"], False
        name = f"Reported: {REPORT_CATEGORIES[category][0]}"
        zid = self.db.add_zone(name, "circle", {"lat": lat, "lon": lon, "radius_m": CAUTION_RADIUS_M}, True, by,
                               expires=_utc(datetime.now(timezone.utc) + timedelta(hours=hours)), source=source)
        return zid, True

    def review_report(self, rid: int, action: str, officer: str) -> Dict:
        rep = self.db.report(rid)
        if action == "verify":
            zid, _ = self._caution_zone(rep["lat"], rep["lon"], rep["category"], f"police ({officer})",
                                        VERIFIED_HOURS, "verified")
            self.db.update_report(rid, status="verified", handled_by=officer, zone_id=zid)
            if rep["incident_id"]:
                self.db.update_incident(rep["incident_id"], status="acknowledged", handled_by=officer,
                                        notes="report verified by police")
        else:   # reject: false or duplicate report
            self.db.update_report(rid, status="rejected", handled_by=officer)
            if rep["incident_id"]:
                self.db.update_incident(rep["incident_id"], status="resolved", handled_by=officer,
                                        notes="report rejected by police")
            if rep["zone_id"]:
                others = [r for r in self.db.reports(limit=1000)
                          if r["zone_id"] == rep["zone_id"] and r["id"] != rid and r["status"] != "rejected"]
                if len({r["tourist_id"] for r in others}) < MIN_REPORTERS and \
                        not any(r["status"] == "verified" for r in others):
                    self.db.delete_zone(rep["zone_id"])
        return self.db.report(rid)

    # ---------------------------------------------------------------- police actions
    def set_status(self, iid: int, status: str, officer: str, notes: str = ""):
        self.db.update_incident(iid, status=status, handled_by=officer, notes=notes)
        return self.db.incident(iid)

    def verify(self, credential_json: str) -> Dict:
        cred = Credential.from_json(credential_json)
        r = self.registry.verify(cred)
        t = self.db.tourist(cred.tourist_id)
        return {"valid": r.valid, "reason": r.reason, "tourist_id": r.tourist_id, "block": r.block,
                "name": cred.name, "nationality": cred.nationality, "valid_from": cred.valid_from,
                "valid_to": cred.valid_to, "phone": t["phone"] if t else None,
                "emergency_contact": t["emergency_contact"] if t else None,
                "incidents": self.registry.incidents(cred.tourist_id)}

    def revoke(self, tourist_id: str, reason: str):
        with self.chain_lock:
            self.registry.revoke(tourist_id, reason, "City Police")
            self._save_chain()

    def chain_summary(self, limit=50) -> Dict:
        with self.chain_lock:
            problems = self.chain.verify_chain()
            blocks = [{"index": b.index, "hash": b.hash, "prev_hash": b.prev_hash, "validator": b.validator,
                       "timestamp": b.timestamp, "merkle_root": b.merkle_root,
                       "transactions": [{"kind": t.kind, "tx_hash": t.tx_hash, "issuer": t.issuer,
                                         "payload": t.payload} for t in b.transactions]}
                      for b in self.chain.chain[::-1][:limit]]
            return {"intact": not problems, "problems": problems, "height": len(self.chain.chain),
                    "transactions": sum(len(b.transactions) for b in self.chain.chain),
                    "authorities": list(self.chain.public_keys), "blocks": blocks}
