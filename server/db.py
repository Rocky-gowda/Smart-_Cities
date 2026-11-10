"""SQLite storage for users, tourists, GPS positions, incidents and geofence zones."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS tourists (
    tourist_id TEXT PRIMARY KEY, name TEXT NOT NULL, nationality TEXT, phone TEXT, emergency_contact TEXT,
    credential TEXT NOT NULL, route TEXT NOT NULL, itinerary TEXT NOT NULL, token_hash TEXT NOT NULL,
    valid_from TEXT, valid_to TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tourist_id TEXT NOT NULL, t TEXT NOT NULL,
    lat REAL NOT NULL, lon REAL NOT NULL, accuracy REAL, received TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS ix_pos_tourist ON positions(tourist_id, id);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT, client_event_id TEXT UNIQUE, tourist_id TEXT NOT NULL,
    source TEXT NOT NULL, t TEXT NOT NULL, lat REAL, lon REAL, zones TEXT, features TEXT,
    local_class TEXT, local_confidence REAL, escalated INTEGER, escalation_reason TEXT,
    final_class TEXT NOT NULL, decided_by TEXT, cloud_reason TEXT, action TEXT,
    status TEXT DEFAULT 'open', handled_by TEXT, notes TEXT, synced_late INTEGER DEFAULT 0,
    chain_tx TEXT, timings TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP, updated TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS ix_inc_status ON incidents(status, id);
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, kind TEXT NOT NULL, geometry TEXT NOT NULL,
    hazard INTEGER NOT NULL DEFAULT 1, created_by TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, client_report_id TEXT UNIQUE, tourist_id TEXT NOT NULL, t TEXT NOT NULL,
    lat REAL NOT NULL, lon REAL NOT NULL, category TEXT NOT NULL, description TEXT, photo_file TEXT,
    photo_sha256 TEXT, app_said_safe INTEGER, zones TEXT, status TEXT DEFAULT 'pending', corroborations INTEGER DEFAULT 1,
    incident_id INTEGER, zone_id INTEGER, handled_by TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS ix_rep_time ON reports(created);
"""

# columns added after the first release (older databases are upgraded on start)
MIGRATIONS = ["ALTER TABLE zones ADD COLUMN expires TEXT", "ALTER TABLE zones ADD COLUMN source TEXT"]

JSON_COLS = {"credential", "route", "itinerary", "zones", "features", "geometry", "timings"}


class DB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            for m in MIGRATIONS:
                try:
                    self.conn.execute(m)
                except sqlite3.OperationalError:
                    pass                                 # column already there
            self.conn.commit()

    # ---------------------------------------------------------------- helpers
    def execute(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    def all(self, sql: str, args=()) -> List[Dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [self._row(r) for r in rows]

    def one(self, sql: str, args=()) -> Optional[Dict[str, Any]]:
        rows = self.all(sql, args)
        return rows[0] if rows else None

    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        for k in JSON_COLS & d.keys():
            if isinstance(d[k], str):
                try:
                    d[k] = json.loads(d[k])
                except json.JSONDecodeError:
                    pass
        return d

    # ---------------------------------------------------------------- users
    def add_user(self, username, password_hash, role):
        self.execute("INSERT OR REPLACE INTO users(username, password_hash, role) VALUES (?,?,?)",
                     (username, password_hash, role))

    def user(self, username):
        return self.one("SELECT * FROM users WHERE username=?", (username,))

    # ---------------------------------------------------------------- tourists
    def add_tourist(self, **t):
        self.execute("""INSERT INTO tourists(tourist_id, name, nationality, phone, emergency_contact, credential,
                        route, itinerary, token_hash, valid_from, valid_to) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                     (t["tourist_id"], t["name"], t["nationality"], t["phone"], t["emergency_contact"],
                      json.dumps(t["credential"]), json.dumps(t["route"]), json.dumps(t["itinerary"]),
                      t["token_hash"], t["valid_from"], t["valid_to"]))

    def tourist(self, tourist_id):
        return self.one("SELECT * FROM tourists WHERE tourist_id=?", (tourist_id,))

    def tourists_overview(self):
        return self.all("""
            SELECT t.tourist_id, t.name, t.nationality, t.phone, t.emergency_contact, t.valid_from, t.valid_to,
                   p.lat, p.lon, p.t AS last_seen,
                   (SELECT final_class FROM incidents i WHERE i.tourist_id=t.tourist_id ORDER BY i.id DESC LIMIT 1) AS last_risk,
                   (SELECT COUNT(*) FROM incidents i WHERE i.tourist_id=t.tourist_id AND i.status!='resolved') AS open_incidents
            FROM tourists t LEFT JOIN positions p ON p.id = (SELECT MAX(id) FROM positions WHERE tourist_id=t.tourist_id)
            ORDER BY t.created DESC""")

    # ---------------------------------------------------------------- positions
    def add_positions(self, tourist_id, fixes):
        with self.lock:
            self.conn.executemany("INSERT INTO positions(tourist_id, t, lat, lon, accuracy) VALUES (?,?,?,?,?)",
                                  [(tourist_id, f["t"], f["lat"], f["lon"], f.get("accuracy")) for f in fixes])
            self.conn.commit()

    def track(self, tourist_id, limit=500):
        return self.all("SELECT t, lat, lon, accuracy FROM positions WHERE tourist_id=? ORDER BY id DESC LIMIT ?",
                        (tourist_id, limit))[::-1]

    # ---------------------------------------------------------------- incidents
    def add_incident(self, inc: Dict) -> int:
        cols = ["client_event_id", "tourist_id", "source", "t", "lat", "lon", "zones", "features", "local_class",
                "local_confidence", "escalated", "escalation_reason", "final_class", "decided_by", "cloud_reason",
                "action", "synced_late", "chain_tx", "timings"]
        vals = [json.dumps(inc.get(c)) if c in JSON_COLS else inc.get(c) for c in cols]
        return self.execute(f"INSERT INTO incidents({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)

    def incident(self, iid):
        return self.one("SELECT i.*, t.name, t.phone, t.emergency_contact FROM incidents i "
                        "LEFT JOIN tourists t USING(tourist_id) WHERE i.id=?", (iid,))

    def incident_by_client_id(self, cid):
        return self.one("SELECT * FROM incidents WHERE client_event_id=?", (cid,))

    def incidents(self, status=None, tourist_id=None, limit=200):
        q, args = "SELECT i.*, t.name, t.phone, t.emergency_contact FROM incidents i LEFT JOIN tourists t USING(tourist_id) WHERE 1=1", []
        if status:
            q += " AND i.status=?"
            args.append(status)
        if tourist_id:
            q += " AND i.tourist_id=?"
            args.append(tourist_id)
        return self.all(q + " ORDER BY i.id DESC LIMIT ?", (*args, limit))

    def update_incident(self, iid, **kw):
        sets = ", ".join(f"{k}=?" for k in kw) + ", updated=CURRENT_TIMESTAMP"
        self.execute(f"UPDATE incidents SET {sets} WHERE id=?", (*kw.values(), iid))

    def set_chain_tx(self, iid, tx):
        self.execute("UPDATE incidents SET chain_tx=? WHERE id=?", (tx, iid))

    # ---------------------------------------------------------------- zones
    def add_zone(self, name, kind, geometry, hazard, created_by, expires=None, source="police"):
        return self.execute("INSERT INTO zones(name, kind, geometry, hazard, created_by, expires, source) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (name, kind, json.dumps(geometry), int(hazard), created_by, expires, source))

    def zones(self):
        """Active zones only: temporary crowd-reported zones disappear when they expire."""
        return self.all("SELECT * FROM zones WHERE expires IS NULL OR expires > strftime('%Y-%m-%d %H:%M:%S','now') "
                        "ORDER BY id")

    def delete_zone(self, zid):
        self.execute("DELETE FROM zones WHERE id=?", (zid,))

    # ---------------------------------------------------------------- tourist incident reports
    def add_report(self, r: Dict) -> int:
        cols = ["client_report_id", "tourist_id", "t", "lat", "lon", "category", "description", "photo_file",
                "photo_sha256", "app_said_safe", "zones"]
        vals = [json.dumps(r.get(c)) if c in JSON_COLS else r.get(c) for c in cols]
        return self.execute(f"INSERT INTO reports({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)

    _REP = ("SELECT r.*, t.name, t.phone, i.final_class, i.chain_tx FROM reports r LEFT JOIN tourists t USING(tourist_id) "
            "LEFT JOIN incidents i ON i.id = r.incident_id")

    def report(self, rid):
        return self.one(self._REP + " WHERE r.id=?", (rid,))

    def report_by_client_id(self, cid):
        return self.one(self._REP + " WHERE r.client_report_id=?", (cid,))

    def reports(self, status=None, tourist_id=None, since=None, limit=200):
        q, args = self._REP + " WHERE 1=1", []
        for col, v in (("r.status", status), ("r.tourist_id", tourist_id)):
            if v:
                q += f" AND {col}=?"
                args.append(v)
        if since:
            q += " AND r.created >= ?"
            args.append(since)
        return self.all(q + " ORDER BY r.id DESC LIMIT ?", (*args, limit))

    def update_report(self, rid, **kw):
        self.execute(f"UPDATE reports SET {', '.join(f'{k}=?' for k in kw)} WHERE id=?", (*kw.values(), rid))
