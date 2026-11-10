"""Blockchain-based digital tourist ID and tamper-evident incident log.

A small permissioned blockchain run by trusted authority nodes (tourism
department, police, immigration), sealed by proof-of-authority:

  * every transaction is signed by its issuer (Ed25519),
  * transactions are grouped into blocks with a SHA-256 Merkle root,
  * each block stores the hash of the previous block and is signed by the
    authority that sealed it, so changing any past record breaks the chain.

On top of the chain, `DigitalIDRegistry` works like a smart contract:
register a tourist's digital ID, verify it, revoke it, and anchor every
incident alert produced by the AI system.

Privacy: personal data (name, passport/Aadhaar number, phone) never goes on
the chain. Only salted SHA-256 hashes are stored; the readable KYC record stays
in the issuing authority's off-chain vault.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


# --------------------------------------------------------------------------- helpers
def canonical(obj) -> bytes:
    """Deterministic JSON encoding, so the same data always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def merkle_root(hashes: List[str]) -> str:
    """Merkle root of a list of hex hashes (last element duplicated on odd levels)."""
    if not hashes:
        return sha256(b"")
    level = list(hashes)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha256(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


# --------------------------------------------------------------------------- authorities
class Authority:
    """A permissioned node (e.g. Tourism Dept., City Police) with a signing key."""

    def __init__(self, name: str, private_key: Optional[Ed25519PrivateKey] = None):
        self.name = name
        self._sk = private_key or Ed25519PrivateKey.generate()
        self.public_key_hex = self._sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()

    def sign(self, data: bytes) -> str:
        return self._sk.sign(data).hex()

    def private_hex(self) -> str:
        return self._sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                      serialization.NoEncryption()).hex()

    @classmethod
    def from_private_hex(cls, name: str, hx: str) -> "Authority":
        return cls(name, Ed25519PrivateKey.from_private_bytes(bytes.fromhex(hx)))


def verify_sig(public_key_hex: str, signature_hex: str, data: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(bytes.fromhex(signature_hex), data)
        return True
    except (InvalidSignature, ValueError):
        return False


# --------------------------------------------------------------------------- chain structures
@dataclass
class Transaction:
    kind: str            # REGISTER_ID | REVOKE_ID | INCIDENT | VERIFY_LOG
    payload: Dict
    issuer: str
    timestamp: float
    signature: str = ""

    def body(self) -> Dict:
        return {"kind": self.kind, "payload": self.payload, "issuer": self.issuer, "timestamp": self.timestamp}

    @property
    def tx_hash(self) -> str:
        return sha256(canonical(self.body()))


@dataclass
class Block:
    index: int
    timestamp: float
    prev_hash: str
    merkle_root: str
    validator: str
    transactions: List[Transaction] = field(default_factory=list)
    signature: str = ""

    def header(self) -> Dict:
        return {"index": self.index, "timestamp": self.timestamp, "prev_hash": self.prev_hash,
                "merkle_root": self.merkle_root, "validator": self.validator}

    @property
    def hash(self) -> str:
        return sha256(canonical(self.header()))


class Blockchain:
    """Permissioned proof-of-authority chain."""

    def __init__(self, authorities: List[Authority]):
        self.authorities: Dict[str, Authority] = {a.name: a for a in authorities}
        self.public_keys: Dict[str, str] = {a.name: a.public_key_hex for a in authorities}
        self.chain: List[Block] = []
        self.pending: List[Transaction] = []
        self._seal_count = 0
        genesis = Block(0, 0.0, "0" * 64, merkle_root([]), "genesis")
        self.chain.append(genesis)

    # ---------------------------------------------------------------- transactions and blocks
    def submit(self, kind: str, payload: Dict, issuer: str) -> Transaction:
        if issuer not in self.authorities:
            raise PermissionError(f"{issuer} is not an authorised node")
        tx = Transaction(kind, payload, issuer, time.time())
        tx.signature = self.authorities[issuer].sign(canonical(tx.body()))
        self.pending.append(tx)
        return tx

    def seal_block(self, validator: Optional[str] = None) -> Optional[Block]:
        """Proof-of-authority: validators take turns (round robin) to seal the pending pool."""
        if not self.pending:
            return None
        names = sorted(self.authorities)
        validator = validator or names[self._seal_count % len(names)]
        self._seal_count += 1
        prev = self.chain[-1]
        blk = Block(prev.index + 1, time.time(), prev.hash, merkle_root([t.tx_hash for t in self.pending]),
                    validator, list(self.pending))
        blk.signature = self.authorities[validator].sign(canonical(blk.header()))
        self.chain.append(blk)
        self.pending = []
        return blk

    # ---------------------------------------------------------------- integrity
    def verify_chain(self) -> List[str]:
        """Return a list of problems; an empty list means the chain is intact."""
        problems = []
        for i, blk in enumerate(self.chain[1:], start=1):
            prev = self.chain[i - 1]
            if blk.prev_hash != prev.hash:
                problems.append(f"block {i}: previous-hash link broken")
            if blk.merkle_root != merkle_root([t.tx_hash for t in blk.transactions]):
                problems.append(f"block {i}: Merkle root mismatch (a transaction was altered)")
            pk = self.public_keys.get(blk.validator)
            if pk is None or not verify_sig(pk, blk.signature, canonical(blk.header())):
                problems.append(f"block {i}: invalid validator signature")
            for t in blk.transactions:
                tpk = self.public_keys.get(t.issuer)
                if tpk is None or not verify_sig(tpk, t.signature, canonical(t.body())):
                    problems.append(f"block {i}: invalid signature on transaction {t.tx_hash[:12]}")
        return problems

    def transactions(self, kind: Optional[str] = None):
        for blk in self.chain:
            for t in blk.transactions:
                if kind is None or t.kind == kind:
                    yield blk, t

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> Dict:
        return {"public_keys": self.public_keys, "seal_count": self._seal_count,
                "chain": [{**b.header(), "hash": b.hash, "signature": b.signature,
                           "transactions": [{**t.body(), "signature": t.signature, "tx_hash": t.tx_hash}
                                            for t in b.transactions]} for b in self.chain],
                "pending": [{**t.body(), "signature": t.signature} for t in self.pending]}

    def save(self, path: Path, keys_path: Optional[Path] = None):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))
        if keys_path:   # demo only: in production each node keeps its own key in an HSM
            Path(keys_path).write_text(json.dumps({n: a.private_hex() for n, a in self.authorities.items()}, indent=1))

    @classmethod
    def load(cls, path: Path, keys_path: Path) -> "Blockchain":
        d = json.loads(Path(path).read_text())
        keys = json.loads(Path(keys_path).read_text())
        bc = cls([Authority.from_private_hex(n, k) for n, k in keys.items()])
        bc.public_keys = d["public_keys"]
        bc._seal_count = d.get("seal_count", 0)
        mk = lambda t: Transaction(t["kind"], t["payload"], t["issuer"], t["timestamp"], t["signature"])
        bc.chain = [Block(b["index"], b["timestamp"], b["prev_hash"], b["merkle_root"], b["validator"],
                          [mk(t) for t in b["transactions"]], b["signature"]) for b in d["chain"]]
        bc.pending = [mk(t) for t in d.get("pending", [])]
        return bc


# --------------------------------------------------------------------------- digital ID "smart contract"
@dataclass
class Credential:
    """What the tourist carries (e.g. as a QR code in the app)."""
    tourist_id: str
    name: str
    nationality: str
    valid_from: str
    valid_to: str
    issuer: str
    signature: str = ""

    def body(self) -> Dict:
        return {"tourist_id": self.tourist_id, "name": self.name, "nationality": self.nationality,
                "valid_from": self.valid_from, "valid_to": self.valid_to, "issuer": self.issuer}

    def to_json(self) -> str:
        return json.dumps({**self.body(), "signature": self.signature})

    @classmethod
    def from_json(cls, s: str) -> "Credential":
        return cls(**json.loads(s))


@dataclass
class VerificationResult:
    valid: bool
    reason: str
    tourist_id: str
    block: Optional[int] = None
    incidents: int = 0


class DigitalIDRegistry:
    """Smart-contract-like operations over the blockchain."""

    def __init__(self, chain: Blockchain, vault: Optional[Dict] = None):
        self.chain = chain
        self.vault: Dict[str, Dict] = vault if vault is not None else {}   # off-chain KYC, issuer-held

    def _pepper(self) -> str:
        """Network secret derived from the registering authority's key (never published)."""
        a = self.chain.authorities[sorted(self.chain.authorities)[0]]
        return sha256("tid-pepper:" + a.private_hex())

    # ---------------------------------------------------------------- registration
    def register_tourist(self, name: str, nationality: str, doc_type: str, doc_number: str, phone: str,
                         emergency_contact: str, itinerary: List[str], valid_from: date, valid_to: date,
                         issuer: str, seal: bool = True) -> Credential:
        if valid_to < valid_from:
            raise ValueError("valid_to must be on or after valid_from")
        salt = secrets.token_hex(16)
        doc_hash = sha256(f"{doc_type}:{doc_number.strip().upper()}:{salt}")
        # The ID is a keyed hash of the document and trip start: the same document cannot get two
        # IDs for one trip, and nobody without the network's secret key can link an ID to a passport.
        tourist_id = "TID-" + sha256(f"{self._pepper()}:{doc_type}:{doc_number.strip().upper()}:"
                                     f"{nationality}:{valid_from.isoformat()}")[:12].upper()
        if self.lookup(tourist_id) and not self.is_revoked(tourist_id):
            raise ValueError(f"{tourist_id} already registered for this document and trip")
        cred = Credential(tourist_id, name, nationality, valid_from.isoformat(), valid_to.isoformat(), issuer)
        cred.signature = self.chain.authorities[issuer].sign(canonical(cred.body()))
        payload = {
            "tourist_id": tourist_id,
            "credential_hash": sha256(canonical(cred.body())),
            "kyc_hash": sha256(canonical({"name": name, "doc_type": doc_type, "doc_hash": doc_hash,
                                          "phone": phone, "emergency": emergency_contact, "salt": salt})),
            "itinerary_hash": sha256(canonical(itinerary)),
            "nationality": nationality,
            "valid_from": cred.valid_from, "valid_to": cred.valid_to,
        }
        self.chain.submit("REGISTER_ID", payload, issuer)
        self.vault[tourist_id] = {"name": name, "doc_type": doc_type, "doc_last4": doc_number[-4:],
                                  "phone": phone, "emergency_contact": emergency_contact,
                                  "itinerary": itinerary, "salt": salt}
        if seal:
            self.chain.seal_block()
        return cred

    def revoke(self, tourist_id: str, reason: str, issuer: str, seal: bool = True):
        if not self.lookup(tourist_id):
            raise KeyError(tourist_id)
        self.chain.submit("REVOKE_ID", {"tourist_id": tourist_id, "reason": reason}, issuer)
        if seal:
            self.chain.seal_block()

    # ---------------------------------------------------------------- queries
    def lookup(self, tourist_id: str):
        """(block index, REGISTER_ID payload) or None, searching sealed and pending records."""
        for blk, t in self.chain.transactions("REGISTER_ID"):
            if t.payload["tourist_id"] == tourist_id:
                return blk.index, t.payload
        for t in self.chain.pending:
            if t.kind == "REGISTER_ID" and t.payload["tourist_id"] == tourist_id:
                return None, t.payload
        return None

    def is_revoked(self, tourist_id: str) -> bool:
        return any(t.payload["tourist_id"] == tourist_id for _, t in self.chain.transactions("REVOKE_ID"))

    def verify(self, cred: Credential, on: Optional[date] = None, verifier: Optional[str] = None) -> VerificationResult:
        """What a police officer's app does after scanning the tourist's QR code."""
        on = on or date.today()
        found = self.lookup(cred.tourist_id)
        res = lambda ok, why, blk=None: VerificationResult(ok, why, cred.tourist_id, blk, len(self.incidents(cred.tourist_id)))
        if not found:
            return res(False, "ID not found on the blockchain")
        blk, rec = found
        pk = self.chain.public_keys.get(cred.issuer)
        if pk is None or not verify_sig(pk, cred.signature, canonical(cred.body())):
            return res(False, "issuer signature invalid (forged or altered credential)", blk)
        if sha256(canonical(cred.body())) != rec["credential_hash"]:
            return res(False, "credential does not match the on-chain record", blk)
        if self.is_revoked(cred.tourist_id):
            return res(False, "ID has been revoked", blk)
        if not (date.fromisoformat(rec["valid_from"]) <= on <= date.fromisoformat(rec["valid_to"])):
            return res(False, f"ID not valid on {on.isoformat()} (valid {rec['valid_from']} to {rec['valid_to']})", blk)
        if verifier:
            self.chain.submit("VERIFY_LOG", {"tourist_id": cred.tourist_id, "verifier": verifier}, verifier)
        return res(True, "valid", blk)

    # ---------------------------------------------------------------- incidents
    def record_incident(self, tourist_id: str, alert: Dict, issuer: str) -> Transaction:
        """Anchor an AI alert on the chain (hash of the full alert + key fields)."""
        payload = {"tourist_id": tourist_id, "time": alert.get("time"),
                   "lat": round(float(alert.get("lat", 0)), 4), "lon": round(float(alert.get("lon", 0)), 4),
                   "risk_level": alert.get("final_class"), "decided_by": alert.get("decided_by"),
                   "alert_hash": sha256(canonical(alert))}
        evidence = (alert.get("features") or {}).get("photo_sha256") if isinstance(alert.get("features"), dict) else None
        if evidence:
            payload["evidence_sha256"] = evidence      # photo of a tourist report: tamper-evident
        return self.chain.submit("INCIDENT", payload, issuer)

    def incidents(self, tourist_id: str) -> List[Dict]:
        return [dict(t.payload, block=b.index) for b, t in self.chain.transactions("INCIDENT")
                if t.payload["tourist_id"] == tourist_id]


# --------------------------------------------------------------------------- defaults
DEFAULT_AUTHORITIES = ["Tourism Department", "City Police", "Immigration Office"]


def new_network() -> tuple[Blockchain, DigitalIDRegistry]:
    bc = Blockchain([Authority(n) for n in DEFAULT_AUTHORITIES])
    return bc, DigitalIDRegistry(bc)


def credential_qr_png(cred: Credential) -> bytes:
    """PNG bytes of a QR code holding the signed credential."""
    import io
    import qrcode
    img = qrcode.make(cred.to_json())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
