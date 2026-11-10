"""Password hashing and signed login tokens (standard library only)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Dict, Optional

ITERATIONS = 200_000


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt}${dk.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, it, salt, h = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(dk.hex(), h)
    except ValueError:
        return False


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(secret: str, sub: str, role: str, ttl_s: int) -> str:
    """HMAC-SHA256 signed token: base64(payload).base64(signature)."""
    payload = _b64(json.dumps({"sub": sub, "role": role, "exp": int(time.time()) + ttl_s}).encode())
    sig = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def read_token(secret: str, token: str) -> Optional[Dict]:
    try:
        payload, sig = token.split(".")
        good = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(_unb64(payload))
        return data if data["exp"] > time.time() else None
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
