"""Build and load the demo blockchain network (registered tourists + incidents).

Used by validate_system.py (which creates data/chain.json) and by the demo app.
All names and document numbers are randomly generated, not real people.
"""
import json
import random
from datetime import date, timedelta

from . import config as C
from .blockchain import Blockchain, Credential, DigitalIDRegistry, new_network

FIRST = ["Aarav", "Ananya", "Liam", "Sofia", "Kenji", "Amara", "Lucas", "Mei", "Omar", "Elena", "Rohan",
         "Priya", "Noah", "Hana", "Mateo", "Zara", "Ethan", "Leila", "Arjun", "Chloe"]
LAST = ["Sharma", "Iyer", "Müller", "Rossi", "Tanaka", "Okafor", "Silva", "Chen", "Haddad", "Novak",
        "Nair", "Reddy", "Smith", "Kim", "Garcia", "Khan", "Brown", "Rahimi", "Menon", "Dubois"]
NAT = ["IN", "IN", "IN", "DE", "IT", "JP", "NG", "BR", "CN", "AE", "CZ", "US", "KR", "ES", "FR", "GB"]
SPOTS = ["MG Road", "Cubbon Park", "Vidhana Soudha", "Lalbagh", "Bangalore Palace", "Museum Road",
         "Commercial Street", "ISKCON Temple"]

CHAIN = C.DATA_DIR / "chain.json"
KEYS = C.DATA_DIR / "authority_keys.json"
VAULT = C.DATA_DIR / "kyc_vault.json"
CREDS = C.DATA_DIR / "demo_credentials.json"


def random_tourist(rng: random.Random):
    nat = rng.choice(NAT)
    doc_type = "Aadhaar" if nat == "IN" else "Passport"
    doc = "".join(rng.choice("0123456789") for _ in range(12)) if doc_type == "Aadhaar" else \
        rng.choice("CKLMNP") + "".join(rng.choice("0123456789") for _ in range(8))
    start = date(2026, 12, 1) + timedelta(days=rng.randint(0, 20))
    return dict(name=f"{rng.choice(FIRST)} {rng.choice(LAST)}", nationality=nat, doc_type=doc_type, doc_number=doc,
                phone="+91 9" + "".join(rng.choice("0123456789") for _ in range(9)),
                emergency_contact="+91 8" + "".join(rng.choice("0123456789") for _ in range(9)),
                itinerary=rng.sample(SPOTS, 3), valid_from=start, valid_to=start + timedelta(days=rng.randint(3, 21)))


def build(n: int = 200, seed: int = 7):
    rng = random.Random(seed)
    bc, reg = new_network()
    creds = []
    for i in range(n):
        t = random_tourist(rng)
        creds.append(reg.register_tourist(issuer="Tourism Department", seal=False, **t))
        if (i + 1) % 25 == 0:
            bc.seal_block()
    bc.seal_block()
    return bc, reg, creds


def save(bc: Blockchain, reg: DigitalIDRegistry, creds):
    bc.save(CHAIN, KEYS)
    VAULT.write_text(json.dumps(reg.vault, indent=1, ensure_ascii=False))
    CREDS.write_text(json.dumps([json.loads(c.to_json()) for c in creds], indent=1, ensure_ascii=False))


def load():
    bc = Blockchain.load(CHAIN, KEYS)
    reg = DigitalIDRegistry(bc, json.loads(VAULT.read_text()))
    creds = [Credential(**d) for d in json.loads(CREDS.read_text())]
    return bc, reg, creds
