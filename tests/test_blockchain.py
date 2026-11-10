"""Unit, integration and system tests for the blockchain digital-ID layer."""
import copy
import json
from datetime import date

import pytest

from src.blockchain import (Authority, Blockchain, Credential, DigitalIDRegistry, canonical, merkle_root,
                            new_network, sha256)

PERIOD = (date(2026, 12, 10), date(2026, 12, 24))


def _register(reg, doc="C01X00T47", issuer="Tourism Department", **kw):
    args = dict(name="Test Tourist", nationality="DE", doc_type="Passport", doc_number=doc,
                phone="+49 151 000", emergency_contact="+49 30 111", itinerary=["MG Road", "Cubbon Park"],
                valid_from=PERIOD[0], valid_to=PERIOD[1], issuer=issuer)
    args.update(kw)
    return reg.register_tourist(**args)


@pytest.fixture
def net():
    return new_network()


# ---------------------------------------------------------------- unit
@pytest.mark.unit
def test_sha256_known_value():
    assert sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.unit
def test_merkle_root_changes_with_any_leaf():
    leaves = [sha256(str(i)) for i in range(5)]
    r = merkle_root(leaves)
    leaves[3] = sha256("x")
    assert merkle_root(leaves) != r


@pytest.mark.unit
def test_canonical_is_order_independent():
    assert canonical({"a": 1, "b": 2}) == canonical({"b": 2, "a": 1})


@pytest.mark.unit
def test_signature_roundtrip_and_forgery():
    from src.blockchain import verify_sig
    a = Authority("A")
    sig = a.sign(b"hello")
    assert verify_sig(a.public_key_hex, sig, b"hello")
    assert not verify_sig(a.public_key_hex, sig, b"hellO")
    assert not verify_sig(Authority("B").public_key_hex, sig, b"hello")


@pytest.mark.unit
def test_genesis_and_block_linking(net):
    bc, reg = net
    _register(reg)
    _register(reg, doc="P1234567")
    assert len(bc.chain) == 3
    assert bc.chain[2].prev_hash == bc.chain[1].hash
    assert bc.verify_chain() == []


@pytest.mark.unit
def test_unauthorised_node_cannot_write(net):
    bc, _ = net
    with pytest.raises(PermissionError):
        bc.submit("REGISTER_ID", {"tourist_id": "x"}, "Random Hacker")


@pytest.mark.unit
def test_round_robin_validators(net):
    bc, reg = net
    for i in range(3):
        _register(reg, doc=f"P00{i}")
    assert len({b.validator for b in bc.chain[1:]}) == 3


@pytest.mark.unit
def test_no_personal_data_on_chain(net):
    bc, reg = net
    _register(reg, name="Secret Name", doc="Z9876543", phone="+49 999 777")
    raw = json.dumps(bc.to_dict())
    for secret in ("Secret Name", "Z9876543", "+49 999 777"):
        assert secret not in raw


# ---------------------------------------------------------------- integration: ID lifecycle
@pytest.mark.integration
def test_valid_credential_verifies(net):
    _, reg = net
    c = _register(reg)
    r = reg.verify(c, on=date(2026, 12, 15))
    assert r.valid and r.block == 1


@pytest.mark.integration
def test_forged_credential_rejected(net):
    _, reg = net
    c = _register(reg)
    forged = Credential(**{**json.loads(c.to_json()), "name": "Someone Else"})
    assert not reg.verify(forged, on=date(2026, 12, 15)).valid


@pytest.mark.integration
def test_extended_validity_rejected(net):
    _, reg = net
    c = _register(reg)
    fake = Credential(**{**json.loads(c.to_json()), "valid_to": "2027-12-31"})
    assert not reg.verify(fake, on=date(2027, 6, 1)).valid


@pytest.mark.integration
def test_expired_and_future_ids_rejected(net):
    _, reg = net
    c = _register(reg)
    assert "not valid" in reg.verify(c, on=date(2027, 1, 5)).reason
    assert "not valid" in reg.verify(c, on=date(2026, 11, 1)).reason


@pytest.mark.integration
def test_revoked_id_rejected(net):
    _, reg = net
    c = _register(reg)
    reg.revoke(c.tourist_id, "lost phone", "City Police")
    assert reg.verify(c, on=date(2026, 12, 15)).reason == "ID has been revoked"


@pytest.mark.integration
def test_unknown_id_rejected(net):
    _, reg = net
    other_bc, other_reg = new_network()
    c = _register(other_reg)
    assert reg.verify(c, on=date(2026, 12, 15)).reason == "ID not found on the blockchain"


@pytest.mark.integration
def test_duplicate_registration_blocked(net):
    _, reg = net
    _register(reg)
    with pytest.raises(ValueError):
        _register(reg)                                   # same document, same trip
    c3 = _register(reg, valid_from=date(2027, 3, 1), valid_to=date(2027, 3, 9))   # a new trip is fine
    assert reg.lookup(c3.tourist_id)


@pytest.mark.integration
def test_invalid_validity_period(net):
    _, reg = net
    with pytest.raises(ValueError):
        _register(reg, valid_from=date(2026, 12, 20), valid_to=date(2026, 12, 1))


# ---------------------------------------------------------------- tamper evidence
@pytest.mark.integration
def test_tampered_transaction_detected(net):
    bc, reg = net
    c = _register(reg)
    reg.record_incident(c.tourist_id, {"final_class": "CRITICAL", "lat": 12.98, "lon": 77.59}, "City Police")
    bc.seal_block()
    bc.chain[-1].transactions[0].payload["risk_level"] = "LOW"
    problems = bc.verify_chain()
    assert any("Merkle" in p for p in problems)


@pytest.mark.integration
def test_rewritten_block_breaks_link(net):
    bc, reg = net
    _register(reg)
    _register(reg, doc="P2")
    bc.chain[1].timestamp += 1
    assert any("previous-hash" in p for p in bc.verify_chain())


@pytest.mark.integration
def test_save_and_load_roundtrip(tmp_path, net):
    bc, reg = net
    c = _register(reg)
    bc.save(tmp_path / "c.json", tmp_path / "k.json")
    bc2 = Blockchain.load(tmp_path / "c.json", tmp_path / "k.json")
    assert bc2.verify_chain() == [] and len(bc2.chain) == len(bc.chain)
    assert DigitalIDRegistry(bc2).verify(c, on=date(2026, 12, 15)).valid


# ---------------------------------------------------------------- system: AI + blockchain
@pytest.mark.system
def test_alerts_are_anchored_on_chain(artifacts, net):
    from src.system import HybridRiskSystem
    from src.trajectory import PLANNED_ROUTE, SCENARIOS, ZONES, make_trace
    bc, reg = net
    c = _register(reg)
    sysm = HybridRiskSystem(registry=reg)
    sc = next(s for s in SCENARIOS if s.name == "hazard_night")
    alerts = sysm.run_stream("x", make_trace(sc, 0), PLANNED_ROUTE, ZONES,
                             {"context_z": sc.context_z, "incident": sc.incident}, credential=c)
    assert alerts[0].digital_id["valid"]
    inc = reg.incidents(c.tourist_id)
    assert len(inc) == 1 and inc[0]["risk_level"] == alerts[0].final_class
    assert bc.verify_chain() == []


@pytest.mark.system
def test_demo_network_is_intact(artifacts):
    from src import id_demo
    if not id_demo.CHAIN.exists():
        pytest.skip("run validate_system.py first")
    bc, reg, creds = id_demo.load()
    assert bc.verify_chain() == []
    assert len(creds) >= 100


@pytest.mark.performance
def test_verification_is_fast(net):
    import time
    _, reg = net
    c = _register(reg)
    t0 = time.perf_counter()
    for _ in range(100):
        reg.verify(c, on=date(2026, 12, 15))
    assert (time.perf_counter() - t0) / 100 < 0.01   # < 10 ms per verification
