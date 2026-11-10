# Smart Tourist Safety Monitoring & Incident Response System Using AI, Geo-Fencing, and Blockchain-Based Digital ID

Major project by **Rakesh G**, Acharya Institute of Technology, Bangalore. The AI part is published as
ICTCS 2026 Paper ID 492, *Latency-Aware Hybrid Edge–Cloud Risk Classification for Tourist Safety
Monitoring: A Simulation Study of Selective LLM Escalation* (Rakesh G, Ratnakirti Roy).

* **Geo-fencing:** Haversine distance, circular/polygon hazard zones, route deviation, 15-minute idle gate.
* **AI:** a 0.7 KB on-device model classifies each event (LOW / MODERATE / HIGH / CRITICAL) in microseconds;
  only uncertain or severe events are escalated to the cloud LLM (*selective LLM escalation*), cutting alert latency.
* **Blockchain digital ID:** a permissioned proof-of-authority blockchain (Tourism Dept., City Police,
  Immigration) issues signed tourist IDs (QR), lets police verify or revoke them, and records every AI
  alert as a tamper-evident incident log. Only hashes go on-chain. A Solidity smart contract is included.

```
GPS fixes ─► Stage 1: geofence gate ─► Stage 2a: on-device ONNX model ─► escalation policy ─┬─► accept locally ─► alert
            (Haversine, zones, route     (logistic regression, 11 µs)    (θc + severity)    └─► Stage 2b: cloud LLM ─► alert
             deviation, idle ≥ 15 min)
```

## The real application

```bash
pip install -r requirements.txt
python run_server.py            # laptop:  http://localhost:8000   (police login admin / admin123)
python run_server.py --https    # phones on the same Wi-Fi get real GPS (see DEPLOYMENT.md)
```

* **Tourist app** (`/tourist/`, installable on the phone): register and receive a blockchain digital ID with QR code,
  real GPS tracking on a live map, geo-fencing and the AI model running **on the phone** (works offline),
  alerts queued offline and sent when the network returns, SOS button, and **incident reporting** (see below).
* **Police dashboard** (`/police/`): login, live map of tourists, incidents pushed in real time (with sound),
  acknowledge / resolve, verify a digital ID by scanning its QR with the camera, draw hazard zones on the map,
  blockchain explorer with integrity check.
* **Server** (`server/`): FastAPI + SQLite, signed login tokens, WebSocket live feed, the blockchain, and the
  cloud check through a **real LLM** (Claude, OpenAI or Gemini via API key) with an automatic rules-engine fallback.

### New: tourist incident reporting (crowd-sourced safety)

The app may show a place as safe while something real is happening there. While travelling, a tourist taps
**📢 Report an incident**, picks what happened (theft, harassment, assault, scam, medical, accident, overcrowding,
flood/fire, unsafe place, other), adds a short description and an optional photo; the GPS position is attached.

1. **Police get it instantly**: the report appears live in the dashboard's **📢 Reports** tab (with sound), with
   the photo, the reporter's phone number and a note when the app had shown that place as *safe*. It also becomes
   an incident (severity set by the type of incident) so it follows the normal acknowledge / resolve workflow.
2. **Tamper-evident evidence**: the incident is anchored on the blockchain together with the **SHA-256 of the photo**,
   so later changes to the photo can be detected.
3. **Useful to others**: every tourist's map shows recent reports (last 24 h) as warning pins, and a tourist who comes
   within 300 m of one gets a warning and a vibration. Only the type, time, place and description are shared —
   never the reporter's name, phone or photo.
4. **Crowd-verified caution zones**: when **two or more different tourists** report within 300 m and 2 hours, the
   place automatically becomes a temporary caution zone (250 m, expires after 6 h) that all phones download and the
   geo-fencing / AI pipeline treats as a hazard. Police can **Verify** a report (zone for 12 h) or **Reject** a false
   one (its warning and any crowd zone it created are removed).
5. Works offline: a report made without network is saved on the phone and sent automatically later.

API: `POST /api/reports` (tourist), `GET /api/reports/nearby` (tourist, anonymised), `GET /api/reports`,
`GET /api/reports/{id}/photo`, `POST /api/reports/{id}/verify|reject` (police). Tests: `tests/test_reports.py`
and the browser test `test_tourist_incident_report_reaches_police_and_warns_others`.

Full instructions, settings and a viva demo script: **[DEPLOYMENT.md](DEPLOYMENT.md)**.

## Project structure

| Path | What it is |
|---|---|
| `src/config.py` | Every constant (class mix, generator, latency model, thresholds) |
| `src/geofence.py` | **Geofencing**: Haversine, circle/polygon zones, route deviation, idle detection, Stage-1 gate |
| `src/trajectory.py` | GPS trace simulator: Bangalore route, hazard zones, 5 tourist scenarios |
| `src/generator.py` | Synthetic post-gate event generator + feature builder |
| `src/models.py` | 8 on-device candidates, grids, rule baseline |
| `src/stats.py` | Metrics, bootstrap CIs, McNemar, 5×2 cv F-test, ECE, cost-sensitive rule |
| `src/hybrid.py` | Cloud proxy, escalation policies, latency model, simulation |
| `src/system.py` | **Integrated system**: digital-ID check → geofence → ONNX model → escalation → cloud → alert → blockchain |
| `src/blockchain.py` | **Blockchain**: signed transactions, Merkle roots, PoA blocks, digital-ID registry, incident log |
| `src/id_demo.py` | Builds / loads the demo network (200 registered demo tourists) |
| `contracts/TouristDigitalID.sol` | Solidity smart contract (+ `test_contract.js`: 13 tests on a local Ethereum chain) |
| `flutter/risk_classifier.dart` | The trained model embedded in Dart for the Flutter app |
| `src/bench_onnx.py` | ONNX Runtime latency / memory benchmark |
| `run_all.py` | Full study: data, training, ONNX export, hybrid simulation, all tables |
| `make_figures.py` | All figures |
| `validate_system.py` | **Architecture validation and end-to-end execution** |
| `run_tests.py` | **All testing** (unit, integration, system, performance) + HTML report |
| `run_server.py` | Starts the real application (HTTP or HTTPS for phones) |
| `server/` | FastAPI server: `main.py` (API, live feed), `services.py` (logic), `llm.py` (Claude / OpenAI / Gemini / rules), `db.py`, `security.py`, `settings.py` |
| `web/tourist/` | Tourist mobile web app (PWA: works offline, installable) |
| `web/police/` | Police dashboard |
| `web/static/geo.js` | On-device engine in JavaScript: geofence gate + AI model (identical to the Python code) |
| `app/streamlit_app.py` | Research & analytics dashboard (6 pages) |
| `app/leaflet_map.py` | Leaflet / OpenStreetMap map: zones, route, replay of the tourist's movement, alert pins |
| `data/` | Generated train/val/test datasets (CSV) |
| `models/` | Trained models (`.joblib`), ONNX exports, `deployed_model.onnx` + `.json` |
| `results/` | `results.json`, tables (CSV), validation and test reports |
| `figures/` | All figures (PNG, 300 dpi) |

## How to run (Windows / macOS / Linux, Python 3.10+)

```bash
# 1. install
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 2. the trained models and results are already included, so you can go straight to the demo:
streamlit run app/streamlit_app.py

# 3. to reproduce everything from scratch (about 10-15 minutes):
python run_all.py                 # data + training + ONNX + simulation  (add --quick for a 2-minute check)
python make_figures.py            # figures
python validate_system.py         # architecture validation + end-to-end execution
python run_tests.py               # all 139 tests -> results/test_report.html
                                  # (browser tests need once:  python -m playwright install chromium)

# optional: smart-contract tests (needs Node.js 18+)
cd contracts && npm install && npm test
```

## The four project parts

1. **System architecture validation and execution** – `validate_system.py` checks that every stage
   agrees (ONNX = scikit-learn, embedded coefficients = ONNX, geofence features = training features),
   runs 100 simulated tourists through the whole pipeline, tests offline mode, and pushes all 4,000
   test events through the integrated system to confirm it reproduces the simulation.
2. **Geofencing features** – `src/geofence.py`: Haversine distance, circular and polygon hazard zones,
   distance from planned route, stationary/idle detection, one event per idle episode.
3. **Feature and model consolidation** – one feature definition (`CandidateEvent.features` =
   `generator.add_features`), one selected and exported model (`models/deployed_model.onnx`), and the
   same model as 24 coefficients + 10 scaling constants (`deployed_model.json`) for a Flutter app.
4. **Evaluation and testing** – full evaluation in `results/`, plus the pytest suite in `tests/`
   (139 tests: unit, integration, system, end-to-end browser tests of the real app, performance)
   and the smart-contract tests.
5. **Blockchain digital ID** – `src/blockchain.py`, validated in `validate_system.py` (200 IDs, forged /
   expired / revoked IDs rejected, alerts anchored, tampering detected) and shown on the app's Digital ID page.

## Honest limitations

* The AI model was **trained on synthetic data** (no labelled tourist-incident dataset is public), so its
  accuracy numbers show relative trade-offs, not field performance. Retraining on real incident logs is the
  first step before live use.
* In the research study the cloud LLM is a calibrated stand-in; the real application calls a real LLM when an
  API key is set, otherwise a rules engine.
* The blockchain runs as one permissioned network on the server, with all three authority keys in
  `server_data/`; a multi-department deployment gives each department its own node and key store.
* Latency in the study was measured on a CPU, not a phone. Numbers from a fresh run are close to, but not
  identical to, the paper's.

## Map and internet

The Geofence page uses Leaflet with OpenStreetMap, which needs internet to load the map. Geo-fencing and the
on-device AI decision work without internet; cloud verification and blockchain records are queued until the
network returns. `app/leaflet_map.build_map(..., tiles_url=...)` accepts locally stored tiles for offline maps.
