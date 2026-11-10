# Running the real application

The application has three parts, all served by one Python server:

| Part | Address | Used by |
|---|---|---|
| Tourist app (installable mobile web app) | `/tourist/` | Tourists, on their phones |
| Police dashboard | `/police/` | Police / tourism authorities, on a laptop |
| REST API + live feed | `/api/...`, `/ws`, docs at `/docs` | Both apps |

## 1. Install (once)

```
cd smart-tourist-safety
python -m venv venv
venv\Scripts\activate              (macOS / Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## 2. Start the server

**On one laptop only (quick check):**
```
python run_server.py
```
Open http://localhost:8000. The police login is `admin` / `admin123`.

**With phones (real GPS):** phone browsers only allow GPS on **https** pages, so start it with:
```
python run_server.py --https
```
The console prints addresses like `https://192.168.1.20:8443/tourist/`.
1. Connect the phone to the **same Wi-Fi** as the laptop.
2. On the phone, open the tourist address (or scan the QR code on the home page).
3. The browser warns about the certificate (it is self-made). Tap **Advanced → Proceed** once.
4. Allow location access. Tap **Add to Home screen** to install it like an app.
5. If the phone can't connect, allow Python through the Windows Firewall when it asks
   (or: Windows Security → Firewall → Allow an app → Python).

**From anywhere (mobile data, no shared Wi-Fi):** run the normal server and open a free tunnel with
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/):
```
python run_server.py
cloudflared tunnel --url http://localhost:8000
```
It prints a public `https://….trycloudflare.com` address with a valid certificate. Share that address with the phones.

## 3. Settings

Set these before starting the server (cmd: `set NAME=value`, PowerShell: `$env:NAME="value"`):

| Variable | Meaning |
|---|---|
| `ADMIN_PASSWORD` | Police / admin password. **Change it** for any real use. |
| `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` or `GEMINI_API_KEY` | Turns on the real cloud LLM check. Without a key, a transparent rules engine is used. |
| `LLM_MODEL` | Model name for that provider (defaults: `claude-haiku-5-5`, `gpt-4o-mini`, `gemini-2.0-flash`). |
| `TIME_SCALE` | For demonstrations: `--time-scale 15` makes 1 real minute count as 15 minutes, so a 15-minute stop triggers after 1 minute. |
| `DATA_DIR` | Where the database and blockchain are kept (default `server_data/`). |

Example with Gemini and a demo time scale:
```
set GEMINI_API_KEY=your-key-here
set ADMIN_PASSWORD=MyStrongPassword
python run_server.py --https --time-scale 15
```
The console line "Cloud check:" shows which LLM is active. If the LLM call fails (no internet, wrong key,
quota), the server falls back to the rules engine automatically and records that on the incident.

Extra police accounts: log in as admin and call `POST /api/officers` from the API docs page (`/docs`).

## 4. What a viva demonstration can look like

1. Start `python run_server.py --https --time-scale 15` and open the police dashboard on the laptop.
2. On a phone: open the tourist app, register, show the digital-ID QR code.
3. Police dashboard → **Verify ID** → scan the phone's QR with the laptop camera → VALID (block number shown).
4. Police dashboard → **Zones** → click the map where you are standing → save a hazard zone.
5. Phone: **Start tracking**, stand still for about a minute → the phone shows the risk level, the
   police dashboard beeps and shows the incident live. Acknowledge and resolve it.
6. Phone: turn on airplane mode, use **Demonstration tools → Simulate a stop**, turn airplane mode off →
   the saved alert arrives at the police dashboard marked "sent after reconnecting".
7. Phone: press **SOS** twice → CRITICAL incident on the dashboard.
8. Phone: **📢 Report an incident** → choose "Theft", type a line, take a photo → **Send report**. The police
   dashboard beeps; open the **📢 Reports** tab to see the photo, the description and "App showed this place as safe".
9. A second phone (another registered tourist) at the same place: a warning pin appears on its map and it shows
   "Reported by tourists near you". If the second phone also reports, the place turns into an orange dashed
   caution zone on every phone. Or press **Verify → caution zone** on the dashboard.
10. Police dashboard → **Blockchain** → chain intact, every registration and incident recorded (reports carry the
    photo's SHA-256 as `evidence_sha256`).

## 5. Permanent hosting

Any Linux server or cloud VM with Python works: copy the folder, `pip install -r requirements.txt`, and run
`python run_server.py --host 0.0.0.0 --port 8000` behind an HTTPS reverse proxy (for example Caddy:
`caddy reverse-proxy --from yourdomain.com --to localhost:8000`). Keep `server_data/` on persistent disk and
back it up: it holds the database, the blockchain and the authority keys.
