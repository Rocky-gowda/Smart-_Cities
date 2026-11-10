"""Start the Smart Tourist Safety server.

  python run_server.py                 # http://localhost:8000 (laptop only; phones need HTTPS for GPS)
  python run_server.py --https         # https on your Wi-Fi address, so phones on the same network can use GPS
  python run_server.py --time-scale 15 # demonstrations: 1 real minute counts as 15 minutes of idle time

Environment variables (API keys, admin password, ...) are described in server/settings.py.
"""
import argparse
import ipaddress
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def free_port(host: str, start: int) -> int:
    """First free port from `start` upward (an old server may still hold the default one)."""
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {start} and {start + 49}. Close other servers and try again.")


def make_cert(folder: Path, ip: str):
    """Self-signed certificate for this computer's Wi-Fi address (browsers will show a warning once)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Smart Tourist Safety")])
    san = x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address(ip)),
                                       x509.IPAddress(ipaddress.ip_address("127.0.0.1"))])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=365)).add_extension(san, critical=False).sign(key, hashes.SHA256()))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                                       serialization.NoEncryption()))
    (folder / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return folder / "cert.pem", folder / "key.pem"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--https", action="store_true")
    ap.add_argument("--time-scale", type=float, default=None)
    a = ap.parse_args()
    if a.time_scale:
        os.environ["TIME_SCALE"] = str(a.time_scale)
    import uvicorn
    from server.main import create_app
    from server.settings import Settings
    s = Settings()
    app = create_app(s)
    ip = lan_ip()
    host = a.host or ("0.0.0.0" if a.https else "127.0.0.1")
    wanted = a.port or (8443 if a.https else 8000)
    port = free_port(host, wanted)
    if port != wanted:
        print(f"\n NOTE: port {wanted} is busy (probably an older server window is still open), using {port} instead.")
    ssl = {}
    scheme = "http"
    if a.https:
        cert, key = make_cert(s.data_dir / "tls", ip)
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        scheme = "https"
    base = f"{scheme}://{ip if a.https else 'localhost'}:{port}"
    print("\n" + "=" * 70)
    print(" Smart Tourist Safety Monitoring & Incident Response System")
    print("=" * 70)
    print(f" Home page        : {base}/")
    print(f" Tourist app      : {base}/tourist/     (open on the phone)")
    print(f" Police dashboard : {base}/police/      login: {s.admin_user} / {'*' * len(s.admin_password)}")
    print(f" API docs         : {base}/docs")
    print(f" Cloud check      : {app.state.core.verifier.name}")
    if s.time_scale != 1:
        print(f" Demo time scale  : 1 real minute = {s.time_scale:g} minutes")
    if s.admin_password == "admin123":
        print(" WARNING: default admin password in use. Set ADMIN_PASSWORD before real use.")
    if a.https:
        print(" Phones: connect to the same Wi-Fi, open the tourist address, accept the certificate warning once.")
    print("=" * 70 + "\n")
    import asyncio

    def quiet_handler(loop, context):
        # On Windows, phones/browsers often drop connections abruptly; asyncio then
        # prints a harmless "WinError 10054 ConnectionResetError" traceback. Ignore it.
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        loop.default_exception_handler(context)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", **ssl))

    async def serve():
        asyncio.get_running_loop().set_exception_handler(quiet_handler)
        await server.serve()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
