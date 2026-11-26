"""Anteprima live della tesi: ricompila main.tex a ogni salvataggio e serve il PDF.

Uso: python3 server.py [porta]   (default 8765)
"""
import http.server, os, pathlib, subprocess, sys, threading, time, json

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
EXT = {".tex", ".bib", ".cls", ".pdf", ".png", ".jpg"}
state = {"version": 0, "status": "avvio", "log": ""}


def snapshot():
    out = {}
    for p in ROOT.rglob("*"):
        if p.suffix in EXT and p.name != "main.pdf" and "anteprima" not in p.parts:
            try:
                out[str(p)] = p.stat().st_mtime
            except FileNotFoundError:
                pass
    return out


def build():
    state["status"] = "compilazione"
    r = subprocess.run(["tectonic", "-X", "compile", "main.tex"], cwd=ROOT,
                       capture_output=True, text=True)
    errs = [l for l in (r.stdout + r.stderr).splitlines() if "error" in l.lower()]
    state["log"] = "\n".join(errs[-15:])
    state["status"] = "ok" if r.returncode == 0 else "errore"
    if r.returncode == 0:
        state["version"] = int((ROOT / "main.pdf").stat().st_mtime * 1000)
    print(time.strftime("%H:%M:%S"), state["status"], flush=True)


def watcher():
    last = None
    while True:
        snap = snapshot()
        if snap != last:
            last = snap
            build()
        time.sleep(1)


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def do_GET(self):
        if self.path.startswith("/stato"):
            b = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path.startswith("/elenco"):
            fl = sorted([p for p in ROOT.rglob("*") if p.suffix in (".tex", ".bib", ".cls", ".py")
                         and "anteprima" not in p.parts and not any(x.startswith(".") for x in p.parts)],
                        key=lambda p: str(p))
            b = json.dumps([{"p": str(p.relative_to(ROOT)), "t": p.stat().st_mtime} for p in fl]).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b); return
        if self.path.startswith("/sorgente?p="):
            from urllib.parse import unquote
            p = (ROOT / unquote(self.path.split("=", 1)[1])).resolve()
            if ROOT not in p.parents or not p.is_file():
                self.send_error(404); return
            b = p.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/plain; charset=utf-8"); self.end_headers()
            self.wfile.write(b); return
        if self.path in ("/", "/index.html"):
            self.path = "/anteprima/anteprima.html"
        return super().do_GET()


    def do_POST(self):
        from urllib.parse import unquote
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
        if self.path.startswith("/salva?p="):
            p = (ROOT / unquote(self.path.split("=", 1)[1])).resolve()
            if ROOT not in p.parents or p.suffix not in (".tex", ".bib") or not p.is_file():
                self.send_error(403); return
            p.write_bytes(body)
            b = json.dumps({"t": p.stat().st_mtime}).encode()
        elif self.path.startswith("/compila"):
            threading.Thread(target=build, daemon=True).start(); b = b"{}"
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=watcher, daemon=True).start()
    print(f"Anteprima su http://localhost:{PORT}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
