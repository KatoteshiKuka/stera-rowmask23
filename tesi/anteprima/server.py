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
    r = subprocess.run(["tectonic", "-X", "compile", "--synctex", "main.tex"], cwd=ROOT,
                       capture_output=True, text=True)
    errs = [l for l in (r.stdout + r.stderr).splitlines() if "error" in l.lower()]
    state["log"] = "\n".join(errs[-15:])
    state["status"] = "ok" if r.returncode == 0 else "errore"
    if r.returncode == 0:
        state["version"] = int((ROOT / "main.pdf").stat().st_mtime * 1000)
    print(time.strftime("%H:%M:%S"), state["status"], flush=True)


_sync = {"mtime": None, "pagine": {}, "input": {}}


def carica_sync():
    import gzip, re
    f = ROOT / "main.synctex.gz"
    if not f.exists():
        return
    m = f.stat().st_mtime
    if _sync["mtime"] == m:
        return
    inp, pagine, cur = {}, {}, None
    rec = re.compile(r"^[hvxkg$(\[](\d+),(\d+):(-?\d+),(-?\d+)")
    with gzip.open(f, "rt", errors="replace") as fh:
        for l in fh:
            if l.startswith("Input:"):
                _, tag, path = l.rstrip("\n").split(":", 2)
                inp[tag] = path
            elif l.startswith("{"):
                cur = int(l[1:]); pagine[cur] = []
            elif l.startswith("}"):
                cur = None
            elif cur is not None:
                r = rec.match(l)
                if r:
                    pagine[cur].append((r.group(1), int(r.group(2)), int(r.group(3)), int(r.group(4))))
    _sync.update(mtime=m, pagine=pagine, input=inp)


def sincro(pagina, x, y):
    carica_sync()
    k = 65536 * 72.27 / 72
    hx, vy = x * k, y * k
    best = None
    for tag, line, h, v in _sync["pagine"].get(pagina, []):
        p = _sync["input"].get(tag, "")
        if not p.endswith(".tex") or line <= 0 or "/figure/" in p:
            continue
        dv = v - vy  # baseline sotto il punto cliccato: 0 .. 12 pt
        d = abs(h - hx) / 20 + (abs(dv) if 0 <= dv <= 12 * 65536 else 10 * abs(dv) + 10 ** 7)
        if best is None or d < best[0]:
            best = (d, p, line)
    if not best:
        return None
    try:
        rel = str(pathlib.Path(best[1]).resolve().relative_to(ROOT))
    except ValueError:
        return None
    return {"p": rel, "riga": best[2]}


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
        if self.path.startswith("/sincro?"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            r = sincro(int(q["pagina"][0]), float(q["x"][0]), float(q["y"][0]))
            b = json.dumps(r).encode()
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
