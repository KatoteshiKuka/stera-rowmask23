"""Scrive il file CONTRACT nella cartella dei dati (STERA_DATI).

CONTRACT contiene le impronte del pacchetto, dell'indice e del protocollo di addestramento
(dati/protocollo_contratto.json). Il trainer lo legge e lo registra nei checkpoint, che possono
essere ripresi solo con lo stesso contratto. Con i dati della tesi il risultato deve coincidere
con dati/CONTRACT.
"""
import hashlib, json, os
from pathlib import Path

QUI = Path(__file__).resolve().parent
DATI = Path(os.environ["STERA_DATI"])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


m = json.loads((DATI / "MANIFEST.json").read_text())
assert sha(DATI / "stories.blocks") == m["pack_sha256"] and sha(DATI / "stories.index") == m["index_sha256"]
protocollo = json.loads((QUI / "dati/protocollo_contratto.json").read_text())
assert protocollo["pack_sha256"] == m["pack_sha256"] and protocollo["index_sha256"] == m["index_sha256"]
contratto = hashlib.sha256(json.dumps(protocollo, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
riga = f"{m['pack_sha256']} {m['index_sha256']} {contratto}\n"
(DATI / "CONTRACT").write_text(riga)
atteso = (QUI / "dati/CONTRACT").read_text()
print("CONTRACT scritto:", contratto, "| coincide con la tesi:", riga == atteso)
