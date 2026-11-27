"""Write the CONTRACT file into the data folder (STERA_DATA).

CONTRACT holds the hashes of the pack, of the index and of the training protocol
(data/protocollo_contratto.json). The trainer reads it and records it in every checkpoint;
a checkpoint can only be resumed with the same contract. With the thesis data the result
must match data/CONTRACT.
"""
import hashlib, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STERA_DATA") or os.environ["STERA_DATI"])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


m = json.loads((DATA / "MANIFEST.json").read_text())
assert sha(DATA / "stories.blocks") == m["pack_sha256"] and sha(DATA / "stories.index") == m["index_sha256"]
protocol = json.loads((HERE / "data/protocollo_contratto.json").read_text())
assert protocol["pack_sha256"] == m["pack_sha256"] and protocol["index_sha256"] == m["index_sha256"]
contract = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
line = f"{m['pack_sha256']} {m['index_sha256']} {contract}\n"
(DATA / "CONTRACT").write_text(line)
expected = (HERE / "data/CONTRACT").read_text()
print("CONTRACT written:", contract, "| matches the thesis:", line == expected)
