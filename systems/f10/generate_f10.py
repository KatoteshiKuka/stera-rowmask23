"""Generate the fictitious STERA F10 model from the final STERA 30M checkpoint.

F10 is only a resource reference (memory, disk reads) for the comparison with Qwen3.5-9B:
it is NOT a trained model. It widens STERA 30M by a factor F=10 (hidden size 7,680, 160 heads,
feed-forward 30,720, 6 layers, same vocabulary): 4.63 billion values in the dense-equivalent form.

Deterministic rule, using IEEE arithmetic only (no trigonometric functions), so the file is
bit-identical on every machine:
  - every new block (r, c) copies the coordinates of the trained block (r mod R, c mod C), divided by F
    and perturbed by a small combination of two other coordinates of the same block, with a pseudo-random
    coefficient taken from an integer hash of (block, index), so that every block has its own coordinates;
  - scales, gains, U, V, LayerNorm, biases and positions are repeated F times (V divided by F in the layers),
    as in the function-preserving construction used in the measurements.
Usage: python generate_f10.py systems/models/H122000_fp32_weights.npz systems/models/F10_H122000.npz
The keys of the JSON manifest written next to the output are kept as in the archived F10_H122000.json.
"""
import hashlib, json, struct, sys, time
from pathlib import Path
import numpy as np

F = 10
BASE = [("embedding", 50304, 768)] + [(f"layers.{l}.{k}", r, c) for l in range(6)
        for k, r, c in [("q", 768, 768), ("k", 768, 768), ("v", 768, 768), ("o", 768, 768), ("fc", 3072, 768), ("proj", 768, 3072)]]
NC = 3176
LIMIT = (1 << 31) - 1   # zipfile.ZIP64_LIMIT


def coordinates(src, rt, ct, rf):
    """src: (rt*ct, 3176) float32 -> (rt*rf * ct*F, 3176) float32."""
    ctb = ct * F
    n = rt * rf * ctb
    out = np.empty((n, NC), np.float32)
    ci = np.arange(NC, dtype=np.uint64)
    ia, ib = (ci + 37) % NC, (ci + 151) % NC
    for tile in range(n):
        old = (tile // ctb % rt) * ct + (tile % ctb % ct)
        s = src[old].astype(np.float64)
        h = (np.uint64(tile) * np.uint64(2654435761) + ci * np.uint64(40503) + np.uint64(12345)) % np.uint64(2**32)
        coef = h.astype(np.float64) / 2**32 * 2 - 1
        out[tile] = ((s + 0.009 * coef * (s[ia] + 0.25 * s[ib])) / F).astype(np.float32)
    return out


class NpzWriter:
    """Minimal uncompressed .npz (zip) writer with a fixed byte layout.

    Python's zipfile writes different local headers depending on the Python version (for example 3.9 and 3.11
    differ with force_zip64), which changes the SHA-256 of the file although the arrays are identical.
    This writer always produces the layout of the published F10_H122000.npz: local header version 2.0 with a
    ZIP64 extra field holding the sizes, date 1980-01-01, central directory without extra fields.
    np.load reads the result as a normal .npz file.
    """

    def __init__(self, path):
        self.f = open(path, "wb"); self.entries = []

    def add(self, name, arr):
        import io, zlib
        buf = io.BytesIO(); np.lib.format.write_array(buf, arr, allow_pickle=False)
        data = buf.getbuffer(); size = len(data); crc = zlib.crc32(data) & 0xFFFFFFFF
        fname = name.encode(); offset = self.f.tell()
        assert size <= LIMIT, "the fixed layout supports entries below 2 GiB"
        extra = struct.pack("<HHQQ", 1, 16, size, size)
        self.f.write(struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0x21, crc, size, size, len(fname), len(extra)))
        self.f.write(fname); self.f.write(extra); self.f.write(data)
        self.entries.append((fname, crc, size, offset))

    def close(self):
        # same rules as CPython's zipfile: ZIP64 fields only for offsets beyond 2 GiB
        start = self.f.tell()
        for fname, crc, size, offset in self.entries:
            big = offset > LIMIT
            extra = struct.pack("<HHQ", 1, 8, offset) if big else b""
            ver = 45 if big else 20
            self.f.write(struct.pack("<IBBHHHHHIIIHHHHHII", 0x02014B50, ver, 3, ver, 0, 0, 0, 0x21, crc, size, size,
                                     len(fname), len(extra), 0, 0, 0, 0o600 << 16, 0xFFFFFFFF if big else offset))
            self.f.write(fname); self.f.write(extra)
        end = self.f.tell(); n = len(self.entries); size = end - start
        if start > LIMIT or size > LIMIT or n > 0xFFFF:
            self.f.write(struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, n, n, size, start))
            self.f.write(struct.pack("<IIQI", 0x07064B50, 0, end, 1))
        self.f.write(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, min(n, 0xFFFF), min(n, 0xFFFF),
                                 min(size, 0xFFFFFFFF), min(start, 0xFFFFFFFF), 0))
        self.f.close()


def main(source, target):
    t0 = time.time()
    z = np.load(source)
    info = {"coordinate": 0, "byte_pesi_fp32": 0}
    zf = NpzWriter(target)
    if True:
        def write(name, arr):
            arr = np.ascontiguousarray(arr)
            zf.add(name + ".npy", arr)
            info["byte_pesi_fp32"] += arr.nbytes
        write("position.weight", np.tile(z["position.weight"], (1, F)))
        for n in ["final_norm"] + [f"layers.{l}.{k}" for l in range(6) for k in ("ln1", "ln2")]:
            for a in ("weight", "bias"):
                write(f"{n}.{a}", np.tile(z[f"{n}.{a}"], F))
        for l in range(6):
            for k in ("o", "fc", "proj"):
                write(f"layers.{l}.{k}.bias", np.tile(z[f"layers.{l}.{k}.bias"], F))
        for name, r, c in BASE:
            rf = 1 if name == "embedding" else F
            rt, ct = r // 128, c // 128
            co = coordinates(z[name + ".coords"].reshape(-1, NC), rt, ct, rf)
            info["coordinate"] += co.size
            write(name + ".coords", co); del co
            write(name + ".scales", np.tile(z[name + ".scales"].reshape(rt, ct), (rf, F)).reshape(-1).astype(np.float32))
            write(name + ".gout", np.tile(z[name + ".gout"], rf))
            write(name + ".gin", np.tile(z[name + ".gin"], F))
            write(name + ".U", np.tile(z[name + ".U"], (rf, 1)))
            V = np.tile(z[name + ".V"], (F, 1))
            write(name + ".V", (V / F).astype(np.float32) if rf != 1 else V)
            print(name, round(time.time() - t0), "s", flush=True)
    zf.close()
    sha = hashlib.sha256()
    with open(target, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            sha.update(chunk)
    man = {"file": Path(target).name, "sha256": sha.hexdigest(), "sorgente": Path(source).name,
           "sha256_sorgente": hashlib.sha256(Path(source).read_bytes()).hexdigest(), "fattore": F,
           "coordinate": int(info["coordinate"]), "byte_pesi_fp32": int(info["byte_pesi_fp32"]),
           "valori_forma_densa": 4633067520}
    Path(target).with_suffix(".json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
