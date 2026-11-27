#!/usr/bin/env python3
"""Tokenize parquet or jsonl code files into train.bin / val.bin (uint16) for stera_code.py.

Every source is a folder of .parquet or .jsonl[.gz] files with the text in one column (content or text).
A document goes to VAL if the SHA-256 of its text, modulo 1000, is below 5 (0.5%), otherwise to TRAIN;
identical documents therefore always land in the same part. Exact duplicates are dropped.
Every document ends with the end-of-text token. Writes MANIFEST.json with counts and hashes
(manifest keys are kept in Italian, as in the existing data folders).

Usage: python prepare_code.py --tokenizer tokenizer.json --output DATA \
        --source name=folder[:column][:max_tokens][:languages] [--source ...]
(languages: comma-separated list, compared with the "lang" field of the jsonl files)
The Italian option names --uscita, --sorgente and --lotto are accepted as aliases.
"""
import argparse, glob, gzip, hashlib, json, os, time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def documents(folder, column, languages=None):
    for f in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(f)
            name = column or next(c for c in ("content", "text") if c in pf.schema.names)
            for rg in range(pf.num_row_groups):
                for t in pf.read_row_group(rg, columns=[name]).column(0).to_pylist():
                    if t:
                        yield t
        elif f.endswith((".jsonl", ".jsonl.gz", ".json.gz")):
            with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as h:
                for line in h:
                    r = json.loads(line)
                    if languages and str(r.get("lang", "")).lower() not in languages:
                        continue
                    t = r.get(column) if column else r.get("content", r.get("text"))
                    if t:
                        yield t


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True); p.add_argument("--output", "--uscita", dest="output", required=True)
    p.add_argument("--source", "--sorgente", dest="source", action="append", required=True); p.add_argument("--batch", "--lotto", dest="batch", type=int, default=2000)
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    eos = tok.token_to_id("<|endoftext|>")
    assert eos is not None and tok.get_vocab_size() <= 65536
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    ftr, fva = open(out / "train.bin.partial", "wb"), open(out / "val.bin.partial", "wb")
    seen, man, t0 = set(), {"tokenizer_sha256": sha(a.tokenizer), "eos": eos, "vocab": tok.get_vocab_size(),
                             "regola_val": "sha256(testo) mod 1000 < 5", "sorgenti": []}, time.time()
    for s in a.source:
        name, rest = s.split("=", 1)
        parts = rest.split(":")
        folder, column = parts[0], (parts[1] if len(parts) > 1 and parts[1] else None)
        quota = int(float(parts[2])) if len(parts) > 2 and parts[2] else None
        languages = set(parts[3].lower().split(",")) if len(parts) > 3 and parts[3] else None
        counts = {"nome": name, "cartella": folder, "documenti": 0, "duplicati": 0, "token_train": 0, "token_val": 0, "quota": quota,
                "lingue": sorted(languages) if languages else None}
        buf = []

        def write(buf):
            for t, ids in zip(buf, tok.encode_batch(buf)):
                h = hashlib.sha256(t.encode()).digest()
                arr = np.array(ids.ids + [eos], dtype=np.uint16)
                if int.from_bytes(h[:8], "big") % 1000 < 5:
                    fva.write(arr.tobytes()); counts["token_val"] += len(arr)
                else:
                    ftr.write(arr.tobytes()); counts["token_train"] += len(arr)

        for t in documents(folder, column, languages):
            h = hashlib.sha256(t.encode()).digest()[:16]
            if h in seen:
                counts["duplicati"] += 1; continue
            seen.add(h); buf.append(t); counts["documenti"] += 1
            if len(buf) == a.batch:
                write(buf); buf = []
                if counts["documenti"] % 200000 == 0:
                    print(json.dumps({**counts, "secondi": round(time.time() - t0)}), flush=True)
                if quota and counts["token_train"] >= quota:
                    break
        if buf:
            write(buf)
        man["sorgenti"].append(counts); print(json.dumps(counts), flush=True)
    ftr.close(); fva.close()
    for n in ("train", "val"):
        (out / f"{n}.bin.partial").rename(out / f"{n}.bin")
        man[f"{n}_sha256"] = sha(out / f"{n}.bin"); man[f"{n}_token"] = (out / f"{n}.bin").stat().st_size // 2
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "sorgenti"}), flush=True)


if __name__ == "__main__":
    main()
