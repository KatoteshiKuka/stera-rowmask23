#!/usr/bin/env python3
"""Tokenizza file parquet o jsonl di codice in train.bin / val.bin (uint16) per stera_codice.py.

Ogni sorgente e una cartella di file .parquet o .jsonl[.gz] con il testo in una colonna (content o text).
Un documento va in VAL se l'hash SHA-256 del testo, modulo 1000, e minore di 5 (0,5%), altrimenti in TRAIN;
documenti identici finiscono quindi sempre nella stessa parte. Duplicati esatti sono scartati.
Ogni documento termina con il token di fine testo. Scrive MANIFEST.json con conteggi e impronte.

Uso: python prepara_codice.py --tokenizer tokenizer.json --uscita DATI \
        --sorgente nome=cartella[:colonna][:quota_massima_token][:lingue] [--sorgente ...]
(lingue: elenco separato da virgole, confrontato con il campo "lang" dei file jsonl)
"""
import argparse, glob, gzip, hashlib, json, os, time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def documenti(cartella, colonna, lingue=None):
    for f in sorted(glob.glob(os.path.join(cartella, "**", "*"), recursive=True)):
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(f)
            nome = colonna or next(c for c in ("content", "text") if c in pf.schema.names)
            for rg in range(pf.num_row_groups):
                for t in pf.read_row_group(rg, columns=[nome]).column(0).to_pylist():
                    if t:
                        yield t
        elif f.endswith((".jsonl", ".jsonl.gz", ".json.gz")):
            with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as h:
                for riga in h:
                    r = json.loads(riga)
                    if lingue and str(r.get("lang", "")).lower() not in lingue:
                        continue
                    t = r.get(colonna) if colonna else r.get("content", r.get("text"))
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
    p.add_argument("--tokenizer", required=True); p.add_argument("--uscita", required=True)
    p.add_argument("--sorgente", action="append", required=True); p.add_argument("--lotto", type=int, default=2000)
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    fine = tok.token_to_id("<|endoftext|>")
    assert fine is not None and tok.get_vocab_size() <= 65536
    out = Path(a.uscita); out.mkdir(parents=True, exist_ok=True)
    ftr, fva = open(out / "train.bin.partial", "wb"), open(out / "val.bin.partial", "wb")
    visti, man, t0 = set(), {"tokenizer_sha256": sha(a.tokenizer), "eos": fine, "vocab": tok.get_vocab_size(),
                             "regola_val": "sha256(testo) mod 1000 < 5", "sorgenti": []}, time.time()
    for s in a.sorgente:
        nome, resto = s.split("=", 1)
        parti = resto.split(":")
        cartella, colonna = parti[0], (parti[1] if len(parti) > 1 and parti[1] else None)
        quota = int(float(parti[2])) if len(parti) > 2 and parti[2] else None
        lingue = set(parti[3].lower().split(",")) if len(parti) > 3 and parti[3] else None
        cont = {"nome": nome, "cartella": cartella, "documenti": 0, "duplicati": 0, "token_train": 0, "token_val": 0, "quota": quota,
                "lingue": sorted(lingue) if lingue else None}
        buf = []

        def scrivi(buf):
            for t, ids in zip(buf, tok.encode_batch(buf)):
                h = hashlib.sha256(t.encode()).digest()
                arr = np.array(ids.ids + [fine], dtype=np.uint16)
                if int.from_bytes(h[:8], "big") % 1000 < 5:
                    fva.write(arr.tobytes()); cont["token_val"] += len(arr)
                else:
                    ftr.write(arr.tobytes()); cont["token_train"] += len(arr)

        for t in documenti(cartella, colonna, lingue):
            h = hashlib.sha256(t.encode()).digest()[:16]
            if h in visti:
                cont["duplicati"] += 1; continue
            visti.add(h); buf.append(t); cont["documenti"] += 1
            if len(buf) == a.lotto:
                scrivi(buf); buf = []
                if cont["documenti"] % 200000 == 0:
                    print(json.dumps({**cont, "secondi": round(time.time() - t0)}), flush=True)
                if quota and cont["token_train"] >= quota:
                    break
        if buf:
            scrivi(buf)
        man["sorgenti"].append(cont); print(json.dumps(cont), flush=True)
    ftr.close(); fva.close()
    for n in ("train", "val"):
        (out / f"{n}.bin.partial").rename(out / f"{n}.bin")
        man[f"{n}_sha256"] = sha(out / f"{n}.bin"); man[f"{n}_token"] = (out / f"{n}.bin").stat().st_size // 2
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "sorgenti"}), flush=True)


if __name__ == "__main__":
    main()
