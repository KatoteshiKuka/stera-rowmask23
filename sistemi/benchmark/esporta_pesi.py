"""Esporta i pesi di un checkpoint STERA 30M nei due formati letti dai motori MLX.

  <nome>_fp32_weights.npz : tutti i tensori allenabili in FP32 (coordinate RowMask23 comprese)
  <nome>_int8_weights.npz : coordinate quantizzate INT8 simmetriche per blocco (massimo assoluto / 127,
                            scala FP32 per blocco); gli altri tensori restano FP32

Uso: python esporta_pesi.py <checkpoint.ckpt.gz> <sha256 atteso> <nome>   (es. nome = H80000)
I file vengono scritti in sistemi/modelli/.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parents[1] / "addestramento"))
import ts68_v3 as V
import ts68_core as T

V.LAYERS = 6
T.PHYSICAL_PARAMS = 30346182
T.CONFIG = (6,) + V.CFG256[1:7] + (512,)


def main(ck, sha, nome):
    W = T.read_native(ck, expected_sha=sha)["weights"]
    fp = {k: v.numpy() for k, v in W.items()}
    q = {}; err2 = ref2 = 0.0
    for k, a in fp.items():
        if k.endswith(".coords"):
            scala = np.max(np.abs(a), axis=1, keepdims=True) / np.float32(127)
            scala = np.where(scala > 0, scala, np.float32(1)).astype(np.float32)
            b = np.clip(np.rint(a / scala), -127, 127).astype(np.int8)
            q[k] = b; q[k + ".qscale"] = scala
            e = b.astype(np.float32) * scala - a
            err2 += float(np.sum(e.astype(np.float64) ** 2)); ref2 += float(np.sum(a.astype(np.float64) ** 2))
        else:
            q[k] = a
    out = QUI.parent / "modelli"
    info = {}
    for modo, arrays in (("fp32", fp), ("int8", q)):
        p = out / f"{nome}_{modo}_weights.npz"
        assert not p.exists(), f"{p} esiste gia"
        with open(str(p) + ".part", "wb") as f:
            np.savez(f, **arrays)
        os.replace(str(p) + ".part", p)
        info[modo] = {"payload_byte": sum(a.nbytes for a in arrays.values()), "sha256": T.file_sha256(p)}
    info["rmse_relativo_coordinate_int8"] = math.sqrt(err2 / ref2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
