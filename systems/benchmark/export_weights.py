"""Export the weights of a STERA 30M checkpoint into the two formats read by the MLX engines.

  <name>_fp32_weights.npz : all trainable tensors in FP32 (RowMask23 coordinates included)
  <name>_int8_weights.npz : coordinates quantized to symmetric INT8 per block (max absolute value / 127,
                            one FP32 scale per block); the other tensors stay FP32

Usage: python export_weights.py <checkpoint.ckpt.gz> <expected sha256> <name>   (e.g. name = H122000)
The files are written to systems/models/.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "training"))
import ts68_v3 as V
import ts68_core as T

V.LAYERS = 6
T.PHYSICAL_PARAMS = 30346182
T.CONFIG = (6,) + V.CFG256[1:7] + (512,)


def main(ck, sha, name):
    W = T.read_native(ck, expected_sha=sha)["weights"]
    fp = {k: v.numpy() for k, v in W.items()}
    q = {}; err2 = ref2 = 0.0
    for k, a in fp.items():
        if k.endswith(".coords"):
            scale = np.max(np.abs(a), axis=1, keepdims=True) / np.float32(127)
            scale = np.where(scale > 0, scale, np.float32(1)).astype(np.float32)
            b = np.clip(np.rint(a / scale), -127, 127).astype(np.int8)
            q[k] = b; q[k + ".qscale"] = scale
            e = b.astype(np.float32) * scale - a
            err2 += float(np.sum(e.astype(np.float64) ** 2)); ref2 += float(np.sum(a.astype(np.float64) ** 2))
        else:
            q[k] = a
    out = HERE.parent / "models"
    info = {}
    for mode, arrays in (("fp32", fp), ("int8", q)):
        p = out / f"{name}_{mode}_weights.npz"
        assert not p.exists(), f"{p} already exists"
        with open(str(p) + ".part", "wb") as f:
            np.savez(f, **arrays)
        os.replace(str(p) + ".part", p)
        info[mode] = {"payload_byte": sum(a.nbytes for a in arrays.values()), "sha256": T.file_sha256(p)}
    info["rmse_relativo_coordinate_int8"] = math.sqrt(err2 / ref2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
