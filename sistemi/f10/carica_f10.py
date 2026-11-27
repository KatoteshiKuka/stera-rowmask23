"""Carica nel motore compatto MLX il modello fittizio F10 generato da genera_f10.py.

Sostituisce la costruzione al volo in GPU (stera_independent_fp32_variable_width_v23.DirectIndependentModel)
con la lettura del file generato, così che le misure usino esattamente il modello riproducibile.
"""
import gc, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import mlx.core as mx
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "motori"))
import stera4b_stream_mlx as S
import stera4b_wht_sparse_v3 as H

FILE = Path(os.environ.get("STERA_F10", R / "modelli" / "F10_H122000.npz"))
BASE = S.BASE


class F10DaFile(S.CoordModel):
    def __init__(self, f=10):
        assert f == 10
        man = json.loads(FILE.with_suffix(".json").read_text())
        self.f = f; self.h = 768 * f; self.ffn = 3072 * f; self.nhead = 16 * f
        self.map = S.make_map(); self.p = {}; self.w = {}; self.cow = {}; self.kv = [None] * 6
        with np.load(FILE) as z:
            for k in z.files:
                if not any(k.endswith(s) for s in (".coords", ".scales", ".gout", ".gin", ".U", ".V")):
                    self.p[k] = mx.array(z[k])
            for name, _, _ in BASE:
                parts = tuple(mx.array(z[name + s]) for s in (".coords", ".scales", ".gout", ".gin", ".U", ".V"))
                self.w[name] = parts
                co = parts[0]
                self.cow[name] = H.fwht(mx.concatenate([mx.zeros((co.shape[0], 1), mx.float32), co[:, 128:255]], axis=1))
                mx.eval(*parts, self.cow[name]); gc.collect()
        mx.eval(*self.p.values())
        self.weight_bytes = sum(x.nbytes for parts in self.w.values() for x in parts) + sum(x.nbytes for x in self.p.values())
        self.unique_coord_count = sum(v[0].size for v in self.w.values())
        self.sourcehash = man["sha256"]
        assert self.weight_bytes == man["byte_pesi_fp32"], (self.weight_bytes, man["byte_pesi_fp32"])
        print("F10_DA_FILE", FILE.name, man["sha256"][:12], "byte", self.weight_bytes, "MLX", S.get_mem(), flush=True)
