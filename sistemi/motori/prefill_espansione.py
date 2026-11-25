"""Prefill di STERA 30M con espansione al volo, strato per strato.

Nel prefill molti token usano gli stessi pesi: conviene ricostruire la parte strutturata di ogni proiezione
(coordinate RowMask23, scala del blocco, guadagni) in una matrice densa temporanea e usare il prodotto
matrice per matrice della GPU. Il termine a basso rango non viene espanso: si applica in forma fattorizzata
(x V) U^T, in FP32 oppure a 8 bit (motore stera_rango8_mlx). Le matrici di uno strato servono solo a quello
strato e vengono liberate dopo l'uso; il decode resta quello del motore compatto.

Il kernel di espansione legge le coordinate nella disposizione del motore compatto (correzioni trasposte
[23 x 127] da stera4b_wht_splitk_transposed_v28.repack_model).
"""
import mlx.core as mx
import numpy as np

import stera4b_wht_splitk_transposed_v28 as F


def mappe_blocco():
    """Per ogni posizione (i, j) del blocco: indice della diagonale o della base XOR, e indice della correzione (-1 se assente)."""
    princ = np.zeros(16384, np.int32); corr = np.full(16384, -1, np.int32)
    for i in range(128):
        for j in range(128):
            s = i ^ j; p = i * 128 + j
            princ[p] = i if s == 0 else 128 + s - 1
    for i in range(1, 128):
        for k in range(23):
            s = 1 + (13 * k + 7 * i) % 127
            corr[i * 128 + (i ^ s)] = 255 + k * 127 + (i - 1)      # disposizione trasposta del motore compatto
    return mx.array(princ), mx.array(corr)


KESPANDI = mx.fast.metal_kernel(
    name="stera_espansione_righe_v1",
    input_names=["coords", "scales", "gout", "gin", "princ", "corr", "righe", "dim"],
    output_names=["W"],
    source=r"""
    uint gid = thread_position_in_grid.x;
    uint cols = dim[0], nrighe = dim[1];
    if (gid >= nrighe * cols) return;
    uint t = gid / cols, c = gid - t * cols;
    uint r = righe[t];
    uint tile = (r >> 7) * (cols >> 7) + (c >> 7);
    uint p = ((r & 127u) << 7) | (c & 127u);
    uint base = tile * 3176u;
    float v = coords[base + princ[p]];
    int ci = corr[p];
    if (ci >= 0) v += coords[base + (uint)ci];
    W[gid] = v * scales[tile] * gout[r] * gin[c];
    """)


def espandi(m, name, righe=None):
    """Parte strutturata della proiezione `name` (senza basso rango); con `righe` solo le righe indicate."""
    co, sc, gout, gin = m.w[name][:4]
    rows, cols = int(gout.size), int(gin.size)
    if righe is None:
        righe = mx.arange(rows, dtype=mx.uint32)
    n = int(righe.size)
    return KESPANDI(inputs=[co.reshape(-1), sc, gout, gin, m.princ, m.corr, righe.astype(mx.uint32), mx.array([cols, n], mx.uint32)],
                    grid=(n * cols, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(n, cols)], output_dtypes=[mx.float32])[0]


class PrefillEspansione:
    def __init__(self, base, capacity=512):
        self.base = base; m = base.m; self.capacity = capacity
        if not hasattr(m, "princ"):
            m.princ, m.corr = mappe_blocco()
        q8 = hasattr(m, "lr")
        if q8:
            import stera_rango8_mlx as R8

        def basso_rango(name, x):
            if q8:
                return R8.basso_rango(m, name, x)
            U, V = m.w[name][4], m.w[name][5]
            return (x @ V) @ U.T

        def riga_emb(ids):
            if q8:
                return R8.riga_embedding(m, ids)
            U, V = m.w["embedding"][4], m.w["embedding"][5]
            return mx.take(U, ids, axis=0) @ V.T

        def lin(name, x):
            y = x @ espandi(m, name).T + basso_rango(name, x)
            return y + m.p[name + ".bias"] if name + ".bias" in m.p else y

        def norm(x, prefix):
            d = x - mx.mean(x, axis=-1, keepdims=True)
            return d * mx.rsqrt(mx.mean(d * d, axis=-1, keepdims=True) + 1e-5) * m.p[prefix + ".weight"] + m.p[prefix + ".bias"]

        pos = mx.arange(capacity, dtype=mx.int32)

        def strato(l, x, t, start, Kl, Vl):
            B = x.shape[0]; n = f"layers.{l}."
            z = norm(x, n + "ln1")
            q, k, v = (lin(n + s, z).reshape(B, 16, 48) for s in "qkv")
            Kl = mx.slice_update(Kl, k, start.reshape(1), axes=[0]); Vl = mx.slice_update(Vl, v, start.reshape(1), axes=[0])
            ok = pos[None, :] <= t[:, None]
            if l % 2:
                ok = ok & (pos[None, :] > t[:, None] - 256)
            mask = mx.where(ok, 0.0, -1e9)[None, None]
            a = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], Kl.transpose(1, 0, 2)[None], Vl.transpose(1, 0, 2)[None],
                                                     scale=1.0, mask=mask)[0].transpose(1, 0, 2).reshape(B, 768)
            x = x + lin(n + "o", a)
            y = lin(n + "fc", norm(x, n + "ln2"))
            y = .5 * y * (1 + mx.tanh(.7978845608028654 * (y + .044715 * y * y * y)))
            return x + lin(n + "proj", y), Kl, Vl

        def ingresso(ids, t):
            return espandi(m, "embedding", ids) + riga_emb(ids) + mx.take(m.p["position.weight"], t, axis=0)

        def uscita(x):
            logits = m.matvec("embedding", norm(x[-1:], "final_norm")[0])
            return mx.where(mx.arange(50304, dtype=mx.int32) < 50257, logits, -1e9)

        # uno strato alla volta: ogni grafo compilato contiene le matrici espanse di un solo strato,
        # che vengono liberate prima di espandere lo strato successivo
        self._strati = [mx.compile(lambda x, t, start, Kl, Vl, l=l: strato(l, x, t, start, Kl, Vl)) for l in range(6)]
        self._ingresso = mx.compile(ingresso); self._uscita = mx.compile(uscita)

    def process(self, ids, t):
        assert t + len(ids) <= self.capacity
        idsa = mx.array(ids, mx.uint32); start = mx.array(t, mx.int32); tt = start + mx.arange(len(ids), dtype=mx.int32)
        x = self._ingresso(idsa, tt); mx.eval(x)
        K, V = list(self.base.k), list(self.base.v)
        for l in range(6):
            x, K[l], V[l] = self._strati[l](x, tt, start, K[l], V[l]); mx.eval(x, K[l], V[l])
        self.base.k, self.base.v = tuple(K), tuple(V)
        return self._uscita(x)
