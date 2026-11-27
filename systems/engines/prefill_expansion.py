"""Prefill of STERA 30M with on-the-fly expansion, one layer at a time.

In the prefill many tokens use the same weights: it pays to rebuild the structured part of every projection
(RowMask23 coordinates, block scale, gains) into a temporary dense matrix and use the GPU matrix-matrix
product. The low-rank term is not expanded: it is applied in factored form (x V) U^T, in FP32 or at 8 bits
(engine stera_rank8_mlx). The matrices of a layer are only needed by that layer and are freed after use;
the decode is the one of the compact engine.

The expansion kernel reads the coordinates in the layout of the compact engine (transposed corrections
[23 x 127] from stera4b_wht_splitk_transposed_v28.repack_model).
"""
import mlx.core as mx
import numpy as np

import stera4b_wht_splitk_transposed_v28 as F


def block_maps():
    """For every position (i, j) of the block: index of the diagonal or XOR-base coordinate, and index of the correction (-1 if none)."""
    princ = np.zeros(16384, np.int32); corr = np.full(16384, -1, np.int32)
    for i in range(128):
        for j in range(128):
            s = i ^ j; p = i * 128 + j
            princ[p] = i if s == 0 else 128 + s - 1
    for i in range(1, 128):
        for k in range(23):
            s = 1 + (13 * k + 7 * i) % 127
            corr[i * 128 + (i ^ s)] = 255 + k * 127 + (i - 1)      # transposed layout of the compact engine
    return mx.array(princ), mx.array(corr)


KEXPAND = mx.fast.metal_kernel(
    name="stera_expand_rows_v1",
    input_names=["coords", "scales", "gout", "gin", "princ", "corr", "row_ids", "dim"],
    output_names=["W"],
    source=r"""
    uint gid = thread_position_in_grid.x;
    uint cols = dim[0], nrows = dim[1];
    if (gid >= nrows * cols) return;
    uint t = gid / cols, c = gid - t * cols;
    uint r = row_ids[t];
    uint tile = (r >> 7) * (cols >> 7) + (c >> 7);
    uint p = ((r & 127u) << 7) | (c & 127u);
    uint base = tile * 3176u;
    float v = coords[base + princ[p]];
    int ci = corr[p];
    if (ci >= 0) v += coords[base + (uint)ci];
    W[gid] = v * scales[tile] * gout[r] * gin[c];
    """)


def expand(m, name, row_ids=None):
    """Structured part of the projection `name` (without the low-rank term); with `row_ids`, only those rows."""
    co, sc, gout, gin = m.w[name][:4]
    rows, cols = int(gout.size), int(gin.size)
    if row_ids is None:
        row_ids = mx.arange(rows, dtype=mx.uint32)
    n = int(row_ids.size)
    return KEXPAND(inputs=[co.reshape(-1), sc, gout, gin, m.princ, m.corr, row_ids.astype(mx.uint32), mx.array([cols, n], mx.uint32)],
                    grid=(n * cols, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(n, cols)], output_dtypes=[mx.float32])[0]


class PrefillExpansion:
    def __init__(self, base, capacity=512):
        self.base = base; m = base.m; self.capacity = capacity
        if not hasattr(m, "princ"):
            m.princ, m.corr = block_maps()
        q8 = hasattr(m, "lr")
        if q8:
            import stera_rank8_mlx as R8

        def low_rank(name, x):
            if q8:
                return R8.low_rank(m, name, x)
            U, V = m.w[name][4], m.w[name][5]
            return (x @ V) @ U.T

        def emb_row(ids):
            if q8:
                return R8.embedding_row(m, ids)
            U, V = m.w["embedding"][4], m.w["embedding"][5]
            return mx.take(U, ids, axis=0) @ V.T

        def lin(name, x):
            y = x @ expand(m, name).T + low_rank(name, x)
            return y + m.p[name + ".bias"] if name + ".bias" in m.p else y

        def norm(x, prefix):
            d = x - mx.mean(x, axis=-1, keepdims=True)
            return d * mx.rsqrt(mx.mean(d * d, axis=-1, keepdims=True) + 1e-5) * m.p[prefix + ".weight"] + m.p[prefix + ".bias"]

        pos = mx.arange(capacity, dtype=mx.int32)

        def layer(l, x, t, start, Kl, Vl):
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

        def embed_in(ids, t):
            return expand(m, "embedding", ids) + emb_row(ids) + mx.take(m.p["position.weight"], t, axis=0)

        def head_out(x):
            logits = m.matvec("embedding", norm(x[-1:], "final_norm")[0])
            return mx.where(mx.arange(50304, dtype=mx.int32) < 50257, logits, -1e9)

        # one layer at a time: every compiled graph holds the expanded matrices of a single layer,
        # which are freed before the next layer is expanded
        self._layers = [mx.compile(lambda x, t, start, Kl, Vl, l=l: layer(l, x, t, start, Kl, Vl)) for l in range(6)]
        self._embed_in = mx.compile(embed_in); self._head_out = mx.compile(head_out)

    def process(self, ids, t):
        assert t + len(ids) <= self.capacity
        idsa = mx.array(ids, mx.uint32); start = mx.array(t, mx.int32); tt = start + mx.arange(len(ids), dtype=mx.int32)
        x = self._embed_in(idsa, tt); mx.eval(x)
        K, V = list(self.base.k), list(self.base.v)
        for l in range(6):
            x, K[l], V[l] = self._layers[l](x, tt, start, K[l], V[l]); mx.eval(x, K[l], V[l])
        self.base.k, self.base.v = tuple(K), tuple(V)
        return self._head_out(x)


PrefillEspansione = PrefillExpansion   # name used in the archived logs
