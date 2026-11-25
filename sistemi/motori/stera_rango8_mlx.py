"""Motore compatto di STERA 30M con il termine a basso rango U V^T memorizzato a 8 bit anche in memoria.

Coordinate RowMask23, scale, guadagni, posizioni e normalizzazioni restano FP32 e sono usati dagli stessi kernel
del motore compatto (stera_h80000_fp32_tuned_sdpa_v63.py, prefill_f1.py). Cambia solo il termine a basso rango:
U e V sono quantizzati con il formato di MLX (8 bit, una scala e un offset FP32 ogni 32 valori) e moltiplicati
con mx.quantized_matmul, senza riconvertirli in FP32. Le versioni FP32 di U e V vengono eliminate.
"""
import mlx.core as mx

import stera_h80000_fp32_tuned_sdpa_v63 as C
import stera4b_wht_splitk_transposed_v28 as F
import stera_batch_rowmask_fp32_v38 as BK
import prefill_f1 as PF

GS, BIT = 32, 8


def qmm(x, q):
    w, s, b = q
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=BIT)


def quantizza_rango(m):
    """Sostituisce U, V di ogni proiezione con le loro versioni a 8 bit; restituisce i byte del solo basso rango."""
    m.lr = {}
    for name, (co, sc, gout, gin, U, V) in list(m.w.items()):
        e = {"U": mx.quantize(U, group_size=GS, bits=BIT),          # righe x r, gruppi lungo r: y += t @ U^T
             "Vt": mx.quantize(V.T, group_size=GS, bits=BIT)}       # r x colonne, gruppi lungo le colonne: t = x @ V
        if name == "embedding":
            e["V"] = mx.quantize(V, group_size=GS, bits=BIT)        # colonne x r: riga di U per V^T
        mx.eval(e)
        m.lr[name] = e
        m.w[name] = (co, sc, gout, gin, None, None)
    mx.clear_cache()
    return sum(t.nbytes for e in m.lr.values() for q in e.values() for t in q)


def basso_rango(m, name, x2d):
    e = m.lr[name]
    return qmm(qmm(x2d, e["Vt"]), e["U"])


def riga_embedding(m, ids):
    w, s, b = m.lr["embedding"]["U"]
    u = mx.dequantize(mx.take(w, ids, axis=0), mx.take(s, ids, axis=0), mx.take(b, ids, axis=0), group_size=GS, bits=BIT)
    return qmm(u, m.lr["embedding"]["V"])


def fast_matvec(m, name, x, split=8):
    co, sc, gout, gin, _, _ = m.w[name]
    cols, rows = int(gin.size), int(gout.size)
    vec = x * gin
    xw = F.MT.fwht_metal(vec)
    part = F.KSPLIT(inputs=[co.reshape(-1), sc, vec, xw, m.cow[name].reshape(-1), mx.array([split], mx.int32)],
                    grid=(rows * split, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, split)], output_dtypes=[mx.float32])[0]
    y = mx.sum(part, axis=1) * gout + basso_rango(m, name, x[None])[0]
    return y + m.p[name + ".bias"] if name + ".bias" in m.p else y


def packed_matmul(m, name, x, split=2):
    co, sc, gout, gin, _, _ = m.w[name]
    B, N = x.shape; rows = int(gout.size)
    vec = x * gin[None, :]
    wht = BK.MT.fwht_metal(vec.reshape(-1)).reshape(B, N)
    part = BK.K(inputs=[co.reshape(-1), sc, vec, wht, m.cow[name].reshape(-1), mx.array([split], mx.int32)],
                grid=(rows * split * 8, B // 8, 1), threadgroup=(1024, 1, 1), output_shapes=[(B, rows, split)], output_dtypes=[mx.float32])[0]
    y = mx.sum(part, axis=2) * gout[None, :] + basso_rango(m, name, x)
    return y + m.p[name + ".bias"][None, :] if name + ".bias" in m.p else y


class CompressedRango8(C.CompressedCompiled):
    """Stesso grafo di decode del motore compatto; l'embedding e tutte le proiezioni usano il basso rango a 8 bit."""
    def __init__(self, context=512):
        # il costruttore originale prepara modello e grafo; il grafo e tracciato solo alla prima chiamata,
        # quindi sostituire funzioni e pesi qui, prima di usarlo, cambia il calcolo eseguito
        super().__init__(context)
        m = self.m
        self.byte_rango = quantizza_rango(m)
        F.fast_matvec = fast_matvec; BK.packed_matmul = packed_matmul
        KEMB = F.KEMB_PACKED
        orig_take = mx.take

        position_range = mx.arange(context, dtype=mx.int32)
        mask_vocab = mx.arange(50304, dtype=mx.int32) < 50257

        def ln(x, name):
            d = x - mx.mean(x)
            return d * mx.rsqrt(mx.mean(d * d) + 1e-5) * m.p[name + ".weight"] + m.p[name + ".bias"]

        def st(token, t, oldk, oldv):
            co, sc, gout, gin, _, _ = m.w["embedding"]
            e = KEMB(inputs=[co.reshape(-1), sc, gout, gin, m.map, token.reshape((1,))], grid=(768, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(768,)], output_dtypes=[mx.float32])[0]
            x = e + riga_embedding(m, token.reshape((1,)))[0] + orig_take(m.p["position.weight"], t.reshape((1,)), axis=0)[0]
            nk, nv = [], []
            for l in range(6):
                pre = f"layers.{l}."
                z = ln(x, pre + "ln1")
                q = m.matvec(pre + "q", z).reshape((16, 48)); k = m.matvec(pre + "k", z).reshape((16, 48)); v = m.matvec(pre + "v", z).reshape((16, 48))
                K = mx.slice_update(oldk[l], k.reshape((1, 16, 48)), t.reshape((1,)), axes=[0])
                V = mx.slice_update(oldv[l], v.reshape((1, 16, 48)), t.reshape((1,)), axes=[0])
                nk.append(K); nv.append(V)
                good = position_range <= t
                if l % 2:
                    good = good & (position_range > t - 256)
                mask = mx.where(good, mx.array(0.0, mx.float32), mx.array(-1e9, mx.float32))[None, None, None, :]
                a = mx.fast.scaled_dot_product_attention(q[None, :, None, :], K.transpose(1, 0, 2)[None], V.transpose(1, 0, 2)[None],
                                                         scale=1.0, mask=mask)[0, :, 0, :].reshape((768,))
                x = x + m.matvec(pre + "o", a)
                y = m.matvec(pre + "fc", ln(x, pre + "ln2"))
                y = .5 * y * (1 + mx.tanh(.7978845608028654 * (y + .044715 * y * y * y)))
                x = x + m.matvec(pre + "proj", y)
            logits = m.matvec("embedding", ln(x, "final_norm"))
            return mx.where(mask_vocab, logits, -1e9), tuple(nk), tuple(nv)
        self.compiled = mx.compile(st)


class PrefillRango8(PF.BatchedPrefill):
    """Prefill a blocchi del motore compatto (stessa struttura di prefill_f1.BatchedPrefill) con il basso rango a 8 bit."""
    def __init__(self, capacity=512, BLOCK=64, shared_base=None):
        self.base = shared_base; self.m = shared_base.m; self.batch = BLOCK; self.capacity = capacity
        self.position_mode = "strict"; self.split = 1; self.kv_precision = "fp32"
        self._costruisci(BLOCK, capacity)

    def _costruisci(self, BS, capacity):
        m = self.base.m
        gpos = mx.arange(capacity, dtype=mx.int32); bpos = mx.arange(BS, dtype=mx.int32)

        def embed(ids):
            co, sc, gout, gin, _, _ = m.w["embedding"]
            embs = [F.KEMB_PACKED(inputs=[co.reshape(-1), sc, gout, gin, m.map, mx.take(ids, mx.array([i], mx.int32), axis=0)],
                                  grid=(768, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(768,)], output_dtypes=[mx.float32])[0] for i in range(BS)]
            return mx.stack(embs, axis=0) + riga_embedding(m, ids)

        def norm(x, prefix):
            d = x - mx.mean(x, axis=-1, keepdims=True)
            return d * mx.rsqrt(mx.mean(d * d, axis=-1, keepdims=True) + 1e-5) * m.p[prefix + ".weight"][None, :] + m.p[prefix + ".bias"][None, :]

        def step(ids, t, oldK, oldV):
            ts = t + bpos
            x = embed(ids) + mx.take(m.p["position.weight"], ts, axis=0)
            nk, nv = [], []
            for l in range(6):
                n = f"layers.{l}."
                z = norm(x, n + "ln1")
                q = packed_matmul(m, n + "q", z, 1).reshape((BS, 16, 48)); k = packed_matmul(m, n + "k", z, 1).reshape((BS, 16, 48))
                v = packed_matmul(m, n + "v", z, 1).reshape((BS, 16, 48))
                K = mx.slice_update(oldK[l], k, t.reshape((1,)), axes=[0]); V = mx.slice_update(oldV[l], v, t.reshape((1,)), axes=[0])
                mask = gpos[None, :] <= ts[:, None]
                if l % 2:
                    mask = mask & (gpos[None, :] > ts[:, None] - 256)
                nk.append(K); nv.append(V)
                bias = mx.where(mask, mx.array(0.0, mx.float32), mx.array(-1e9, mx.float32))[None, None, :, :]
                a = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], K.transpose(1, 0, 2)[None], V.transpose(1, 0, 2)[None],
                                                         scale=1.0, mask=bias)[0].transpose(1, 0, 2).reshape((BS, 768))
                x = x + packed_matmul(m, n + "o", a, 1)
                y = packed_matmul(m, n + "fc", norm(x, n + "ln2"), 1)
                y = .5 * y * (1 + mx.tanh(.7978845608028654 * (y + .044715 * y * y * y)))
                x = x + packed_matmul(m, n + "proj", y, 1)
            logits = m.matvec("embedding", norm(x, "final_norm")[-1])
            return mx.where(mx.arange(50304, dtype=mx.int32) < 50257, logits, -1e9), tuple(nk), tuple(nv)
        self.compiled = mx.compile(step)
