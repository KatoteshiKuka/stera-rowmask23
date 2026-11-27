"""Dense MLX/Metal engine for decoder-only Transformers with the GPT-Neo structure (and STERA 30M in dense form).

Same optimizations as the dense engine used for STERA 30M (stera_h80000_fp32_dense_sdpa_v62.py and prefill_dense.py):
FP32, resident weights, static KV cache, decode step compiled with mx.compile, SDPA attention without scaling
(as in GPT-Neo), even layers with global attention and odd layers with a local window, compiled block prefill.
The sizes (layers, width, heads, vocabulary) are read from the weights, so the same code runs
dense STERA 30M, TinyStories-8M and TinyStories-33M.
"""
from pathlib import Path

import mlx.core as mx

GELU = 0.7978845608028654


def load_stera_dense(path):
    """Dense-equivalent weights of STERA 30M (H122000_DENSE_FP32.safetensors)."""
    w = mx.load(str(path)); L = 1 + max(int(k.split(".")[1]) for k in w if k.startswith("layers."))
    layers = [{"ln1": (w[f"layers.{l}.ln1.weight"], w[f"layers.{l}.ln1.bias"]), "ln2": (w[f"layers.{l}.ln2.weight"], w[f"layers.{l}.ln2.bias"]),
               "qkv": mx.concatenate([w[f"layers.{l}.{s}"] for s in "qkv"], axis=0), "o": w[f"layers.{l}.o"], "o_b": w[f"layers.{l}.o.bias"],
               "fc": w[f"layers.{l}.fc"], "fc_b": w[f"layers.{l}.fc.bias"], "proj": w[f"layers.{l}.proj"], "proj_b": w[f"layers.{l}.proj.bias"]} for l in range(L)]
    return {"E": w["embedding"][:50257], "pos": w["position.weight"], "layers": layers, "lnf": (w["final_norm.weight"], w["final_norm.bias"]), "heads": 16}


def load_tinystories(folder):
    """Public GPT-Neo TinyStories checkpoint (pytorch_model.bin); the output head is the tied embedding."""
    import json, torch
    cfg = json.loads((Path(folder) / "config.json").read_text())
    sd = torch.load(Path(folder) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    a = lambda k: mx.array(sd[k].float().numpy())
    layers = []
    for l in range(cfg["num_layers"]):
        p = f"transformer.h.{l}."
        layers.append({"ln1": (a(p + "ln_1.weight"), a(p + "ln_1.bias")), "ln2": (a(p + "ln_2.weight"), a(p + "ln_2.bias")),
                       "qkv": mx.concatenate([a(p + f"attn.attention.{s}_proj.weight") for s in "qkv"], axis=0),
                       "o": a(p + "attn.attention.out_proj.weight"), "o_b": a(p + "attn.attention.out_proj.bias"),
                       "fc": a(p + "mlp.c_fc.weight"), "fc_b": a(p + "mlp.c_fc.bias"), "proj": a(p + "mlp.c_proj.weight"), "proj_b": a(p + "mlp.c_proj.bias")})
    assert cfg["window_size"] == 256 and cfg["activation_function"] == "gelu_new"
    return {"E": a("transformer.wte.weight"), "pos": a("transformer.wpe.weight"), "layers": layers,
            "lnf": (a("transformer.ln_f.weight"), a("transformer.ln_f.bias")), "heads": cfg["num_heads"]}


class DenseMLX:
    def __init__(self, weights, capacity=512, window=256):
        self.P = weights; E = weights["E"]; self.L = len(weights["layers"]); self.d = E.shape[1]; self.h = weights["heads"]; self.hd = self.d // self.h
        self.capacity, self.window = capacity, window
        mx.eval(weights)
        self.reset()
        pos = mx.arange(capacity, dtype=mx.int32)
        L, d, h, hd, S = self.L, self.d, self.h, self.hd, weights["layers"]

        def ln(x, wb):
            c = x - mx.mean(x, axis=-1, keepdims=True)
            return c * mx.rsqrt(mx.mean(c * c, axis=-1, keepdims=True) + 1e-5) * wb[0] + wb[1]

        def step_fn(token, t, K, V):
            x = E[token] + weights["pos"][t]
            nk, nv = [], []
            for l in range(L):
                s = S[l]
                qkv = (s["qkv"] @ ln(x, s["ln1"])).reshape(3, h, hd)
                k = mx.slice_update(K[l], qkv[1].reshape(1, h, hd), t.reshape(1), axes=[0])
                v = mx.slice_update(V[l], qkv[2].reshape(1, h, hd), t.reshape(1), axes=[0])
                nk.append(k); nv.append(v)
                ok = pos <= t
                if l % 2:
                    ok = ok & (pos > t - window)
                m = mx.where(ok, 0.0, -1e9)[None, None, None, :]
                a = mx.fast.scaled_dot_product_attention(qkv[0][None, :, None, :], k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None],
                                                         scale=1.0, mask=m)[0, :, 0, :].reshape(d)
                x = x + s["o"] @ a + s["o_b"]
                y = s["fc"] @ ln(x, s["ln2"]) + s["fc_b"]
                y = 0.5 * y * (1 + mx.tanh(GELU * (y + 0.044715 * y * y * y)))
                x = x + s["proj"] @ y + s["proj_b"]
            return E @ ln(x, weights["lnf"]), tuple(nk), tuple(nv)

        self._step = mx.compile(step_fn)
        self._prefill = {}

        def build_prefill(B):
            def pf(ids, start, K, V):
                t = start + mx.arange(B, dtype=mx.int32)
                x = E[ids] + weights["pos"][t]
                nk, nv = [], []
                for l in range(L):
                    s = S[l]
                    qkv = (ln(x, s["ln1"]) @ s["qkv"].T).reshape(B, 3, h, hd)
                    k = mx.slice_update(K[l], qkv[:, 1], start.reshape(1), axes=[0])
                    v = mx.slice_update(V[l], qkv[:, 2], start.reshape(1), axes=[0])
                    nk.append(k); nv.append(v)
                    ok = pos[None, :] <= t[:, None]
                    if l % 2:
                        ok = ok & (pos[None, :] > t[:, None] - window)
                    m = mx.where(ok, 0.0, -1e9)[None, None]
                    a = mx.fast.scaled_dot_product_attention(qkv[:, 0].transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
                                                             v.transpose(1, 0, 2)[None], scale=1.0, mask=m)[0].transpose(1, 0, 2).reshape(B, d)
                    x = x + a @ s["o"].T + s["o_b"]
                    y = ln(x, s["ln2"]) @ s["fc"].T + s["fc_b"]
                    y = 0.5 * y * (1 + mx.tanh(GELU * (y + 0.044715 * y * y * y)))
                    x = x + y @ s["proj"].T + s["proj_b"]
                return E @ ln(x[-1], weights["lnf"]), tuple(nk), tuple(nv)
            return mx.compile(pf)
        self._build_prefill = build_prefill

    def reset(self):
        self.K = tuple(mx.zeros((self.capacity, self.h, self.hd), mx.float32) for _ in range(self.L))
        self.V = tuple(mx.zeros((self.capacity, self.h, self.hd), mx.float32) for _ in range(self.L))
        mx.eval(self.K, self.V)

    def step(self, token, t):
        lg, self.K, self.V = self._step(token, mx.array(t, mx.int32), self.K, self.V)
        return lg

    def prefill(self, ids, offset):
        B = len(ids)
        if B not in self._prefill:
            self._prefill[B] = self._build_prefill(B)
        lg, self.K, self.V = self._prefill[B](mx.array(ids, mx.int32), mx.array(offset, mx.int32), self.K, self.V)
        return lg


# names used in the archived logs and in older scripts
carica_stera_denso, carica_tinystories, DensoMLX = load_stera_dense, load_tinystories, DenseMLX
