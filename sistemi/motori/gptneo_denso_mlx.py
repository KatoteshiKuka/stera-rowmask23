"""Motore denso MLX/Metal per Transformer decoder-only con la struttura di GPT-Neo (e di STERA 30M in forma densa).

Stesse ottimizzazioni del motore denso usato per STERA 30M (stera_h80000_fp32_dense_sdpa_v62.py e prefill_dense.py):
FP32, pesi residenti, cache KV statica, passo di decode compilato con mx.compile, attenzione SDPA senza scala
(come GPT-Neo), strati pari con attenzione globale e dispari con finestra locale, prefill a blocchi compilato.
Le dimensioni (strati, larghezza, teste, vocabolario) sono lette dai pesi, cosi lo stesso codice esegue
STERA 30M denso, TinyStories-8M e TinyStories-33M.
"""
from pathlib import Path

import mlx.core as mx

GELU = 0.7978845608028654


def carica_stera_denso(path):
    """Pesi densi equivalenti di STERA 30M (H122000_DENSE_FP32.safetensors)."""
    w = mx.load(str(path)); L = 1 + max(int(k.split(".")[1]) for k in w if k.startswith("layers."))
    strati = [{"ln1": (w[f"layers.{l}.ln1.weight"], w[f"layers.{l}.ln1.bias"]), "ln2": (w[f"layers.{l}.ln2.weight"], w[f"layers.{l}.ln2.bias"]),
               "qkv": mx.concatenate([w[f"layers.{l}.{s}"] for s in "qkv"], axis=0), "o": w[f"layers.{l}.o"], "o_b": w[f"layers.{l}.o.bias"],
               "fc": w[f"layers.{l}.fc"], "fc_b": w[f"layers.{l}.fc.bias"], "proj": w[f"layers.{l}.proj"], "proj_b": w[f"layers.{l}.proj.bias"]} for l in range(L)]
    return {"E": w["embedding"][:50257], "pos": w["position.weight"], "strati": strati, "lnf": (w["final_norm.weight"], w["final_norm.bias"]), "teste": 16}


def carica_tinystories(cartella):
    """Checkpoint pubblico GPT-Neo di TinyStories (pytorch_model.bin); la testa di uscita e l'embedding condiviso."""
    import json, torch
    cfg = json.loads((Path(cartella) / "config.json").read_text())
    sd = torch.load(Path(cartella) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    a = lambda k: mx.array(sd[k].float().numpy())
    strati = []
    for l in range(cfg["num_layers"]):
        p = f"transformer.h.{l}."
        strati.append({"ln1": (a(p + "ln_1.weight"), a(p + "ln_1.bias")), "ln2": (a(p + "ln_2.weight"), a(p + "ln_2.bias")),
                       "qkv": mx.concatenate([a(p + f"attn.attention.{s}_proj.weight") for s in "qkv"], axis=0),
                       "o": a(p + "attn.attention.out_proj.weight"), "o_b": a(p + "attn.attention.out_proj.bias"),
                       "fc": a(p + "mlp.c_fc.weight"), "fc_b": a(p + "mlp.c_fc.bias"), "proj": a(p + "mlp.c_proj.weight"), "proj_b": a(p + "mlp.c_proj.bias")})
    assert cfg["window_size"] == 256 and cfg["activation_function"] == "gelu_new"
    return {"E": a("transformer.wte.weight"), "pos": a("transformer.wpe.weight"), "strati": strati,
            "lnf": (a("transformer.ln_f.weight"), a("transformer.ln_f.bias")), "teste": cfg["num_heads"]}


class DensoMLX:
    def __init__(self, pesi, capacita=512, finestra=256):
        self.P = pesi; E = pesi["E"]; self.L = len(pesi["strati"]); self.d = E.shape[1]; self.h = pesi["teste"]; self.hd = self.d // self.h
        self.capacita, self.finestra = capacita, finestra
        mx.eval(pesi)
        self.azzera()
        pos = mx.arange(capacita, dtype=mx.int32)
        L, d, h, hd, S = self.L, self.d, self.h, self.hd, pesi["strati"]

        def ln(x, wb):
            c = x - mx.mean(x, axis=-1, keepdims=True)
            return c * mx.rsqrt(mx.mean(c * c, axis=-1, keepdims=True) + 1e-5) * wb[0] + wb[1]

        def passo(token, t, K, V):
            x = E[token] + pesi["pos"][t]
            nk, nv = [], []
            for l in range(L):
                s = S[l]
                qkv = (s["qkv"] @ ln(x, s["ln1"])).reshape(3, h, hd)
                k = mx.slice_update(K[l], qkv[1].reshape(1, h, hd), t.reshape(1), axes=[0])
                v = mx.slice_update(V[l], qkv[2].reshape(1, h, hd), t.reshape(1), axes=[0])
                nk.append(k); nv.append(v)
                ok = pos <= t
                if l % 2:
                    ok = ok & (pos > t - finestra)
                m = mx.where(ok, 0.0, -1e9)[None, None, None, :]
                a = mx.fast.scaled_dot_product_attention(qkv[0][None, :, None, :], k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None],
                                                         scale=1.0, mask=m)[0, :, 0, :].reshape(d)
                x = x + s["o"] @ a + s["o_b"]
                y = s["fc"] @ ln(x, s["ln2"]) + s["fc_b"]
                y = 0.5 * y * (1 + mx.tanh(GELU * (y + 0.044715 * y * y * y)))
                x = x + s["proj"] @ y + s["proj_b"]
            return E @ ln(x, pesi["lnf"]), tuple(nk), tuple(nv)

        self._passo = mx.compile(passo)
        self._prefill = {}

        def costruisci_prefill(B):
            def pf(ids, start, K, V):
                t = start + mx.arange(B, dtype=mx.int32)
                x = E[ids] + pesi["pos"][t]
                nk, nv = [], []
                for l in range(L):
                    s = S[l]
                    qkv = (ln(x, s["ln1"]) @ s["qkv"].T).reshape(B, 3, h, hd)
                    k = mx.slice_update(K[l], qkv[:, 1], start.reshape(1), axes=[0])
                    v = mx.slice_update(V[l], qkv[:, 2], start.reshape(1), axes=[0])
                    nk.append(k); nv.append(v)
                    ok = pos[None, :] <= t[:, None]
                    if l % 2:
                        ok = ok & (pos[None, :] > t[:, None] - finestra)
                    m = mx.where(ok, 0.0, -1e9)[None, None]
                    a = mx.fast.scaled_dot_product_attention(qkv[:, 0].transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
                                                             v.transpose(1, 0, 2)[None], scale=1.0, mask=m)[0].transpose(1, 0, 2).reshape(B, d)
                    x = x + a @ s["o"].T + s["o_b"]
                    y = ln(x, s["ln2"]) @ s["fc"].T + s["fc_b"]
                    y = 0.5 * y * (1 + mx.tanh(GELU * (y + 0.044715 * y * y * y)))
                    x = x + y @ s["proj"].T + s["proj_b"]
                return E @ ln(x[-1], pesi["lnf"]), tuple(nk), tuple(nv)
            return mx.compile(pf)
        self._costruisci_prefill = costruisci_prefill

    def azzera(self):
        self.K = tuple(mx.zeros((self.capacita, self.h, self.hd), mx.float32) for _ in range(self.L))
        self.V = tuple(mx.zeros((self.capacita, self.h, self.hd), mx.float32) for _ in range(self.L))
        mx.eval(self.K, self.V)

    def step(self, token, t):
        lg, self.K, self.V = self._passo(token, mx.array(t, mx.int32), self.K, self.V)
        return lg

    def prefill(self, ids, offset):
        B = len(ids)
        if B not in self._prefill:
            self._prefill[B] = self._costruisci_prefill(B)
        lg, self.K, self.V = self._prefill[B](mx.array(ids, mx.int32), mx.array(offset, mx.int32), self.K, self.V)
        return lg
