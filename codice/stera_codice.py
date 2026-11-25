#!/usr/bin/env python3
"""STERA per il codice: Transformer decoder-only con tutte le proiezioni RowMask23.

Stessa parametrizzazione di STERA 30M (blocchi 128x128 da 3.176 coordinate, guadagni per riga e colonna,
termine a basso rango), con dimensioni configurabili. La ricostruzione dei blocchi e quella di
addestramento/ts68_core.py (RowMask23), quindi le coordinate sono gli unici parametri delle matrici.

Dati: cartella con train.bin e val.bin (token uint16 concatenati, documenti separati da EOS) e meta.json.
Ogni passo usa finestre contigue di ctx+1 token; l'ordine delle finestre e una permutazione deterministica,
quindi la ripresa da un checkpoint continua esattamente la stessa sequenza.

Avvio: torchrun --nproc_per_node=N stera_codice.py --data DATI --output RUN [opzioni]
Con --dati-sintetici si misura la velocita senza dati reali.
"""
import argparse, hashlib, json, math, os, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "addestramento"))
from ts68_core import RowMask23, COORD_COUNT  # stessa ricostruzione di STERA 30M

DENSO_PER_BLOCCO = 128 * 128


def configurazione(a):
    return {"strati": a.strati, "d": a.d, "ffn": a.ffn, "teste": a.teste, "vocab": a.vocab, "ctx": a.ctx, "rango": a.rango}


def matrici(c):
    vp = -(-c["vocab"] // 128) * 128
    out = [("embedding", vp, c["d"])]
    for l in range(c["strati"]):
        n = f"strati.{l}."
        out += [(n + k, c["d"], c["d"]) for k in ("q", "k", "v", "o")]
        out += [(n + "fc", c["ffn"], c["d"]), (n + "proj", c["d"], c["ffn"])]
    return out


def pesi_iniziali(c, seme=20261125):
    g = torch.Generator().manual_seed(seme)
    W = OrderedDict()
    sd = 0.02 / math.sqrt(1 + 2921 / DENSO_PER_BLOCCO)   # stessa scala iniziale di STERA 30M
    for n, r, k in matrici(c):
        nt = (r // 128) * (k // 128)
        W[n + ".coords"] = torch.randn(nt, COORD_COUNT, generator=g) * sd
        W[n + ".scales"] = torch.ones(nt, 1, 1)
        W[n + ".gout"] = torch.ones(r)
        W[n + ".gin"] = torch.ones(k)
        if c["rango"]:
            W[n + ".U"] = torch.zeros(r, c["rango"])
            W[n + ".V"] = torch.randn(k, c["rango"], generator=g) * 0.02
    W["posizioni"] = torch.randn(c["ctx"], c["d"], generator=g) * 0.02
    for n in ["norma_finale"] + [f"strati.{l}.{k}" for l in range(c["strati"]) for k in ("ln1", "ln2")]:
        W[n + ".weight"] = torch.ones(c["d"])
        W[n + ".bias"] = torch.zeros(c["d"])
    for l in range(c["strati"]):
        n = f"strati.{l}."
        W[n + "o.bias"] = torch.zeros(c["d"])
        W[n + "fc.bias"] = torch.zeros(c["ffn"])
        W[n + "proj.bias"] = torch.zeros(c["d"])
    return W


class SteraCodice:
    def __init__(self, c, pesi, device, ricalcolo=False):
        self.c, self.device, self.ricalcolo = c, device, ricalcolo
        self.weight = OrderedDict((k, torch.nn.Parameter(v.to(device, torch.float32))) for k, v in pesi.items())
        self.rm = RowMask23(device)

    def w(self, nome, righe, colonne):
        P = self.weight
        W = self.rm.reconstruct(P[nome + ".coords"], P[nome + ".scales"], righe, colonne)
        W = W * P[nome + ".gout"].reshape(righe, 1) * P[nome + ".gin"].reshape(1, colonne)
        if self.c["rango"]:
            W = W + P[nome + ".U"] @ P[nome + ".V"].t()
        return W

    def lineare(self, x, nome, righe, colonne, bias):
        y = F.linear(x, self.w(nome, righe, colonne))
        return y + self.weight[nome + ".bias"] if bias else y

    def ln(self, x, nome):
        return F.layer_norm(x, (self.c["d"],), self.weight[nome + ".weight"], self.weight[nome + ".bias"], eps=1e-5)

    def strato(self, x, l):
        c = self.c; d, h = c["d"], c["teste"]; B, T, _ = x.shape; n = f"strati.{l}."
        z = self.ln(x, n + "ln1")
        q, k, v = (self.lineare(z, n + s, d, d, False).view(B, T, h, d // h).transpose(1, 2) for s in "qkv")
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.lineare(a.transpose(1, 2).reshape(B, T, d), n + "o", d, d, True)
        y = F.gelu(self.lineare(self.ln(x, n + "ln2"), n + "fc", c["ffn"], d, True), approximate="tanh")
        return x + self.lineare(y, n + "proj", d, c["ffn"], True)

    def logits(self, ids):
        c = self.c; B, T = ids.shape
        vp = -(-c["vocab"] // 128) * 128
        emb = self.w("embedding", vp, c["d"])
        x = F.embedding(ids, emb) + self.weight["posizioni"][:T]
        for l in range(c["strati"]):
            x = checkpoint(self.strato, x, l, use_reentrant=False) if self.ricalcolo else self.strato(x, l)
        return F.linear(self.ln(x, "norma_finale"), emb[:c["vocab"]])


class Dati:
    """Finestre contigue di ctx+1 token da un file uint16; ordine dato da una permutazione affine deterministica."""
    def __init__(self, cartella, ctx, sintetici=0, vocab=49152):
        self.ctx = ctx
        if sintetici:
            g = np.random.default_rng(0)
            self.train = g.integers(0, vocab, sintetici, dtype=np.uint16)
            self.val = g.integers(0, vocab, ctx * 64 + 1, dtype=np.uint16)
        else:
            cartella = Path(cartella)
            self.train = np.memmap(cartella / "train.bin", dtype=np.uint16, mode="r")
            self.val = np.memmap(cartella / "val.bin", dtype=np.uint16, mode="r")
        self.n = (len(self.train) - 1) // ctx
        self.passo = 1_000_003
        while math.gcd(self.passo, self.n) != 1:
            self.passo += 2

    def finestra(self, indice):
        s = ((indice * self.passo + 12_345) % self.n) * self.ctx
        z = np.asarray(self.train[s:s + self.ctx + 1], dtype=np.int64)
        return z[:-1], z[1:]

    def lotto(self, primo, quanti):
        xs, ys = zip(*(self.finestra(primo + j) for j in range(quanti)))
        return np.stack(xs), np.stack(ys)


def tasso(t, picco, riscaldamento, totale, frazione):
    """Warmup lineare, costante, poi discesa lineare fino al 10% nell'ultima frazione dei passi (come STERA 30M)."""
    if t < riscaldamento:
        return picco * (t + 1) / riscaldamento
    d0 = int(totale * (1 - frazione))
    return picco if t < d0 else picco * (1 - 0.9 * min(1.0, (t - d0) / max(1, totale - d0)))


def con_decadimento(nome):
    return nome.endswith((".coords", ".U", ".V")) or nome == "posizioni"


@torch.no_grad()
def nll_val(modello, dati, finestre, micro, dev):
    tot = cnt = 0.0
    n = min(finestre, (len(dati.val) - 1) // dati.ctx)
    for i in range(0, n, micro):
        idx = range(i, min(n, i + micro))
        z = np.stack([np.asarray(dati.val[j * dati.ctx:(j + 1) * dati.ctx + 1], dtype=np.int64) for j in idx])
        x, y = torch.from_numpy(z[:, :-1]).to(dev), torch.from_numpy(z[:, 1:]).to(dev)
        lg = modello.logits(x).float()
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum")); cnt += y.numel()
    return tot / cnt


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def salva(cartella, stato):
    cartella.mkdir(parents=True, exist_ok=True)
    out = cartella / f"STERA_CODICE_PASSO{stato['passo']}.pt"
    tmp = out.with_suffix(".pt.partial")
    torch.save({k: (OrderedDict((n, t.detach().cpu()) for n, t in v.items()) if isinstance(v, dict) and k in ("pesi", "m", "v") else v)
                for k, v in stato.items()}, tmp)
    tmp.rename(out)
    return out, sha256(out)


def main():
    p = argparse.ArgumentParser()
    for nome, tipo, default in [("--data", str, ""), ("--output", str, None), ("--riprendi", str, ""), ("--passi", int, 10**9),
                                ("--passi-totali", int, 20000), ("--lotto", int, 256), ("--micro", int, 4), ("--ctx", int, 1024),
                                ("--strati", int, 24), ("--d", int, 1024), ("--ffn", int, 4096), ("--teste", int, 16), ("--vocab", int, 49152),
                                ("--rango", int, 96), ("--lr", float, 3e-4), ("--riscaldamento", int, 1000), ("--frazione-discesa", float, 0.2),
                                ("--wd", float, 0.1), ("--beta2", float, 0.95), ("--salva-ogni", int, 1000), ("--val-finestre", int, 256),
                                ("--dati-sintetici", int, 0), ("--stampa-ogni", int, 10)]:
        p.add_argument(nome, type=tipo, default=default, required=nome == "--output")
    p.add_argument("--ricalcolo", action="store_true", help="ricalcola le attivazioni di ogni strato nel passo all'indietro")
    a = p.parse_args()

    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(dev)
    torch.manual_seed(20261125)

    c = configurazione(a)
    if a.riprendi:
        stato = torch.load(a.riprendi, map_location="cpu", weights_only=False)
        assert stato["configurazione"] == c, "configurazione diversa dal checkpoint"
    else:
        pesi = pesi_iniziali(c)
        stato = {"configurazione": c, "pesi": pesi, "m": OrderedDict((k, torch.zeros_like(v)) for k, v in pesi.items()),
                 "v": OrderedDict((k, torch.zeros_like(v)) for k, v in pesi.items()), "passo": 0, "token": 0}
    modello = SteraCodice(c, stato["pesi"], dev, a.ricalcolo)
    m = OrderedDict((k, v.to(dev)) for k, v in stato["m"].items()); v2 = OrderedDict((k, v.to(dev)) for k, v in stato["v"].items())
    nomi = list(modello.weight)
    dati = Dati(a.data, a.ctx, a.dati_sintetici, a.vocab)
    assert a.lotto % (world * a.micro) == 0, "il lotto deve essere multiplo di processi x micro"

    if rank == 0:
        salvati = sum(t.numel() for t in modello.weight.values())
        denso = sum(r * k for _, r, k in matrici(c)) + sum(t.numel() for n, t in modello.weight.items() if not n.split(".")[-1] in ("coords", "scales", "gout", "gin", "U", "V"))
        print(json.dumps({"configurazione": c, "parametri_salvati": salvati, "parametri_densi_equivalenti": denso, "processi": world,
                          "passo_iniziale": stato["passo"], **{k: v for k, v in vars(a).items() if k not in ("data",)}}), flush=True)
        if not a.dati_sintetici:
            print(f"VAL passo={stato['passo']} nll={nll_val(modello, dati, a.val_finestre, a.micro, dev):.6f}", flush=True)

    t0, tok0, t = time.time(), stato["token"], 0
    while t < a.passi:
        passo = stato["passo"]
        lr = tasso(passo, a.lr, a.riscaldamento, a.passi_totali, a.frazione_discesa)
        per_proc = a.lotto // world
        x, y = dati.lotto(passo * a.lotto + rank * per_proc, per_proc)
        for w in modello.weight.values():
            w.grad = None
        L = torch.zeros((), device=dev)
        for i in range(0, per_proc, a.micro):
            xs, ys = torch.from_numpy(x[i:i + a.micro]).to(dev), torch.from_numpy(y[i:i + a.micro]).to(dev)
            lg = modello.logits(xs)
            Lm = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ys.reshape(-1), reduction="sum") / (a.lotto * a.ctx)
            Lm.backward(); L += Lm.detach(); del lg, Lm
        with torch.no_grad():
            piatto = torch.cat([modello.weight[k].grad.reshape(-1) for k in nomi] + [L.reshape(1)])
            if world > 1:
                dist.all_reduce(piatto)
            perdita = float(piatto[-1]); off = 0; grad = []
            for k in nomi:
                q = modello.weight[k].numel(); grad.append(piatto[off:off + q].view_as(modello.weight[k])); off += q
            norma = torch.sqrt(sum(torch.sum(g * g) for g in grad)); fac = torch.clamp(1.0 / (norma + 1e-6), max=1.0)
            c1, c2 = 1 - 0.9 ** (passo + 1), 1 - a.beta2 ** (passo + 1)
            for k, g in zip(nomi, grad):
                w = modello.weight[k]; g = g * fac
                m[k].mul_(0.9).add_(g, alpha=0.1); v2[k].mul_(a.beta2).addcmul_(g, g, value=1 - a.beta2)
                u = (m[k] / c1) / (torch.sqrt(v2[k] / c2) + 1e-8)
                if con_decadimento(k):
                    u = u + a.wd * w
                w.add_(u, alpha=-lr)
        stato["passo"] += 1; stato["token"] += a.lotto * a.ctx; t += 1
        if rank == 0 and t % a.stampa_ogni == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated(dev) / 2**30 if cuda else 0
            print(f"PASSO {stato['passo']} token={stato['token']} perdita={perdita:.5f} norma={float(norma):.4f} lr={lr:.6g} "
                  f"token_s={(stato['token'] - tok0) / el:.0f} memoria_gib={mem:.2f}", flush=True)
        if t % a.salva_ogni == 0 or t == a.passi:
            dist.barrier()
            if rank == 0 and not a.dati_sintetici:
                stato.update(pesi=modello.weight, m=m, v=v2)
                out, h = salva(Path(a.output), stato)
                print(f"CHECKPOINT {out} sha256={h}", flush=True)
                print(f"VAL passo={stato['passo']} nll={nll_val(modello, dati, a.val_finestre, a.micro, dev):.6f}", flush=True)
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
