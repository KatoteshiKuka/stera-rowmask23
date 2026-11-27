#!/usr/bin/env python3
"""STERA for code: decoder-only Transformer with every projection parametrized with RowMask23 blocks.

Same parametrization as STERA 30M (128x128 blocks with 3,176 coordinates, row and column gains,
low-rank term), with configurable sizes. The block reconstruction is the one of
training/ts68_core.py (RowMask23), so the coordinates are the only parameters of the matrices.

Data: folder with train.bin and val.bin (concatenated uint16 tokens, documents separated by EOS).
Every step uses contiguous windows of ctx+1 tokens; the window order is a deterministic permutation,
so resuming from a checkpoint continues exactly the same sequence.

Launch: torchrun --nproc_per_node=N stera_code.py --data DATA --output RUN [options]
With --synthetic-data N the throughput is measured without real data.
Checkpoint keys and parameter names (configurazione, pesi, passo, strati.*, posizioni, norma_finale) are kept
unchanged so that existing checkpoints remain readable; the Italian option names are accepted as aliases.
"""
import argparse, hashlib, json, math, os, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from ts68_core import RowMask23, COORD_COUNT  # same reconstruction as STERA 30M

DENSE_PER_BLOCK = 128 * 128


def configuration(a):
    return {"strati": a.layers, "d": a.d, "ffn": a.ffn, "teste": a.heads, "vocab": a.vocab, "ctx": a.ctx, "rango": a.rank}


def matrices(c):
    vp = -(-c["vocab"] // 128) * 128
    out = [("embedding", vp, c["d"])]
    for l in range(c["strati"]):
        n = f"strati.{l}."
        out += [(n + k, c["d"], c["d"]) for k in ("q", "k", "v", "o")]
        out += [(n + "fc", c["ffn"], c["d"]), (n + "proj", c["d"], c["ffn"])]
    return out


def initial_weights(c, seed=20261125):
    g = torch.Generator().manual_seed(seed)
    W = OrderedDict()
    sd = 0.02 / math.sqrt(1 + 2921 / DENSE_PER_BLOCK)   # same initial scale as STERA 30M
    for n, r, k in matrices(c):
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


class SteraCode:
    def __init__(self, c, weights, device, recompute=False):
        self.c, self.device, self.recompute = c, device, recompute
        self.weight = OrderedDict((k, torch.nn.Parameter(v.to(device, torch.float32))) for k, v in weights.items())
        self.rm = RowMask23(device)

    def w(self, name, rows, cols):
        P = self.weight
        W = self.rm.reconstruct(P[name + ".coords"], P[name + ".scales"], rows, cols)
        W = W * P[name + ".gout"].reshape(rows, 1) * P[name + ".gin"].reshape(1, cols)
        if self.c["rango"]:
            W = W + P[name + ".U"] @ P[name + ".V"].t()
        return W

    def linear(self, x, name, rows, cols, bias):
        y = F.linear(x, self.w(name, rows, cols))
        return y + self.weight[name + ".bias"] if bias else y

    def ln(self, x, name):
        return F.layer_norm(x, (self.c["d"],), self.weight[name + ".weight"], self.weight[name + ".bias"], eps=1e-5)

    def layer(self, x, l):
        c = self.c; d, h = c["d"], c["teste"]; B, T, _ = x.shape; n = f"strati.{l}."
        z = self.ln(x, n + "ln1")
        q, k, v = (self.linear(z, n + s, d, d, False).view(B, T, h, d // h).transpose(1, 2) for s in "qkv")
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.linear(a.transpose(1, 2).reshape(B, T, d), n + "o", d, d, True)
        y = F.gelu(self.linear(self.ln(x, n + "ln2"), n + "fc", c["ffn"], d, True), approximate="tanh")
        return x + self.linear(y, n + "proj", d, c["ffn"], True)

    def logits(self, ids):
        c = self.c; B, T = ids.shape
        vp = -(-c["vocab"] // 128) * 128
        emb = self.w("embedding", vp, c["d"])
        x = F.embedding(ids, emb) + self.weight["posizioni"][:T]
        for l in range(c["strati"]):
            x = checkpoint(self.layer, x, l, use_reentrant=False) if self.recompute else self.layer(x, l)
        return F.linear(self.ln(x, "norma_finale"), emb[:c["vocab"]])


class Data:
    """Contiguous windows of ctx+1 tokens from a uint16 file; order given by a deterministic affine permutation."""
    def __init__(self, folder, ctx, synthetic=0, vocab=49152):
        self.ctx = ctx
        if synthetic:
            g = np.random.default_rng(0)
            self.train = g.integers(0, vocab, synthetic, dtype=np.uint16)
            self.val = g.integers(0, vocab, ctx * 64 + 1, dtype=np.uint16)
        else:
            folder = Path(folder)
            self.train = np.memmap(folder / "train.bin", dtype=np.uint16, mode="r")
            self.val = np.memmap(folder / "val.bin", dtype=np.uint16, mode="r")
        self.n = (len(self.train) - 1) // ctx
        self.stride = 1_000_003
        while math.gcd(self.stride, self.n) != 1:
            self.stride += 2

    def window(self, index):
        s = ((index * self.stride + 12_345) % self.n) * self.ctx
        z = np.asarray(self.train[s:s + self.ctx + 1], dtype=np.int64)
        return z[:-1], z[1:]

    def batch(self, first, count):
        xs, ys = zip(*(self.window(first + j) for j in range(count)))
        return np.stack(xs), np.stack(ys)


def lr_at(t, peak, warmup, total, frac):
    """Linear warmup, constant, then linear decay to 10% over the last fraction of the steps (as in STERA 30M)."""
    if t < warmup:
        return peak * (t + 1) / warmup
    d0 = int(total * (1 - frac))
    return peak if t < d0 else peak * (1 - 0.9 * min(1.0, (t - d0) / max(1, total - d0)))


def with_decay(name):
    return name.endswith((".coords", ".U", ".V")) or name == "posizioni"


@torch.no_grad()
def nll_val(model, data, windows, micro, dev):
    tot = cnt = 0.0
    n = min(windows, (len(data.val) - 1) // data.ctx)
    for i in range(0, n, micro):
        idx = range(i, min(n, i + micro))
        z = np.stack([np.asarray(data.val[j * data.ctx:(j + 1) * data.ctx + 1], dtype=np.int64) for j in idx])
        x, y = torch.from_numpy(z[:, :-1]).to(dev), torch.from_numpy(z[:, 1:]).to(dev)
        lg = model.logits(x).float()
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum")); cnt += y.numel()
    return tot / cnt


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(folder, state):
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"STERA_CODICE_PASSO{state['passo']}.pt"
    tmp = out.with_suffix(".pt.partial")
    torch.save({k: (OrderedDict((n, t.detach().cpu()) for n, t in v.items()) if isinstance(v, dict) and k in ("pesi", "m", "v") else v)
                for k, v in state.items()}, tmp)
    tmp.rename(out)
    return out, sha256(out)


def main():
    p = argparse.ArgumentParser()
    options = [("--data", None, str, ""), ("--output", None, str, None), ("--resume", "--riprendi", str, ""), ("--steps", "--passi", int, 10**9),
               ("--total-steps", "--passi-totali", int, 20000), ("--batch", "--lotto", int, 256), ("--micro", None, int, 4), ("--ctx", None, int, 1024),
               ("--layers", "--strati", int, 24), ("--d", None, int, 1024), ("--ffn", None, int, 4096), ("--heads", "--teste", int, 16), ("--vocab", None, int, 49152),
               ("--rank", "--rango", int, 96), ("--lr", None, float, 3e-4), ("--warmup", "--riscaldamento", int, 1000), ("--decay-frac", "--frazione-discesa", float, 0.2),
               ("--wd", None, float, 0.1), ("--beta2", None, float, 0.95), ("--save-every", "--salva-ogni", int, 1000), ("--val-windows", "--val-finestre", int, 256),
               ("--synthetic-data", "--dati-sintetici", int, 0), ("--print-every", "--stampa-ogni", int, 10)]
    for name, alias, kind, default in options:
        p.add_argument(*([name, alias] if alias else [name]), type=kind, default=default, required=name == "--output")
    p.add_argument("--recompute", "--ricalcolo", action="store_true", help="recompute the activations of every layer in the backward pass")
    a = p.parse_args()

    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(dev)
    torch.manual_seed(20261125)

    c = configuration(a)
    if a.resume:
        state = torch.load(a.resume, map_location="cpu", weights_only=False)
        assert state["configurazione"] == c, "configuration differs from the checkpoint"
    else:
        weights = initial_weights(c)
        state = {"configurazione": c, "pesi": weights, "m": OrderedDict((k, torch.zeros_like(v)) for k, v in weights.items()),
                 "v": OrderedDict((k, torch.zeros_like(v)) for k, v in weights.items()), "passo": 0, "token": 0}
    model = SteraCode(c, state["pesi"], dev, a.recompute)
    m = OrderedDict((k, v.to(dev)) for k, v in state["m"].items()); v2 = OrderedDict((k, v.to(dev)) for k, v in state["v"].items())
    names = list(model.weight)
    data = Data(a.data, a.ctx, a.synthetic_data, a.vocab)
    assert a.batch % (world * a.micro) == 0, "the batch must be a multiple of processes x micro"

    if rank == 0:
        stored = sum(t.numel() for t in model.weight.values())
        dense = sum(r * k for _, r, k in matrices(c)) + sum(t.numel() for n, t in model.weight.items() if not n.split(".")[-1] in ("coords", "scales", "gout", "gin", "U", "V"))
        print(json.dumps({"configurazione": c, "parametri_salvati": stored, "parametri_densi_equivalenti": dense, "processi": world,
                          "passo_iniziale": state["passo"], **{k: v for k, v in vars(a).items() if k not in ("data",)}}), flush=True)
        if not a.synthetic_data:
            print(f"VAL step={state['passo']} nll={nll_val(model, data, a.val_windows, a.micro, dev):.6f}", flush=True)

    t0, tok0, t = time.time(), state["token"], 0
    while t < a.steps:
        step = state["passo"]
        lr = lr_at(step, a.lr, a.warmup, a.total_steps, a.decay_frac)
        per_proc = a.batch // world
        x, y = data.batch(step * a.batch + rank * per_proc, per_proc)
        for w in model.weight.values():
            w.grad = None
        L = torch.zeros((), device=dev)
        for i in range(0, per_proc, a.micro):
            xs, ys = torch.from_numpy(x[i:i + a.micro]).to(dev), torch.from_numpy(y[i:i + a.micro]).to(dev)
            lg = model.logits(xs)
            Lm = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ys.reshape(-1), reduction="sum") / (a.batch * a.ctx)
            Lm.backward(); L += Lm.detach(); del lg, Lm
        with torch.no_grad():
            flat = torch.cat([model.weight[k].grad.reshape(-1) for k in names] + [L.reshape(1)])
            if world > 1:
                dist.all_reduce(flat)
            loss = float(flat[-1]); off = 0; grad = []
            for k in names:
                q = model.weight[k].numel(); grad.append(flat[off:off + q].view_as(model.weight[k])); off += q
            norm = torch.sqrt(sum(torch.sum(g * g) for g in grad)); fac = torch.clamp(1.0 / (norm + 1e-6), max=1.0)
            c1, c2 = 1 - 0.9 ** (step + 1), 1 - a.beta2 ** (step + 1)
            for k, g in zip(names, grad):
                w = model.weight[k]; g = g * fac
                m[k].mul_(0.9).add_(g, alpha=0.1); v2[k].mul_(a.beta2).addcmul_(g, g, value=1 - a.beta2)
                u = (m[k] / c1) / (torch.sqrt(v2[k] / c2) + 1e-8)
                if with_decay(k):
                    u = u + a.wd * w
                w.add_(u, alpha=-lr)
        state["passo"] += 1; state["token"] += a.batch * a.ctx; t += 1
        if rank == 0 and t % a.print_every == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated(dev) / 2**30 if cuda else 0
            print(f"STEP {state['passo']} tokens={state['token']} loss={loss:.5f} norm={float(norm):.4f} lr={lr:.6g} "
                  f"tokens_s={(state['token'] - tok0) / el:.0f} memory_gib={mem:.2f}", flush=True)
        if t % a.save_every == 0 or t == a.steps:
            dist.barrier()
            if rank == 0 and not a.synthetic_data:
                state.update(weights=model.weight, m=m, v=v2)
                out, h = save(Path(a.output), state)
                print(f"CHECKPOINT {out} sha256={h}", flush=True)
                print(f"VAL step={state['passo']} nll={nll_val(model, data, a.val_windows, a.micro, dev):.6f}", flush=True)
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
