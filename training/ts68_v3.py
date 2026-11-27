#!/usr/bin/env python3
"""STERA TS68 RowMask23 — refined training recipe (prototype, PyTorch CUDA, 1..N GPUs via torchrun).

Same model (ts68_core.TS68, RowMask23 coordinates). Recipe: context 512, B stories per update (token-mean loss),
AdamW beta2=0.95, weight decay 0.1 only on coordinates/position table (not LayerNorm, biases, scales),
LR schedule WSD (warmup -> constant -> linear decay to 10% over the last DECAY fraction of the phase budget).
Starts from any native TS68 full-state checkpoint (Adam moments kept). Corpus order continues from the checkpoint cursor
and wraps to further epochs. Rank 0 evaluates token-mean NLL on the first N_VAL validation stories at every save.
Save decisions depend only on the step counter (identical on all ranks).
"""
import argparse, math, os, time, json
from collections import OrderedDict
from pathlib import Path
import numpy as np, torch, torch.distributed as dist
import ts68_core as T

CFG256 = (4, 768, 3072, 16, 50257, 2048, 256, 256)

class TS68X(T.TS68):
    """RowMask23 + optional expressivity extensions (all compatible with compressed-domain inference):
    signs: fixed per-tile +-1 diagonals on input and output of every tile (seeded, no parameters);
    gains: learned per-row (gout) and per-column (gin) scaling of the whole matrix;
    rank:  learned matrix-wide low-rank term U @ V^T coupling all tiles."""
    def __init__(self, state, device, signs=False, gains=False, rank=0):
        super().__init__(state, device); self.signs, self.gains, self.rank = signs, gains, rank; self.sg = {}
    def sign_pair(self, name, rows, cols):
        if name not in self.sg:
            g = torch.Generator().manual_seed(abs(hash(name)) % (2**31))
            nt = (rows // 128) * (cols // 128)
            so = (torch.randint(0, 2, (nt, 128, 1), generator=g) * 2 - 1).float()
            si = (torch.randint(0, 2, (nt, 1, 128), generator=g) * 2 - 1).float()
            self.sg[name] = (so.to(self.device), si.to(self.device))
        return self.sg[name]
    def logits(self, ids):
        F_ = T.F; B, Tn = ids.shape
        emb = self.w("embedding", 50304, 768)[:T.VOCAB]
        x = F_.embedding(ids, emb) + self.weight["position.weight"][:Tn]
        for layer in range(LAYERS):
            n = f"layers.{layer}."
            z = self.ln(x, n + "ln1")
            q = self.linear(z, n + "q", 768, 768, False).view(B, Tn, 16, 48).transpose(1, 2)
            k = self.linear(z, n + "k", 768, 768, False).view(B, Tn, 16, 48).transpose(1, 2)
            v = self.linear(z, n + "v", 768, 768, False).view(B, Tn, 16, 48).transpose(1, 2)
            if layer % 2:
                i = torch.arange(Tn, device=ids.device)
                allowed = (i[:, None] >= i[None, :]) & ((i[:, None] - i[None, :]) < 256)
                a = F_.scaled_dot_product_attention(q, k, v, attn_mask=torch.where(allowed, 0.0, -1e9).to(torch.float32), scale=1.0)
            else:
                a = F_.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
            x = x + self.linear(a.transpose(1, 2).contiguous().view(B, Tn, 768), n + "o", 768, 768, True)
            y = self.linear(self.ln(x, n + "ln2"), n + "fc", 3072, 768, True)
            y = .5 * y * (1 + torch.tanh(.7978845608028654 * (y + .044715 * y * y * y)))
            x = x + self.linear(y, n + "proj", 768, 3072, True)
        return F_.linear(self.ln(x, "final_norm").reshape(B * Tn, 768), emb)

    def w(self, name, rows, cols):
        rm = self.rowmask; c = self.weight[name + ".coords"]; nt = c.shape[0]
        tile = (c.index_select(1, rm.base) + c.index_select(1, rm.residual) * rm.mask).reshape(nt, 128, 128) * self.weight[name + ".scales"]
        if self.signs:
            so, si = self.sign_pair(name, rows, cols); tile = tile * so * si
        W = tile.reshape(rows // 128, cols // 128, 128, 128).permute(0, 2, 1, 3).reshape(rows, cols)
        if self.gains: W = W * self.weight[name + ".gout"].reshape(rows, 1) * self.weight[name + ".gin"].reshape(1, cols)
        if self.rank: W = W + self.weight[name + ".U"] @ self.weight[name + ".V"].t()
        return W

LAYERS = 4
def matrices():
    out = [("embedding", 50304, 768)]
    for l in range(LAYERS):
        n = f"layers.{l}."; out += [(n + k, 768, 768) for k in ("q", "k", "v", "o")] + [(n + "fc", 3072, 768), (n + "proj", 768, 3072)]
    return out

def scratch_state(contract, seed=20261121):
    g = torch.Generator().manual_seed(seed); W = OrderedDict()
    sd = 0.02 / math.sqrt(1 + 2921 / 16384)
    for n, r, c in matrices():
        nt = (r // 128) * (c // 128); W[n + ".coords"] = torch.randn(nt, 3176, generator=g) * sd; W[n + ".scales"] = torch.ones(nt, 1, 1)
    W["position.weight"] = torch.randn(2048, 768, generator=g) * 0.02
    for n in ["final_norm"] + [f"layers.{l}.{k}" for l in range(LAYERS) for k in ("ln1", "ln2")]:
        W[n + ".weight"] = torch.ones(768); W[n + ".bias"] = torch.zeros(768)
    for l in range(LAYERS):
        n = f"layers.{l}."; W[n + "o.bias"] = torch.zeros(768); W[n + "fc.bias"] = torch.zeros(3072); W[n + "proj.bias"] = torch.zeros(768)
    return {"weights": W, "m": OrderedDict((k, torch.zeros_like(v)) for k, v in W.items()), "v": OrderedDict((k, torch.zeros_like(v)) for k, v in W.items()),
            "step": 0, "ordinal": 0, "offset": 0, "targets": 0, "contract": contract}

def add_extensions(state, gains, rank, seed=7):
    g = torch.Generator().manual_seed(seed)
    for n, r, c in matrices():
        new = {}
        if gains and n + ".gout" not in state["weights"]: new[n + ".gout"] = torch.ones(r); new[n + ".gin"] = torch.ones(c)
        if rank and n + ".U" not in state["weights"]: new[n + ".U"] = torch.zeros(r, rank); new[n + ".V"] = torch.randn(c, rank, generator=g) * 0.02
        for k, v in new.items(): state["weights"][k] = v; state["m"][k] = torch.zeros_like(v); state["v"][k] = torch.zeros_like(v)


def read_any(path, contract, sha):
    for ctx in (256, 512):
        T.CONFIG = CFG256[:7] + (ctx,)
        try: return T.read_native(path, contract, sha)
        except ValueError as e:
            if "architecture mismatch" not in str(e): raise
    raise ValueError("unreadable checkpoint")

def batch(corpus, ordinal, offset, B, ctx):
    N = len(corpus.parts[0]); x = np.full((B, ctx), T.EOS, np.int64); y = np.full((B, ctx), T.EOS, np.int64); m = np.zeros((B, ctx), np.float32); tok = 0
    for r in range(B):
        doc = int(corpus.parts[0][(194723 + (ordinal % N) * corpus.stride) % N]); s = corpus.story(doc)
        v = min(ctx, len(s) - 1 - offset)
        x[r, :v] = s[offset:offset + v]; y[r, :v] = s[offset + 1:offset + v + 1]; m[r, :v] = 1.0; tok += v; offset += v
        if offset + 1 >= len(s): ordinal, offset = ordinal + 1, 0
    return x, y, m, ordinal, offset, tok

def decayed(name): return name.endswith(".coords") or name.endswith(".U") or name.endswith(".V") or name == "position.weight"

def lr_at(t, peak, warm, total, frac):
    if t < warm: return peak * (t + 1) / warm
    d0 = int(total * (1 - frac))
    if t < d0: return peak
    return peak * (1 - 0.9 * min(1.0, (t - d0) / max(1, total - d0)))

@torch.no_grad()
def val_nll(model, corpus, n, ctx, dev):
    tot = cnt = 0.0
    for j in range(n):
        s = corpus.story(int(corpus.parts[1][j])); v = min(ctx, len(s) - 1)
        x = torch.full((1, ctx), T.EOS, dtype=torch.long, device=dev); y = x.clone()
        x[0, :v] = torch.from_numpy(s[:v].astype(np.int64)); y[0, :v] = torch.from_numpy(s[1:v + 1].astype(np.int64))
        lg = model.logits(x).float(); nll = torch.logsumexp(lg, -1) - lg.gather(-1, y.reshape(-1, 1)).reshape(-1)
        tot += float(nll[:v].sum()); cnt += v
    return tot / cnt

def main():
    p = argparse.ArgumentParser()
    for a, t, d in [("--data", str, None), ("--ckpt", str, None), ("--sha256", str, ""), ("--output", str, None), ("--steps", int, 10**9),
                    ("--batch", int, 64), ("--ctx", int, 512), ("--lr", float, 6e-4), ("--warmup", int, 300), ("--budget-steps", int, 60000),
                    ("--decay-frac", float, 0.2), ("--wd", float, 0.1), ("--beta2", float, 0.95), ("--save-steps", int, 2000), ("--nval", int, 256), ("--micro", int, 8), ("--rank", int, 0), ("--layers", int, 4), ("--sched-offset", int, -1)]:
        p.add_argument(a, type=t, default=d, required=d is None and a in ("--data", "--output"))
    p.add_argument("--signs", action="store_true"); p.add_argument("--gains", action="store_true"); p.add_argument("--scratch", action="store_true")
    a = p.parse_args()
    cuda = torch.cuda.is_available(); dist.init_process_group("nccl" if cuda else "gloo"); rank, world = dist.get_rank(), dist.get_world_size()
    if cuda: torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank))); dev = torch.device("cuda", torch.cuda.current_device())
    else: dev = torch.device("cpu")
    torch.manual_seed(T.SEED); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    global LAYERS; LAYERS = a.layers
    data = Path(a.data); contract = (data / "CONTRACT").read_text().split()[2]
    if a.scratch:
        state = scratch_state(contract)
    elif a.layers == 4 and not (a.gains or a.rank):
        state = read_any(a.ckpt, contract, a.sha256 or None)
    else:   # resume a checkpoint written by this trainer: expected config and parameter count are known in advance
        probe = scratch_state(contract); add_extensions(probe, a.gains, a.rank)
        T.PHYSICAL_PARAMS = sum(v.numel() for v in probe["weights"].values()); T.CONFIG = (a.layers,) + CFG256[1:7] + (a.ctx,)
        state = T.read_native(a.ckpt, contract, a.sha256 or None); del probe
    T.CONFIG = (a.layers,) + CFG256[1:7] + (a.ctx,)
    add_extensions(state, a.gains, a.rank)
    for key in ("weights", "m", "v"): state[key] = OrderedDict(sorted(state[key].items()))   # native format requires sorted names
    T.PHYSICAL_PARAMS = sum(v.numel() for v in state["weights"].values())   # writer validates against the actual parameter set
    corpus = T.Corpus(data, verify=(rank == 0)); model = TS68X(state, dev, a.signs, a.gains, a.rank)
    state["m"] = OrderedDict((n, v.to(dev)) for n, v in state["m"].items()); state["v"] = OrderedDict((n, v.to(dev)) for n, v in state["v"].items()); state["weights"] = model.weight
    names = list(model.weight); b2 = a.beta2; s0 = state["step"]; t = 0
    if rank == 0:
        ntrain = sum(v.numel() for v in model.weight.values())
        print(json.dumps({"trainable_params": ntrain, "start_step": s0, "targets": state["targets"], "ordinal": state["ordinal"], "world": world, **{k: v for k, v in vars(a).items() if k not in ("data",)}}), flush=True)
        print(f"VAL step={s0} nll={val_nll(model, corpus, a.nval, a.ctx, dev):.6f}", flush=True)
    t0 = time.time(); tok0 = state["targets"]
    while t < a.steps:
        x, y, m, o1, f1, n = batch(corpus, state["ordinal"], state["offset"], a.batch, a.ctx)
        rows = np.array_split(np.arange(a.batch), world)[rank]; denom = float(m.sum())
        ts = t if a.sched_offset < 0 else state["step"] - a.sched_offset   # resume-safe schedule position
        lr = lr_at(ts, a.lr, a.warmup, a.budget_steps, a.decay_frac)
        for w in model.weight.values(): w.grad = None
        L = torch.zeros((), device=dev)
        rows = rows[np.argsort(np.count_nonzero(m[rows], axis=1))]   # group similar lengths -> less padding
        for mb in np.array_split(rows, max(1, math.ceil(len(rows) / a.micro))):   # gradient accumulation (memory only)
            Tm = int(np.max(np.count_nonzero(m[mb], axis=1))); Tm = max(16, -(-Tm // 16) * 16)   # trim padding (exact: causal model)
            xs, ys, ms = (torch.from_numpy(np.ascontiguousarray(z[mb, :Tm])).to(dev) for z in (x, y, m))
            lg = model.logits(xs); nll = torch.logsumexp(lg, -1) - lg.gather(-1, ys.reshape(-1, 1)).reshape(-1)
            Lm = torch.sum(nll * ms.reshape(-1)) / denom; Lm.backward(); L = L + Lm.detach(); del lg, nll, Lm
        with torch.no_grad():
            flat = torch.cat([model.weight[k].grad.reshape(-1) for k in names] + [L.detach().reshape(1)])
            if world > 1: dist.all_reduce(flat)
            Lg = float(flat[-1]); off = 0; grads = []
            for k in names: q = model.weight[k].numel(); grads.append(flat[off:off + q].view_as(model.weight[k])); off += q
            norm = torch.sqrt(sum(torch.sum(g * g) for g in grads)); fac = torch.clamp(1.0 / (norm + 1e-6), max=1.0)
            step = state["step"] + 1
            c1, c2 = (1 - .9 ** (ts + 1), 1 - b2 ** (ts + 1)) if (a.scratch or a.sched_offset >= 0) else (1.0, 1.0)   # bias correction only for fresh moments
            for k, g in zip(names, grads):
                w = model.weight[k]; g = g * fac; mm = state["m"][k]; vv = state["v"][k]
                mm.mul_(.9).add_(g, alpha=.1); vv.mul_(b2).addcmul_(g, g, value=1 - b2)
                u = (mm / c1) / (torch.sqrt(vv / c2) + 1e-8)
                if decayed(k): u = u + a.wd * w
                w.add_(u, alpha=-lr)
            state["step"] = step
        state["ordinal"], state["offset"] = o1, f1; state["targets"] += n; t += 1
        if rank == 0 and t % 20 == 0:
            el = time.time() - t0
            print(f"V2 step={state['step']} t={t} labels={state['targets']} loss={Lg:.5f} grad_norm={float(norm):.4f} lr={lr:.6g} tok_s={(state['targets'] - tok0) / el:.0f}", flush=True)
        if t % a.save_steps == 0 or t == a.steps:
            dist.barrier()
            if rank == 0:
                out = Path(a.output) / f"TS68V2_STEP{state['step']}.ckpt.gz"; sha = T.write_native(out, state)
                print(f"V2_CHECKPOINT_OK path={out} sha256={sha}", flush=True)
                print(f"VAL step={state['step']} t={t} labels={state['targets']} nll={val_nll(model, corpus, a.nval, a.ctx, dev):.6f}", flush=True)
            dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
