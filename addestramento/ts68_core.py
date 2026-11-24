#!/usr/bin/env python3
"""Experimental CUDA/PyTorch port of STERA TS68. NOT yet cross-backend qualified.

Matches source recovery_v2/train_resume.cpp and performance_v1/batch_step.hpp.
Never writes to the Mac's original data/ or run/ directories.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import struct
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MAGIC = b"STERATS68NATV1\n\x00"
CONFIG = (4, 768, 3072, 16, 50257, 2048, 256, 256)
COORD_COUNT = 3176
PHYSICAL_PARAMS = 14586342
TOTAL_TARGETS = 530189191
SEED = 20261117
VOCAB = 50257
EOS = 50256


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_exact(f, size):
    b = f.read(size)
    if len(b) != size:
        raise ValueError("truncated TS68 checkpoint")
    return b


def read_native(path, expected_contract=None, expected_sha=None):
    path = Path(path)
    if expected_sha and file_sha256(path) != expected_sha:
        raise ValueError("checkpoint SHA256 mismatch")
    tensors, moments, variances = OrderedDict(), OrderedDict(), OrderedDict()
    with gzip.open(path, "rb") as f:
        if read_exact(f, 16) != MAGIC:
            raise ValueError("wrong TS68 magic")
        if struct.unpack("<B", read_exact(f, 1))[0] != 1:
            raise ValueError("checkpoint has no full optimizer state")
        cfg = struct.unpack("<8i", read_exact(f, 32))
        if cfg != CONFIG:
            raise ValueError(f"architecture mismatch: {cfg}")
        step = struct.unpack("<Q", read_exact(f, 8))[0]
        ordinal, offset, targets = struct.unpack("<3Q", read_exact(f, 24))
        contract = read_exact(f, 64).decode("ascii")
        if expected_contract is not None and contract != expected_contract:
            raise ValueError("provenance contract mismatch")
        n = struct.unpack("<I", read_exact(f, 4))[0]
        if not (10 < n < 1000):
            raise ValueError("invalid tensor count")
        for _ in range(n):
            name = read_exact(f, struct.unpack("<H", read_exact(f, 2))[0]).decode()
            ndim = struct.unpack("<B", read_exact(f, 1))[0]
            if not 1 <= ndim <= 4:
                raise ValueError("invalid ndim")
            shape = struct.unpack("<" + "I"*ndim, read_exact(f, 4*ndim))
            num = struct.unpack("<Q", read_exact(f, 8))[0]
            if num != math.prod(shape) or num > 100_000_000:
                raise ValueError(f"invalid tensor dims {name}")
            arrs = []
            for _m in range(3):
                arr = np.frombuffer(read_exact(f, 4*num), dtype="<f4").copy().reshape(shape)
                if not np.isfinite(arr).all():
                    raise ValueError(f"nonfinite in {name}")
                arrs.append(torch.from_numpy(arr))
            if name in tensors:
                raise ValueError("duplicate parameter")
            tensors[name], moments[name], variances[name] = arrs
        if f.read(1):
            raise ValueError("trailing bytes in checkpoint")
    total = sum(t.numel() for t in tensors.values())
    if total != PHYSICAL_PARAMS:
        raise ValueError(f"parameter count {total} != {PHYSICAL_PARAMS}")
    if list(tensors) != sorted(tensors):
        raise ValueError("unexpected parameter ordering")
    return dict(step=step, ordinal=ordinal, offset=offset, targets=targets,
                contract=contract, weights=tensors, m=moments, v=variances)


def write_native(path, state):
    """Same cross-platform little-endian TS68 native full-state format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with gzip.open(temp, "wb", compresslevel=3) as f:
        f.write(MAGIC)
        f.write(struct.pack("<B8iQ3Q", 1, *CONFIG, state["step"],
                            state["ordinal"], state["offset"], state["targets"]))
        f.write(state["contract"].encode("ascii"))
        names = list(state["weights"])
        f.write(struct.pack("<I", len(names)))
        for name in names:
            b = name.encode("utf-8")
            w = state["weights"][name]
            f.write(struct.pack("<H", len(b)) + b + struct.pack("<B", w.ndim))
            f.write(struct.pack("<" + "I"*w.ndim, *w.shape))
            f.write(struct.pack("<Q", w.numel()))
            for key in ("weights", "m", "v"):
                t = state[key][name].detach().to("cpu", dtype=torch.float32).contiguous()
                f.write(t.numpy().astype("<f4", copy=False).tobytes())
    # Fully verify the new file before publishing its final name.
    validate = read_native(temp, state["contract"])
    if (validate["step"], validate["ordinal"], validate["offset"], validate["targets"]) != (
            state["step"], state["ordinal"], state["offset"], state["targets"]):
        raise ValueError("checkpoint roundtrip metadata mismatch")
    if not all(torch.equal(validate["weights"][k], state["weights"][k].detach().cpu()) and
               torch.equal(validate["m"][k], state["m"][k].detach().cpu()) and
               torch.equal(validate["v"][k], state["v"][k].detach().cpu())
               for k in validate["weights"]):
        raise ValueError("checkpoint roundtrip tensors mismatch")
    os.replace(temp, path)
    return file_sha256(path)


class RowMask23:
    def __init__(self, device):
        # Code layout exactly as stera_rowmask23_codes.hpp + rowmask23_mlx.hpp
        rid = [[-1]*128 for _ in range(128)]
        idx = 255
        for i in range(1, 128):
            for k in range(23):
                s = 1 + (13*k + 7*i) % 127
                if rid[i][s] != -1:
                    raise ValueError("duplicate rowmask coordinate")
                rid[i][s] = idx
                idx += 1
        assert idx == COORD_COUNT
        base, residual, mask = [], [], []
        for i in range(128):
            for j in range(128):
                s = i ^ j
                base.append(i if s == 0 else 128 + s - 1)
                r = rid[i][s] if s else -1
                residual.append(max(r, 0))
                mask.append(1.0 if r >= 0 else 0.0)
        self.base = torch.tensor(base, dtype=torch.long, device=device)
        self.residual = torch.tensor(residual, dtype=torch.long, device=device)
        self.mask = torch.tensor(mask, dtype=torch.float32, device=device).reshape(1, 16384)

    def reconstruct(self, coords, scales, rows, cols):
        if (rows % 128 or cols % 128 or
                coords.shape != ((rows//128)*(cols//128), COORD_COUNT)):
            raise ValueError("RowMask23 shape error")
        nt = coords.shape[0]
        tile = (coords.index_select(1, self.base) +
                coords.index_select(1, self.residual)*self.mask).reshape(nt,128,128)
        tile = tile * scales
        return tile.reshape(rows//128, cols//128, 128, 128).permute(0,2,1,3).reshape(rows,cols)


class TS68:
    def __init__(self, state, device):
        self.device = torch.device(device)
        self.weight = OrderedDict((k, torch.nn.Parameter(v.to(self.device, dtype=torch.float32)))
                                  for k, v in state["weights"].items())
        self.rowmask = RowMask23(self.device)

    def w(self, name, rows, cols):
        return self.rowmask.reconstruct(
            self.weight[name+".coords"], self.weight[name+".scales"], rows, cols)

    def linear(self, x, prefix, rows, cols, bias):
        out = F.linear(x, self.w(prefix, rows, cols))
        return out + self.weight[prefix+".bias"] if bias else out

    def ln(self, x, prefix):
        return F.layer_norm(x, (768,), self.weight[prefix+".weight"],
                            self.weight[prefix+".bias"], eps=1e-5)

    def logits(self, ids):
        assert ids.ndim == 2
        B, T = ids.shape
        emb = self.w("embedding", 50304, 768)[:VOCAB]
        x = F.embedding(ids, emb) + self.weight["position.weight"][:T]
        for layer in range(4):
            n = f"layers.{layer}."
            z = self.ln(x, n+"ln1")
            q = self.linear(z, n+"q", 768, 768, False).view(B,T,16,48).transpose(1,2)
            k = self.linear(z, n+"k", 768, 768, False).view(B,T,16,48).transpose(1,2)
            v = self.linear(z, n+"v", 768, 768, False).view(B,T,16,48).transpose(1,2)
            if layer % 2:
                i = torch.arange(T,device=ids.device)
                allowed = (i[:,None] >= i[None,:]) & ((i[:,None]-i[None,:])<256)
                mask = torch.where(allowed, 0.0, -1e9).to(torch.float32)
                a = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=1.0)
            else:
                a = F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=1.0)
            x = x + self.linear(a.transpose(1,2).contiguous().view(B,T,768),
                                n+"o",768,768,True)
            y = self.linear(self.ln(x,n+"ln2"),n+"fc",3072,768,True)
            y = .5*y*(1+torch.tanh(.7978845608028654*(y+.044715*y*y*y)))
            x = x + self.linear(y,n+"proj",768,3072,True)
        return F.linear(self.ln(x,"final_norm").reshape(B*T,768),emb)

    def objective(self, x, y, mask):
        logits = self.logits(x)
        nll = torch.logsumexp(logits, dim=-1) - logits.gather(-1,y.reshape(-1,1)).reshape(-1)
        return torch.sum(nll * mask.reshape(-1)) / torch.sum(mask)


class Corpus:
    def __init__(self, path, verify=True):
        self.root = Path(path)
        self.manifest = json.loads((self.root/"MANIFEST.json").read_text())
        self.records = np.fromfile(self.root/"stories.index",
                                   dtype=np.dtype([("block","<u8"),("token_offset","<u4"),
                                                   ("length","<u4"),("source_offset","<u4"),
                                                   ("split","<u4")]))
        self.parts = [np.flatnonzero(self.records["split"]==s) for s in range(3)]
        assert len(self.records)==2717495 and len(self.parts[0])==2660865
        assert self.manifest["split_target_tokens"][0] == TOTAL_TARGETS
        self.stride = 104729
        while math.gcd(self.stride,len(self.parts[0]))!=1:
            self.stride += 2
        self.pack = open(self.root/"stories.blocks","rb")
        self.prev_block, self.cached = None, None
        if verify:
            if (file_sha256(self.root/"stories.blocks") != self.manifest["pack_sha256"] or
                    file_sha256(self.root/"stories.index") != self.manifest["index_sha256"]):
                raise ValueError("corpus SHA mismatch")

    def story(self, index):
        rec = self.records[index]
        block = int(rec["block"])
        if block != self.prev_block:
            self.pack.seek(block)
            compressed, raw = struct.unpack("<II",read_exact(self.pack,8))
            if not (0<compressed<16000000 and 0<raw<16000000 and raw%2==0):
                raise ValueError("bad corpus block sizes")
            self.cached = np.frombuffer(zlib.decompress(read_exact(self.pack,compressed)),
                                        dtype="<u2")
            if self.cached.nbytes != raw:
                raise ValueError("corrupt corpus block")
            self.prev_block = block
        off, n = int(rec["token_offset"]),int(rec["length"])
        s = self.cached[off:off+n]
        if len(s)!=n or s[0]!=EOS or s[-1]!=EOS:
            raise ValueError("bad story")
        return s

    def batch(self, ordinal, offset, chunks=4, context=256):
        x=np.full((chunks,context),EOS,np.int64)
        y=np.full((chunks,context),EOS,np.int64)
        mask=np.zeros((chunks,context),np.float32)
        tokens=0
        for row in range(chunks):
            if ordinal >= len(self.parts[0]):
                break
            doc_index=int(self.parts[0][(194723 + ordinal*self.stride)%len(self.parts[0])])
            s=self.story(doc_index)
            valid=min(context,len(s)-1-offset)
            if valid<=0:
                raise ValueError("zero target chunk")
            x[row,:valid]=s[offset:offset+valid]
            y[row,:valid]=s[offset+1:offset+valid+1]
            mask[row,:valid]=1.0/valid
            tokens += valid
            offset += valid
            if offset+1 >= len(s):
                ordinal,offset=ordinal+1,0
        if tokens == 0:
            raise StopIteration
        return x,y,mask,ordinal,offset,tokens


def update(model, state, x, y, mask, lr):
    for w in model.weight.values():
        w.grad = None
    L = model.objective(x,y,mask)
    L.backward()
    with torch.no_grad():
        grads = [model.weight[name].grad for name in model.weight]
        norm = torch.sqrt(sum(torch.sum(g*g) for g in grads))
        factor = torch.minimum(torch.tensor(1.,device=x.device),1.0/(norm+1e-6))
        step=state["step"]+1
        c1 = float(np.float32(1.0 - float(np.float32(.9))**step))
        c2 = float(np.float32(1.0 - float(np.float32(.999))**step))
        for name, w in model.weight.items():
            g=grads.pop(0)*factor
            m=state["m"][name]
            v=state["v"][name]
            m.mul_(.9).add_(g,alpha=(1-.9))
            v.mul_(.999).addcmul_(g,g,value=(1-.999))
            u = (m/c1)/(torch.sqrt(v/c2)+1e-8) + .01*w
            w.add_(u,alpha=-lr)
        state["step"]=step
    return float(L.detach()),float(norm.detach())


def run(args):
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=Path(args.data)
    contract=(data/"CONTRACT").read_text().split()[2]
    state=read_native(args.ckpt,contract, args.sha256 or None)
    print(json.dumps({k:state[k] for k in ("step","ordinal","offset","targets","contract")},
                     indent=2),flush=True)
    if args.codec_only:
        print("FULLSTATE_CODEC_READ_OK",flush=True)
        return
    corpus=Corpus(data,verify=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA GPU required for full-model training; use --codec-only for Mac preflight")
    print("DEVICE",device,torch.cuda.get_device_name() if device=="cuda" else "",flush=True)
    model=TS68(state,device)
    state["m"]=OrderedDict((n,v.to(device)) for n,v in state["m"].items())
    state["v"]=OrderedDict((n,v.to(device)) for n,v in state["v"].items())
    state["weights"]=model.weight
    start=state["step"]
    lastsave=time.monotonic()
    while state["ordinal"]<len(corpus.parts[0]) and state["step"]-start<args.steps:
        x,y,m,ord1,off1,target_count = corpus.batch(state["ordinal"],state["offset"])
        phase=min(1.0,state["targets"]/TOTAL_TARGETS)
        warm=min(1.0,(state["step"]+1)/1000.0)
        lr=float(np.float32(3e-4*warm*(.1+.9*.5*(1+math.cos(math.pi*phase)))))
        x=torch.from_numpy(x).to(device); y=torch.from_numpy(y).to(device)
        m=torch.from_numpy(m).to(device)
        loss,norm=update(model,state,x,y,m,lr)
        state["ordinal"],state["offset"]=ord1,off1
        state["targets"]+=target_count
        if state["step"]%10==0:
            print(f"TS68_CUDA step={state['step']} labels={state['targets']} "
                  f"loss={loss:.7f} grad_norm={norm:.6f} lr={lr:.9f}",flush=True)
        if time.monotonic()-lastsave>=args.save_seconds:
            out=Path(args.output)/f"TS68_FULLSTATE_STEP{state['step']}.ckpt.gz"
            digest=write_native(out,state)
            print(f"CUDA_CHECKPOINT_OK path={out} sha256={digest}",flush=True)
            lastsave=time.monotonic()
    out=Path(args.output)/f"TS68_FULLSTATE_STEP{state['step']}.ckpt.gz"
    digest=write_native(out,state)
    print(f"CUDA_FINAL_CHECKPOINT_OK path={out} step={state['step']} "
          f"labels={state['targets']} sha256={digest}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--ckpt",required=True)
    p.add_argument("--sha256",default="")
    p.add_argument("--output",default="/kaggle/working")
    p.add_argument("--steps",type=int,default=2)
    p.add_argument("--save-seconds",type=int,default=900)
    p.add_argument("--codec-only",action="store_true")
    p.add_argument("--allow-cpu",action="store_true")
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
