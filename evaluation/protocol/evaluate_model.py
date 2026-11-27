"""TEST300 evaluation: NLL on the reference stories and generation of the continuations.

Same procedure for the three models of the thesis:
  - teacher-forcing NLL on the 300 stories (at most 512 target tokens each);
  - top-k 40 sampling, temperature 0.75, at most 180 new tokens (256 in total),
    stop at token 50256, CPU generator seeded by the prompt.

Usage:
  python evaluate_model.py stera30m  --checkpoint ../../weights/stera30m_step122000.ckpt.gz --out stera30m.jsonl
  python evaluate_model.py hf        --model <TinyStories HF checkpoint folder> --out tinystories33m.jsonl
Options: --device cpu|cuda|mps  --n 300
         --rank-bits 8 (stera30m only): quantize to 8 bits only the factors U, V of the low-rank term,
         with one symmetric FP32 scale per column; every other parameter stays FP32.
         --reduced-rank k (stera30m only): replace U V^T with its best rank-k approximation (SVD);
         k = 0 removes the low-rank term. --nll-only: compute the NLL only, without generation.
         --rank-scheme colonna|mlx: per-column scales, or the group-32 format of the MLX 8-bit engine.
         --ablation scale1|guadagni1|scale8|guadagni8|piccole8: post-hoc tests on scales and gains (see ablation()).
The Italian option names used in the archived logs (--modello, --rango-bit, --rango-ridotto, --solo-nll,
--rango-schema, --ablazione) are still accepted. Keys of the printed and written JSON records are unchanged,
so the outputs match the archived files.
"""
import argparse, hashlib, json, math, pathlib, sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "training"))

SHA_STERA30M = "418becfbcaddcefd56f828f295138bbc865f9405a9060930c060ff7e6ced0887"
SHA_TOKENIZER = "f6ed3d307010c244c22aeffbde05f419cf277c23e64cf98b673cac5449cfeff5"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def quantize_rank_mlx(model, matrices, bits, group=32):
    """Same format as the 8-bit MLX engine (systems/engines/stera_rank8_mlx.py): U quantized in groups along the rank,
    V^T in groups along the columns (for the embedding also V in groups along the rank, used for the rows of U)."""
    import mlx.core as mx
    import numpy as np
    rt = lambda a: np.array(mx.dequantize(*mx.quantize(mx.array(a), group_size=group, bits=bits), group_size=group, bits=bits))
    err2 = ref2 = 0.0
    with torch.no_grad():
        for n, _, _ in matrices:
            U, V = model.weight[n + ".U"], model.weight[n + ".V"]
            u, v = U.detach().cpu().numpy(), V.detach().cpu().numpy()
            uq, vq = rt(u), rt(v.T).T
            for t, q in ((U, uq), (V, vq)):
                err2 += float(((torch.from_numpy(q) - t.cpu()) ** 2).sum()); ref2 += float((t ** 2).sum()); t.copy_(torch.from_numpy(q))
    print(json.dumps({"rango_bit": bits, "schema": f"mlx gruppi da {group}", "rmse_relativo_rango": math.sqrt(err2 / ref2)}), flush=True)


def quantize_rank(model, matrices, bits):
    """Symmetric per-column quantization of the factors U and V only; values are dequantized to FP32 for compute."""
    q = 2 ** (bits - 1) - 1; err2 = ref2 = 0.0; values = scales = 0
    with torch.no_grad():
        for n, _, _ in matrices:
            for k in (".U", ".V"):
                t = model.weight[n + k]
                s = t.abs().amax(dim=0, keepdim=True).clamp_min(1e-30) / q
                tq = torch.round(t / s).clamp(-q, q) * s
                err2 += float(((tq - t) ** 2).sum()); ref2 += float((t ** 2).sum()); values += t.numel(); scales += s.numel()
                t.copy_(tq)
    print(json.dumps({"rango_bit": bits, "valori_quantizzati": values, "scale_fp32": scales,
                      "rmse_relativo_rango": math.sqrt(err2 / ref2)}), flush=True)


def reduce_rank(model, matrices, k):
    """Replace U V^T with its best rank-k approximation (Eckart-Young theorem), computed in FP64."""
    err2 = ref2 = 0.0
    with torch.no_grad():
        for n, _, _ in matrices:
            U, V = model.weight[n + ".U"], model.weight[n + ".V"]
            W = model.w(n, U.shape[0], V.shape[0]).double()
            qu, ru = torch.linalg.qr(U.double()); qv, rv = torch.linalg.qr(V.double())
            a, sv, bh = torch.linalg.svd(ru @ rv.T)
            Uk = qu @ a[:, :k] * sv[:k]; Vk = qv @ bh[:k].T
            U.zero_(); V.zero_(); U[:, :k] = Uk.float(); V[:, :k] = Vk.float()
            Wk = model.w(n, U.shape[0], V.shape[0]).double()
            err2 += float(((Wk - W) ** 2).sum()); ref2 += float((W ** 2).sum())
    print(json.dumps({"rango_ridotto": k, "errore_relativo_matrici": math.sqrt(err2 / ref2)}), flush=True)


def ablation(model, matrices, kind):
    """Post-hoc tests on the small components of the projections (option values keep the thesis names):
    scale1 / guadagni1: block scales or gains reset to 1 (component removed);
    scale8 / guadagni8: component quantized to 8 bits (symmetric, one FP32 scale per matrix);
    piccole8: scales, gains and low-rank term at 8 bits together (FP32 coordinates)."""
    def q8(t):
        s = t.abs().max().clamp_min(1e-30) / 127
        return torch.round(t / s).clamp(-127, 127) * s
    with torch.no_grad():
        for n, _, _ in matrices:
            W = model.weight
            if kind == "scale1": W[n + ".scales"].fill_(1.0)
            if kind == "guadagni1": W[n + ".gout"].fill_(1.0); W[n + ".gin"].fill_(1.0)
            if kind in ("scale8", "piccole8"): W[n + ".scales"].copy_(q8(W[n + ".scales"]))
            if kind in ("guadagni8", "piccole8"): W[n + ".gout"].copy_(q8(W[n + ".gout"])); W[n + ".gin"].copy_(q8(W[n + ".gin"]))
    if kind == "piccole8":
        quantize_rank(model, matrices, 8)
    print(json.dumps({"ablazione": kind}), flush=True)


def load_stera30m(path, device, rank_bits=None, reduced_rank=None, scheme="colonna", abl=None):
    import ts68_v3 as V
    import ts68_core as T
    sha = sha256(path)
    assert sha == SHA_STERA30M, "checkpoint hash differs from the one in the thesis"
    contract = (REPO / "training/data/CONTRACT").read_text().split()[2]
    V.LAYERS = 6
    probe = V.scratch_state(contract); V.add_extensions(probe, True, 96)
    T.PHYSICAL_PARAMS = sum(t.numel() for t in probe["weights"].values())
    T.CONFIG = (6,) + V.CFG256[1:7] + (512,)
    st = T.read_native(path, contract, sha)
    assert st["step"] == 122000
    assert sum(t.numel() for t in st["weights"].values()) == 30346182
    model = V.TS68X(st, torch.device(device), False, True, 96)
    if abl:
        ablation(model, V.matrices(), abl)
    if reduced_rank is not None:
        reduce_rank(model, V.matrices(), reduced_rank)
    if rank_bits and scheme == "mlx":
        quantize_rank_mlx(model, V.matrices(), rank_bits)
    elif rank_bits:
        quantize_rank(model, V.matrices(), rank_bits)
    with torch.inference_mode():
        w = {n: model.w(n, r, c).contiguous() for n, r, c in V.matrices()}
    model.w = lambda name, rows, cols: w[name]
    return lambda ids: model.logits(ids).reshape(ids.shape[0], ids.shape[1], -1)


def load_hf(folder, device):
    from transformers import AutoModelForCausalLM
    import transformers
    # the argument is called torch_dtype up to Transformers 4.5x and dtype in later versions
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    m = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, **{key: torch.float32}).eval().to(device)
    return lambda ids: m(input_ids=ids, use_cache=False).logits


def main():
    a = argparse.ArgumentParser()
    a.add_argument("kind", choices=["stera30m", "hf"])
    a.add_argument("--checkpoint"); a.add_argument("--model", "--modello", dest="model")
    a.add_argument("--tokenizer", default=None, help="tokenizer.json (default: the one of the HF model)")
    a.add_argument("--device", default="cpu"); a.add_argument("--n", type=int, default=300)
    a.add_argument("--out", required=True)
    a.add_argument("--rank-bits", "--rango-bit", dest="rank_bits", type=int, default=None)
    a.add_argument("--reduced-rank", "--rango-ridotto", dest="reduced_rank", type=int, default=None)
    a.add_argument("--nll-only", "--solo-nll", dest="nll_only", action="store_true")
    a.add_argument("--rank-scheme", "--rango-schema", dest="rank_scheme", choices=["colonna", "mlx"], default="colonna")
    a.add_argument("--ablation", "--ablazione", dest="ablation", choices=["scale1", "guadagni1", "scale8", "guadagni8", "piccole8"], default=None)
    x = a.parse_args()
    torch.set_grad_enabled(False)
    tokp = x.tokenizer or (pathlib.Path(x.model) / "tokenizer.json" if x.model else None)
    assert tokp and sha256(tokp) == SHA_TOKENIZER, "the TinyStories GPT-Neo tokenizer is required (--tokenizer)"
    tok = Tokenizer.from_file(str(tokp))
    forward = load_stera30m(x.checkpoint, x.device, x.rank_bits, x.reduced_rank, x.rank_scheme, x.ablation) if x.kind == "stera30m" else load_hf(x.model, x.device)
    rows = json.loads((HERE / "incipit_test300.json").read_text())[: x.n]
    tot_nll = tot_tok = 0.0
    with torch.inference_mode(), open(x.out, "w") as f:
        for r in rows:
            ref = r["reference_ids"]
            ids_ref = torch.tensor([ref[:-1]], device=x.device)
            nll = float(F.cross_entropy(forward(ids_ref).reshape(-1, 50257).float().cpu(), torch.tensor(ref[1:]), reduction="sum"))
            ids = r["tokens"][:]; start = len(ids); g = torch.Generator(device="cpu").manual_seed(r["seed"]); eos = False
            for _ in range(0 if x.nll_only else min(180, 256 - start)):
                lg = forward(torch.tensor([ids], device=x.device))[0, -1].float().cpu()
                v, k = torch.topk(lg, 40)
                nx = int(k[torch.multinomial(torch.softmax(v / .75, dim=0), 1, generator=g)])
                if nx == 50256:
                    eos = True; break
                ids.append(nx)
            z = {"id": r["id"], "prompt": r["prompt"], "seed": r["seed"], "generated_ids": ids[start:],
                 "continuation": tok.decode(ids[start:]), "eos": eos, "generated_tokens": len(ids) - start,
                 "ref_nll_sum": nll, "ref_target_tokens": len(ref) - 1}
            f.write(json.dumps(z, ensure_ascii=False) + "\n"); f.flush()
            tot_nll += nll; tot_tok += len(ref) - 1
    print(f"stories {len(rows)}  NLL {tot_nll/tot_tok:.6f}  perplexity {math.exp(tot_nll/tot_tok):.4f}")


if __name__ == "__main__":
    main()
