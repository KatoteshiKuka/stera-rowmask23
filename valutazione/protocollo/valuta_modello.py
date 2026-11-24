"""Valutazione TEST300: NLL sui racconti di riferimento e generazione delle continuazioni.

Stessa procedura per i tre modelli della tesi:
  - NLL in teacher forcing sui 300 racconti (max 512 token bersaglio ciascuno);
  - generazione top-k 40, temperatura 0,75, al più 180 nuovi token (256 totali),
    arresto al token 50256, generatore CPU con seme dell'incipit.

Uso:
  python valuta_modello.py stera30m  --checkpoint ../../pesi/stera30m_step122000.ckpt.gz --out stera30m.jsonl
  python valuta_modello.py hf        --modello <cartella checkpoint HF TinyStories> --out tinystories33m.jsonl
Opzioni: --device cpu|cuda|mps  --n 300
"""
import argparse, hashlib, json, math, pathlib, sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

QUI = pathlib.Path(__file__).resolve().parent
REPO = QUI.parent.parent
sys.path.insert(0, str(REPO / "addestramento"))

SHA_STERA30M = "418becfbcaddcefd56f828f295138bbc865f9405a9060930c060ff7e6ced0887"
SHA_TOKENIZER = "f6ed3d307010c244c22aeffbde05f419cf277c23e64cf98b673cac5449cfeff5"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def carica_stera30m(path, device):
    import ts68_v3 as V
    import ts68_core as T
    sha = sha256(path)
    assert sha == SHA_STERA30M, "impronta del checkpoint diversa da quella della tesi"
    contratto = (REPO / "addestramento/dati/CONTRACT").read_text().split()[2]
    V.LAYERS = 6
    probe = V.scratch_state(contratto); V.add_extensions(probe, True, 96)
    T.PHYSICAL_PARAMS = sum(t.numel() for t in probe["weights"].values())
    T.CONFIG = (6,) + V.CFG256[1:7] + (512,)
    st = T.read_native(path, contratto, sha)
    assert st["step"] == 122000
    assert sum(t.numel() for t in st["weights"].values()) == 30346182
    model = V.TS68X(st, torch.device(device), False, True, 96)
    with torch.inference_mode():
        w = {n: model.w(n, r, c).contiguous() for n, r, c in V.matrices()}
    model.w = lambda name, rows, cols: w[name]
    return lambda ids: model.logits(ids).reshape(ids.shape[0], ids.shape[1], -1)


def carica_hf(cartella, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(cartella, local_files_only=True, dtype=torch.float32).eval().to(device)
    return lambda ids: m(input_ids=ids, use_cache=False).logits


def main():
    a = argparse.ArgumentParser()
    a.add_argument("tipo", choices=["stera30m", "hf"])
    a.add_argument("--checkpoint"); a.add_argument("--modello")
    a.add_argument("--tokenizer", default=None, help="tokenizer.json (default: quello del modello HF)")
    a.add_argument("--device", default="cpu"); a.add_argument("--n", type=int, default=300)
    a.add_argument("--out", required=True)
    x = a.parse_args()
    torch.set_grad_enabled(False)
    tokp = x.tokenizer or (pathlib.Path(x.modello) / "tokenizer.json" if x.modello else None)
    assert tokp and sha256(tokp) == SHA_TOKENIZER, "serve il tokenizer GPT-Neo di TinyStories (--tokenizer)"
    tok = Tokenizer.from_file(str(tokp))
    forward = carica_stera30m(x.checkpoint, x.device) if x.tipo == "stera30m" else carica_hf(x.modello, x.device)
    righe = json.loads((QUI / "incipit_test300.json").read_text())[: x.n]
    tot_nll = tot_tok = 0.0
    with torch.inference_mode(), open(x.out, "w") as f:
        for r in righe:
            ref = r["reference_ids"]
            ids_ref = torch.tensor([ref[:-1]], device=x.device)
            nll = float(F.cross_entropy(forward(ids_ref).reshape(-1, 50257).float().cpu(), torch.tensor(ref[1:]), reduction="sum"))
            ids = r["tokens"][:]; start = len(ids); g = torch.Generator(device="cpu").manual_seed(r["seed"]); eos = False
            for _ in range(min(180, 256 - start)):
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
    print(f"racconti {len(righe)}  NLL {tot_nll/tot_tok:.6f}  perplessita {math.exp(tot_nll/tot_tok):.4f}")


if __name__ == "__main__":
    main()
