"""Genera il modello fittizio STERA F10 a partire dal checkpoint finale di STERA 30M.

F10 serve solo come termine di confronto di risorse (memoria, lettura da disco) con Qwen3.5-9B:
NON è un modello addestrato. Allarga STERA 30M di un fattore F=10 (hidden size 7.680, 160 head,
feed-forward 30.720, 6 layer, stesso vocabolario): 4,63 miliardi di valori nella forma densa equivalente.

Regola deterministica, in sola aritmetica IEEE (nessuna funzione trigonometrica), quindi il file è
identico bit per bit su ogni macchina:
  - ogni blocco nuovo (r, c) copia le coordinate del blocco addestrato (r mod R, c mod C), divise per F
    e perturbate da una piccola combinazione di altre due sue coordinate con un coefficiente pseudo-casuale
    ricavato da un hash intero di (blocco, indice), così che ogni blocco abbia coordinate proprie;
  - scale, guadagni, U, V, LayerNorm, bias e posizioni sono ripetuti F volte (V diviso per F negli strati),
    come nella costruzione a funzione preservata usata nelle misure.
Uso: python genera_f10.py pesi/H122000_fp32_weights.npz sistemi/f10/F10_H122000.npz
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

F = 10
BASE = [("embedding", 50304, 768)] + [(f"layers.{l}.{k}", r, c) for l in range(6)
        for k, r, c in [("q", 768, 768), ("k", 768, 768), ("v", 768, 768), ("o", 768, 768), ("fc", 3072, 768), ("proj", 768, 3072)]]
NC = 3176


def coordinate(src, rt, ct, rf):
    """src: (rt*ct, 3176) float32 -> (rt*rf * ct*F, 3176) float32."""
    ctb = ct * F
    n = rt * rf * ctb
    out = np.empty((n, NC), np.float32)
    ci = np.arange(NC, dtype=np.uint64)
    ia, ib = (ci + 37) % NC, (ci + 151) % NC
    for tile in range(n):
        old = (tile // ctb % rt) * ct + (tile % ctb % ct)
        s = src[old].astype(np.float64)
        h = (np.uint64(tile) * np.uint64(2654435761) + ci * np.uint64(40503) + np.uint64(12345)) % np.uint64(2**32)
        coef = h.astype(np.float64) / 2**32 * 2 - 1
        out[tile] = ((s + 0.009 * coef * (s[ia] + 0.25 * s[ib])) / F).astype(np.float32)
    return out


def main(ingresso, uscita):
    import zipfile
    t0 = time.time()
    z = np.load(ingresso)
    info = {"coordinate": 0, "byte_pesi_fp32": 0}
    with zipfile.ZipFile(uscita, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        def scrivi(nome, arr):
            arr = np.ascontiguousarray(arr)
            with zf.open(nome + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, arr, allow_pickle=False)
            info["byte_pesi_fp32"] += arr.nbytes
        scrivi("position.weight", np.tile(z["position.weight"], (1, F)))
        for n in ["final_norm"] + [f"layers.{l}.{k}" for l in range(6) for k in ("ln1", "ln2")]:
            for a in ("weight", "bias"):
                scrivi(f"{n}.{a}", np.tile(z[f"{n}.{a}"], F))
        for l in range(6):
            for k in ("o", "fc", "proj"):
                scrivi(f"layers.{l}.{k}.bias", np.tile(z[f"layers.{l}.{k}.bias"], F))
        for name, r, c in BASE:
            rf = 1 if name == "embedding" else F
            rt, ct = r // 128, c // 128
            co = coordinate(z[name + ".coords"].reshape(-1, NC), rt, ct, rf)
            info["coordinate"] += co.size
            scrivi(name + ".coords", co); del co
            scrivi(name + ".scales", np.tile(z[name + ".scales"].reshape(rt, ct), (rf, F)).reshape(-1).astype(np.float32))
            scrivi(name + ".gout", np.tile(z[name + ".gout"], rf))
            scrivi(name + ".gin", np.tile(z[name + ".gin"], F))
            scrivi(name + ".U", np.tile(z[name + ".U"], (rf, 1)))
            V = np.tile(z[name + ".V"], (F, 1))
            scrivi(name + ".V", (V / F).astype(np.float32) if rf != 1 else V)
            print(name, round(time.time() - t0), "s", flush=True)
    sha = hashlib.sha256()
    with open(uscita, "rb") as fh:
        for blocco in iter(lambda: fh.read(1 << 24), b""):
            sha.update(blocco)
    man = {"file": Path(uscita).name, "sha256": sha.hexdigest(), "sorgente": Path(ingresso).name,
           "sha256_sorgente": hashlib.sha256(Path(ingresso).read_bytes()).hexdigest(), "fattore": F,
           "coordinate": int(info["coordinate"]), "byte_pesi_fp32": int(info["byte_pesi_fp32"]),
           "valori_forma_densa": 4633067520}
    Path(uscita).with_suffix(".json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
