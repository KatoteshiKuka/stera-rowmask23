"""Figure delle componenti della proiezione STERA (Capitolo 4) e dello studio del rango (Capitolo 7).
Legge i pesi FP32 del checkpoint finale (sistemi/modelli/H122000_fp32_weights.npz) e i risultati in valutazione/varianti/."""
import json, pathlib
import numpy as np
from stile import plt, C, it
from matplotlib.ticker import FuncFormatter

R = pathlib.Path(__file__).resolve().parent.parent
P = R.parent
z = np.load(P / "sistemi/modelli/H122000_fp32_weights.npz")
proj = sorted({k.rsplit(".", 1)[0] for k in z.files if k.endswith(".coords")})
(R / "figure/formalismo").mkdir(exist_ok=True); (R / "figure/risultati").mkdir(exist_ok=True)
fmt = FuncFormatter(lambda v, _: it(v, 1) if abs(v - round(v)) > 1e-9 else it(v))

# 1. scale
s = np.concatenate([z[p + ".scales"].ravel() for p in proj])
fig, ax = plt.subplots(figsize=(6.0, 2.4))
ax.hist(s, bins=70, color=C["stera"])
ax.axvline(1, color="#333", ls=":", lw=1); ax.text(1.6, ax.get_ylim()[1] * .85, "valore iniziale: 1", fontsize=8.5)
ax.set_xlabel(r"scala $\alpha^{(p,q)}$"); ax.set_ylabel("numero di blocchi"); ax.xaxis.set_major_formatter(fmt)
fig.savefig(R / "figure/formalismo/scale_apprese.pdf"); plt.close(fig)

# 2. guadagni
go = np.concatenate([z[p + ".gout"] for p in proj]); gi = np.concatenate([z[p + ".gin"] for p in proj])
fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.4), sharey=True)
for ax, v, nome, col in [(axs[0], go, r"$g_{\mathrm{out}}$ (righe)", C["stera"]), (axs[1], gi, r"$g_{\mathrm{in}}$ (colonne)", C["ts33"])]:
    ax.hist(v, bins=80, color=col, log=True); ax.axvline(1, color="#333", ls=":", lw=1)
    ax.set_xlabel(nome); ax.xaxis.set_major_formatter(fmt)
axs[0].set_ylabel("numero (scala log.)")
fig.savefig(R / "figure/formalismo/guadagni_appresi.pdf"); plt.close(fig)

# 3. spettro del basso rango: valori singolari di U V^T normalizzati
def spettro(p):
    _, ru = np.linalg.qr(z[p + ".U"].astype(np.float64)); _, rv = np.linalg.qr(z[p + ".V"].astype(np.float64))
    sv = np.linalg.svd(ru @ rv.T, compute_uv=False); return sv / sv[0]
gruppi = {"attenzione ($W_Q,W_K,W_V,W_O$)": [p for p in proj if p.split(".")[-1] in "qkvo"],
          "feed-forward ($W_1,W_2$)": [p for p in proj if p.split(".")[-1] in ("fc", "proj")],
          "embedding": ["embedding"]}
fig, ax = plt.subplots(figsize=(6.0, 2.6))
for (nome, ps), col in zip(gruppi.items(), [C["stera"], C["ts33"], C["ts8"]]):
    S = np.array([spettro(p) for p in ps]); k = np.arange(1, S.shape[1] + 1)
    ax.plot(k, np.median(S, 0), color=col, lw=1.8, label=nome)
    if len(ps) > 1: ax.fill_between(k, S.min(0), S.max(0), color=col, alpha=.15, lw=0)
ax.set_xlabel("indice del valore singolare $k$"); ax.set_ylabel(r"$\sigma_k/\sigma_1$")
ax.set_xlim(1, 96); ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(fmt); ax.legend(frameon=False)
fig.savefig(R / "figure/formalismo/spettro_basso_rango.pdf"); plt.close(fig)

# 4. studio del rango ridotto (Capitolo 7)
d = json.load(open(P / "valutazione/varianti/rango_ridotto/risultati_giudizio_rango_ridotto.json"))["ranghi"]
k = sorted(int(x) for x in d); nll = [d[str(x)]["nll_test"] for x in k]; pc = [d[str(x)]["pct"] for x in k]
lo = [d[str(x)]["pct"] - d[str(x)]["wilson95"][0] for x in k]; hi = [d[str(x)]["wilson95"][1] - d[str(x)]["pct"] for x in k]
fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.5))
axs[0].plot(k, nll, "o-", color=C["stera"]); axs[0].set_xlabel("rango usato $k$"); axs[0].set_ylabel("NLL TEST (nat/token)")
axs[1].errorbar(k, pc, yerr=[lo, hi], fmt="o-", color=C["ts33"], capsize=2); axs[1].set_xlabel("rango usato $k$"); axs[1].set_ylabel("racconti corretti (%)")
for ax in axs: ax.set_xticks(k); ax.yaxis.set_major_formatter(fmt)
fig.savefig(R / "figure/risultati/rango_ridotto.pdf"); plt.close(fig)
print("ok", len(proj), round(float(np.median(spettro('layers.0.q'))), 3))
