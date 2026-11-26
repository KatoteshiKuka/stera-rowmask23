import json, pathlib
from stile import plt, C, it
from matplotlib.ticker import FuncFormatter
R = pathlib.Path(__file__).resolve().parent.parent
d = json.loads((R/"dati_figure/CONFRONTO_DECODIFICATO_3MODELLI.json").read_text())
M = [("TinyStories-8M", "TinyStories-8M", C["ts8"]), ("STERA30M_H122000", "STERA 30M", C["stera"]), ("TinyStories-33M", "TinyStories-33M", C["ts33"])]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"width_ratios": [1.15, 1], "wspace": .55})
# (a) storie corrette con Wilson
for i, (k, lab, col) in enumerate(M):
    m = d["models"][k]; lo, hi = m["wilson95"]
    a.bar(i, m["pct"], color=col, width=.62)
    a.errorbar(i, m["pct"], yerr=[[m["pct"]-lo], [hi-m["pct"]]], color="#333", capsize=3, lw=.8)
    a.text(i, hi+1.5, f"{m['correct']}/300", ha="center", fontsize=8.5)
a.set_xticks(range(3)); a.set_xticklabels([l for _, l, _ in M], fontsize=8.5)
a.set_ylabel("Racconti corretti (%)"); a.set_ylim(0, 62)
a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v)))
a.set_title("(a) Giudizio cieco, 300 prompt", fontsize=9.5)
a.grid(axis="x", visible=False)
# (b) NLL
for i, (k, lab, col) in enumerate(M):
    v = d["nll"][k]["nll"]
    b.bar(i, v, color=col, width=.62); b.text(i, v+.03, it(v, 3), ha="center", fontsize=8.5)
b.set_xticks(range(3)); b.set_xticklabels([l for _, l, _ in M], fontsize=8.5)
b.set_ylabel("NLL (nat per token)"); b.set_ylim(0, 1.75)
b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v, 1)))
b.set_title("(b) Racconti di riferimento", fontsize=9.5)
b.grid(axis="x", visible=False)
for ax in (a, b):
    plt.setp(ax.get_xticklabels(), rotation=12)
fig.savefig(R/"figure/risultati/narrativa_nll_3modelli.pdf")
# figura 2: categorie S/P/N impilate
fig, ax = plt.subplots(figsize=(6.0, 1.9))
cols = {"S": "#4c8c6b", "P": "#d9a441", "N": "#b5473a"}; names = {"S": "S (coerente)", "P": "P (criticità sostanziale)", "N": "N (incoerente)"}
for i, (k, lab, _) in enumerate(M[::-1]):
    left = 0
    for c in "SPN":
        v = d["models"][k]["categories"].get(c, 0)
        ax.barh(i, v, left=left, color=cols[c], height=.6, label=names[c] if i == 0 else None)
        if v >= 12: ax.text(left+v/2, i, str(v), ha="center", va="center", color="white", fontsize=8.5)
        left += v
ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l, _ in M[::-1]])
ax.set_xlim(0, 300); ax.set_xlabel("Numero di racconti")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.32), fontsize=8.5)
ax.grid(axis="y", visible=False)
fig.savefig(R/"figure/risultati/categorie_spn_3modelli.pdf")
print("ok")
