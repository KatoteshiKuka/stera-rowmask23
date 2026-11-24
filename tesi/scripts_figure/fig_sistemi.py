import pathlib
from stile import plt, C, it
from matplotlib.ticker import FuncFormatter
R = pathlib.Path(__file__).resolve().parent.parent
out = R/"figure/risultati"
# 1) byte dei pesi
rows = [("STERA 30M, coordinate INT8", 74.241, C["stera"]), ("TinyStories-8M, FP32", 78.810, C["ts8"]),
        ("STERA 30M, archivio senza perdita", 102.353, C["chiaro"]), ("STERA 30M, FP32", 121.385, C["stera"]),
        ("TinyStories-33M, FP32", 274.056, C["ts33"]), ("STERA 30M, espansione densa FP32", 330.885, C["neutro"])]
fig, ax = plt.subplots(figsize=(6.0, 2.6))
for i, (lab, v, col) in enumerate(rows):
    ax.barh(i, v, color=col, height=.62, hatch="///" if "espansione" in lab else None, edgecolor="white" if "espansione" not in lab else "#ffffff")
    ax.text(v+4, i, it(v, 1), va="center", fontsize=8.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlabel("Milioni di byte"); ax.set_xlim(0, 380); ax.grid(axis="y", visible=False)
fig.savefig(out/"byte_pesi_stera30m.pdf")
# 2) decode appaiato
# mediane delle cinque campagne di misura (sistemi/risultati/campagne_20261124/protocollo/riepilogo.json)
import json
RJ = json.loads((R.parent/"sistemi/risultati/campagne_20261124/protocollo/riepilogo.json").read_text())["confronto_appaiato"]
inp = ["10", "25", "40", "55", "70", "85"]; var = [RJ["variazione_decode_pct_per_input"][o] for o in inp]
td = [RJ["totale_s_per_input"][o]["denso"] for o in inp]; ts = [RJ["totale_s_per_input"][o]["compatto"] for o in inp]
med = RJ["variazione_decode_pct"]["mediana"]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"wspace": .45})
a.bar(range(6), var, color=C["stera"], width=.6); a.axhline(med, color=C["ts33"], lw=1, ls="--")
a.text(5.45, med, "mediana\n$" + it(med, 1).replace(",", "{,}") + "\\%$", va="center", ha="left", fontsize=8, color=C["ts33"])
a.set_xticks(range(6)); a.set_xticklabels([f"P{i+1}" for i in range(6)]); a.set_ylabel("Variazione tempo per token (%)")
a.set_ylim(-40, 0); a.set_xlim(-.6, 6.6); a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v)))
a.set_title("(a) Decode, compatto rispetto a denso", fontsize=9.5); a.grid(axis="x", visible=False)
for i in range(6):
    b.plot([0, 1], [td[i], ts[i]], color=C["chiaro"], lw=.9, zorder=1)
b.scatter([0]*6, td, color=C["neutro"], zorder=2, s=16, label="Denso"); b.scatter([1]*6, ts, color=C["stera"], zorder=2, s=16, label="Compatto")
b.set_xticks([0, 1]); b.set_xticklabels(["Denso", "Compatto"]); b.set_xlim(-.4, 1.4); b.set_ylim(.5, .9)
b.set_ylabel("Tempo totale (s)"); b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v, 2)))
b.set_title("(b) Tempo totale della richiesta", fontsize=9.5); b.grid(axis="x", visible=False)
fig.savefig(out/"decode_appaiato_stera30m.pdf")
# 3) soglia RAM e lettura SSD
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.4), gridspec_kw={"width_ratios": [1.5, 1], "wspace": .55})
rows = [("F10 compatto FP32", 3.73, C["stera"]), ("Qwen3.5-9B BF16", 16.68, C["ts33"]), ("F10 espanso FP32", 17.32, C["neutro"])]
for i, (lab, v, col) in enumerate(rows):
    a.barh(i, v, color=col, height=.6); a.text(v+.3, i, it(v, 2), va="center", fontsize=8.5)
a.axvline(16, color="#333", ls=":", lw=1); a.text(16, 2.55, "16 GiB", ha="center", fontsize=8.5)
a.set_yticks(range(3)); a.set_yticklabels([r[0] for r in rows], fontsize=8.5); a.set_xlim(0, 21); a.set_ylim(-.5, 2.9)
a.set_xlabel("Pesi (GiB)"); a.set_title("(a) Spazio dei soli pesi", fontsize=9.5); a.grid(axis="y", visible=False)
b.bar([0, 1], [1.685, 7.821], color=[C["stera"], C["neutro"]], width=.55)
for x, v in zip([0, 1], [1.685, 7.821]): b.text(x, v+.2, it(v, 2), ha="center", fontsize=8.5)
b.set_xticks([0, 1]); b.set_xticklabels(["4,0 GB\ncompatto", "18,6 GB\nespanso"], fontsize=8.5)
b.set_ylabel("Lettura da SSD (s)"); b.set_ylim(0, 9.2); b.set_title("(b) Lettura sequenziale", fontsize=9.5); b.grid(axis="x", visible=False)
b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v)))
fig.savefig(out/"soglia_ram_m1.pdf")
print("ok")
