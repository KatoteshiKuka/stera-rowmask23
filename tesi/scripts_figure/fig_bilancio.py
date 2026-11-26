"""Figura del bilancio dei parametri (Capitolo 4): risparmio di un modello STERA rispetto alla forma densa
al variare della dimensione d, per rango fisso e per rango proporzionale alla dimensione.
Modello di riferimento: 6 layer, feed-forward 4d, vocabolario 50.304, 2.048 posizioni (come STERA 30M)."""
import pathlib
from fractions import Fraction
from stile import plt, C, it
from matplotlib.ticker import FuncFormatter

R = pathlib.Path(__file__).resolve().parent.parent
K = Fraction(3177, 16384)


def risparmio(d, r, L=6, V=50304, T=2048):
    f = 4 * d
    M = V * d + L * (4 * d * d + 2 * f * d)
    Rr = (V + d) + L * (8 * d + 2 * (f + d))
    A = T * d + 2 * d * (2 * L + 1) + L * (2 * d + f)
    return 100 * float(1 - (K * M + (1 + r) * Rr + A) / (M + A))


ds = [256 * 2 ** (i / 4) for i in range(0, 25)]
ds = [128 * round(x / 128) for x in ds]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.axhline(100 * float(1 - K), color="#333", ls=":", lw=1)
ax.text(ds[0], 100 * float(1 - K) + 1.2, "tetto del formato RowMask23: 80,6%", fontsize=8.5, color="#333")
serie = [("senza low-rank", lambda d: 0, C["ts8"], "-"), ("$r=32$", lambda d: 32, C["chiaro"], "-"),
         ("$r=96$", lambda d: 96, C["stera"], "-"), ("$r=d/8$", lambda d: d // 8, C["ts33"], "--")]
for nome, rf, col, ls in serie:
    ax.plot(ds, [risparmio(d, rf(d)) for d in ds], color=col, ls=ls, lw=1.8, label=nome)
ax.scatter([768], [risparmio(768, 96)], color=C["stera"], zorder=3, s=22)
ax.annotate("STERA 30M\n63,3%", (768, risparmio(768, 96)), xytext=(1100, 47), fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color="#666", lw=.7))
ax.set_xscale("log", base=2)
ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384]); ax.set_xticklabels(["256", "512", "1024", "2048", "4096", "8192", "16384"])
ax.set_xlabel("Dimensione del modello $d$"); ax.set_ylabel("Risparmio rispetto alla forma densa (%)")
ax.set_ylim(30, 85); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v)))
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
fig.savefig(R / "figure/formalismo/risparmio_crescita.pdf")
print("ok", round(risparmio(768, 96), 2))
