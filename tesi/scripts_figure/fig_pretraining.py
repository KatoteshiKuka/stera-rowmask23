import re, pathlib, statistics
from stile import plt, C, it
from matplotlib.ticker import FuncFormatter
R = pathlib.Path(__file__).resolve().parent.parent
tr, dv, lr = [], [], []
for l in (R/"dati_figure/log_training_0_122k.txt").read_text().splitlines():
    st = int(re.search(r"step=(\d+)", l).group(1))
    if st > 122000: continue
    if l.startswith("VAL"):
        dv.append((st, float(re.search(r"nll=([\d.]+)", l).group(1))))
    else:
        tr.append((st, float(re.search(r"loss=([\d.]+)", l).group(1))))
        lr.append((st, float(re.search(r"lr=([\deE.+-]+)", l).group(1))))
tr.sort(); dv.sort(); lr.sort()
w = 50
med = [(tr[i][0], statistics.median(x for _, x in tr[max(0, i-w+1):i+1])) for i in range(len(tr))]
fig, (a, b) = plt.subplots(2, 1, figsize=(6.0, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": .12})
k = lambda s: s/1000
a.plot([k(s) for s, _ in tr], [x for _, x in tr], color=C["chiaro"], lw=.4, label="Loss del batch (TRAIN)")
a.plot([k(s) for s, _ in med], [x for _, x in med], color=C["stera"], lw=1.2, label="Mediana mobile, 50 misure")
a.plot([k(s) for s, _ in dv], [x for _, x in dv], "o-", color=C["ts33"], ms=2.6, lw=1, label="NLL su DEV (512 racconti)")
a.set_ylim(1.0, 2.2); a.set_ylabel("nat per token")
a.legend(frameon=False, loc="upper right")
fin = dv[-1]
a.annotate(f"{it(fin[1],3)}", (k(fin[0]), fin[1]), xytext=(-4, -14), textcoords="offset points", ha="right", fontsize=9, color=C["ts33"])
b.plot([k(s) for s, _ in lr], [x*1e4 for _, x in lr], color=C["neutro"], lw=1)
b.set_ylabel(r"Learning rate ($\times10^{-4}$)"); b.set_ylim(0, 7)
b.set_xlabel("Aggiornamenti (migliaia)")
b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v)))
a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v, 1))); b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: it(v, 1))); b.set_yticks([0, 3, 6])
b.set_xlim(0, 123)
fig.savefig(R/"figure/STERA30M/06_pretraining_stera30m.pdf")
print("ok", len(tr), len(dv), dv[0], dv[-1])
