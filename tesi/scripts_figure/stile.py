import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "Times"],
    "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "axes.grid": True, "grid.color": "#e3e5e8", "grid.linewidth": .6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": .02,
    "axes.formatter.use_locale": False,
})
# palette: STERA, TinyStories-33M, TinyStories-8M, neutro
C = {"stera": "#1f6f8b", "ts33": "#c2672b", "ts8": "#7a8b3a", "neutro": "#6b7280", "chiaro": "#b9c7d2"}
def it(x, nd=0):
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")
