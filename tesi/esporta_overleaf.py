#!/usr/bin/env python3
"""Prepara una copia della tesi per Overleaf, con le figure TikZ precompilate in PDF.

Le figure di figure/tikz/ sono compilate una volta con tectonic (classe standalone, 11 pt, stessi colori e
librerie della tesi) e nella copia vengono incluse con \\includegraphics, cosi la compilazione su Overleaf
non deve ridisegnarle. Il resto dei sorgenti e identico. Produce <uscita>/ e <uscita>.zip.

Uso: python esporta_overleaf.py [cartella_uscita]
"""
import re, shutil, subprocess, sys, tempfile
from pathlib import Path

TESI = Path(__file__).resolve().parent
ESCLUSI = {"anteprima", "scripts_figure", "dati_figure", "main.pdf", "tesi_STERA_Nardoni.pdf", "esporta_overleaf.py"}
INVOLUCRO = r"""\documentclass[11pt,border=0pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[italian]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,decorations.pathreplacing}
\definecolor{steraBlu}{HTML}{1F6F8B}
\definecolor{steraArancio}{HTML}{C2672B}
\definecolor{steraVerde}{HTML}{7A8B3A}
\definecolor{steraGrigio}{HTML}{6B7280}
\begin{document}
\input{%s}
\end{document}
"""


def main(uscita=None):
    uscita = Path(uscita or TESI.parent / "tesi_overleaf").resolve()
    if uscita.exists():
        shutil.rmtree(uscita)
    shutil.copytree(TESI, uscita, ignore=lambda d, nomi: [n for n in nomi if n in ESCLUSI or n.startswith(".")
                                                         or n.endswith((".aux", ".log", ".toc", ".out", ".bbl", ".blg"))])
    pdf = uscita / "figure" / "tikz_pdf"; pdf.mkdir()
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted((TESI / "figure" / "tikz").glob("*.tex")):
            sorgente = Path(tmp) / f"{f.stem}.tex"
            sorgente.write_text(INVOLUCRO % f.resolve().as_posix())
            subprocess.run(["tectonic", "-X", "compile", str(sorgente)], check=True, capture_output=True)
            shutil.copy(Path(tmp) / f"{f.stem}.pdf", pdf / f"{f.stem}.pdf")
    sostituite = 0
    for tex in (uscita / "capitoli").glob("*.tex"):
        s = tex.read_text()
        s, a = re.subn(r"\\resizebox\{([^{}]+)\}\{!\}\{\\input\{figure/tikz/([\w]+)\}\}", r"\\includegraphics[width=\1]{figure/tikz_pdf/\2}", s)
        s, b = re.subn(r"\\input\{figure/tikz/([\w]+)\}", r"\\includegraphics{figure/tikz_pdf/\1}", s)
        sostituite += a + b
        tex.write_text(s)
    archivio = shutil.make_archive(str(uscita), "zip", uscita)
    print(f"figure precompilate: {len(list(pdf.glob('*.pdf')))}, inclusioni sostituite: {sostituite}")
    print(archivio)


if __name__ == "__main__":
    main(*sys.argv[1:])
