#!/usr/bin/env python3
"""Prepare a copy of the thesis for Overleaf, with the TikZ figures precompiled to PDF.

The figures in figure/tikz/ are compiled once with tectonic (standalone class, 11 pt, same colours and
libraries as the thesis) and included in the copy with \\includegraphics, so that the Overleaf build
does not have to redraw them. The rest of the sources is identical. Produces <output>/ and <output>.zip.

Usage: python export_overleaf.py [output_folder]   (default: ../thesis_overleaf)
"""
import re, shutil, subprocess, sys, tempfile
from pathlib import Path

THESIS = Path(__file__).resolve().parent
EXCLUDED = {"preview", "figure_scripts", "figure_data", "main.pdf", "thesis_STERA_Nardoni.pdf", "export_overleaf.py"}
WRAPPER = r"""\documentclass[11pt,border=0pt]{standalone}
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


def main(output=None):
    output = Path(output or THESIS.parent / "thesis_overleaf").resolve()
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(THESIS, output, ignore=lambda d, names: [n for n in names if n in EXCLUDED or n.startswith(".")
                                                         or n.endswith((".aux", ".log", ".toc", ".out", ".bbl", ".blg"))])
    pdf = output / "figure" / "tikz_pdf"; pdf.mkdir()
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted((THESIS / "figure" / "tikz").glob("*.tex")):
            source = Path(tmp) / f"{f.stem}.tex"
            source.write_text(WRAPPER % f.resolve().as_posix())
            subprocess.run(["tectonic", "-X", "compile", str(source)], check=True, capture_output=True)
            shutil.copy(Path(tmp) / f"{f.stem}.pdf", pdf / f"{f.stem}.pdf")
    replaced = 0
    for tex in (output / "capitoli").glob("*.tex"):
        s = tex.read_text()
        s, a = re.subn(r"\\resizebox\{([^{}]+)\}\{!\}\{\\input\{figure/tikz/([\w]+)\}\}", r"\\includegraphics[width=\1]{figure/tikz_pdf/\2}", s)
        s, b = re.subn(r"\\input\{figure/tikz/([\w]+)\}", r"\\includegraphics{figure/tikz_pdf/\1}", s)
        replaced += a + b
        tex.write_text(s)
    archive = shutil.make_archive(str(output), "zip", output)
    print(f"precompiled figures: {len(list(pdf.glob('*.pdf')))}, replaced inclusions: {replaced}")
    print(archive)


if __name__ == "__main__":
    main(*sys.argv[1:])
