#!/usr/bin/env python3
"""Scarica da Hugging Face, alle revisioni fissate, i file usati per il pre-addestramento di STERA per il codice.

StarCoderData richiede di accettare le condizioni d'uso sulla pagina del dataset e un login (hf auth login).
I file vanno in esterni/codice/grezzi/<sorgente>/; dopo la tokenizzazione si possono cancellare.
"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

STARCODER = ("bigcode/starcoderdata", "9fc30b578cedaec69e47302df72cf00feed7c8c4")
OPENCODER = ("OpenCoder-LLM/opc-annealing-corpus", "078e4bc1d39725cccc18514d2e3278a3fe191110")
TOKENIZER = ("bigcode/starcoder2-3b", "733247c55e3f73af49ce8e9c7949bf14af205928")

FILE = {
    "c":        (STARCODER, [f"c/train-{i:05d}-of-00053.parquet" for i in range(5)]),
    "cpp":      (STARCODER, [f"cpp/train-{i:05d}-of-00048.parquet" for i in range(2)]),
    "python":   (STARCODER, [f"python/train-{i:05d}-of-00059.parquet" for i in range(1)]),
    "algoritmi": (OPENCODER, [f"algorithmic_corpus/{i}.jsonl" for i in range(1, 13)]),
    "frammenti": (OPENCODER, [f"synthetic_code_snippet/{i}.jsonl" for i in range(1, 5)]),
}


def main(radice="esterni/codice"):
    radice = Path(radice)
    hf_hub_download(TOKENIZER[0], "tokenizer.json", revision=TOKENIZER[1], local_dir=radice / "tokenizer")
    for nome, ((repo, rev), file) in FILE.items():
        for f in file:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=radice / "grezzi" / nome)
            print(nome, p, Path(p).stat().st_size, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
