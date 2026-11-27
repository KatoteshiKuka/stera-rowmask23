#!/usr/bin/env python3
"""Download from Hugging Face, at pinned revisions, the files used to pre-train STERA for code.

StarCoderData requires accepting the terms of use on the dataset page and a login (hf auth login).
The files go to external/code/grezzi/<source>/ ("raw"); they can be deleted after tokenization.
The source names (c, cpp, python, algoritmi, frammenti) are folder names and are kept as they are.
"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

STARCODER = ("bigcode/starcoderdata", "9fc30b578cedaec69e47302df72cf00feed7c8c4")
OPENCODER = ("OpenCoder-LLM/opc-annealing-corpus", "078e4bc1d39725cccc18514d2e3278a3fe191110")
TOKENIZER = ("bigcode/starcoder2-3b", "733247c55e3f73af49ce8e9c7949bf14af205928")

FILES = {
    "c":        (STARCODER, [f"c/train-{i:05d}-of-00053.parquet" for i in range(5)]),
    "cpp":      (STARCODER, [f"cpp/train-{i:05d}-of-00048.parquet" for i in range(2)]),
    "python":   (STARCODER, [f"python/train-{i:05d}-of-00059.parquet" for i in range(1)]),
    "algoritmi": (OPENCODER, [f"algorithmic_corpus/{i}.jsonl" for i in range(1, 13)]),
    "frammenti": (OPENCODER, [f"synthetic_code_snippet/{i}.jsonl" for i in range(1, 5)]),
}


def main(root="external/code"):
    root = Path(root)
    hf_hub_download(TOKENIZER[0], "tokenizer.json", revision=TOKENIZER[1], local_dir=root / "tokenizer")
    for name, ((repo, rev), files) in FILES.items():
        for f in files:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=root / "grezzi" / name)
            print(name, p, Path(p).stat().st_size, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
