# Official TinyStories models

Weights, configurations and tokenizers of TinyStories-8M and TinyStories-33M go in `systems/models/TinyStories/8M`
and `systems/models/TinyStories/33M` (not versioned). The public revisions and SHA-256 hashes are recorded in
[`systems/models/manifest_tinystories.json`](../../models/manifest_tinystories.json). The benchmark uses the
Transformers/PyTorch MPS environment of `requirements_systems.txt`.

From the `systems/` folder:

```sh
GPTNEO_SDPA=1 PROMPT_OFFSET=15 ../.venv-systems/bin/python benchmark/TinyStories/measure.py 33M 1
```

`measure.py` takes the model (`STERA`, `8M`, `33M`, the MLX variants `8M-MLX`, `33M-MLX`, `STERA-DENSO-MLX`, and the
STERA variants `STERA-R8`, `STERA-ESP`, `STERA-R8-ESP`) and a repetition index, and prints one JSON line. The model
labels are kept as in the archived results. The `run_*.sh` scripts run the campaigns used in the thesis.
