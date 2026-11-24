# TinyStories ufficiali

Pesi, configurazioni e tokenizer sono in [modelli/TinyStories](../../modelli/TinyStories/). Il [manifest](../../modelli/TinyStories/manifest.json) registra revisioni pubbliche, parametri e SHA-256. Il benchmark usa l’ambiente Transformers/PyTorch MPS collegato come `.venv-tinystories`.

Dalla root del workspace:

```sh
GPTNEO_SDPA=1 PROMPT_OFFSET=15 ./.venv-tinystories/bin/python benchmark/TinyStories/misura.py 33M 1
```

Lo script accetta anche `8M` e `28M`. Le campagne storiche di velocità presenti qui riguardano 8M e 28M. I materiali del 33M includono le valutazioni storiche di qualità e narrativa. Gli script originali della qualità e i report sono conservati come archivio, con i riferimenti della campagna originale.
