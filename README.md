# STERA – a structured RowMask23 weight parametrization for Transformers

Repository of the Master's thesis in Computer Engineering (Università degli Studi Roma Tre, academic year 2025/2026)

**STERA: Progettazione e valutazione di una parametrizzazione strutturata dei pesi per modelli linguistici Transformer**
(*STERA: design and evaluation of a structured weight parametrization for Transformer language models*)
Andrea Nardoni — advisor Prof. Fabio Gasparetti

The thesis (in Italian) is in [`thesis/thesis_STERA_Nardoni.pdf`](thesis/thesis_STERA_Nardoni.pdf).
Code, folders and documentation are in English.

---

## In short

**STERA** (*Structured Training, Execution and Reconstruction Architecture*) parametrizes every linear projection
of a Transformer as

$$W = \mathrm{diag}(g_\text{out})\,\widetilde W\,\mathrm{diag}(g_\text{in}) + U V^\top,$$

where $\widetilde W$ is a mosaic of $128\times128$ **RowMask23** blocks, each multiplied by a learned scale.
A RowMask23 block is described by **3,176 coordinates** instead of 16,384 independent coefficients:

- a free **diagonal** (128 values);
- an **XOR base** shared along the generalized diagonals $\{(i, i\oplus s)\}$ (127 values);
- **23 corrections per row** at fixed positions $\sigma(i,k) = 1 + ((13k+7i) \bmod 127)$ (2,921 values).

The map from coordinates to block is linear and injective, the gradient with respect to the coordinates has a closed
form, and the product with a vector is computed with the Walsh–Hadamard transform, without rebuilding the matrix.
The optimizer updates the coordinates directly: there is no dense matrix to approximate.

**STERA 30M** is a 6-layer decoder-only Transformer (hidden size 768, 16 heads, feed-forward 3,072, GPT-Neo
vocabulary of 50,257 tokens) in which every projection, including the embedding matrix tied with the output head,
is a STERA projection with row and column gains and a low-rank term of rank 96. It has **30,346,182 trainable
parameters** (82,721,280 values in the dense-equivalent form) and was trained from scratch on TinyStoriesV2-GPT4 for
**122,000 updates** (1,529,514,422 target tokens) on two NVIDIA P100 GPUs.

## Main results

On 300 TEST stories, with the same prompts, the same tokenizer and the same generation procedure for the three models:

| Model | Parameters | TEST NLL | Correct stories (blind judgment) |
|---|---:|---:|---:|
| TinyStories-8M | 19,702,528 | 1.515 | 65/300 (21.7%) |
| **STERA 30M** | **30,346,182** | **1.174** | **132/300 (44.0%)** |
| TinyStories-33M | 68,514,048 | 1.324 | 145/300 (48.3%) |

- STERA 30M − TinyStories-8M: **+22.3 points** (95% CI [+15.3; +29.3], McNemar with Holm p < 10⁻⁸).
- STERA 30M − TinyStories-33M: **−4.3 points** (95% CI [−12.0; +3.7], p = 0.32, not significant).
- Compact FP32 weights: **121.385 MB**, against 330.885 MB for the dense expansion (−63.3%) and 274.056 MB for
  TinyStories-33M (−55.7%); 86.097 MB with the low-rank term at 8 bits (−74.0%).
- Decode on the same model: **−34.8%** median time per token for the compact engine relative to the dense engine
  (Mac M1, MLX/Metal, 30 paired measurements in five campaigns), with identical tokens.
- 8-bit low-rank term: NLL unchanged (+0.007%), decode 17.8% faster.
- The RowMask23 blocks guarantee −80.6% on the structured part; the saving of a STERA model as a function of the
  rank is derived in closed form in Chapter 4.
- A fictitious, untrained model the size of Qwen3.5-9B built with STERA (F10, 4.63 billion dense-equivalent values)
  runs with weights and KV cache resident in 16 GiB up to 64,000 tokens of context, at 12.6 tokens/s with short
  contexts and 3.9 tokens/s at 64,000 tokens; its dense form and Qwen3.5-9B exceed 16 GiB with the weights alone.

Full details and all numbers: [`docs/RESULTS.md`](docs/RESULTS.md).

## Project map

```
STERA-RowMask23/
├── README.md                         this page
├── docs/
│   ├── REPRODUCTION.md               full step-by-step procedure
│   ├── RESULTS.md                    every number of the thesis and where it comes from
│   ├── REPRODUCIBILITY_CHECK.md      outcome of rerunning every step
│   └── ARTIFACTS.md                  SHA-256 hashes of data, models and results
├── training/
│   ├── prepare_data.py               corpus tokenization and hash-based TRAIN/DEV/TEST split
│   ├── seal_data.py                  data contract recorded in the checkpoints
│   ├── ts68_core.py                  RowMask23 blocks, model, checkpoint reading and writing
│   ├── ts68_v3.py                    distributed trainer (PyTorch, 2 GPUs) with gains and low-rank term
│   ├── train_stera30m.sh             training command, 0 → 122,000 updates
│   ├── training_log_0_122k.txt       TRAIN loss every 20 steps and DEV NLL every 2,000
│   ├── check_20261124/               logs of the training-engine checks (20 steps on CPU, 200 steps on 2 × P100)
│   └── data/                         split manifest, data contract, stories reserved for DEV
├── evaluation/
│   ├── protocol/
│   │   ├── protocollo_test300.json    full description of the protocol
│   │   ├── incipit_test300.json       the 300 prompts with reference stories and seeds
│   │   ├── select_prompts.py          deterministic prompt selection
│   │   ├── evaluate_model.py          NLL + generation, same code for the three models
│   │   └── esclusioni/                prompts used during development, excluded from TEST
│   ├── generations/                  continuations of the three models (300 each), with per-story NLL
│   ├── blind_judgment/
│   │   ├── rubrica.txt                judge instructions and rubric (rubric_en.txt: English translation)
│   │   ├── prepare_blind_set.py       anonymization and shuffling of the 900 texts
│   │   ├── racconti_anonimi_900.json  the judged texts (T000–T899)
│   │   ├── giudizi_blocchi/           raw judgments, 30 batches of 30 texts
│   │   ├── giudizi_900.json           all judgments
│   │   ├── abbinamento_modelli.json   mapping T### → model and prompt
│   │   └── aggregate_and_decode.py    statistics: Wilson, bootstrap, McNemar, Holm, NLL
│   ├── variants/                     post-hoc studies of the components (8-bit low rank, reduced rank, scales and gains)
│   ├── checks/                       numerical check of forward and backpropagation
│   └── risultati_3modelli.json       final result of the comparison
├── systems/
│   ├── engines/                      MLX/Metal engines: direct product on the coordinates and dense equivalent
│   ├── benchmark/                    paired measurements, TinyStories, F10 context study, weight export, lossless archive
│   ├── f10/                          generation and loading of the fictitious F10 model
│   ├── models/                       tokenizer and manifests (weights are not versioned)
│   └── results/                      raw data and reports of the system measurements (Chapter 10)
├── code/                             STERA for source code (separate experiment, not part of the thesis)
├── presentation/                     slides of the thesis defense (in Italian) and the script that builds them
├── thesis/
│   ├── main.tex, capitoli/, sommario.tex, bibliografia.bib, TesiDiaUniroma3.cls   (in Italian)
│   ├── figure/                       figures (PDF and TikZ sources)
│   ├── figure_scripts/               scripts that draw the plots from the data
│   ├── figure_data/                  data used by the plots
│   ├── preview/                      local editor with live recompilation and click-to-source
│   ├── export_overleaf.py            copy of the thesis for Overleaf with precompiled TikZ figures
│   └── thesis_STERA_Nardoni.pdf      compiled PDF
├── weights/                          (not versioned) final STERA 30M checkpoint
├── requirements_training.txt
└── requirements_systems.txt
```

**Naming of data files.** Data and result files keep the names under which they were recorded (for example
`risultati_3modelli.json` = results of the three models, `giudizi_900.json` = the 900 judgments,
`abbinamento_*.json` = assignment, `campagne_*` = campaigns, `protocollo` = protocol, `riepilogo.json` = summary),
because these names appear in manifests, logs and the hash tables of the thesis. JSON keys are kept for the same
reason. Archived folders under `systems/results/` are left exactly as recorded.

## From the thesis to the code

| Thesis chapter | Where it is in the repository |
|---|---|
| 4 – The STERA parametrization | `thesis/figure_scripts/fig_bilancio.py` (parameter budget) |
| 5 – Forward and backpropagation | `evaluation/checks/check_backward_stera.py` (numerical check), `training/ts68_core.py` (reconstruction) |
| 6 – The training and inference engine | `training/ts68_v3.py`, `systems/engines/stera4b_wht_*`, `systems/engines/prefill_expansion.py`, `systems/engines/stera_rank8_mlx.py` |
| 7 – STERA 30M: the trained model | `training/prepare_data.py`, `training/train_stera30m.sh`, `thesis/figure_scripts/fig_componenti.py` |
| 8 – Validation methodology | `evaluation/protocol/` |
| 9 – Learning and story quality | `training/training_log_0_122k.txt`, `evaluation/` |
| 10 – Memory and execution time | `systems/` (F10: `systems/f10/`, `systems/benchmark/f10_*.py`) |
| Figures | `thesis/figure_scripts/` (plots), `thesis/figure/tikz/` (diagrams) |

## Quick reproduction

```bash
# 1. data: download corpus and tokenizer (see docs/REPRODUCTION.md), then
export STERA_CORPUS=external/TinyStoriesV2-GPT4-train.txt
export STERA_TOKENIZER=external/tokenizer.json
export STERA_DATA=external/tokenized_corpus
python training/prepare_data.py && python training/seal_data.py

# 2. training (2 GPUs)
export STERA_RUN=external/checkpoint
bash training/train_stera30m.sh
cp $STERA_RUN/TS68V2_STEP122000.ckpt.gz weights/stera30m_step122000.ckpt.gz
# (instead of step 2: download the checkpoint from Release v1.0, see weights/README.md)

# 3. evaluation of the three models
python evaluation/protocol/evaluate_model.py stera30m --checkpoint weights/stera30m_step122000.ckpt.gz \
       --tokenizer $STERA_TOKENIZER --device cuda --out evaluation/generations/stera30m.jsonl
python evaluation/protocol/evaluate_model.py hf --model external/TinyStories-33M --out evaluation/generations/tinystories33m.jsonl

# 4. statistics of the blind judgment
python evaluation/blind_judgment/aggregate_and_decode.py

# 5. thesis
cd thesis && tectonic -X compile main.tex
```

The full procedure, with indicative times and intermediate checks, is in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Licenses of the data and reference models

- TinyStories corpus (`roneneldan/TinyStories`): CDLA-Sharing-1.0 license.
- Reference checkpoints `roneneldan/TinyStories-8M` and `roneneldan/TinyStories-33M`: downloaded from Hugging Face,
  under the licenses stated on their model cards.
- Figure 2.1 of the thesis reproduces Figure 1 of Vaswani et al. (2017), with attribution (see `thesis/FIGURE_SOURCES.txt`).
