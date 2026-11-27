# Full reproduction

This guide goes from nothing to the numbers of the thesis. Every step says what it produces and how to check it.
Large files go into an `external/` folder (ignored by git) at the root of the repository.

**Naming.** Code, folders and documentation are in English. Data and result files keep the names under which they
were recorded (for example `risultati_3modelli.json`, `giudizi_900.json`, `campagne_20261124/protocollo/`), because
those names appear in manifests, logs and the hash tables of the thesis; their content is unchanged. JSON keys
written by the scripts are also kept, so a rerun produces files that can be compared byte for byte with the archive.
The scripts accept the Italian option and variable names used in the archived logs as aliases
(for example `STERA_DATI` for `STERA_DATA`, `STERA_PESI` for `STERA_WEIGHTS`, `--modello` for `--model`).

## 0. Environment

| Part | Hardware used in the thesis | Software |
|---|---|---|
| Data preparation, training, evaluation | server with 2 × NVIDIA P100 16 GB, Slurm | Python 3.12, PyTorch, tokenizers 0.20, transformers 4.46 (`requirements_training.txt`) |
| Evaluation of TinyStories-8M, statistics | Mac with Apple M1, 16 GB | Python 3.9+, PyTorch 2.8 on CPU |
| System measurements, backward check | Mac with Apple M1, 16 GB | Python 3.11, MLX 0.32.4, PyTorch 2.15 MPS, Transformers 5 (`requirements_systems.txt`) |
| Figures and thesis | any | matplotlib 3.9.4 (pixel-identical figures), [Tectonic](https://tectonic-typesetting.github.io) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_training.txt
```

## 1. Data

Download the corpus and the tokenizer at the revisions used in the thesis:

```bash
mkdir -p external
curl -L -o external/TinyStoriesV2-GPT4-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-train.txt
curl -L -o external/tokenizer.json \
  https://huggingface.co/roneneldan/TinyStories-33M/resolve/2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8/tokenizer.json
shasum -a 256 external/TinyStoriesV2-GPT4-train.txt external/tokenizer.json
```

The expected hashes are `6418d412…dec90443` (corpus, 2,227,753,162 bytes) and `f6ed3d30…cfeff5` (tokenizer).

Tokenization and split:

```bash
export STERA_CORPUS=external/TinyStoriesV2-GPT4-train.txt
export STERA_TOKENIZER=external/tokenizer.json
export STERA_DATA=external/tokenized_corpus
python training/prepare_data.py
```

The script assigns every story to TRAIN, DEV or TEST from the SHA-256 of the normalized text (uniform whitespace,
lower case), modulo 10,000: values 0–99 go to TEST, 100–199 to DEV, the rest to TRAIN. The 2,000 stories of
`training/data/racconti_riservati_dev_2000.jsonl` are forced into DEV. Identical stories always land in the same split.

Then write the data contract, which the trainer records in every checkpoint:

```bash
python training/seal_data.py
```

**Check:** `external/tokenized_corpus/MANIFEST.json` must match `training/data/MANIFEST_DATI.json` in the fields
`pack_sha256` (`5ae58177…`), `index_sha256` (`bbadffc0…`), `split_stories` (2,660,865 / 29,279 / 27,351) and
`split_target_tokens`. The `CONTRACT` file must match `training/data/CONTRACT` (the script checks it and prints
`matches the thesis: True`).

## 2. Training STERA 30M

```bash
export STERA_RUN=external/checkpoint
sbatch training/train_stera30m.sh        # or: bash training/train_stera30m.sh
```

Configuration: 6 layers, row and column gains, low-rank term of rank 96; batches of 64 sequences of at most
512 tokens, AdamW ($\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1), learning rate with 300 warmup steps up to
$6\cdot10^{-4}$, constant until step 64,000, linear decay to $6\cdot10^{-5}$ at step 80,000, then constant.
A full checkpoint (weights, Adam moments, data cursor) is saved every 2,000 steps; the script resumes from the
last one when relaunched.

Indicative time: about 12,000 tokens/s on two P100s, i.e. about 35 hours for 122,000 updates. The run of the
thesis was executed as three consecutive jobs of the same script.

At the end, copy the final checkpoint `external/checkpoint/TS68V2_STEP122000.ckpt.gz` to the path used by all
later steps (if you do not retrain, download it from the repository Release instead, see `weights/README.md`):

```bash
cp external/checkpoint/TS68V2_STEP122000.ckpt.gz weights/stera30m_step122000.ckpt.gz
shasum -a 256 weights/stera30m_step122000.ckpt.gz   # 418becfb…6ed0887
```

**Quick check without a GPU.** The DEV NLL of the final checkpoint can be obtained without training: the trainer
evaluates before the first step, so with `--steps 0` it prints only `VAL`. On a machine without a working
`torchrun`, set the single-process variables by hand:

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
  python training/ts68_v3.py --data "$STERA_DATA" --output /tmp/stera_val \
  --ckpt weights/stera30m_step122000.ckpt.gz --sha256 418becfbcaddcefd56f828f295138bbc865f9405a9060930c060ff7e6ced0887 \
  --layers 6 --gains --rank 96 --steps 0 --ctx 512 --nval 512
```

The expected result is `VAL step=122000 nll=1.152155`. In the same way, 20 steps from scratch on CPU reproduce
exactly the first lines of the log (loss 9.34693 at step 20; see `training/check_20261124/`).

**Check:** the `VAL` lines of the log must follow `training/training_log_0_122k.txt` (DEV NLL 10.928 at step 0,
1.172 at 80,000, 1.152 at 122,000). The final checkpoint of the thesis has hash `418becfb…6ed0887`.

**Forward and backpropagation check** (thesis, Chapter 5):

```bash
.venv-systems/bin/python evaluation/checks/check_backward_stera.py
```

It compares the formulas of the thesis with autograd and with finite differences, in double precision, and writes
`evaluation/checks/backward_results.json`. With PyTorch 2.15 the file is reproduced exactly (finite-difference
error 7.6e-8); other PyTorch versions give the same autograd agreement (≤1.4e-15) and a finite-difference error
of the same order (3.8e-7 with PyTorch 2.8).

## 3. Test prompts

The file `evaluation/protocol/incipit_test300.json` is already in the repository (SHA-256 `95bcffdb…032ec`).
To regenerate it:

```bash
python evaluation/protocol/select_prompts.py     # uses STERA_DATA and STERA_TOKENIZER
```

## 4. NLL and generation of the three models

Download the two references into `external/` at the revisions listed in `systems/models/manifest_tinystories.json`
(8M: `8612e3b1…`, 33M: `2ad0a164…`), for example with
`huggingface-cli download roneneldan/TinyStories-33M --revision 2ad0a164… --local-dir external/TinyStories-33M`.

```bash
cd evaluation/protocol
python evaluate_model.py stera30m --checkpoint ../../weights/stera30m_step122000.ckpt.gz \
       --tokenizer ../../external/tokenizer.json --device cuda --out stera30m.jsonl
python evaluate_model.py hf --model ../../external/TinyStories-33M --device cuda --out tinystories33m.jsonl
python evaluate_model.py hf --model ../../external/TinyStories-8M  --device cpu  --out tinystories8m.jsonl
```

The same script computes the NLL of the 300 reference stories (57,841 target tokens) and generates one
continuation per prompt with top-$k$ 40, temperature 0.75, at most 180 new tokens. The random generator runs on
the CPU with the seed of the prompt, so the continuations do not depend on the device.

**Check:** the continuations match token by token those in `evaluation/generations/`; rerunning the three models
on P100 reproduces all 900 continuations; on CPU 899 of 900 match (prompt F156 of TinyStories-33M diverges from
token 84 because of rounding). The mean NLLs differ by at most a few hundred-thousandths of a nat.

## 5. Blind judgment

```bash
python evaluation/blind_judgment/prepare_blind_set.py
```

creates `racconti_anonimi_900.json` (anonymized stories), the 30 batches of 30 texts and `abbinamento_modelli.json`
(the assignment to the models). The judge receives only the batches and `rubrica.txt` (the rubric, in Italian with
English score names; an English translation is in `rubric_en.txt`); for every text it returns the grammar,
consistency and plot scores (1–10), the S/P/N category and a reason. In the thesis the judge is Claude Opus 5.5,
which rated the 900 texts in a single session without access to the assignment; its judgments are in
`giudizi_blocchi/`.

```bash
python evaluation/blind_judgment/aggregate_and_decode.py
```

checks that all 900 texts were rated consistently with the rubric, opens the assignment and computes rates with
Wilson intervals, paired differences with bootstrap (25,000 resamples), McNemar tests with Holm correction and NLL
differences with bootstrap (10,000 resamples). The result is `evaluation/risultati_3modelli.json`; the hash of the
judgments is `40760aca…a5e8ea`.

## 6. System measurements (Mac with Apple Silicon)

The engines in `systems/engines/` read the weights from `systems/models/`:

| File | Content | How to obtain it |
|---|---|---|
| `H122000_fp32_weights.npz` | FP32 coordinates and tensors of the final checkpoint | `python systems/benchmark/export_weights.py weights/stera30m_step122000.ckpt.gz 418becfb…6ed0887 H122000` |
| `H122000_DENSE_FP32.safetensors` | dense-equivalent matrices | `python systems/benchmark/export_dense.py` |

All the measurements of the thesis use the final checkpoint. The engines read the weights named by the variable
`STERA_WEIGHTS` (or `STERA_PESI`; default `H122000`).

```bash
python3.11 -m venv .venv-systems && .venv-systems/bin/pip install -r requirements_systems.txt
sh systems/benchmark/paired_comparison.sh                 # one campaign: compact vs dense engine, 6 paired inputs
sh systems/benchmark/TinyStories/run_thesis_comparison.sh  # one campaign: STERA 30M and the two TinyStories on MPS
```

The numbers of the thesis come from five full campaigns on the weights of the final checkpoint, each started when
the one-minute load average drops below 1.5 and after a 60 s pause, with the Mac on mains power and no other
applications running:

```bash
LOAD_THRESHOLD=1.5 PAUSE=60 STERA_WEIGHTS=H122000 PYTHON=.venv-systems/bin/python \
  sh systems/benchmark/campaigns_timing.sh systems/results/campagne_20261124/protocollo 5
.venv-systems/bin/python systems/benchmark/summarize_campaigns.py systems/results/campagne_20261124/protocollo
```

The summary first checks that the tokens match between the two engines, with the eager implementation of the
TinyStories models and across all campaigns, then computes the medians. Rerun on the archived campaigns, it
reproduces `riepilogo.json` byte for byte (SHA-256 `6e1afd9f…6bac1`). The generated tokens do not depend on the
machine; the times do, and another Mac will give different values.

The results of the thesis, with the reports that describe them, are in `systems/results/`:

- `campagne_20261124/protocollo/`: the five campaigns on the final checkpoint used in the thesis and
  `riepilogo.json` (paired decode −34.8%, times and peaks of the three models);
  `protocollo_H122000_sotto_carico/` holds five campaigns on the same weights run on a loaded system;
  `protocollo_H80000/` and `preliminari/` are earlier campaigns on the weights at 80,000 updates;
- `01_motore_compatto_vs_denso/`: earlier paired measurements and attribution of time per operator;
- `02_tre_modelli_mac/`: TTFT, decode and peak memory of the three models (earlier checkpoint);
- `03_f10_ssd_qwen/`: validated SSD reads and Qwen3.5-9B with layers loaded from disk, run with
  [SWLP](https://github.com/Rishav-Upadhaya/SWLP) at commit `3941963812850f9ad6821f3bec78754d695143ac`
  (scripts `prepara_swlp.py`, `misura_swlp.py`, `riepiloga_swlp.py`); model `Qwen/Qwen3.5-9B` downloaded from
  Hugging Face (revision recorded in the download metadata: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`).
  The F10 runs in this folder predate section 6.1 and are not used in the thesis.

The archived folders under `systems/results/` are kept as they were recorded: their scripts and reports may still
use the old Italian paths (`sistemi/…`).

The lossless archive of the FP32 weights of the final checkpoint (102,352,624 bytes, not used in the thesis) is
built with `systems/benchmark/lossless_archive/steraf32_byteplane.py` and restored bit for bit with
`restore_lossless_archive.py`.

### 6.1 Fictitious STERA F10 model (resource comparison with Qwen3.5-9B)

F10 is not trained: it only serves to compare execution resources. It is generated from the final checkpoint by a
deterministic script; the file is bit-identical on every machine and Python version (the script writes the `.npz`
container with a fixed byte layout):

```bash
python systems/f10/generate_f10.py systems/models/H122000_fp32_weights.npz systems/models/F10_H122000.npz
shasum -a 256 systems/models/F10_H122000.npz   # c56d2f90fab47efcdb250b231394b531c0b11944faf3dffc5e6e166ba4fa9162
```

Decode for varying context lengths (cache filled with test values, three runs per context, each started with a load
below 1.5), control test with the cache filled by the model itself, and thermal check:

```bash
cd systems/benchmark
CONTEXTS="128:fp32 512:fp32 2000:fp32 8000:fp32 16000:fp32 32000:fp32 32000:fp16 64000:fp16" ./campaigns_f10_context.sh
STERA_WEIGHTS=H122000 python f10_real_fill_control.py --tokens 8000 --out ../results/campagne_20261127/f10_contesto/controllo_reale_8000.json
STERA_WEIGHTS=H122000 python f10_context_sustained.py --context 8000 --seconds 800 --out ../results/campagne_20261127/f10_contesto/verifica_termica_8000.json
```

Results are in `systems/results/campagne_20261127/f10_contesto/`. The `diagnostica_*` folders hold earlier
measurements not used in the thesis (model generated on the GPU; cache copied at every step).

## 7. Figures and thesis

```bash
cd thesis/figure_scripts
python fig_pretraining.py && python fig_narrativa.py && python fig_sistemi.py && python fig_bilancio.py && python fig_componenti.py
cd .. && tectonic -X compile main.tex
```

With matplotlib 3.9.4 the figures are pixel-identical to the committed ones (only the creation date in the PDF
metadata changes). The figure labels and the thesis text are in Italian, the language of the thesis.

To work on the text with automatic preview: `python thesis/preview/server.py` and open `http://localhost:8765`.

## 8. Components of the projections and additional system studies

```bash
cd evaluation/protocol
python evaluate_model.py stera30m --checkpoint ../../weights/stera30m_step122000.ckpt.gz --tokenizer ../../external/tokenizer.json \
       --rank-bits 8 --rank-scheme mlx --nll-only --out /tmp/rank8.jsonl      # 8-bit low-rank term
#   --reduced-rank K (K = 64 ... 0), --ablation scale1|guadagni1|scale8|guadagni8|piccole8
```

Blind judgments of the variants: `evaluation/variants/` (anonymized sets, assignments, `aggregate*.py`; rerun,
the aggregation scripts reproduce the archived results byte for byte).
Timing: `systems/benchmark/campaigns_mlx.sh` (like-for-like comparison), `campaigns_context.sh` (input length),
`campaigns_prefill.sh` (prefill with expansion), `campaigns_rank8.sh` (8-bit low-rank term); engines
`systems/engines/gptneo_dense_mlx.py`, `prefill_expansion.py`, `stera_rank8_mlx.py`.
