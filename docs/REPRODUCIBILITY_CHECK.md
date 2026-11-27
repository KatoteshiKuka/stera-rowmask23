# Reproducibility check (24 November 2026, updated 27 November 2026)

Every step of the thesis was rerun starting only from the repository and from public sources: the corpus and the
tokenizer downloaded from Hugging Face at the pinned revisions, the two TinyStories models at the revisions of
`systems/models/manifest_tinystories.json`, and the final STERA 30M checkpoint. The table gives the outcome of each item.

| Item | Where it was run | Outcome |
|---|---|---|
| Download of corpus, tokenizer, TinyStories-8M and 33M | Mac | SHA-256 identical to the expected ones (`6418d412…`, `f6ed3d30…`, `22c355bf…`, `41316d3b…`) |
| Tokenization and split (`prepare_data.py`) | Mac CPU and P100 server | `MANIFEST.json` identical in every field (pack `5ae58177…`, index `bbadffc0…`, per-split counts, target tokens) except path and duration |
| Data contract (`seal_data.py`) | Mac, CPU | `CONTRACT` identical (`b8a68cad…`); the final checkpoint accepts it |
| DEV NLL of the final checkpoint (`ts68_v3.py --steps 0`) | Mac CPU and P100 | 1.152155 on both, identical to the log |
| Training engine: 20 steps from scratch | Mac, CPU | DEV NLL at step 0 (10.927510), loss at step 20 (9.34693), gradient norm (5.7446), learning rate and number of labels identical to the log; logs in `training/check_20261124/` |
| Training engine: 200 steps from scratch with `sbatch train_stera30m.sh` | Slurm server, 2 × P100 | DEV NLL at step 0 and log lines identical up to step 120, with the same labels at every step; from step 140, differences of order 10⁻⁵ in the loss (3.71362 against 3.71367 at step 200), due to the non-deterministic order of GPU reductions. Logs in `training/check_20261124/` |
| Forward and backpropagation check (`check_backward_stera.py`) | Mac, PyTorch 2.15 | `backward_results.json` reproduced exactly: autograd agreement ≤1.4·10⁻¹⁵, finite differences 7.6·10⁻⁸ |
| TEST300 evaluation, TinyStories-8M | Mac CPU and P100 server | 300/300 continuations identical on CPU and GPU; NLL 1.515349 (CPU) and 1.515352 (GPU), thesis 1.515349 |
| TEST300 evaluation, STERA 30M | Mac CPU and P100 server | 300/300 continuations identical on CPU and GPU; NLL 1.173975 (CPU) and 1.173986 (GPU), thesis 1.173997 |
| TEST300 evaluation, TinyStories-33M | Mac CPU and P100 server | on GPU 300/300 continuations identical, NLL 1.324452 (thesis 1.324456); on CPU 299/300, with prompt F156 diverging from token 84 because of CPU rounding |
| Blind-judgment aggregation (`aggregate_and_decode.py`) | Mac | `risultati_3modelli.json` and `giudizi_900.json` byte-identical |
| Aggregation of the variant judgments (`aggregate_rank8.py`, `reduced_rank/aggregate.py`) | Mac | all output files byte-identical |
| Summary of the timing campaigns (`summarize_campaigns.py`) | Mac | `riepilogo.json` byte-identical (`6e1afd9f…`) |
| Export of the FP32 weights | Mac | payload of 121,384,728 bytes, as in the thesis |
| Fictitious F10 model (`generate_f10.py`) | Mac, Python 3.11 (the published file was generated with Python 3.9) | `F10_H122000.npz` with SHA-256 `c56d2f90…` and identical manifest |
| Lossless archive | Mac | rebuilt on the final checkpoint, 102,352,624 bytes; bit-for-bit restore of all 267 tensors. An `.npz` rebuilt by `restore_lossless_archive.py` holds the same tensors but may have a different hash, because the zip container records its own metadata |
| Figures | Mac, matplotlib 3.9.4 | pixel-identical; with matplotlib 3.11 they differ by 1–2 pixels because of text rendering |
| Times and memory (Chapter 10) | Mac M1 | rerun on the final checkpoint, see below |

## Times, archive and F10 on the final checkpoint

All the system results of the thesis refer to the final checkpoint (122,000 updates, weights `H122000_*`, SHA-256 of
the FP32 weights `864d03dc…`). The engines read the checkpoint named by `STERA_WEIGHTS` (or `STERA_PESI`), with
`H122000` as default.

- **Times.** Five campaigns (`systems/benchmark/campaigns_timing.sh`), each started when the one-minute load average
  had dropped below 1.5, followed by a 60 s pause; the load at the actual start is recorded in `condizioni.log`
  (between 1.25 and 1.66). In all 30 pairs the two engines produce the same tokens, identical across campaigns; the
  TinyStories models match the eager implementation. The median decode change is −34.8%, with per-campaign medians
  between −33.7% and −36.0%. Summary: `systems/results/campagne_20261124/protocollo/riepilogo.json`. Five campaigns
  on the same weights with a loaded system (average between 2.7 and 4.6, `protocollo_H122000_sotto_carico/`) give
  −35.9% on the paired decode but slower absolute times.
- **Dense-equivalent form.** `export_dense.py` produces `H122000_DENSE_FP32.safetensors` (330,885,120 bytes of
  payload, bit-for-bit check after saving).
- **Lossless archive.** 102,352,624 bytes on the final checkpoint, with bit-for-bit restore of all 267 tensors (on
  the weights at 80,000 updates the archive was 102,298,136 bytes).
- **F10.** Generated from the final checkpoint by `systems/f10/generate_f10.py`; decode campaigns for contexts from
  128 to 64,000 tokens, control with the cache filled by the model and thermal check in
  `systems/results/campagne_20261127/f10_contesto/`. The SSD read test measures byte volumes that depend only on
  the shape of the model, so it holds for any checkpoint.

The earlier campaigns on the weights at 80,000 updates remain in `campagne_20261124/protocollo_H80000/` and
`preliminari/`, with −36.1% and between −31.3% and −36.5% decode change.

## Items not rerun

- **Full training.** Not repeated (about 35 hours on two P100s). The training engine is checked with 20 steps from
  scratch on CPU, with 200 steps launched with `sbatch` on the two P100s and with the DEV NLL of the final checkpoint.
- **Qwen3.5-9B** (`03_f10_ssd_qwen/`): not rerun (it requires downloading about 18 GB); it does not depend on STERA.
