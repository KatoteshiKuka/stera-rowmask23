# Results of the thesis and where they come from

Chapter, table and figure numbers refer to the compiled thesis (`thesis/thesis_STERA_Nardoni.pdf`).
Numbers are written with a decimal point here; the thesis, in Italian, uses a decimal comma.

## Training (Chapter 9, Figure 9.1)

| Quantity | Value | Source |
|---|---|---|
| Updates | 122,000 | `training/training_log_0_122k.txt` |
| Target tokens | 1,529,514,422 | same file, field `labels` |
| Initial DEV NLL | 10.928 | line `VAL step=0` |
| DEV NLL at step 80,000 | 1.172 | line `VAL step=80000` |
| Final DEV NLL | 1.152 (perplexity 3.17) | line `VAL step=122000` |

## Likelihood on the 300 TEST stories (Table 9.1)

57,841 target tokens, same tokenizer. Source: `evaluation/risultati_3modelli.json`, field `nll`.

| Model | NLL | Perplexity | Difference from STERA 30M (95% CI) |
|---|---:|---:|---|
| TinyStories-8M | 1.515 | 4.55 | +0.341 [+0.319; +0.365] |
| STERA 30M | 1.174 | 3.23 | — |
| TinyStories-33M | 1.324 | 3.76 | +0.150 [+0.115; +0.186] |

## Blind judgment (Tables 9.2–9.3, Figures 9.2–9.3)

900 anonymous continuations, one judge (Claude Opus 5.5), rubric in `evaluation/blind_judgment/rubrica.txt`
(English translation: `rubric_en.txt`). Source: `evaluation/risultati_3modelli.json`.

| Model | Correct | 95% Wilson CI | S / P / N | Grammar | Consistency | Plot |
|---|---:|---|---|---:|---:|---:|
| TinyStories-8M | 65/300 (21.7%) | [17.4; 26.7] | 65 / 200 / 35 | 7.43 | 4.74 | 4.96 |
| STERA 30M | 132/300 (44.0%) | [38.5; 49.7] | 132 / 152 / 16 | 7.67 | 5.26 | 5.37 |
| TinyStories-33M | 145/300 (48.3%) | [42.7; 54.0] | 145 / 140 / 15 | 7.91 | 5.54 | 5.49 |

| Paired comparison | Difference (points) | 95% CI | Only A / only B | Holm p |
|---|---:|---|---|---:|
| STERA 30M − TinyStories-8M | +22.3 | [+15.3; +29.3] | 98 / 31 | 5.4·10⁻⁹ |
| STERA 30M − TinyStories-33M | −4.3 | [−12.0; +3.7] | 66 / 79 | 0.32 |
| TinyStories-33M − TinyStories-8M | +26.7 | [+19.7; +33.7] | 109 / 29 | 1.3·10⁻¹¹ |

## Memory (Chapter 10, Table 10.2, Figure 10.1)

| Representation | Weight bytes |
|---|---:|
| TinyStories-8M FP32 | 78,810,112 |
| STERA 30M, low-rank U, V at 8 bits (MLX format), rest FP32 | 86,096,664 |
| STERA 30M FP32 | 121,384,728 |
| TinyStories-33M FP32 | 274,056,192 |
| STERA 30M, dense expansion FP32 | 330,885,120 |

Sources: parameter count (`training/ts68_v3.py`) and files in `systems/models/`.

Parameter budget (Chapter 4, Section 4.7): the RowMask23 blocks guarantee −80.61% on the structured part; for a
layer with rank $r=\kappa d$ the saving is $80.61\% - 1.5\kappa - 1.5/d$; STERA 30M ($\kappa = 1/8$) saves 63.32%
(30,346,182 values against 82,721,280). Components: coordinates 15,721,200, scales 4,950, gains 134,016,
low-rank 12,865,536, positions/normalizations/biases 1,620,480.

## Components of the projections (Chapter 7, Section 7.5; Chapter 9, Section 9.4)

NLL on the 300 TEST stories of the final model with one component changed after training (`evaluation/variants/`):

| Variant | NLL | Correct stories |
|---|---:|---:|
| Model of the thesis (FP32) | 1.174 | 44.0% |
| 8-bit low-rank term (groups of 32, engine format) | 1.174056 (+0.007%) | — |
| 8-bit low-rank term (one scale per column) | 1.174302 (+0.03%) | 41.3% (−2.7 points, p = 0.52) |
| 8-bit scales / 8-bit gains | +0.02% / +0.17% | — |
| Scales, gains and low-rank term at 8 bits | +0.21% | — |
| Scales reset to 1 / gains reset to 1 | 4.42 / 11.38 | — |
| Low-rank term reduced to rank 64 / 48 / 32 / 16 / 8 / 0 | 1.474 / 1.981 / 2.893 / 4.236 / 4.980 / 5.813 | 20.0% / 0% / 0% / 0% / 0% / 0% |

Relative error of the dense-equivalent matrix with U, V at 8 bits: 0.61% (one scale per column). Judgments of the
variants: separate sessions, same judge and same rubric.

## Execution times (Chapter 10)

Mac with Apple M1, 16 GB, MLX/Metal, FP32, 256 input tokens, 100 decode steps, weights of the final checkpoint.
Five measurement campaigns, each started with a one-minute load average below 1.5
(`systems/benchmark/campaigns_timing.sh`); warm medians over 30 pairs (6 inputs × 5 campaigns).
Source: `systems/results/campagne_20261124/protocollo/riepilogo.json`, produced by
`systems/benchmark/summarize_campaigns.py`.

| Engine | TTFT | Decode | Total |
|---|---:|---:|---:|
| Dense equivalent | 28.8 ms | 130.26 tokens/s | 0.797 s |
| Compact (coordinates) | 61.0 ms | 200.58 tokens/s | 0.559 s |

Median paired change of the time per token in decode: −34.8% (from −18.0% to −39.6% over the 30 pairs; per-campaign
medians from −33.7% to −36.0%). The dense engine emits the first token earlier and the compact engine finishes the
request earlier in all 30 pairs. The two engines produce the same 101 tokens on every input, identical in all
campaigns.

Like-for-like comparison on the same MLX engine (TinyStories in the dense engine `systems/engines/gptneo_dense_mlx.py`,
tokens identical to the reference implementation), 3 inputs × 5 campaigns
(`systems/results/campagne_20261124/confronto_mlx/`):

| Model | TTFT | Decode | Total | Peak RAM |
|---|---:|---:|---:|---:|
| TinyStories-8M | 13.3 ms | 311.9 tokens/s | 0.334 s | 511 MiB |
| STERA 30M compact | 100.9 ms | 201.0 tokens/s | 0.599 s | 413 MiB |
| TinyStories-33M | 20.8 ms | 158.0 tokens/s | 0.654 s | 879 MiB |
| STERA 30M dense | 29.5 ms | 130.1 tokens/s | 0.800 s | 569 MiB |

Input length from 32 to 384 tokens (`contesto_mlx/`): compact STERA decode 191–198 tokens/s, reduction relative to
the dense engine constant between −35.4% and −36.3%. Prefill (`prefill_mlx/`): with on-demand expansion layer by
layer the TTFT on 256 tokens drops from 65.6 to 41.8 ms (dense 30.7). 8-bit low-rank term (`rango8_mlx/`): decode
from 198.9 to 241.9 tokens/s (−17.8% time per token), TTFT unchanged; with expansion and the 8-bit low-rank term the
256 + 100 token request takes 0.47 s.

Three models with the standard TinyStories engine (STERA 30M on MLX/Metal, TinyStories on Transformers/PyTorch 2.15
MPS), 3 inputs × 5 campaigns, warm medians:

| Model | TTFT | Decode | Total | Peak RAM |
|---|---:|---:|---:|---:|
| TinyStories-8M | 22.9 ms | 149.84 tokens/s | 0.691 s | 1,526.1 MiB |
| STERA 30M | 99.4 ms | 202.41 tokens/s | 0.593 s | 413.1 MiB |
| TinyStories-33M | 31.7 ms | 90.56 tokens/s | 1.136 s | 1,493.3 MiB |

The tokens of the two TinyStories models match those of the eager implementation in every campaign.

## A model the size of Qwen3.5-9B: STERA F10 (Chapter 10, Tables 10.6–10.9)

F10 is a fictitious, untrained model generated from the final checkpoint by `systems/f10/generate_f10.py`
(hidden size 7,680, 160 heads, feed-forward 30,720, 6 layers; 898,109,280 coordinates, 4.63 billion values in the
dense-equivalent form, 4,002,707,952 bytes of FP32 weights). It is used only to compare execution resources.

Decode with resident weights for varying context lengths (KV cache filled with test values; medians of three runs on
an idle system, 24 tokens each; memory is the GPU peak for weights and KV cache).
Source: `systems/results/campagne_20261127/f10_contesto/`.

| Context (tokens) | KV cache | Decode (tokens/s) | Time per token (ms) | Memory (GiB) |
|---:|---|---:|---:|---:|
| 128 | FP32 | 12.6 | 79 | 4.0 |
| 512 | FP32 | 12.3 | 81 | 4.3 |
| 2,000 | FP32 | 11.5 | 87 | 4.3 |
| 8,000 | FP32 | 9.2 | 109 | 5.3 |
| 16,000 | FP32 | 7.4 | 136 | 6.7 |
| 32,000 | FP32 | 5.1 | 196 | 9.6 |
| 32,000 | FP16 | 6.0 | 166 | 6.7 |
| 64,000 | FP16 | 3.9 | 257 | 9.6 |

Controls: with the cache filled by the model itself, one token at a time, the time per token at 8,000 tokens is
about 120 ms (`controllo_reale_8000.json`); at a fixed context of 8,000 the time per token rises from 107.5 to
112–121 ms over 13 minutes of continuous decode (`verifica_termica_8000.json`), so the gap is thermal.

Memory required for weights and KV cache (16-bit cache), computed from the model sizes, in GiB; the device has 16 GiB:

| Context (tokens) | F10 compact | F10 dense | Qwen3.5-9B (BF16) |
|---:|---:|---:|---:|
| 8,000 | 4.4 | 18.0 | 17.0 |
| 32,000 | 6.5 | 20.1 | 17.7 |
| 64,000 | 9.2 | 22.8 | 18.7 |
| 128,000 | 14.7 | 28.3 | 20.6 |
| 200,000 | 20.9 | 34.5 | 22.8 |

Qwen3.5-9B in BF16 run with SWLP, layers read from disk, prefetch on, one run per condition, four generated tokens
(`systems/results/03_f10_ssd_qwen/`):

| Input (tokens) | TTFT (s) | Decode (tokens/s) | Process peak (GiB) | SSD reads (GB) |
|---:|---:|---:|---:|---:|
| 256 | 12.06 | 0.153 | 7.63 | 57.15 |
| 4,096 | 95.55 | 0.147 | 7.45 | 154.50 |

The comparison describes two memory regimes (resident weights against weights read from disk), not speed for the
same model: models, precisions and runtimes differ.
