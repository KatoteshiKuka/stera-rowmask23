# STERA H6 Rank96 — LOSSLESS FP32 byteplane archive
Date: 2026-11-22

This experiment NEVER quantizes. Original FP32 tensors are reorganized reversibly by byte planes and encoded with xz/LZMA2 -6 (2 threads). It preserves EVERY original parameter bit. It is NOT the lossy INT8 coordinate codec.

## Best archives
| Model | Original STERA FP32 array payload | Lossless archive | Exact STERA dense-equivalent weight payload | Saving vs dense | Saving vs FP32 compressed STERA |
|---|---:|---:|---:|---:|---:|
| H74000 / 927M | 121,384,728 B | 102,290,056 B | 330,885,120 B | 69.0859% | 15.7307% |
| H80000 / 1B | 121,384,728 B | 102,298,136 B | 330,885,120 B | 69.0835% | 15.7240% |

Dense equivalent: 37 decoded RowMask23 dense weight matrices x float32 = 81,100,800 * 4 = 324,403,200 B, plus 6,481,920 B of unchanged non-reconstructed positional/bias/norm tensors. No TinyStories official weights in denominator.

The compression is lossless **for stored weights**. Once fully reconstructed, the dense matrices occupy the same 330,885,120 B in both cases, unless a separate streaming / lazy decompression inference design is implemented.

The 70% saving target against the native STERA dense-equivalent FP32 package is 99,265,536 B. Current 1B archive is 3,032,600 B above threshold; not yet 70%.

## Verification performed
- All 267 arrays verified byte-for-byte identical from both H74000 and H80000 best xz6 archives.
- H80000 archive restored to new NPZ (121,454,954 B), compared complete SHA-256 with original. Both were identical:
  028fa67574bdecc51778349fe779d081ee53e570a5d45cdcb799069edb56b728.
- Original checkpoints and weights WERE NOT overwritten.
- No training and no quantization occurred.
- A restored original dense STERA has mathematically identical weights, therefore no generation-quality or NLL changes should arise under identical deterministic inference.

## Rebuild/recover
From experiment directory:
  /usr/bin/python3 LOSSLESS_STERA_20261122/steraf32_byteplane.py compress H80000_fp32_weights.npz LOSSLESS_STERA_20261122/temp.sz 9
  /opt/homebrew/bin/zstd -dc LOSSLESS_STERA_20261122/temp.sz | /opt/homebrew/bin/xz -6 -T2 -c > LOSSLESS_STERA_20261122/H80000_fp32_byteplane.xz6
  /usr/bin/python3 LOSSLESS_STERA_20261122/steraf32_byteplane.py verify LOSSLESS_STERA_20261122/H80000_fp32_byteplane.xz6 H80000_fp32_weights.npz
  /usr/bin/python3 LOSSLESS_STERA_20261122/restore_lossless_archive.py LOSSLESS_STERA_20261122/H80000_fp32_byteplane.xz6 LOSSLESS_STERA_20261122/RESTORED_H80000.npz

IMPORTANT: This is archival lossless compression. Files are not GPU-ready packed structures with direct random access; to claim the same savings during inference, a streaming decompressor that does not cache all dense matrices must be designed and benchmarked.

## Other codecs tested (checkpoint 1B, approximate)
- Plain NPZ + zstd-9: 112,401,455 B.
- Plain NPZ + xz-6: 111,805,744 B.
- Float byte-plane + zstd-9: 103,236,186 B.
- Float bit-plane + zstd-9: 103,638,806 B.
- Float bit-plane + xz-6: 103,225,924 B.
- Float byte-plane + Brotli-9: 103,306,562 B.
- Float byte-plane + xz-9: 102,326,716 B.
- **Float byte-plane + xz-6: 102,298,136 B** (best measured).
