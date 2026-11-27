# STERA — SSD sequential I/O budget, MacBook Air M1, 23 November 2026

**Important:** this is an SSD I/O measurement, not a Dense model inference measurement. It does not execute model arithmetic or reconstruct the 18.6GB Dense matrix.

| Data volume | Bytes | Median sequential SSD read time | Median throughput |
|---|---:|---:|---:|
| compact_active_weights | 4,147,491,312 | 0.766 s | 5.412 GB/s |
| dense_equivalent_weights | 18,597,089,280 | 3.434 s | 5.416 GB/s |

The data source was an existing 2.23GB real file repeatedly read through macOS F_NOCACHE, with 16MiB reads. No additional 18.6GB file was written, because free disk space is insufficient. Repeated reads of a source smaller than the nominal weight file are a limitation; cold-read latency, random access and GPU compute are not included.

If all 18.597 GB had to be fetched from SSD once for **every token**, the **I/O-only** median time would be about **3.43 s/token**, corresponding to **0.291 tokens/s** before GPU computation and transfers. This is not a measured fully optimized out-of-core Dense model.

Compact F10's previously measured 11–12 token/s is a GPU end-to-end prototype with weights resident, so this comparison is a scenario contrast, not a matched kernel microbenchmark.

To prove a direct out-of-core Dense inference speed, a tiled GPU Dense matvec against real tensor values, parity checked against the RowMask oracle, with measured disk-to-GPU scheduling is still required.
