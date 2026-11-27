"""Verify paired inputs and greedy output parity, then summarize the timing campaign."""

import argparse
import json
import re
import statistics
from pathlib import Path


def read_rows(path, expected_pairs):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {(row["model"], row["rep"]): row for row in rows}
    assert len(rows) == len(indexed) and set(indexed) == expected_pairs
    return indexed


def summary(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(path):
    models = ("8M", "STERA", "33M")
    inputs = (5, 35, 65)
    optimized = read_rows(path / "ottimizzati.jsonl", {(m, i) for m in models for i in inputs})
    eager = read_rows(path / "eager_parita.jsonl", {(m, i) for m in ("8M", "33M") for i in inputs})
    for offset in inputs:
        assert len({optimized[model, offset]["prompt_sha256"] for model in models}) == 1
    for key, baseline in eager.items():
        candidate = optimized[key]
        assert candidate["prompt_sha256"] == baseline["prompt_sha256"]
        assert candidate["outputs"] == baseline["outputs"] and len(candidate["outputs"]) == 101, key
    output = {}
    for model in models:
        rows = [optimized[model, offset] for offset in inputs]
        assert all(r["dtype"] == "float32" and r["batch"] == 1
                   and r["prompt_tokens"] == 256 and r["decode_steps"] == 100 for r in rows)
        peaks = []
        for offset in inputs:
            log = (path / f"{model}_{offset}.stderr").read_text()
            peaks.append(int(re.search(r"(\d+)\s+peak memory footprint", log)[1]))
        memory = {
            label: summary([r[label]["peak_footprint_sampled_bytes"] / 2**20 for r in rows])
            for label in ("first_request_memory", "compiled_request_memory")
        }
        output[model] = {
            "replicates": len(rows),
            "framework": rows[0]["framework"],
            "load_s": summary([r["load_s"] for r in rows]),
            "warm_ttft_ms": summary([1000 * r["warm"]["ttft_s"] for r in rows]),
            "warm_tps": summary([r["warm"]["decode_tps"] for r in rows]),
            "warm_total_s": summary([r["warm"]["total_s"] for r in rows]),
            "sampled_peak_footprint_mib": memory,
            "lifetime_peak_footprint_mib": summary([peak / 2**20 for peak in peaks]),
            "native_eager_parity": len(inputs) if model != "STERA" else None,
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), indent=2))
