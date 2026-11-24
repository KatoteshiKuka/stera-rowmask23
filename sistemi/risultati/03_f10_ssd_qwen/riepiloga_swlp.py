"""Extract decode-only timing and macOS process footprint from SWLP runs."""

import argparse
import json
import re
from pathlib import Path


def read_run(directory, name):
    raw = (directory / f"qwen_swlp_{name}.json").read_text()
    start = 0 if raw.lstrip().startswith("{") else raw.rindex("\n{\n") + 1
    run = json.loads(raw[start:])
    stats = run["metrics"]
    log = (directory / f"qwen_swlp_{name}.stderr").read_text()
    peak_bytes = int(re.search(r"(\d+)\s+peak memory footprint", log)[1])
    assert stats["backend"] == "swlp" and stats["device"] == "mps"
    assert stats["degradation_count"] == 0 and stats["generated_tokens"] > 1
    decode_seconds = stats["generate_seconds"] - stats["time_to_first_token_seconds"]
    assert abs(decode_seconds / (stats["generated_tokens"] - 1)
               - stats["per_token_latency_seconds"]) < 1e-6
    return {
        "run": name, "input_tokens": stats["input_tokens"],
        "generated_tokens": stats["generated_tokens"],
        "layer_prefetch": run.get("layer_prefetch", True),
        "ttft_s": stats["time_to_first_token_seconds"], "decode_s": decode_seconds,
        "decode_toks_s": (stats["generated_tokens"] - 1) / decode_seconds,
        "generation_s": stats["generate_seconds"], "load_s": stats["load_seconds"],
        "os_peak_footprint_gib": peak_bytes / 2**30,
        "drive_reads_gb_including_load": run["disk_read_device_bytes_including_load"] / 1e9,
        "prompt_sha256": run["prompt_sha256"],
        "completion_sha256": run["generated_text_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    print(json.dumps([read_run(args.directory, name) for name in args.runs], indent=2))
