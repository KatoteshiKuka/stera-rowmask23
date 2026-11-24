"""Run the upstream SWLP decoder on the local Qwen3.5-9B checkpoint."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
from swlp.config import AppConfig
from swlp.runner.base import build_runner
from tokenizers import Tokenizer


def device_read_bytes():
    tree = subprocess.check_output(
        ["/usr/sbin/ioreg", "-r", "-c", "IOBlockStorageDriver", "-l", "-w0"], text=True)
    counters = re.findall(r'^  \|   "Statistics" = .*?"Bytes \(Read\)"=(\d+)', tree, re.M)
    if not counters:
        raise RuntimeError("Missing device read counters")
    return sum(map(int, counters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("shards", type=Path)
    parser.add_argument("length", type=int)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--serial", action="store_true", help="Disable layer prefetch")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    assert (args.shards / "shard_manifest.json").is_file()
    tokenizer = Tokenizer.from_file(str(args.source / "tokenizer.json"))
    workspace = Path(__file__).resolve().parents[2]
    manuscript = workspace / "thesis/riferimenti/STERA_TESI_MANOSCRITTO_INTEGRATO_REV14_20261122.md"
    ids = tokenizer.encode(manuscript.read_text(), add_special_tokens=False).ids[:args.length]
    prompt = tokenizer.decode(ids, skip_special_tokens=False)
    assert len(ids) == args.length == len(tokenizer.encode(prompt, add_special_tokens=False).ids)

    config = AppConfig()
    config.model.model_id = str(args.source)
    config.model.local_model_path = args.source
    config.cache.offline = True
    config.generation.max_new_tokens = args.tokens
    runtime = config.runtime
    runtime.backend = "swlp"
    runtime.device = args.device
    runtime.dtype = "bfloat16"
    runtime.allow_mock_fallback = False
    runtime.swlp_fallback_to_baseline = False
    runtime.swlp_strict = True
    runtime.shard_dir = args.shards
    runtime.swlp_direct_io = "on"
    runtime.swlp_residency = "off"
    runtime.swlp_window_size = 2
    runtime.swlp_prefetch_depth = 2
    runtime.swlp_prefetch = not args.serial
    runtime.swlp_prefill_chunk = 512
    runtime.kv_memory_budget_mb = 3072
    runtime.swlp_activation_cache = False
    runtime.swlp_prealloc_buffer = False

    runner = build_runner(config)
    before = device_read_bytes()
    with redirect_stdout(sys.stderr):
        result = runner.run(prompt, profile=True)
    if args.device == "mps":
        torch.mps.synchronize()
    after = device_read_bytes()
    stats = result.metrics.to_dict()
    assert stats["backend"] == "swlp" and stats["input_tokens"] == args.length
    assert not stats.get("degradation_count") and stats["generated_tokens"] == args.tokens
    print(json.dumps({
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "disk_read_device_bytes_including_load": after - before,
        "window_layers": 2, "pinned_layers": 0, "prefill_chunk_tokens": 512,
        "layer_prefetch": runtime.swlp_prefetch,
        "generated_text_sha256": hashlib.sha256(result.completion.encode()).hexdigest(),
        "metrics": stats,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
