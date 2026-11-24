"""Prepare regular SWLP shards from the already downloaded checkpoint."""

import argparse
import shutil
from pathlib import Path

from simula_qwen_f10 import LAYER, checkpoint_tensors
from swlp.model.shard import shard_model_by_layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    tensors, _ = checkpoint_tensors(args.source)
    required = sum(size for key, size in tensors.items() if not key.startswith("model.visual."))
    reserve = 1_500_000_000
    metadata = 16 * 2**20
    layer_bytes = {}
    for key, size in tensors.items():
        match = LAYER.match(key)
        if match:
            layer = int(match.group(1))
            layer_bytes[layer] = layer_bytes.get(layer, 0) + size

    def check_space(bytes_needed):
        free = shutil.disk_usage(args.destination.parent).free
        if free < bytes_needed + reserve:
            raise RuntimeError(f"SSD free {free}, need {bytes_needed} plus reserve {reserve}")

    check_space(required + metadata)
    print(f"Layer preparation expects {required} payload bytes, reserve {reserve}", flush=True)

    def progress(layer, total, _):
        remaining = required - sum(layer_bytes[index] for index in range(layer))
        check_space(remaining + metadata)
        print(f"Prepared {layer}/{total} layers", flush=True)

    shard_model_by_layer(str(args.source), args.destination, progress=progress)
    print("SHARDS_READY", args.destination, flush=True)


if __name__ == "__main__":
    main()
