"""Calculate memory floors and I/O ceilings from local checkpoint metadata."""

import argparse
import json
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")


def checkpoint_tensors(directory):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    tensors = {}
    for name in shards:
        with (directory / name).open("rb") as handle:
            header_size, = struct.unpack("<Q", handle.read(8))
            if header_size > 100_000_000:
                raise ValueError("Invalid Safetensors header size")
            header = json.loads(handle.read(header_size))
        for tensor, metadata in header.items():
            if tensor == "__metadata__":
                continue
            if tensor in tensors or index["weight_map"][tensor] != name:
                raise ValueError(f"Inconsistent shard index for {tensor}")
            start, end = metadata["data_offsets"]
            tensors[tensor] = end - start
    assert tensors.keys() == index["weight_map"].keys()
    return tensors, sum((directory / name).stat().st_size for name in shards)


def simulate(qwen, bandwidth):
    config = json.loads((qwen / "config.json").read_text())["text_config"]
    tensors, disk_file_bytes = checkpoint_tensors(qwen)
    layer_sizes = [0] * config["num_hidden_layers"]
    resident = {}
    unused = {}
    for key, size in tensors.items():
        match = LAYER.match(key)
        if match:
            layer_sizes[int(match[1])] += size
        elif key in ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight",
                     "lm_head.weight"):
            resident[key] = size
        else:
            unused[key] = size
    assert min(layer_sizes) > 0 and len(resident) == 3 and bandwidth > 0
    assert all(key.startswith(("model.visual.", "mtp.")) for key in unused), list(unused)[:20]

    full_layers = config["layer_types"].count("full_attention")
    kv_per_token = full_layers * 2 * config["num_key_value_heads"] * config["head_dim"] * 2
    linear_layers = config["layer_types"].count("linear_attention")
    linear_keys = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    linear_values = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    recurrent = linear_values * config["linear_key_head_dim"] * 4
    conv = (2 * linear_keys + linear_values) * config["linear_conv_kernel_dim"] * 2
    qwen_constant_cache = linear_layers * (recurrent + conv)

    ssd = json.loads((HERE / "evidenze_f10/ssd_validato/ssd_validato.json").read_text())
    f10_payload = ssd["scenarios"]["compact_weight_payload"]["payload_bytes"]
    dense_equivalent = ssd["scenarios"]["dense_equivalent_weights"]["payload_bytes"]
    reserve = GIB
    buffers = 2 * max(layer_sizes)
    full_scan = sum(layer_sizes)
    rows = []
    for context in (512, 2048, 8192, 32768, 131072):
        qwen_cache = context * kv_per_token + qwen_constant_cache
        qwen_floor = sum(resident.values()) + buffers + qwen_cache
        # F10 v47 has three global and three 256-token ring caches, 160 x 48 head width.
        f10_kv = 3 * 2 * 160 * 48 * 4 * (context + min(context, 256))
        for budget_gib in (8, 12):
            budget = budget_gib * GIB
            if context > config["max_position_embeddings"]:
                raise ValueError("Qwen context exceeds the declared capacity")
            available = budget - qwen_floor - reserve
            pinned = []
            if available >= 0:
                for layer, size in sorted(enumerate(layer_sizes), key=lambda item: -item[1]):
                    if size <= available:
                        pinned.append(layer)
                        available -= size
            streamed = full_scan - sum(layer_sizes[layer] for layer in pinned)
            io_s = streamed / bandwidth if available >= 0 else None
            rows.append({
                "context_tokens": context, "budget_gib": budget_gib,
                "qwen_kv_plus_recurrent_bytes": qwen_cache,
                "qwen_floor_before_pinned_layers_bytes": qwen_floor,
                "qwen_pinned_layers": pinned if available >= 0 else None,
                "qwen_streamed_bytes_per_token": streamed if available >= 0 else None,
                "qwen_io_seconds_per_token": io_s,
                "qwen_io_only_ceiling_tps": 1 / io_s if io_s else None,
                "f10_fp32_kv_bytes": f10_kv if context <= 32768 else None,
                "f10_fp32_weights_plus_kv_bytes": f10_payload + f10_kv if context <= 32768 else None,
                "f10_resident_with_reserve": f10_payload + f10_kv + reserve <= budget
                    if context <= 32768 else None,
            })
    return {
        "qwen_safetensors_files_bytes": disk_file_bytes,
        "qwen_tensors_bytes": sum(tensors.values()),
        "qwen_text_greedy_weights_bytes": full_scan + sum(resident.values()),
        "qwen_text_block_bytes": full_scan,
        "qwen_full_layer_scan_io_s": full_scan / bandwidth,
        "qwen_largest_layer_bytes": max(layer_sizes),
        "qwen_always_resident_tensor_bytes": resident,
        "qwen_excluded_vision_and_mtp_bytes": sum(unused.values()),
        "qwen_kv_bytes_per_token": kv_per_token,
        "qwen_recurrent_and_conv_bytes": qwen_constant_cache,
        "qwen_two_layer_buffers_bytes": buffers,
        "per_model_workspace_reserve_bytes": reserve,
        "f10_compact_payload_bytes": f10_payload,
        "f10_dense_fp32_payload_bytes": dense_equivalent,
        "f10_dense_fp32_full_payload_scan_s": dense_equivalent / bandwidth,
        "ssd_baseline_bytes_per_second": bandwidth,
        "scenarios": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qwen_directory", type=Path)
    parser.add_argument("--bandwidth-gb-s", type=float, default=None)
    args = parser.parse_args()
    ssd = json.loads((HERE / "evidenze_f10/ssd_validato/ssd_validato.json").read_text())
    bandwidth = args.bandwidth_gb_s or ssd["scenarios"]["dense_equivalent_weights"]["median_GB_s"]
    print(json.dumps(simulate(args.qwen_directory, bandwidth * 10 ** 9), indent=2))


if __name__ == "__main__":
    main()
