import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent / "motori"
sys.path.insert(0, str(BENCH))
import dense_clone_fp32_v8 as dense
from stera_h80000_fp32_dense_sdpa_v62 import CompiledStatic
from thesis_timing_study_h80000_fp32_v64 import prompt_tokens

def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()

def generate(model, ids):
    output = []
    for position in range(len(ids) + 100):
        token = mx.array(ids[position], mx.int32) if position < len(ids) else token
        token = mx.argmax(model.step(token, position)).astype(mx.int32)
        mx.eval(token)
        if position >= len(ids) - 1:
            output.append(int(token))
    return output

def main():
    archive = ROOT.parent / "modelli" / "H80000_DENSE_FP32.safetensors"
    assert not archive.exists(), "Existing export protected"
    temporary = ROOT.parent / "modelli" / "H80000_DENSE_FP32.partial.safetensors"
    original_sha = sha256(dense.NPZ)
    model = dense.Model(1)
    weights = {**model.w, **model.p}
    assert len(weights) == len(model.w) + len(model.p)
    nbytes = sum(t.nbytes for t in weights.values())
    assert nbytes == 330_885_120 and all(t.dtype == mx.float32 for t in weights.values())
    try:
        mx.save_safetensors(str(temporary), weights, metadata={"checkpoint":"STERA H80000", "dtype":"float32"})
        restored = mx.load(str(temporary))
        assert restored.keys() == weights.keys()
        for name, weight in weights.items():
            got = restored[name]
            assert got.dtype == weight.dtype and got.shape == weight.shape
            assert np.array_equal(np.asarray(got).view(np.uint32), np.asarray(weight).view(np.uint32)), name
        matrix_names = set(model.w)
        ids = prompt_tokens(256)
        # Reuse the normal compiled dense decoder with a loader for the saved tensors.
        def from_export(factor):
            assert factor == 1
            instance = type("ExportedWeights", (), {})()
            instance.w = {k: restored[k] for k in matrix_names}
            instance.p = {k: v for k,v in restored.items() if k not in matrix_names}
            return instance

        expected = json.loads((BENCH/"THESIS_TIMING_STUDY_F1_dense_sdpa_P256_D100_R01.json").read_text())
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == expected["prompt_sha256"]
        with patch.object(dense, "Model", from_export):
            exported_ids = generate(CompiledStatic(512), ids)
        assert exported_ids[:100] == expected["output_ids"]
        assert sha256(dense.NPZ) == original_sha
        info = {"checkpoint":"STERA H80000", "source":str(dense.NPZ), "source_sha256":original_sha,
                "sha256":sha256(temporary), "serialized_bytes":temporary.stat().st_size,
                "tensor_payload_bytes":nbytes, "fp32_elements":nbytes//4,
                "tensors":{k:list(v.shape) for k,v in restored.items()},
                "architecture":{"layers":6,"hidden":768,"ffn":3072,"heads":16,
                                "vocab":50257,"padded_vocab":50304,"local_window":256,
                                "tied_embedding_head":True},
                "roundtrip_bitwise_equal":True,
                "greedy_reference_ids_equal":100,"prompt_sha256":expected["prompt_sha256"]}
        temporary.rename(archive)
        (ROOT.parent/"modelli"/"manifest_denso.json").write_text(json.dumps(info,indent=2)+"\n")
        print("EXPORT_VERIFIED", archive, info["sha256"], flush=True)
    finally:
        temporary.unlink(missing_ok=True)

if __name__=="__main__":
    main()
