"""Real-capture D64/D128 core latency, Dense quality, and native resources."""
from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_sm120_d64_cuda import CAPTURE, FAMILIES, phase_call, prepare_case, quality
from anemoi.layers.attention.mpa.backends.sm120_q64 import prepare_h3_sm120_operands
from scripts.check_sm120_q64_resources import _collect_records, _phase_name


def resources(binary, build_log):
    records = _collect_records(build_log.read_text(), ("mixed_attention_sm120_",))
    driver = C.CDLL("libcuda.so.1")
    driver.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
    driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
    driver.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
    driver.cuFuncSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_int]
    driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
    driver.cuModuleUnload.argtypes = [C.c_void_p]

    def check(result):
        if result:
            raise RuntimeError(f"CUDA driver error {result}")

    torch.cuda.init()
    torch.empty(1, device="cuda")
    result = {}
    with tempfile.TemporaryDirectory(prefix="sm120-resources-") as directory:
        subprocess.run(["/usr/local/cuda-13.1/bin/cuobjdump", "-xelf", "all", str(binary)],
                       cwd=directory, check=True, capture_output=True)
        modules = []
        try:
            for cubin in Path(directory).iterdir():
                module = C.c_void_p()
                check(driver.cuModuleLoad(C.byref(module), str(cubin).encode()))
                modules.append(module)
            for record in records:
                symbol = record["symbol"]
                match = re.search(r"ILj(64|128)ELb([01])ELb([01])", symbol)
                if not match:
                    continue
                dim, has_low, has_high = map(int, match.groups())
                block = 128 if "sm120_q128" in symbol else 64
                name = _phase_name(symbol)
                if name is None:
                    name = f"q{block}_int8_dense" if "int8_dense" in symbol else f"q{block}_mxfp8_compact"
                key = f"d{dim}_{name}"
                function = C.c_void_p()
                for module in modules:
                    if driver.cuModuleGetFunction(C.byref(function), module, symbol.encode()) == 0:
                        break
                else:
                    raise RuntimeError(f"kernel missing from binary: {symbol}")
                dynamic = max((block + 128) * dim + 64 * (dim // 32) + dim * 2
                              if has_low else 0,
                              (block + 64) * dim * 2 if has_high else 0,
                              block * dim * 2)
                attrs = {}
                for field, attribute in (("static_smem_bytes", 1), ("local_bytes", 3), ("registers", 4)):
                    value = C.c_int()
                    check(driver.cuFuncGetAttribute(C.byref(value), attribute, function))
                    attrs[field] = value.value
                check(driver.cuFuncSetAttribute(function, 8, dynamic))
                active = C.c_int()
                check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    C.byref(active), function, 128, dynamic))
                result[key] = {**record, **attrs, "dynamic_smem_bytes": dynamic,
                               "active_ctas_per_sm": active.value,
                               "active_warps_per_sm": active.value * 4}
        finally:
            for module in modules:
                check(driver.cuModuleUnload(module))
    return result


def benchmark(call, rounds=21, repeats=32):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            output = call()
    for _ in range(5):
        graph.replay()
    samples = []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(rounds):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return {"median_us": statistics.median(samples), "samples_us": samples}, output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--sparsity", type=float, default=.75)
    parser.add_argument("--preparation-only", action="store_true")
    parser.add_argument("--build-log", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"capture": str(CAPTURE), "input": "captured Cosmos3 channels [0:D], Hq=4/Hkv=2",
              "torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "requested_sparsity": args.sparsity, "timing": "CUDA Graph, 21 rounds x 32 calls",
              "resources": resources(args.binary.resolve(), args.build_log) if args.binary else {},
              "cases": []}
    for dim in args.dims:
        for tokens in args.tokens:
            for block in (64, 128):
                prepared, raw, indices, valid, counts, scales = prepare_case(block, head_dim=dim, tokens=tokens)
                q, k, v = raw[0], raw[1][:, ::2].contiguous(), raw[2][:, ::2].contiguous()
                valid_k = valid.view(1, -1, 64).sum(-1).int()
                stages = valid_k.size(-1)
                retained = int(stages * (1 - args.sparsity))
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    dense_timing, dense = benchmark(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
                row = {"head_dim": dim, "query_block": block, "tokens": tokens,
                       "effective_sparsity": 1 - retained / stages, "dense": dense_timing,
                       "preparation": {}, "phases": {}}
                for family in ("int8", "mxfp8", "nvfp4_int8_fp16"):
                    nv, integer, mx, half = FAMILIES[family]
                    for maxpool in (False, True):
                        timing, _ = benchmark(lambda: prepare_h3_sm120_operands(
                            *raw, indices, valid, counts, prefix_tokens=0,
                            query_block_size=block, has_nvfp4=bool(nv), has_int8=bool(integer),
                            has_mxfp8=bool(mx), has_fp16=bool(half),
                            has_prefix_query_int8=False,
                            has_maxpool=maxpool, global_scales=scales))
                        row["preparation"][family + ("_maxpool" if maxpool else "")] = timing
                for family in (() if args.preparation_only else FAMILIES):
                    call, _, _ = phase_call(prepared, scales, block, family, valid_k, retained=retained)
                    timing, (out, _) = benchmark(call)
                    assert torch.isfinite(out).all(), family
                    row["phases"][family] = {**timing, "vs_dense": quality(out[:, :, :tokens], dense),
                                               "speedup_vs_dense": dense_timing["median_us"] / timing["median_us"]}
                result["cases"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(f"D{dim} Q{block} N{tokens}: Dense {dense_timing['median_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
