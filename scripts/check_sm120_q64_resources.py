"""Fail closed on the SM120 Q64 FP16, MXFP8, and INT8/FP16 resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


_KERNELS = (
    "mixed_attention_sm120_q64_kernel",
    "mixed_attention_sm120_q64_int8_fp16_kernel",
)
_COMPILE_RE = re.compile(
    r"Compiling entry function '([^']+)' for '(sm_[0-9]+a?)'"
)
_PROPERTIES_RE = re.compile(r"Function properties for (\S+)")
_STACK_RE = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
    r"(\d+) bytes spill loads"
)
_REGISTERS_RE = re.compile(r"Used (\d+) registers")
_SASS_SECTION_RE = re.compile(
    r"//-+\s+\.text\.([^\s]+).*?\n(.*?)(?=//-+\s+\.text\.|\Z)",
    re.DOTALL,
)


def _collect_records(text: str, kernels: tuple[str, ...]) -> list[dict[str, object]]:
    lines = text.splitlines()
    arches: dict[str, list[str]] = {}
    for line in lines:
        match = _COMPILE_RE.search(line)
        if match:
            arches.setdefault(match.group(1), []).append(match.group(2))

    records: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        properties = _PROPERTIES_RE.search(line)
        if properties is None:
            continue
        symbol = properties.group(1)
        if not any(kernel in symbol for kernel in kernels):
            continue
        stack = None
        registers = None
        for following in lines[index + 1 : index + 8]:
            if _PROPERTIES_RE.search(following) or _COMPILE_RE.search(following):
                break
            stack = stack or _STACK_RE.search(following)
            registers = registers or _REGISTERS_RE.search(following)
        symbol_arches = arches.get(symbol, [])
        if stack is None or registers is None or len(symbol_arches) != 1:
            raise ValueError(f"incomplete ptxas record for {symbol}")
        records.append(
            {
                "symbol": symbol,
                "arch": symbol_arches[0],
                "stack_frame_bytes": int(stack.group(1)),
                "spill_store_bytes": int(stack.group(2)),
                "spill_load_bytes": int(stack.group(3)),
                "registers": int(registers.group(1)),
            }
        )
    return records


def _phase_name(symbol: str) -> str | None:
    query = "q64" if "sm120_q64" in symbol else "q128" if "sm120_q128" in symbol else None
    if query is None:
        return None
    has_fp16 = "ELb1ELb1ELb0EE" in symbol
    if f"sm120_{query}_nv_int8_fp16_kernel" in symbol:
        phases = "nvfp4_int8"
    elif f"sm120_{query}_nv_mx_fp16_kernel" in symbol:
        phases = "nvfp4_mxfp8"
    elif f"sm120_{query}_nvfp4_kernel" in symbol:
        phases = "nvfp4"
    elif f"sm120_{query}_mxfp8_kernel" in symbol:
        phases = "mxfp8"
    elif f"sm120_{query}_int8_fp16_kernel" in symbol:
        phases = "int8"
        has_fp16 = True
    elif f"sm120_{query}_int8_kernel" in symbol:
        phases = "int8"
    elif f"sm120_{query}_fp16_kernel" in symbol:
        return f"{query}_fp16"
    elif query == "q64" and "mixed_attention_sm120_q64_kernel" in symbol:
        if "ELb0ELb1ELb0EE" in symbol:
            return "q64_fp16"
        phases = "mxfp8"
    else:
        return None
    return f"{query}_{phases}{'_fp16' if has_fp16 else ''}"


def collect_phase_resources(text: str) -> dict[str, dict[str, object]]:
    kernels = (
        "mixed_attention_sm120_q64",
        "mixed_attention_sm120_q128",
    )
    result: dict[str, dict[str, object]] = {}
    for record in _collect_records(text, kernels):
        name = _phase_name(str(record["symbol"]))
        if name is None:
            continue
        if name in result:
            raise ValueError(f"duplicate ptxas record for {name}")
        result[name] = record
    return result


def validate_resources(text: str) -> dict[str, object]:
    records = _collect_records(text, _KERNELS)

    variants: dict[str, dict[str, object]] = {}
    for record in records:
        symbol = str(record["symbol"])
        if "mixed_attention_sm120_q64_int8_fp16_kernel" in symbol:
            if "ELb1ELb1ELb0EE" not in symbol:
                continue
            name = "int8_fp16"
        elif "ELb0ELb1ELb0EE" in symbol:
            name = "fp16"
        elif "ELb1ELb1ELb0EE" in symbol:
            name = "mxfp8"
        else:
            continue
        if name in variants:
            raise ValueError(f"duplicate ptxas record for {name}")
        variants[name] = record
    if set(variants) != {"fp16", "mxfp8", "int8_fp16"}:
        raise ValueError(
            "expected FP16, unified MXFP8, and INT8/FP16 ptxas records, found "
            + ", ".join(sorted(variants))
        )
    for name, record in variants.items():
        failures = []
        stack_limit, store_limit, load_limit = (
            (24, 36, 32) if name == "int8_fp16" else (0, 0, 0)
        )
        if record["arch"] != "sm_120a":
            failures.append(f"architecture {record['arch']}")
        if record["stack_frame_bytes"] > stack_limit:
            failures.append(f"stack frame {record['stack_frame_bytes']} bytes")
        if record["spill_store_bytes"] > store_limit:
            failures.append(f"spill stores {record['spill_store_bytes']} bytes")
        if record["spill_load_bytes"] > load_limit:
            failures.append(f"spill loads {record['spill_load_bytes']} bytes")
        if int(record["registers"]) > 168:
            failures.append(f"registers {record['registers']} exceed 168")
        if failures:
            raise ValueError(
                f"SM120 Q64 {name} resource gate failed: "
                + ", ".join(failures)
            )
    return variants


def validate_phase_boundary_sass(
    text: str,
    *,
    q_head_row_line: int,
) -> dict[str, dict[str, int]]:
    targets: dict[str, str] = {}
    for symbol, body in _SASS_SECTION_RE.findall(text):
        if "ELb1ELb1ELb0EE" not in symbol:
            continue
        if "mixed_attention_sm120_q64_int8_fp16_kernel" in symbol:
            targets["q64_int8_fp16"] = body
        elif "mixed_attention_sm120_q128_int8_kernel" in symbol:
            targets["q128_int8_fp16"] = body
    if not targets:
        raise ValueError("no plain INT8 -> FP16 SASS function found")

    result: dict[str, dict[str, int]] = {}
    marker = re.compile(
        rf'q64_attention_phase_composer\.inl", line {q_head_row_line}\b'
    )
    for name, body in targets.items():
        q_head_row = marker.search(body)
        if q_head_row is None:
            raise ValueError(f"{name}: FP16 Q-head row line not found")
        prefix = body[: q_head_row.start()]
        counts = {}
        for axis in ("Z", "Y"):
            counts[axis] = len(
                re.findall(rf"\bS2(?:U)?R\b[^\n]*SR_CTAID\.{axis}\b", prefix)
            )
            if counts[axis] < 2:
                raise ValueError(
                    f"{name}: fresh CTAID.{axis} read missing before FP16 Q"
                )
        result[name] = {
            "ctaid_z_reads_before_fp16_q": counts["Z"],
            "ctaid_y_reads_before_fp16_q": counts["Y"],
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_log", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--sass", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = validate_resources(args.build_log.read_text())
    payload: dict[str, object] = result
    if args.sass:
        source = (
            Path(__file__).resolve().parents[1]
            / "csrc/attention/cuda/sm120/q64_attention_phase_composer.inl"
        )
        q_head_row_line = next(
            index
            for index, line in enumerate(source.read_text().splitlines(), 1)
            if "fp16_batch_id * fp16_num_qo_heads + fp16_head_id" in line
        )
        sass_result: dict[str, dict[str, int]] = {}
        for path in args.sass:
            for name, record in validate_phase_boundary_sass(
                path.read_text(), q_head_row_line=q_head_row_line
            ).items():
                if name in sass_result:
                    raise ValueError(f"duplicate SASS record for {name}")
                sass_result[name] = record
        payload = {"resources": result, "phase_boundary_sass": sass_result}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
