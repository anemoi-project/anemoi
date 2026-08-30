"""Fail closed on SM89 Q64 precision-phase kernel resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


_KERNEL = "mixed_attention_sm89_k64_kernel"
_VARIANT_TAGS = {
    "int8": "ELb1ELb0ELb0EE",
    "int8_fp16": "ELb1ELb1ELb0EE",
    "fp16": "ELb0ELb1ELb0EE",
}
_RESOURCE_LIMITS = {
    "fp16": {
        "registers": 168,
        "stack_frame_bytes": 0,
        "spill_store_bytes": 0,
        "spill_load_bytes": 0,
    },
    "int8": {
        "registers": 168,
        "stack_frame_bytes": 0,
        "spill_store_bytes": 0,
        "spill_load_bytes": 0,
    },
    # The current Q64 composition is faster than the former spill-free
    # implementation. Keep its measured CUDA-12.9 local-memory footprint
    # bounded; the phase-trend benchmark remains the latency authority.
    "int8_fp16": {
        "registers": 168,
        "stack_frame_bytes": 24,
        "spill_store_bytes": 40,
        "spill_load_bytes": 52,
    },
}
_COMPILE_RE = re.compile(
    r"Compiling entry function '([^']+)' for '(sm_[0-9]+a?)'"
)
_PROPERTIES_RE = re.compile(r"Function properties for (\S+)")
_STACK_RE = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
    r"(\d+) bytes spill loads"
)
_REGISTERS_RE = re.compile(r"Used (\d+) registers")


def parse_resources(text: str) -> dict[str, dict[str, object]]:
    """Extract the three compile-time Q64 precision specializations."""

    lines = text.splitlines()
    arches: dict[str, list[str]] = {}
    for line in lines:
        match = _COMPILE_RE.search(line)
        if match:
            arches.setdefault(match.group(1), []).append(match.group(2))

    variants: dict[str, dict[str, object]] = {}
    for index, line in enumerate(lines):
        properties = _PROPERTIES_RE.search(line)
        if properties is None or _KERNEL not in properties.group(1):
            continue
        symbol = properties.group(1)
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
        name = next(
            (name for name, tag in _VARIANT_TAGS.items() if tag in symbol),
            None,
        )
        if name is None:
            continue
        if name in variants:
            raise ValueError(f"duplicate ptxas record for {name}")
        variants[name] = {
            "symbol": symbol,
            "arch": symbol_arches[0],
            "stack_frame_bytes": int(stack.group(1)),
            "spill_store_bytes": int(stack.group(2)),
            "spill_load_bytes": int(stack.group(3)),
            "registers": int(registers.group(1)),
        }

    expected = set(_VARIANT_TAGS)
    if set(variants) != expected:
        raise ValueError(
            "expected SM89 Q64 FP16, INT8, and INT8+FP16 records; found "
            + ", ".join(sorted(variants))
        )
    return variants


def validate_resources(
    text: str,
    baseline: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Require sm_89 and stay within the audited three-CTA resource envelope."""

    variants = parse_resources(text)
    if baseline is not None and set(baseline) != set(_VARIANT_TAGS):
        raise ValueError("baseline must contain fp16, int8, and int8_fp16")
    for name, record in variants.items():
        failures = []
        if record["arch"] != "sm_89":
            failures.append(f"architecture {record['arch']}")
        limits = _RESOURCE_LIMITS[name]
        for field, label in (
            ("registers", "registers"),
            ("stack_frame_bytes", "stack frame"),
            ("spill_store_bytes", "spill stores"),
            ("spill_load_bytes", "spill loads"),
        ):
            if int(record[field]) > limits[field]:
                unit = "" if field == "registers" else " bytes"
                failures.append(
                    f"{label} {record[field]}{unit} exceed limit {limits[field]}{unit}"
                )
        if baseline is not None:
            old_registers = int(baseline[name]["registers"])
            if int(record["registers"]) > old_registers:
                failures.append(
                    f"registers {record['registers']} exceed baseline {old_registers}"
                )
        if failures:
            raise ValueError(
                f"SM89 Q64 {name} resource gate failed: " + ", ".join(failures)
            )
    return variants


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_log", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
    result = validate_resources(args.build_log.read_text(), baseline)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
