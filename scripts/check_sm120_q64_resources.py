"""Fail closed on the SM120 Q64 FP16 and MXFP8 kernel resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


_KERNEL = "mixed_attention_sm120_q64_kernel"
_COMPILE_RE = re.compile(
    r"Compiling entry function '([^']+)' for '(sm_[0-9]+a?)'"
)
_PROPERTIES_RE = re.compile(r"Function properties for (\S+)")
_STACK_RE = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
    r"(\d+) bytes spill loads"
)
_REGISTERS_RE = re.compile(r"Used (\d+) registers")


def validate_resources(text: str) -> dict[str, object]:
    lines = text.splitlines()
    arches: dict[str, list[str]] = {}
    for line in lines:
        match = _COMPILE_RE.search(line)
        if match:
            arches.setdefault(match.group(1), []).append(match.group(2))

    records: list[dict[str, object]] = []
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

    variants: dict[str, dict[str, object]] = {}
    for record in records:
        symbol = str(record["symbol"])
        if "ELb0ELb1ELb0EE" in symbol:
            name = "fp16"
        elif "ELb1ELb1ELb0EE" in symbol:
            name = "mxfp8"
        else:
            continue
        if name in variants:
            raise ValueError(f"duplicate ptxas record for {name}")
        variants[name] = record
    if set(variants) != {"fp16", "mxfp8"}:
        raise ValueError(
            "expected FP16 and unified MXFP8 ptxas records, found "
            + ", ".join(sorted(variants))
        )
    for name, record in variants.items():
        failures = []
        if record["arch"] != "sm_120a":
            failures.append(f"architecture {record['arch']}")
        if record["stack_frame_bytes"] != 0:
            failures.append(f"stack frame {record['stack_frame_bytes']} bytes")
        if record["spill_store_bytes"] != 0:
            failures.append(f"spill stores {record['spill_store_bytes']} bytes")
        if record["spill_load_bytes"] != 0:
            failures.append(f"spill loads {record['spill_load_bytes']} bytes")
        if int(record["registers"]) > 168:
            failures.append(f"registers {record['registers']} exceed 168")
        if failures:
            raise ValueError(
                f"SM120 Q64 {name} resource gate failed: "
                + ", ".join(failures)
            )
    return variants


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_log", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = validate_resources(args.build_log.read_text())
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
