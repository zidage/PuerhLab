#!/usr/bin/env python3
"""
Parse Nikon lens id_map.txt into id_map.json.

Input format supports:
1) '00 40 3C 8E 2C 40 40 0E' = Lens Name
2) 48 = Lens Name
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict


HEX_LINE_RE = re.compile(
    r"^\s*'(?P<key>[0-9A-Fa-f]{2}(?:\s+[0-9A-Fa-f]{2}){7})'\s*=\s*(?P<value>.+?)\s*$"
)
NUM_LINE_RE = re.compile(r"^\s*(?P<key>\d+)\s*=\s*(?P<value>.+?)\s*$")


def _normalize_hex_key(raw: str) -> str:
    return " ".join(part.upper() for part in raw.strip().split())


def _merge_value(existing: str, new_value: str) -> str:
    if existing == new_value:
        return existing
    return f"{existing} | {new_value}"


def parse_id_map(text: str) -> tuple[Dict[str, str], Dict[str, str]]:
    hex_map: Dict[str, str] = {}
    numeric_map: Dict[str, str] = {}

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        hex_match = HEX_LINE_RE.match(line)
        if hex_match:
            key = _normalize_hex_key(hex_match.group("key"))
            value = hex_match.group("value").strip()
            if key in hex_map:
                hex_map[key] = _merge_value(hex_map[key], value)
            else:
                hex_map[key] = value
            continue

        num_match = NUM_LINE_RE.match(line)
        if num_match:
            key = str(int(num_match.group("key")))
            value = num_match.group("value").strip()
            if key in numeric_map:
                numeric_map[key] = _merge_value(numeric_map[key], value)
            else:
                numeric_map[key] = value
            continue

        # Keep parser strict: fail fast on malformed lines.
        raise ValueError(f"Unsupported line format at line {line_no}: {raw_line}")

    return hex_map, numeric_map


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    src_path = base_dir / "id_map.txt"
    out_path = base_dir / "id_map.json"

    raw_text = src_path.read_text(encoding="utf-8")
    hex_map, numeric_map = parse_id_map(raw_text)

    payload = {
        "meta": {
            "source": src_path.name,
            "generated_by": Path(__file__).name,
            "hex_entries": len(hex_map),
            "numeric_entries": len(numeric_map),
        },
        "hex_id_map": hex_map,
        "numeric_id_map": numeric_map,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated: {out_path}")
    print(f"hex_entries={len(hex_map)} numeric_entries={len(numeric_map)}")


if __name__ == "__main__":
    main()

