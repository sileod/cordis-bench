#!/usr/bin/env python3
"""Replace selected predictions by id while preserving release order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("overrides", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    base = read_jsonl(args.base)
    replacements = {item["id"]: item for item in read_jsonl(args.overrides)}
    known = {item["id"] for item in base}
    unknown = sorted(set(replacements) - known)
    if unknown:
        raise ValueError(f"override ids absent from base predictions: {unknown[:3]}")

    merged = [replacements.get(item["id"], item) for item in base]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in merged:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"replaced {len(replacements)} of {len(base)} predictions in {args.output}")


if __name__ == "__main__":
    main()
