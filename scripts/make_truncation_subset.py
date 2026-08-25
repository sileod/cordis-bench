#!/usr/bin/env python3
"""Select released items whose original completion stopped at the token cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    truncated = {
        row["id"]
        for row in read_jsonl(args.predictions)
        if row.get("finish_reason") == "length"
        or str(row.get("native_finish_reason", "")).lower() in {"length", "max_tokens"}
    }
    selected = [row for row in read_jsonl(args.dataset) if row["id"] in truncated]
    if len(selected) != len(truncated):
        missing = truncated - {row["id"] for row in selected}
        raise ValueError(f"{len(missing)} truncated prediction ids are absent from the dataset")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(selected)} token-capped items to {args.output}")


if __name__ == "__main__":
    main()
