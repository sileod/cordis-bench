#!/usr/bin/env python3
"""Select a deterministic, block-balanced subset for the effort ablation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()

    candidates = [
        row
        for row in read_jsonl(args.source)
        if row.get("metadata", {}).get("paper_role") == "primary"
        and row.get("metadata", {}).get("semantic_size") == args.size
    ]
    systems: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in candidates:
        metadata = row["metadata"]
        realization = "cordis_native" if metadata.get("harness_surface") else "formal"
        key = (int(metadata["release_replicate_index"]), realization)
        systems[key].add(metadata["latent_group_id"])

    selected = set()
    for key, latent_ids in sorted(systems.items()):
        ordered = sorted(latent_ids)
        if len(ordered) % 2:
            raise ValueError(f"{key} has odd latent-system count {len(ordered)}")
        selected.update((key, latent_id) for latent_id in ordered[: len(ordered) // 2])

    rows = []
    for row in candidates:
        metadata = row["metadata"]
        realization = "cordis_native" if metadata.get("harness_surface") else "formal"
        key = (int(metadata["release_replicate_index"]), realization)
        if (key, metadata["latent_group_id"]) in selected:
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} size-{args.size} effort-ablation items to {args.output}")


if __name__ == "__main__":
    main()
