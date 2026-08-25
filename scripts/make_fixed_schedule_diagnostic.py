#!/usr/bin/env python3
"""Build a small native condition diagnostic with two schedules at every size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

# Installs the V2.0.1 short-identifier surface before native sampling.
from cordis_bench import v2_release as release
from cordis_bench import v2_native_anchor as anchor


SIZES = (8, 16, 24, 32)
CAPABILITIES = {"postcondition_guarantee", "reachability"}


def fixed_two_schedule_spec(source):
    base = anchor._base
    spec = dict(source)
    spec["schedules"] = dict(list(source["schedules"].items())[:2])
    spec["native_case_id"] = base.stable_id(
        {"diagnostic": "fixed_two_schedules", "source_case": source["native_case_id"]}
    )
    observations, count = base._native_reference(spec)
    spec["abstract_schedule_observations"] = observations
    spec["abstract_outcome_count"] = count
    spec["abstract_confluent"] = count == 1
    spec.pop("v2_predispose", None)
    spec["options"] = {}
    return base._augment_reconfiguration(spec)


def annotate(records, block_index, seed):
    release._annotate_release_block(records, block_index, seed)
    for item in records:
        metadata = item["metadata"]
        metadata["paper_role"] = "diagnostic"
        metadata["schedule_ablation"] = True
        metadata["fixed_schedule_count"] = 2
        dimensions = metadata["semantic_dimensions"]
        dimensions["paper_role"] = "diagnostic"
        dimensions["schedule_ablation"] = True
        dimensions["fixed_schedule_count"] = 2


def build_block(seed, block_index, worlds_per_size, runner_path, node):
    rng = random.Random(seed + 20_400_101)
    specs = []
    for size in SIZES:
        source = anchor._sample_specs_for_size(
            rng, size, worlds_per_size, require_count_variation=True
        )
        specs.extend(
            fixed_two_schedule_spec(spec) for spec in source[:worlds_per_size]
        )

    runtimes = anchor._base.execute_native_specs(
        specs, runner_path=runner_path, node=node
    )
    records = []
    for spec in specs:
        runtime = runtimes[spec["native_case_id"]]
        validation = anchor._base._validate_spec_runtime(spec, runtime)
        candidates = anchor._base._records_for_spec(
            spec, runtime, validation, "challenge", rng
        )
        records.extend(
            item
            for item in candidates
            if item["metadata"]["capability"] in CAPABILITIES
        )
    annotate(records, block_index, seed)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worlds-per-size", type=int, default=10)
    parser.add_argument("--runner-path", type=Path)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    if not 1 <= args.worlds_per_size <= 50:
        raise ValueError("worlds-per-size must be between 1 and 50")

    records = []
    for block_index, seed in enumerate(release.V2_RELEASE_SEEDS):
        records.extend(
            build_block(
                seed,
                block_index,
                args.worlds_per_size,
                args.runner_path,
                args.node,
            )
        )
    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate fixed-schedule diagnostic ids")
    random.Random(20_260_825).shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} fixed-schedule diagnostic items to {args.output}")


if __name__ == "__main__":
    main()
