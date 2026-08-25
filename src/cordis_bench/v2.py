"""CordisBench V2: formal + actual Cordis, each with core + challenge tracks."""

from __future__ import annotations

import random

from .guards import assert_v2_guards
from .v2_formal import (
    V2_FORMAL_CHALLENGE_RECORDS,
    V2_FORMAL_CHALLENGE_SIZES,
    V2_FORMAL_CHALLENGE_TASKS,
    V2_FORMAL_CHALLENGE_WORLDS_PER_SIZE,
    V2_FORMAL_CORE_RECORDS,
    V2_FORMAL_CORE_SIZES,
    V2_FORMAL_CORE_TASKS,
    V2_FORMAL_CORE_WORLDS_PER_SIZE,
    V2_FORMAL_TASKS,
    generate_v2_formal_challenge_dataset,
    generate_v2_formal_core_dataset,
)
from .v2_native_anchor import (
    V2_NATIVE_CHALLENGE_RECORDS,
    V2_NATIVE_CHALLENGE_SIZES,
    V2_NATIVE_CHALLENGE_TASKS,
    V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE,
    V2_NATIVE_CORE_RECORDS,
    V2_NATIVE_CORE_SIZES,
    V2_NATIVE_CORE_TASKS,
    V2_NATIVE_CORE_WORLDS_PER_SIZE,
    V2_NATIVE_TASKS,
    generate_v2_native_challenge_dataset,
    generate_v2_native_core_dataset,
)


V2_TASKS = V2_FORMAL_TASKS + V2_NATIVE_TASKS
V2_CORE_TASKS = V2_FORMAL_CORE_TASKS + V2_NATIVE_CORE_TASKS
V2_CHALLENGE_TASKS = V2_FORMAL_CHALLENGE_TASKS + V2_NATIVE_CHALLENGE_TASKS

V2_FORMAL_RECORDS = V2_FORMAL_CORE_RECORDS + V2_FORMAL_CHALLENGE_RECORDS
V2_NATIVE_RECORDS = V2_NATIVE_CORE_RECORDS + V2_NATIVE_CHALLENGE_RECORDS
V2_DEFAULT_N = V2_FORMAL_RECORDS + V2_NATIVE_RECORDS


def _annotate_provenance(records, seed, guard):
    for item in records:
        metadata = item["metadata"]
        metadata["generation_seed"] = seed
        metadata["release_status"] = "pilot_unfrozen"
        metadata["hard_guard_passed"] = True
        metadata["hard_guard_version"] = "v2-rc2"
        dimensions = metadata.setdefault("semantic_dimensions", {})
        dimensions["generation_seed"] = seed
        dimensions["release_status"] = "pilot_unfrozen"
        dimensions["hard_guard_passed"] = True
        dimensions["hard_guard_version"] = "v2-rc2"
    return guard


def generate_v2_dataset(
    n=None,
    seed=0,
    task_types=V2_TASKS,
    formal_sizes=V2_FORMAL_CHALLENGE_SIZES,
    native_sizes=V2_NATIVE_CHALLENGE_SIZES,
    formal_worlds=V2_FORMAL_CHALLENGE_WORLDS_PER_SIZE,
    native_worlds=V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE,
    runner_path=None,
    node="node",
    *,
    formal_core_sizes=V2_FORMAL_CORE_SIZES,
    formal_core_worlds=V2_FORMAL_CORE_WORLDS_PER_SIZE,
    native_core_sizes=V2_NATIVE_CORE_SIZES,
    native_core_worlds=V2_NATIVE_CORE_WORLDS_PER_SIZE,
):
    if tuple(task_types) != V2_TASKS:
        raise ValueError(
            "V2 is a fixed capability matrix; custom --tasks is intentionally disabled"
        )
    formal_sizes = tuple(map(int, formal_sizes))
    native_sizes = tuple(map(int, native_sizes))
    formal_core_sizes = tuple(map(int, formal_core_sizes))
    native_core_sizes = tuple(map(int, native_core_sizes))
    if not formal_sizes or not native_sizes or not formal_core_sizes or not native_core_sizes:
        raise ValueError("V2 requires non-empty formal/native core and challenge ladders")
    if min(formal_sizes + formal_core_sizes) < 2:
        raise ValueError("V2 formal semantic sizes must be >= 2")
    if any(size < 2 or size % 2 for size in native_sizes + native_core_sizes):
        raise ValueError("V2 native semantic sizes must be even integers >= 2")
    if min(formal_sizes) <= max(formal_core_sizes):
        raise ValueError("V2 formal challenge sizes must start above the core ladder")
    if min(native_sizes) <= max(native_core_sizes):
        raise ValueError("V2 native challenge sizes must start above the core ladder")
    if min(formal_worlds, native_worlds, formal_core_worlds, native_core_worlds) <= 0:
        raise ValueError("V2 worlds-per-size values must be positive")

    records = []
    records.extend(
        generate_v2_formal_core_dataset(
            sizes=formal_core_sizes,
            worlds_per_size=formal_core_worlds,
            seed=seed,
        )
    )
    records.extend(
        generate_v2_formal_challenge_dataset(
            sizes=formal_sizes,
            worlds_per_size=formal_worlds,
            seed=seed,
        )
    )
    records.extend(
        generate_v2_native_core_dataset(
            sizes=native_core_sizes,
            worlds_per_size=native_core_worlds,
            seed=seed,
            runner_path=runner_path,
            node=node,
        )
    )
    records.extend(
        generate_v2_native_challenge_dataset(
            sizes=native_sizes,
            worlds_per_size=native_worlds,
            seed=seed,
            runner_path=runner_path,
            node=node,
        )
    )

    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate V2 IDs")

    random.Random(seed + 20_999_991).shuffle(records)

    expected = (
        len(formal_core_sizes) * formal_core_worlds * len(V2_FORMAL_CORE_TASKS)
        + len(formal_sizes) * formal_worlds * len(V2_FORMAL_CHALLENGE_TASKS)
        + len(native_core_sizes) * native_core_worlds * len(V2_NATIVE_CORE_TASKS)
        + len(native_sizes) * native_worlds * len(V2_NATIVE_CHALLENGE_TASKS)
    )
    if len(records) != expected:
        raise RuntimeError(f"V2 emitted {len(records)} records, expected {expected}")
    if n is not None and n != len(records):
        raise ValueError(f"requested n={n}, selected V2 configuration emits {len(records)}")

    guard = assert_v2_guards(records)
    _annotate_provenance(records, seed, guard)
    return records
