"""Paper-facing CordisBench V2.0.1 release assembly.

V2.0.1 differs from V2.0 only in the model-facing identifier surface: long
numeric opaque names are alpha-renamed to short opaque names while preserving
the same RNG draws and semantic construction for a fixed seed.
"""

from __future__ import annotations

import random
import re

from .guards import assert_v2_guards
from . import v201_surface as _v201_surface  # install before importing V2 generators
from .v2 import (
    V2_DEFAULT_N as V2_BLOCK_N,
    V2_FORMAL_CHALLENGE_SIZES,
    V2_FORMAL_CHALLENGE_WORLDS_PER_SIZE,
    V2_FORMAL_CORE_SIZES,
    V2_FORMAL_CORE_WORLDS_PER_SIZE,
    V2_NATIVE_CHALLENGE_SIZES,
    V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE,
    V2_NATIVE_CORE_SIZES,
    V2_NATIVE_CORE_WORLDS_PER_SIZE,
    V2_TASKS,
    generate_v2_dataset,
)

V2_RELEASE_VERSION = "2.0.1"
V2_RELEASE_REPLICATES = 3
V2_RELEASE_N = V2_BLOCK_N * V2_RELEASE_REPLICATES
V2_RELEASE_SEEDS = tuple(range(V2_RELEASE_REPLICATES))

_LONG_IDENTIFIER = re.compile(
    r"\b(?:plugin|provider|service|slot)_\d{6,}\b|\bq\d{6}\b"
)


def _annotate_release_block(records, block_index, block_seed):
    for item in records:
        metadata = item["metadata"]
        metadata["benchmark_suite"] = V2_RELEASE_VERSION
        metadata["generator_version"] = V2_RELEASE_VERSION
        metadata["identifier_surface"] = _v201_surface.V201_IDENTIFIER_SURFACE
        metadata["identifier_surface_version"] = V2_RELEASE_VERSION
        metadata["release_status"] = "v2.0.1_frozen"
        metadata["release_replicate_index"] = block_index
        metadata["release_replicate_seed"] = block_seed
        metadata["hard_guard_version"] = "v2.0.1"
        dimensions = metadata.setdefault("semantic_dimensions", {})
        dimensions["benchmark_suite"] = V2_RELEASE_VERSION
        dimensions["generator_version"] = V2_RELEASE_VERSION
        dimensions["identifier_surface"] = _v201_surface.V201_IDENTIFIER_SURFACE
        dimensions["identifier_surface_version"] = V2_RELEASE_VERSION
        dimensions["release_status"] = "v2.0.1_frozen"
        dimensions["release_replicate_index"] = block_index
        dimensions["release_replicate_seed"] = block_seed
        dimensions["hard_guard_version"] = "v2.0.1"


def _assert_v201_identifier_surface(records):
    for item in records:
        metadata = item["metadata"]
        if metadata.get("benchmark_suite") != V2_RELEASE_VERSION:
            raise AssertionError("V2.0.1 release row missing benchmark-suite provenance")
        if metadata.get("identifier_surface") != "short_opaque":
            raise AssertionError("V2.0.1 release row missing short-identifier provenance")
        rendered = f"{item.get('prompt', '')}\n{item.get('answer', '')}"
        match = _LONG_IDENTIFIER.search(rendered)
        if match:
            raise AssertionError(
                f"V2.0.1 long model-facing identifier leaked: {match.group(0)}"
            )


def generate_v2_release_dataset(
    n=None,
    seed=0,
    replicates=V2_RELEASE_REPLICATES,
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
    replicates = int(replicates)
    if replicates <= 0:
        raise ValueError("V2 release replicates must be positive")

    records = []
    for block_index in range(replicates):
        block_seed = int(seed) + block_index
        block = generate_v2_dataset(
            seed=block_seed,
            task_types=task_types,
            formal_sizes=formal_sizes,
            native_sizes=native_sizes,
            formal_worlds=formal_worlds,
            native_worlds=native_worlds,
            runner_path=runner_path,
            node=node,
            formal_core_sizes=formal_core_sizes,
            formal_core_worlds=formal_core_worlds,
            native_core_sizes=native_core_sizes,
            native_core_worlds=native_core_worlds,
        )
        _annotate_release_block(block, block_index, block_seed)
        records.extend(block)

    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate V2 IDs across release replicate blocks")

    random.Random(int(seed) + 20_999_993 + replicates).shuffle(records)

    assert_v2_guards(records)
    _assert_v201_identifier_surface(records)
    if n is not None and int(n) != len(records):
        raise ValueError(
            f"requested n={n}, selected V2 release configuration emits {len(records)}"
        )
    return records
