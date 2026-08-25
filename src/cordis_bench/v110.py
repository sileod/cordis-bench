"""V1.10: append-only replication scale-up of the frozen V1.9 benchmark.

V1.10 changes sample density, not the construct. The first 444 records are
emitted by ``generate_v19_dataset`` unchanged, preserving every V1.9 item ID,
prompt, answer, and metadata field. The release then appends independently
generated matched worlds under the already validated bridge, Cordis-native,
and formal-boundary generators.

Default composition:

- 180 harnessless-core records (inherited unchanged),
- 160 harness-bridge records (48 inherited + 112 new),
- 160 Cordis-native records (64 inherited + 96 new), and
- 360 formal-boundary records (152 inherited + 208 new),

for 860 records total.
"""

from __future__ import annotations

from collections import Counter
import random

from .tasks import stable_id
from .v16_stress import generate_v16_stress_dataset
from .v17 import generate_v17_dataset as generate_v17_bridge_dataset
from .v17_suite import _annotate_bridge
from .v18 import generate_v18_native_dataset
from .v19 import (
    V19_DEFAULT_N,
    V19_OVERLAP_PROFILES,
    V19_TASKS,
    _annotate_disjoint,
    _overlap_records,
    _sample_overlap_curve,
    generate_v19_dataset,
)


V110_TASKS = V19_TASKS
V110_INHERITED_RECORDS = V19_DEFAULT_N
V110_CORE_RECORDS = 180
V110_BRIDGE_RECORDS = 160
V110_NATIVE_RECORDS = 160
V110_BOUNDARY_RECORDS = 360

V110_EXTRA_BRIDGE_RECORDS = V110_BRIDGE_RECORDS - 48
V110_EXTRA_NATIVE_RECORDS = V110_NATIVE_RECORDS - 64
V110_EXTRA_BOUNDARY_RECORDS = V110_BOUNDARY_RECORDS - 152
V110_EXTRA_RECORDS = (
    V110_EXTRA_BRIDGE_RECORDS
    + V110_EXTRA_NATIVE_RECORDS
    + V110_EXTRA_BOUNDARY_RECORDS
)
V110_DEFAULT_N = V110_INHERITED_RECORDS + V110_EXTRA_RECORDS

# Each V1.6-stress curve emits 18 records. Three fresh curves for each size
# contribute 108 additional disjoint-scale questions.
V110_EXTRA_DISJOINT_CURVES_PER_SIZE = 3
V110_EXTRA_DISJOINT_RECORDS_PER_SIZE = 54

# Each overlap curve emits 5 levels x 2 global tasks = 10 records. Add more
# replication to the cyclic profiles, which were among the thinnest and most
# informative V1.9 structural strata, without changing their semantics.
V110_EXTRA_OVERLAP_CURVES = {
    "path_d2_ops1": 2,
    "path_d4_ops1": 2,
    "cycle_d2_ops1": 3,
    "cycle_d4_ops2": 3,
}


def _promote_replication(record, replication_block):
    """Give a fresh V1.10 ID while preserving source provenance and semantics."""
    source_id = record["id"]
    metadata = record.setdefault("metadata", {})
    source_benchmark_suite = metadata.get("benchmark_suite")

    record["id"] = stable_id(
        {
            "schema_version": "1.10-replication",
            "replication_block": replication_block,
            "source_id": source_id,
        }
    )
    record["schema_version"] = "1.10"

    metadata.update(
        {
            "benchmark_suite": "1.10",
            "release_added": "1.10",
            "replication_release": "1.10",
            "replication_block": replication_block,
            "replication_source_id": source_id,
            "replication_source_benchmark_suite": source_benchmark_suite,
            "independent_semantic_world": True,
        }
    )
    dimensions = metadata.setdefault("semantic_dimensions", {})
    dimensions.update(
        {
            "release_added": "1.10",
            "replication_release": "1.10",
            "replication_block": replication_block,
            "independent_semantic_world": True,
        }
    )
    return record


def generate_v110_bridge_replication(seed=0):
    bridge = generate_v17_bridge_dataset(
        V110_EXTRA_BRIDGE_RECORDS,
        seed=seed + 10_170_003,
        n_components=7,
        width=5,
    )
    records = [_annotate_bridge(record) for record in bridge]
    return [_promote_replication(record, "harness_bridge") for record in records]


def generate_v110_native_replication(seed=0, runner_path=None, node="node"):
    native = generate_v18_native_dataset(
        V110_EXTRA_NATIVE_RECORDS,
        seed=seed + 10_180_081,
        runner_path=runner_path,
        node=node,
    )
    return [_promote_replication(record, "cordis_native") for record in native]


def generate_v110_boundary_replication(seed=0):
    records = []

    for index, n_components in enumerate((14, 16)):
        stress = generate_v16_stress_dataset(
            V110_EXTRA_DISJOINT_RECORDS_PER_SIZE,
            seed=seed + 10_190_100 + index,
            n_components=n_components,
            width=n_components - 1,
        )
        for record in stress:
            records.append(
                _promote_replication(
                    _annotate_disjoint(record, n_components),
                    "formal_boundary",
                )
            )

    profile_map = {profile: (topology, depth, ops) for profile, topology, depth, ops in V19_OVERLAP_PROFILES}
    rng = random.Random(seed + 10_190_900)
    for profile, extra_curves in V110_EXTRA_OVERLAP_CURVES.items():
        topology, dep_depth, ops_per_leaf = profile_map[profile]
        for _ in range(extra_curves):
            curve = _sample_overlap_curve(
                rng,
                profile=profile,
                topology=topology,
                dep_depth=dep_depth,
                ops_per_leaf=ops_per_leaf,
            )
            records.extend(
                _promote_replication(record, "formal_boundary")
                for record in _overlap_records(rng, curve)
            )

    if len(records) != V110_EXTRA_BOUNDARY_RECORDS:
        raise RuntimeError(
            f"V1.10 boundary replication emitted {len(records)} records, "
            f"expected {V110_EXTRA_BOUNDARY_RECORDS}"
        )
    return records


def generate_v110_replication_dataset(seed=0, runner_path=None, node="node"):
    bridge = generate_v110_bridge_replication(seed=seed)
    native = generate_v110_native_replication(
        seed=seed,
        runner_path=runner_path,
        node=node,
    )
    boundary = generate_v110_boundary_replication(seed=seed)
    records = bridge + native + boundary
    if len(records) != V110_EXTRA_RECORDS:
        raise RuntimeError(
            f"V1.10 replication emitted {len(records)} records, expected {V110_EXTRA_RECORDS}"
        )
    return records


def generate_v110_dataset(
    n=V110_DEFAULT_N,
    seed=0,
    task_types=V110_TASKS,
    n_components=None,
    width=None,
    runner_path=None,
    node="node",
):
    if n != V110_DEFAULT_N:
        raise ValueError(
            f"V1.10 is a fixed append-only superset of {V19_DEFAULT_N} V1.9 records + "
            f"{V110_EXTRA_RECORDS} replication records; use n={V110_DEFAULT_N}"
        )
    if tuple(task_types) != V110_TASKS:
        raise ValueError("V1.10 is a fixed composite suite; custom --tasks is not supported")
    if n_components is not None or width is not None:
        raise ValueError("V1.10 has per-component sizes; do not pass --components or --width")

    inherited = generate_v19_dataset(
        V19_DEFAULT_N,
        seed=seed,
        runner_path=runner_path,
        node=node,
    )
    appended = generate_v110_replication_dataset(
        seed=seed,
        runner_path=runner_path,
        node=node,
    )

    inherited_ids = {item["id"] for item in inherited}
    appended_ids = {item["id"] for item in appended}
    if len(appended_ids) != len(appended):
        raise RuntimeError("V1.10 replication contains duplicate item IDs")
    if inherited_ids & appended_ids:
        raise RuntimeError("V1.10 replication ID collided with inherited V1.9 item")

    dataset = inherited + appended
    component_counts = Counter(
        item.get("metadata", {}).get("benchmark_component")
        for item in dataset
    )
    expected = {
        "harnessless_core": V110_CORE_RECORDS,
        "harness_bridge": V110_BRIDGE_RECORDS,
        "cordis_native": V110_NATIVE_RECORDS,
        "formal_boundary": V110_BOUNDARY_RECORDS,
    }
    if component_counts != expected:
        raise RuntimeError(
            f"V1.10 component counts {dict(component_counts)} != expected {expected}"
        )
    return dataset
