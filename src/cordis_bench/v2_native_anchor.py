"""Pre-freeze V2 native anchoring fixes applied to the canonical native generator.

This module exists only while the benchmark is explicitly ``pilot_unfrozen``.
Importing it patches the underlying V2 native generator with edge-case and
presentation fixes discovered by the manual pilot review.
"""

from __future__ import annotations

import random

from . import v2_native as _base
from .v2_common import canonical_list, make_record


V2_NATIVE_CORE_SIZES = _base.V2_NATIVE_CORE_SIZES
V2_NATIVE_CORE_WORLDS_PER_SIZE = _base.V2_NATIVE_CORE_WORLDS_PER_SIZE
V2_NATIVE_CHALLENGE_SIZES = _base.V2_NATIVE_CHALLENGE_SIZES
V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE = _base.V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE
V2_NATIVE_PRIMARY_TASKS = _base.V2_NATIVE_PRIMARY_TASKS
V2_NATIVE_DIAGNOSTIC_TASKS = _base.V2_NATIVE_DIAGNOSTIC_TASKS
V2_NATIVE_CORE_TASKS = _base.V2_NATIVE_CORE_TASKS
V2_NATIVE_CHALLENGE_TASKS = _base.V2_NATIVE_CHALLENGE_TASKS
V2_NATIVE_TASKS = _base.V2_NATIVE_TASKS
V2_NATIVE_CORE_RECORDS = _base.V2_NATIVE_CORE_RECORDS
V2_NATIVE_CHALLENGE_RECORDS = _base.V2_NATIVE_CHALLENGE_RECORDS
CORDIS_PACKAGE = _base.CORDIS_PACKAGE
CORDIS_VERSION = _base.CORDIS_VERSION
CORDIS_SOURCE_COMMIT = _base.CORDIS_SOURCE_COMMIT


def _condition_record(spec, runtime, validation, track, rng, *, reachable):
    observations = _base._runtime_observations(runtime)
    slots = _base._representative_slots(spec)
    leaf_values_by_slot = {}
    for leaf in spec["leaves"]:
        leaf_values_by_slot.setdefault(leaf["slot"], set()).add(leaf["value"])

    probes = []
    for slot in slots:
        values = {observation["slots"][slot] for observation in observations.values()}
        candidates = {spec["base_slots"][slot], *leaf_values_by_slot.get(slot, set())}
        present = rng.choice(sorted(values))
        absent = _base._surface_matched_absent(values, candidates, rng)
        if reachable:
            probes.extend(((slot, "==", present, True), (slot, "==", absent, False)))
        else:
            probes.extend(((slot, "!=", absent, True), (slot, "!=", present, False)))
    rng.shuffle(probes)

    conditions = []
    true_labels = []
    width = max(2, len(str(len(probes))))
    for index, (slot, operator, value, truth) in enumerate(probes, start=1):
        label = f"c{index:0{width}d}"
        conditions.append((label, slot, operator, value))
        if truth:
            true_labels.append(label)

    lines = "\n".join(
        f"  {label}: harness[{slot!r}] {operator} {value}"
        for label, slot, operator, value in conditions
    )
    if reachable:
        question = """For each condition independently, include its label if AT LEAST ONE listed
completion order reaches a final state satisfying that condition."""
        family = "reachable_conditions"
        capability = "reachability"
        quantifier = "exists_listed_interleaving_per_condition"
        relevance = "identify_possible_application_conditions"
    else:
        question = """For each condition independently, include its label if EVERY listed completion
order ends in a final state satisfying that condition."""
        family = "guaranteed_conditions"
        capability = "postcondition_guarantee"
        quantifier = "all_listed_interleavings_per_condition"
        relevance = "identify_guaranteed_application_conditions"

    prompt = _base._native_prefix(spec) + f"""
Named application conditions:
{lines}

{question}
Return a sorted JSON array of condition labels and nothing else."""
    return make_record(
        task_type=f"cordis_{family}",
        prompt=prompt,
        answer=canonical_list(true_labels, sort=True),
        answer_type="string_set",
        metadata=_base._metadata(
            spec, runtime, validation, track, family, capability, "string_set",
            f"cordis.{family}.v2",
            queried_conditions=len(conditions),
            query_quantifier=quantifier,
            oracle_type="differential_runtime_condition_set",
            operational_relevance=relevance,
        ),
        identity={"case": spec["native_case_id"], "task": family, "conditions": conditions},
    )


def _sample_specs_for_size(rng, size, worlds_per_size, require_count_variation):
    specs = []
    counts = set()
    attempts = 0
    while len(specs) < worlds_per_size:
        attempts += 1
        if attempts > max(100, worlds_per_size * 50):
            raise RuntimeError(f"could not sample varied V2 native worlds for size={size}")
        spec = _base._augment_reconfiguration(_base._sample_native_spec(rng, int(size)))
        count = spec["abstract_outcome_count"]
        if (
            require_count_variation
            and worlds_per_size > 1
            and len(specs) == worlds_per_size - 1
            and len(counts) == 1
            and count in counts
        ):
            continue
        specs.append(spec)
        counts.add(count)
    return specs


def _generate_native_dataset(sizes, worlds_per_size, seed, track, runner_path, node):
    rng = random.Random(seed)
    require_count_variation = track == "challenge"
    specs = [
        spec
        for size in sizes
        for spec in _sample_specs_for_size(
            rng,
            int(size),
            worlds_per_size,
            require_count_variation=require_count_variation,
        )
    ]
    runtimes = _base.execute_native_specs(specs, runner_path=runner_path, node=node)
    records = []
    for spec in specs:
        runtime = runtimes[spec["native_case_id"]]
        validation = _base._validate_spec_runtime(spec, runtime)
        records.extend(_base._records_for_spec(spec, runtime, validation, track, rng))
    return records


def generate_v2_native_core_dataset(
    sizes=V2_NATIVE_CORE_SIZES,
    worlds_per_size=V2_NATIVE_CORE_WORLDS_PER_SIZE,
    seed=0,
    runner_path=None,
    node="node",
):
    return _generate_native_dataset(
        tuple(map(int, sizes)),
        worlds_per_size,
        seed + 20_300_101,
        "core",
        runner_path,
        node,
    )


def generate_v2_native_challenge_dataset(
    sizes=V2_NATIVE_CHALLENGE_SIZES,
    worlds_per_size=V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE,
    seed=0,
    runner_path=None,
    node="node",
):
    return _generate_native_dataset(
        tuple(map(int, sizes)),
        worlds_per_size,
        seed + 20_400_101,
        "challenge",
        runner_path,
        node,
    )


# The base record constructor resolves this global at call time.
_base._condition_record = _condition_record
