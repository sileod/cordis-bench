"""Hardened actual-Cordis implementation for the V1.12 challenge track.

Gold cardinalities and schedule sets are sampled independently of generator
position.  The neutral non-queried branch is the same size as the queried
branch, and the number of controlled schedules grows with challenge size.
"""

import random

from .tasks import stable_id
from . import v112_native as _base


CORDIS_VERSION = _base.CORDIS_VERSION
V112_NATIVE_COMPONENT = _base.V112_NATIVE_COMPONENT
V112_NATIVE_TASKS = _base.V112_NATIVE_TASKS
V112_NATIVE_SIZES = _base.V112_NATIVE_SIZES
V112_NATIVE_WORLDS_PER_SIZE = _base.V112_NATIVE_WORLDS_PER_SIZE
V112_NATIVE_RECORDS = _base.V112_NATIVE_RECORDS
V112_NATIVE_MIN_SCHEDULES = 12


def _schedule_limit(size):
    return min(32, max(V112_NATIVE_MIN_SCHEDULES, size))


def _native_reference(spec):
    return _base._native_reference(spec)


def _sample_native_spec(rng, size, profile=None, desired_outcomes=None):
    if size < 4 or size % 2:
        raise ValueError("native V1.12 size must be an even integer >= 4")

    query_provider = _base._label(rng, "provider")
    other_provider = _base._label(rng, "provider")
    query_service = _base._label(rng, "service")
    other_service = _base._label(rng, "service")
    decoy = _base._label(rng, "plugin")
    query_names = [_base._label(rng, "plugin") for _ in range(size)]
    other_count = size
    other_names = [_base._label(rng, "plugin") for _ in range(other_count)]

    n_values = size // 2 + size + 2 * other_count
    unique_values = iter(rng.sample(range(10_000, 9_999_999), n_values))
    leaves = []
    base_slots = {}
    pairs = []
    for pair_index in range(size // 2):
        slot = _base._label(rng, "slot")
        base_slots[slot] = next(unique_values)
        pair = tuple(query_names[2 * pair_index : 2 * pair_index + 2])
        pairs.append(pair)
        for name in pair:
            leaves.append(
                {
                    "name": name,
                    "service": query_service,
                    "slot": slot,
                    "value": next(unique_values),
                }
            )
    for name in other_names:
        slot = _base._label(rng, "slot")
        base_slots[slot] = next(unique_values)
        leaves.append(
            {
                "name": name,
                "service": other_service,
                "slot": slot,
                "value": next(unique_values),
            }
        )

    startup_order = query_names + other_names
    rng.shuffle(startup_order)
    limit = _schedule_limit(size)
    max_outcomes = min(limit, 2 ** len(pairs))
    if desired_outcomes is None:
        desired_outcomes = rng.randint(max(2, limit // 4), max_outcomes)
    if not 2 <= desired_outcomes <= max_outcomes:
        raise ValueError(
            f"desired_outcomes={desired_outcomes} outside [2, {max_outcomes}]"
        )
    schedules = _base._schedule_orders(
        rng,
        pairs,
        desired_outcomes,
        limit=limit,
    )
    spec = {
        "native_case_id": stable_id(
            {
                "schema_version": "1.12-native-hardened",
                "size": size,
                "providers": (query_provider, other_provider),
                "leaves": leaves,
                "startup": startup_order,
                "schedules": schedules,
            }
        ),
        "base_slots": base_slots,
        "providers": [
            {"name": query_provider, "service": query_service},
            {"name": other_provider, "service": other_service},
        ],
        "decoy": decoy,
        "leaves": leaves,
        "startup_order": startup_order,
        "query_provider": query_provider,
        "schedules": schedules,
        "target": {"slots": base_slots, "active": []},
        "options": {},
        "challenge_size": size,
    }
    observations, count = _base._native_reference(spec)
    if count != desired_outcomes:
        raise RuntimeError(
            f"native construction expected {desired_outcomes} outcomes, got {count}"
        )
    spec["abstract_schedule_observations"] = observations
    spec["abstract_outcome_count"] = count
    spec["abstract_confluent"] = count == 1
    return spec


def _desired_counts(rng, size, worlds_per_size):
    limit = _schedule_limit(size)
    low = max(2, limit // 4)
    possible = list(range(low, limit + 1))
    if worlds_per_size <= len(possible):
        return rng.sample(possible, worlds_per_size)
    values = []
    while len(values) < worlds_per_size:
        cycle = possible[:]
        rng.shuffle(cycle)
        values.extend(cycle)
    return values[:worlds_per_size]


def generate_v112_native_dataset(
    sizes=V112_NATIVE_SIZES,
    worlds_per_size=V112_NATIVE_WORLDS_PER_SIZE,
    seed=0,
    runner_path=None,
    node="node",
):
    rng = random.Random(seed + 12_100_701)
    specs = []
    for size in sizes:
        desired = _desired_counts(rng, int(size), worlds_per_size)
        specs.extend(
            _sample_native_spec(rng, int(size), desired_outcomes=count)
            for count in desired
        )
    runtimes = _base.execute_native_specs(specs, runner_path=runner_path, node=node)
    records = []
    for spec in specs:
        records.extend(_base._records(rng, spec, runtimes[spec["native_case_id"]]))
    return records
