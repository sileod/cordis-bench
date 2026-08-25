"""Actual-Cordis V1.12 challenge with neutral labels and fixed schedule count."""

import json
import random

from .tasks import stable_id
from .v18 import (
    CORDIS_SOURCE_COMMIT,
    CORDIS_VERSION,
    execute_native_specs,
    _render_native_code,
)
from .v112_common import CHECKSUM_MODULUS, metadata, record


V112_NATIVE_COMPONENT = "cordis_native_challenge"
V112_NATIVE_TASKS = (
    "cordis_challenge_schedule_checksum",
    "cordis_challenge_outcome_count",
)
V112_NATIVE_SIZES = (8, 16, 24, 32)
V112_NATIVE_WORLDS_PER_SIZE = 4
V112_NATIVE_RECORDS = (
    len(V112_NATIVE_SIZES) * V112_NATIVE_WORLDS_PER_SIZE * 2 * 2
)
V112_NATIVE_SCHEDULES = 12


def _label(rng, prefix):
    return f"{prefix}_{rng.randrange(10**9):09d}"


def _linear_extension(rng, ordered_pairs):
    state = [0] * len(ordered_pairs)
    output = []
    while len(output) < 2 * len(ordered_pairs):
        available = []
        for index, pair in enumerate(ordered_pairs):
            if state[index] == 0:
                available.append((index, pair[0]))
            elif state[index] == 1:
                available.append((index, pair[1]))
        index, name = rng.choice(available)
        output.append(name)
        state[index] += 1
    return output


def _schedule_orders(rng, pairs, desired_outcomes, limit=V112_NATIVE_SCHEDULES):
    if desired_outcomes > 2 ** len(pairs):
        raise ValueError("not enough pairwise outcome patterns")
    patterns = []
    seen_patterns = set()
    while len(patterns) < desired_outcomes:
        bits = tuple(rng.randrange(2) for _ in pairs)
        if bits not in seen_patterns:
            seen_patterns.add(bits)
            patterns.append(bits)

    schedules = {}
    seen_orders = set()
    for index in range(limit):
        bits = patterns[index % desired_outcomes]
        ordered_pairs = [
            pair if bit else tuple(reversed(pair))
            for pair, bit in zip(pairs, bits)
        ]
        for _ in range(20_000):
            order = _linear_extension(rng, ordered_pairs)
            key = tuple(order)
            if key not in seen_orders:
                seen_orders.add(key)
                schedules[f"schedule_{index + 1}"] = order
                break
        else:
            raise RuntimeError("could not sample enough distinct native schedules")
    return schedules


def _native_reference(spec):
    slots = dict(spec["base_slots"])
    witnesses = {}
    by_name = {leaf["name"]: leaf for leaf in spec["leaves"]}
    for name in spec["startup_order"]:
        leaf = by_name[name]
        witnesses[name] = slots[leaf["slot"]]
        slots[leaf["slot"]] = leaf["value"]
    initial_slots = dict(slots)
    initial_active = {
        spec["decoy"],
        *(provider["name"] for provider in spec["providers"]),
        *(leaf["name"] for leaf in spec["leaves"]),
    }
    observations = {}
    for label, order in spec["schedules"].items():
        current = dict(initial_slots)
        active = set(initial_active)
        for name in order:
            leaf = by_name[name]
            current[leaf["slot"]] = witnesses[name]
            active.remove(name)
        active.remove(spec["query_provider"])
        observations[label] = {
            "slots": dict(sorted(current.items())),
            "active": sorted(active),
        }
    distinct = {json.dumps(value, sort_keys=True) for value in observations.values()}
    return observations, len(distinct)


def _slot_checksum(observation):
    values = list(observation["slots"].values())
    return (sum(values) + 257 * sum(value * value for value in values)) % CHECKSUM_MODULUS


def _sample_native_spec(rng, size, profile=0):
    if size < 4 or size % 2:
        raise ValueError("native V1.12 size must be an even integer >= 4")
    query_provider = _label(rng, "provider")
    other_provider = _label(rng, "provider")
    query_service = _label(rng, "service")
    other_service = _label(rng, "service")
    decoy = _label(rng, "plugin")
    query_names = [_label(rng, "plugin") for _ in range(size)]
    other_count = max(4, size // 2)
    other_names = [_label(rng, "plugin") for _ in range(other_count)]

    unique_values = iter(rng.sample(range(10_000, 999_999), size + other_count + size))
    leaves = []
    base_slots = {}
    pairs = []
    for pair_index in range(size // 2):
        slot = _label(rng, "slot")
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
        slot = _label(rng, "slot")
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
    desired = 3 + (profile % 4)
    schedules = _schedule_orders(rng, pairs, desired)
    spec = {
        "native_case_id": stable_id(
            {
                "schema_version": "1.12-native",
                "size": size,
                "profile": profile,
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
    observations, count = _native_reference(spec)
    if count != desired:
        raise RuntimeError(f"native construction expected {desired} outcomes, got {count}")
    spec["abstract_schedule_observations"] = observations
    spec["abstract_outcome_count"] = count
    spec["abstract_confluent"] = count == 1
    return spec


def _normalize(value):
    return {
        "slots": dict(sorted(value["slots"].items())),
        "active": sorted(value["active"]),
    }


def _validate_runtime(spec, runtime):
    runtime_obs = {
        run["label"]: _normalize(run["observation"])
        for run in runtime["schedule_runs"]
    }
    expected = {
        label: _normalize(value)
        for label, value in spec["abstract_schedule_observations"].items()
    }
    distinct = len({json.dumps(value, sort_keys=True) for value in runtime_obs.values()})
    schedule_agreement = runtime_obs == expected
    confluence_agreement = runtime["confluent"] == spec["abstract_confluent"]
    outcome_count_agreement = distinct == spec["abstract_outcome_count"]
    if not (schedule_agreement and confluence_agreement and outcome_count_agreement):
        raise RuntimeError(
            f"V1.12 Cordis differential disagreement for {spec['native_case_id']}"
        )
    return {
        "oracle_agreement": True,
        "schedule_agreement": schedule_agreement,
        "confluence_agreement": confluence_agreement,
        "outcome_count_agreement": outcome_count_agreement,
    }


def _rename_for_render(spec, rng):
    all_names = (
        [provider["name"] for provider in spec["providers"]]
        + [spec["decoy"]]
        + [leaf["name"] for leaf in spec["leaves"]]
    )
    name_map = {name: _label(rng, "unit") for name in all_names}
    services = sorted({provider["service"] for provider in spec["providers"]})
    service_map = {service: _label(rng, "cap") for service in services}
    slots = sorted(spec["base_slots"])
    slot_map = {slot: _label(rng, "slot") for slot in slots}
    renamed = dict(spec)
    renamed["base_slots"] = {
        slot_map[key]: value for key, value in spec["base_slots"].items()
    }
    renamed["providers"] = [
        {
            "name": name_map[item["name"]],
            "service": service_map[item["service"]],
        }
        for item in spec["providers"]
    ]
    renamed["decoy"] = name_map[spec["decoy"]]
    renamed["leaves"] = [
        {
            "name": name_map[item["name"]],
            "service": service_map[item["service"]],
            "slot": slot_map[item["slot"]],
            "value": item["value"],
        }
        for item in spec["leaves"]
    ]
    rng.shuffle(renamed["leaves"])
    renamed["startup_order"] = [name_map[name] for name in spec["startup_order"]]
    renamed["query_provider"] = name_map[spec["query_provider"]]
    renamed["schedules"] = {
        label: [name_map[name] for name in order]
        for label, order in spec["schedules"].items()
    }
    renamed["target"] = {
        "slots": {slot_map[key]: value for key, value in spec["target"]["slots"].items()},
        "active": [],
    }
    return renamed


def _prompt(spec, task_type):
    schedules = "\n".join(
        f"  {label}: " + " -> ".join(order)
        for label, order in spec["schedules"].items()
    )
    code = _render_native_code(spec)
    prefix = f"""The following program is executed against Cordis {CORDIS_VERSION}. It uses real `ctx.provide`, `inject`, and `ctx.effect` APIs. Every effect records its activation-time previous numeric harness value in its disposer. `gate(name)` changes no state; it realizes the listed dependent-disposer completion order.

Controlled teardown schedules:
{schedules}

```js
{code}```

Maintenance action:
  dispose({spec['query_provider']})
"""
    if task_type == "cordis_challenge_schedule_checksum":
        return prefix + f"""
For schedule_1 only, after teardown finishes, compute
  (sum(v) + 257 * sum(v^2)) mod {CHECKSUM_MODULUS}
over all numeric harness slot values. Return only the integer checksum."""
    return prefix + """
Across all listed controlled schedules, how many distinct final observable configurations (all harness slot values and active plugin fibers) occur?
Return only the integer count."""


def _records(rng, spec, runtime):
    validation = _validate_runtime(spec, runtime)
    rendered = {
        "base": spec,
        "alpha_renamed_reordered": _rename_for_render(spec, rng),
    }
    checksum_gold = _slot_checksum(spec["abstract_schedule_observations"]["schedule_1"])
    tasks = (
        ("cordis_challenge_schedule_checksum", "schedule_checksum", checksum_gold),
        ("cordis_challenge_outcome_count", "outcome_count", spec["abstract_outcome_count"]),
    )
    records = []
    for task_type, family, gold in tasks:
        pair_id = stable_id(
            {
                "schema_version": "1.12-native-pair",
                "case": spec["native_case_id"],
                "task": task_type,
            }
        )
        for perturbation, view in rendered.items():
            md = metadata(
                V112_NATIVE_COMPONENT,
                family,
                spec["challenge_size"],
                "queried_dependent_leaves",
                spec["native_case_id"],
                pair_id,
                perturbation,
                realization="actual_cordis_runtime",
                oracle_type="differential_finite_plus_cordis_runtime",
                query_quantifier=(
                    "named_controlled_native_interleaving"
                    if task_type == "cordis_challenge_schedule_checksum"
                    else "listed_controlled_native_interleavings"
                ),
                native_execution=True,
                differential_oracle=True,
                cordis_version=runtime["cordis_version"],
                cordis_source_commit=CORDIS_SOURCE_COMMIT,
                node_version=runtime.get("node_version"),
                sampled_native_schedules=len(spec["schedules"]),
                queried_dependent_leaves=spec["challenge_size"],
                shared_slots=spec["challenge_size"] // 2,
                problem_size=len(spec["providers"]) + 1 + len(spec["leaves"]),
                **validation,
            )
            records.append(
                record(
                    task_type,
                    _prompt(view, task_type),
                    gold,
                    md,
                    pair_id,
                    perturbation,
                )
            )
    return records


def generate_v112_native_dataset(
    sizes=V112_NATIVE_SIZES,
    worlds_per_size=V112_NATIVE_WORLDS_PER_SIZE,
    seed=0,
    runner_path=None,
    node="node",
):
    rng = random.Random(seed + 12_100_701)
    specs = [
        _sample_native_spec(rng, int(size), profile)
        for size in sizes
        for profile in range(worlds_per_size)
    ]
    runtimes = execute_native_specs(specs, runner_path=runner_path, node=node)
    records = []
    for spec in specs:
        records.extend(_records(rng, spec, runtimes[spec["native_case_id"]]))
    return records
