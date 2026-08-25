"""Actual-Cordis V2 core/challenge from one size-scaled native generator."""

from __future__ import annotations

import json
import random

from .tasks import stable_id
from .v18 import (
    CORDIS_PACKAGE,
    CORDIS_SOURCE_COMMIT,
    CORDIS_VERSION,
    _render_native_code,
    execute_native_specs,
)
from .v112_native import _linear_extension
from .v112_native import _validate_runtime as _validate_runtime
from .v112_native_challenge import _native_reference
from .v2_common import canonical_list, canonical_scalar_sequence, make_metadata, make_record


V2_NATIVE_CORE_SIZES = (2, 4)
V2_NATIVE_CORE_WORLDS_PER_SIZE = 8
V2_NATIVE_CHALLENGE_SIZES = (8, 16, 24, 32)
V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE = 4

V2_NATIVE_PRIMARY_TASKS = (
    "cordis_order_sensitive_slots",
    "cordis_schedule_prediction",
    "cordis_guaranteed_conditions",
    "cordis_reachable_conditions",
    "cordis_reconfiguration_set",
)
V2_NATIVE_DIAGNOSTIC_TASKS = ("cordis_outcome_count_diagnostic",)
V2_NATIVE_CORE_TASKS = V2_NATIVE_PRIMARY_TASKS
V2_NATIVE_CHALLENGE_TASKS = V2_NATIVE_PRIMARY_TASKS + V2_NATIVE_DIAGNOSTIC_TASKS
V2_NATIVE_TASKS = V2_NATIVE_CHALLENGE_TASKS

V2_NATIVE_CORE_RECORDS = (
    len(V2_NATIVE_CORE_SIZES)
    * V2_NATIVE_CORE_WORLDS_PER_SIZE
    * len(V2_NATIVE_CORE_TASKS)
)
V2_NATIVE_CHALLENGE_RECORDS = (
    len(V2_NATIVE_CHALLENGE_SIZES)
    * V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE
    * len(V2_NATIVE_CHALLENGE_TASKS)
)


def _label(rng, prefix):
    return f"{prefix}_{rng.randrange(10**9):09d}"


def _schedule_limit(size):
    if size <= 4:
        return size
    return min(32, max(12, size))


def _balanced_flags(n, rng):
    if n <= 0:
        return []
    if n == 1:
        return [bool(rng.randrange(2))]
    n_true = n // 2
    flags = [True] * n_true + [False] * (n - n_true)
    rng.shuffle(flags)
    return flags


def _normalize(observation):
    return {
        "slots": dict(sorted(observation["slots"].items())),
        "active": sorted(observation["active"]),
    }


def _runtime_observations(runtime):
    return {
        run["label"]: _normalize(run["observation"])
        for run in runtime["schedule_runs"]
    }


def _sample_patterns(rng, n_interfering, desired):
    if n_interfering == 0:
        return [tuple()]
    patterns = [
        tuple(0 for _ in range(n_interfering)),
        tuple(1 for _ in range(n_interfering)),
    ]
    seen = set(patterns)
    while len(patterns) < desired:
        bits = tuple(rng.randrange(2) for _ in range(n_interfering))
        if bits not in seen:
            seen.add(bits)
            patterns.append(bits)
    return patterns


def _schedule_orders(rng, pairs, interfering_indices, desired_outcomes, limit):
    interfering_indices = tuple(sorted(interfering_indices))
    patterns = _sample_patterns(rng, len(interfering_indices), desired_outcomes)
    schedules = {}
    seen_orders = set()
    for index in range(limit):
        full_bits = [rng.randrange(2) for _ in pairs]
        if index == 0:
            full_bits = [0 for _ in pairs]
        elif index == 1:
            full_bits = [1 for _ in pairs]
        pattern = patterns[index % len(patterns)]
        for bit_index, pair_index in enumerate(interfering_indices):
            full_bits[pair_index] = pattern[bit_index]
        ordered_pairs = [
            pair if bit else tuple(reversed(pair))
            for pair, bit in zip(pairs, full_bits)
        ]
        for _ in range(20_000):
            order = _linear_extension(rng, ordered_pairs)
            key = tuple(order)
            if key not in seen_orders:
                seen_orders.add(key)
                schedules[f"schedule_{index + 1}"] = order
                break
        else:
            raise RuntimeError("could not sample enough distinct V2 native schedules")
    return schedules


def _sample_native_spec(rng, size):
    if size < 2 or size % 2:
        raise ValueError("native V2 semantic size must be an even integer >= 2")
    n_pairs = size // 2
    if n_pairs == 1:
        n_interfering = rng.randrange(2)
    else:
        n_interfering = rng.randrange(1, n_pairs)
    interfering_indices = set(rng.sample(range(n_pairs), n_interfering))

    query_provider = _label(rng, "provider")
    other_provider = _label(rng, "provider")
    query_service = _label(rng, "service")
    other_service = _label(rng, "service")
    decoy = _label(rng, "plugin")
    query_names = [_label(rng, "plugin") for _ in range(size)]
    other_names = [_label(rng, "plugin") for _ in range(size)]

    n_values = 4 * n_pairs + 2 * size + 8
    unique_values = iter(rng.sample(range(10_000, 1_000_000), n_values))
    leaves = []
    base_slots = {}
    pairs = []
    pair_units = []

    for pair_index in range(n_pairs):
        pair = tuple(query_names[2 * pair_index : 2 * pair_index + 2])
        pairs.append(pair)
        if pair_index in interfering_indices:
            slot = _label(rng, "slot")
            base_slots[slot] = next(unique_values)
            pair_slots = [slot, slot]
        else:
            pair_slots = [_label(rng, "slot"), _label(rng, "slot")]
            for slot in pair_slots:
                base_slots[slot] = next(unique_values)
        for name, slot in zip(pair, pair_slots):
            leaves.append(
                {
                    "name": name,
                    "service": query_service,
                    "slot": slot,
                    "value": next(unique_values),
                }
            )
        pair_units.append(
            {
                "names": list(pair),
                "slots": pair_slots,
                "representative_slot": pair_slots[0],
                "interfering": pair_index in interfering_indices,
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
    limit = _schedule_limit(size)
    max_outcomes = min(limit, 2 ** n_interfering) if n_interfering else 1
    desired_outcomes = 1 if n_interfering == 0 else rng.randint(2, min(6, max_outcomes))
    schedules = _schedule_orders(
        rng,
        pairs,
        interfering_indices,
        desired_outcomes,
        limit,
    )

    spec = {
        "native_case_id": stable_id(
            {
                "schema_version": "2.0-native-rc2",
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
        "v2_semantic_size": size,
        "v2_pairs": pair_units,
    }
    observations, count = _native_reference(spec)
    if count != desired_outcomes:
        raise RuntimeError(
            f"V2 native construction expected {desired_outcomes} outcomes, got {count}"
        )
    spec["abstract_schedule_observations"] = observations
    spec["abstract_outcome_count"] = count
    spec["abstract_confluent"] = count == 1
    return spec


def _initial_native_state(spec):
    slots = dict(spec["base_slots"])
    witnesses = {}
    by_name = {leaf["name"]: leaf for leaf in spec["leaves"]}
    for name in spec["startup_order"]:
        leaf = by_name[name]
        witnesses[name] = slots[leaf["slot"]]
        slots[leaf["slot"]] = leaf["value"]
    active = {
        spec["decoy"],
        *(provider["name"] for provider in spec["providers"]),
        *(leaf["name"] for leaf in spec["leaves"]),
    }
    return slots, witnesses, active


def _simulate_native_plan(spec, plan, schedule_order):
    slots, witnesses, active = _initial_native_state(spec)
    by_name = {leaf["name"]: leaf for leaf in spec["leaves"]}
    provider_service = {
        provider["name"]: provider["service"] for provider in spec["providers"]
    }

    def dispose_leaf(name):
        if name not in active:
            return
        leaf = by_name[name]
        slots[leaf["slot"]] = witnesses[name]
        active.remove(name)

    for command in plan:
        if command["kind"] != "dispose":
            raise ValueError("V2 native plans contain only dispose operations")
        name = command["name"]
        if name not in active:
            continue
        if name in by_name:
            dispose_leaf(name)
            continue
        if name in provider_service:
            service = provider_service[name]
            for leaf_name in schedule_order:
                if leaf_name in active and by_name[leaf_name]["service"] == service:
                    dispose_leaf(leaf_name)
            active.remove(name)
            continue
        active.remove(name)
    return _normalize({"slots": slots, "active": active})


def _query_slots(spec):
    query_service = next(
        provider["service"]
        for provider in spec["providers"]
        if provider["name"] == spec["query_provider"]
    )
    return sorted(
        {
            leaf["slot"]
            for leaf in spec["leaves"]
            if leaf["service"] == query_service
        }
    )


def _representative_slots(spec):
    return [unit["representative_slot"] for unit in spec["v2_pairs"]]


def _minimal_predisposals(spec):
    positions = {name: index for index, name in enumerate(spec["startup_order"])}
    selected = []
    for unit in spec["v2_pairs"]:
        if unit["interfering"]:
            selected.append(max(unit["names"], key=positions.__getitem__))
    return sorted(selected)


def _augment_reconfiguration(spec):
    predispose = _minimal_predisposals(spec)
    plan = [
        *({"kind": "dispose", "name": name} for name in predispose),
        {"kind": "dispose", "name": spec["query_provider"]},
    ]
    targets = [
        _simulate_native_plan(spec, plan, order)
        for order in spec["schedules"].values()
    ]
    if not targets or any(target != targets[0] for target in targets[1:]):
        raise RuntimeError("V2 native reconfiguration plan is not schedule robust")

    query_slots = _query_slots(spec)
    baseline = {slot: spec["base_slots"][slot] for slot in query_slots}
    if any(targets[0]["slots"][slot] != value for slot, value in baseline.items()):
        raise RuntimeError("V2 native reconfiguration target did not restore query baseline")

    for removed in predispose:
        reduced = [
            *(
                {"kind": "dispose", "name": name}
                for name in predispose
                if name != removed
            ),
            {"kind": "dispose", "name": spec["query_provider"]},
        ]
        if all(
            all(
                _simulate_native_plan(spec, reduced, order)["slots"][slot] == value
                for slot, value in baseline.items()
            )
            for order in spec["schedules"].values()
        ):
            raise RuntimeError(f"V2 native predisposal {removed} was not necessary")

    spec = dict(spec)
    spec["target"] = targets[0]
    spec["options"] = {"GOLD": plan}
    spec["v2_predispose"] = predispose
    return spec


def _validate_spec_runtime(spec, runtime):
    validation = _validate_runtime(spec, runtime)
    if runtime.get("option_success", {}).get("GOLD") is not True:
        raise RuntimeError(
            f"V2 Cordis reconfiguration plan failed runtime validation for {spec['native_case_id']}"
        )
    return {**validation, "reconfiguration_plan_agreement": True}


def _native_prefix(spec):
    schedules = "\n".join(
        f"  {label}: " + " -> ".join(order)
        for label, order in spec["schedules"].items()
    )
    code = _render_native_code(spec)
    return f"""The following program is executed against Cordis {CORDIS_VERSION}. It uses real
`ctx.provide`, `inject`, and `ctx.effect` APIs. Each effect records the
application value present when it activates and restores that witnessed value
when its disposer completes.

The benchmark controls dependent-disposer completion order without changing
application state:
{schedules}

```js
{code}```

Maintenance action:
  dispose({spec['query_provider']})
"""


def _metadata(spec, runtime, validation, track, family, capability, answer_type, prompt_contract, **extra):
    return make_metadata(
        track=track,
        realization="cordis_native",
        family=family,
        capability=capability,
        answer_type=answer_type,
        semantic_size=spec["v2_semantic_size"],
        semantic_size_unit="queried_dependent_leaves",
        latent_group_id=spec["native_case_id"],
        prompt_contract=prompt_contract,
        native_execution=True,
        differential_oracle=True,
        cordis_package=CORDIS_PACKAGE,
        cordis_version=runtime["cordis_version"],
        cordis_source_commit=CORDIS_SOURCE_COMMIT,
        node_version=runtime.get("node_version"),
        problem_size=len(spec["providers"]) + 1 + len(spec["leaves"]),
        state_width=len(spec["base_slots"]),
        sampled_schedules=len(spec["schedules"]),
        **validation,
        **extra,
    )


def _surface_matched_absent(values, pool, rng):
    candidates = sorted(set(pool) - set(values))
    if candidates:
        return rng.choice(candidates)
    candidate = rng.randrange(10_000, 1_000_000)
    while candidate in values:
        candidate = rng.randrange(10_000, 1_000_000)
    return candidate


def _order_sensitive_record(spec, runtime, validation, track):
    observations = list(_runtime_observations(runtime).values())
    slots = _query_slots(spec)
    sensitive = [
        slot
        for slot in slots
        if len({observation["slots"][slot] for observation in observations}) > 1
    ]
    prompt = _native_prefix(spec) + """
Which queried application `harness` slots, if any, can finish with different
values across the listed completion orders?
Return a sorted JSON array of slot names and nothing else."""
    return make_record(
        task_type="cordis_order_sensitive_slots",
        prompt=prompt,
        answer=canonical_list(sensitive, sort=True),
        answer_type="string_set",
        metadata=_metadata(
            spec,
            runtime,
            validation,
            track,
            "order_sensitive_slots",
            "localization",
            "string_set",
            "cordis.order_sensitive_slots.v2",
            query_quantifier="compare_all_listed_controlled_interleavings",
            oracle_type="differential_runtime_localization",
            operational_relevance="localize_order_sensitive_application_state",
        ),
        identity={"case": spec["native_case_id"], "task": "order_sensitive_slots"},
    )


def _prediction_record(spec, runtime, validation, track):
    observations = _runtime_observations(runtime)
    label = next(iter(spec["schedules"]))
    slots = _representative_slots(spec)
    requested = "\n".join(f"  {index + 1}. harness[{slot!r}]" for index, slot in enumerate(slots))
    values = [observations[label]["slots"][slot] for slot in slots]
    prompt = _native_prefix(spec) + f"""
For controlled completion order `{label}`, report the final values of these
application observables, in this exact order:
{requested}

Return only a JSON array of values."""
    return make_record(
        task_type="cordis_schedule_prediction",
        prompt=prompt,
        answer=canonical_scalar_sequence(values),
        answer_type="scalar_sequence",
        metadata=_metadata(
            spec,
            runtime,
            validation,
            track,
            "schedule_prediction",
            "prediction",
            "scalar_sequence",
            "cordis.schedule_prediction.v2",
            queried_observables=len(slots),
            query_quantifier="named_controlled_native_interleaving",
            oracle_type="actual_cordis_observation",
            operational_relevance="predict_application_observables",
        ),
        identity={"case": spec["native_case_id"], "task": "prediction", "schedule": label},
    )


def _condition_record(spec, runtime, validation, track, rng, *, reachable):
    observations = _runtime_observations(runtime)
    slots = _representative_slots(spec)
    pool = {
        observation["slots"][slot]
        for observation in observations.values()
        for slot in _query_slots(spec)
    } | {spec["base_slots"][slot] for slot in _query_slots(spec)}
    n_conditions = max(4, 2 * len(slots))
    flags = _balanced_flags(n_conditions, rng)
    conditions = []
    true_labels = []
    for index, truth in enumerate(flags, start=1):
        slot = slots[(index - 1) % len(slots)]
        values = {observation["slots"][slot] for observation in observations.values()}
        if truth:
            value = rng.choice(sorted(values)) if reachable else _surface_matched_absent(values, pool, rng)
            true_labels.append(f"c{index}")
        else:
            value = _surface_matched_absent(values, pool, rng) if reachable else rng.choice(sorted(values))
        operator = "==" if reachable else "!="
        conditions.append((f"c{index}", slot, operator, value))

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

    prompt = _native_prefix(spec) + f"""
Named application conditions:
{lines}

{question}
Return a sorted JSON array of condition labels and nothing else."""
    return make_record(
        task_type=f"cordis_{family}",
        prompt=prompt,
        answer=canonical_list(true_labels, sort=True),
        answer_type="string_set",
        metadata=_metadata(
            spec,
            runtime,
            validation,
            track,
            family,
            capability,
            "string_set",
            f"cordis.{family}.v2",
            queried_conditions=len(conditions),
            query_quantifier=quantifier,
            oracle_type="differential_runtime_condition_set",
            operational_relevance=relevance,
        ),
        identity={"case": spec["native_case_id"], "task": family, "conditions": conditions},
    )


def _reconfiguration_record(spec, runtime, validation, track):
    baseline = {slot: spec["base_slots"][slot] for slot in _query_slots(spec)}
    baseline_text = "\n".join(
        f"  - harness[{slot!r}] == {value}" for slot, value in sorted(baseline.items())
    )
    prompt = _native_prefix(spec) + f"""
Before disposing provider `{spec['query_provider']}`, the orchestrator may
explicitly dispose some of its dependent plugins and let each disposal settle.
The target after provider disposal is:
{baseline_text}

Return the MINIMUM set of dependent plugin names that must be explicitly
disposed beforehand so this target is reached under EVERY listed controlled
completion order. Do not include the provider or unrelated plugins.
Return a sorted JSON array and nothing else."""

    oracle_spec = dict(spec)
    oracle_spec["options"] = {}
    oracle_spec.pop("v2_predispose", None)
    return make_record(
        task_type="cordis_reconfiguration_set",
        prompt=prompt,
        answer=canonical_list(spec["v2_predispose"], sort=True),
        answer_type="string_set",
        metadata=_metadata(
            spec,
            runtime,
            validation,
            track,
            "reconfiguration_set",
            "act",
            "string_set",
            "cordis.reconfiguration_set.v2",
            query_quantifier="robust_across_all_listed_controlled_interleavings",
            oracle_type="runtime_executable_minimal_plan",
            operational_relevance="use_cordis_to_reach_application_target",
            protected_unrelated_state=True,
        ),
        identity={"case": spec["native_case_id"], "task": "reconfiguration_set"},
        oracle={
            "kind": "cordis_reconfiguration",
            "spec": oracle_spec,
            "minimum_predisposals": len(spec["v2_predispose"]),
            "query_slots": sorted(baseline),
        },
    )


def _count_record(spec, runtime, validation):
    observations = _runtime_observations(runtime)
    distinct = len({json.dumps(value, sort_keys=True) for value in observations.values()})
    prompt = _native_prefix(spec) + """
Diagnostic only: across the listed controlled completion orders, how many
distinct final application-visible configurations (all harness slots and
active plugin fibers) occur?
Return only the integer count."""
    return make_record(
        task_type="cordis_outcome_count_diagnostic",
        prompt=prompt,
        answer=distinct,
        answer_type="integer",
        metadata=_metadata(
            spec,
            runtime,
            validation,
            "challenge",
            "outcome_count",
            "global_cardinality",
            "integer",
            "cordis.outcome_count.v2",
            paper_role="diagnostic",
            query_quantifier="all_listed_controlled_native_interleavings",
            oracle_type="differential_runtime_cardinality",
            operational_relevance="diagnostic_not_primary",
        ),
        identity={"case": spec["native_case_id"], "task": "outcome_count"},
    )


def _records_for_spec(spec, runtime, validation, track, rng):
    records = [
        _order_sensitive_record(spec, runtime, validation, track),
        _prediction_record(spec, runtime, validation, track),
        _condition_record(spec, runtime, validation, track, rng, reachable=False),
        _condition_record(spec, runtime, validation, track, rng, reachable=True),
        _reconfiguration_record(spec, runtime, validation, track),
    ]
    if track == "challenge":
        records.append(_count_record(spec, runtime, validation))
    return records


def _sample_specs_for_size(rng, size, worlds_per_size, require_count_variation=False):
    specs = []
    counts = set()
    attempts = 0
    while len(specs) < worlds_per_size:
        attempts += 1
        if attempts > max(100, worlds_per_size * 50):
            raise RuntimeError(f"could not sample varied V2 native worlds for size={size}")
        spec = _augment_reconfiguration(_sample_native_spec(rng, int(size)))
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
    specs = [
        spec
        for size in sizes
        for spec in _sample_specs_for_size(
            rng,
            int(size),
            worlds_per_size,
            require_count_variation=(track == "challenge"),
        )
    ]
    runtimes = execute_native_specs(specs, runner_path=runner_path, node=node)
    records = []
    for spec in specs:
        runtime = runtimes[spec["native_case_id"]]
        validation = _validate_spec_runtime(spec, runtime)
        records.extend(_records_for_spec(spec, runtime, validation, track, rng))
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
