"""V1.11 challenge track beside the frozen V1.10 benchmark.

V1.11 does not replace or mutate V1.10. It adds scalable formal and actual-
Cordis challenge families intended to retain headroom for stronger models.
Both families expose a size ladder and paired alpha-renamed/component-reordered
renderings to make shortcut sensitivity measurable.
"""

from __future__ import annotations

from math import factorial
import json
import random

from .core import Action, Component, Op, RuntimeState, World
from .generate import PRIMES
from .render import render_state, render_world
from .semantics import apply_ops
from .tasks import LETTERS, stable_id
from .v15 import _pair_effects, _random_keys, _random_names, _rules
from .v16 import _count_choices, _outcome_count_prompt
from .v18 import (
    CORDIS_SOURCE_COMMIT,
    CORDIS_VERSION,
    execute_native_specs,
    _render_native_code,
)


V111_FORMAL_COMPONENT = "formal_challenge"
V111_NATIVE_COMPONENT = "cordis_native_challenge"
V111_TASKS = (
    "formal_challenge_relevance_count",
    "formal_challenge_outcome_count",
    "cordis_challenge_schedule_invariance",
    "cordis_challenge_outcome_count",
)
V111_FORMAL_SIZES = (8, 16, 24, 32)  # noncommuting pair count
V111_NATIVE_SIZES = (4, 8, 12, 16)   # critical dependent-leaf count
V111_FORMAL_WORLDS_PER_SIZE = 8
V111_NATIVE_WORLDS_PER_SIZE = 3
V111_PERTURBATIONS = ("base", "alpha_renamed_reordered")

# 4 sizes x 8 worlds x 2 tasks x 2 renderings.
V111_FORMAL_RECORDS = (
    len(V111_FORMAL_SIZES) * V111_FORMAL_WORLDS_PER_SIZE * 2 * len(V111_PERTURBATIONS)
)
# 4 sizes x 3 worlds x 4 factorial cells x 2 tasks x 2 renderings.
V111_NATIVE_RECORDS = (
    len(V111_NATIVE_SIZES) * V111_NATIVE_WORLDS_PER_SIZE * 4 * 2 * len(V111_PERTURBATIONS)
)
V111_DEFAULT_N = V111_FORMAL_RECORDS + V111_NATIVE_RECORDS


def _challenge_names(rng, n):
    """Return stable unique names for the scalable formal ladder.

    The older generators intentionally use a 26-letter alphabet. V1.11's
    32-pair level needs more than 26 components, so extend that alphabet with
    deterministic suffixed names while retaining the old ordering for smaller
    worlds.
    """
    names = _random_names(rng, min(n, 26))
    suffix = 0
    while len(names) < n:
        candidate = f"C{suffix}"
        suffix += 1
        if candidate not in names:
            names.append(candidate)
    return names


def _count_answer(choices, gold):
    return next(letter for letter, text in choices.items() if int(text) == gold)


def _binary_choices(rng):
    return {"A": "YES", "B": "NO"} if rng.random() < 0.5 else {"A": "NO", "B": "YES"}


def _binary_answer(choices, value):
    desired = "YES" if value else "NO"
    return next(letter for letter, text in choices.items() if text == desired)


def _formal_pair_distinct(world, start, left, right):
    a = world.by_name[left]
    b = world.by_name[right]
    a_inv = a.inverse_effect(world.modulus)
    b_inv = b.inverse_effect(world.modulus)
    ab = apply_ops(apply_ops(start.values, a_inv, world.modulus), b_inv, world.modulus)
    ba = apply_ops(apply_ops(start.values, b_inv, world.modulus), a_inv, world.modulus)
    return ab != ba


def _build_formal_world(rng, n_pairs):
    """Build a factorized but shortcut-resistant large composition world.

    Each noncommuting pair has disjoint coordinate support from every other
    pair. Pairs attach to one of two dependency branches; disabling branch A
    therefore makes only a hidden subset lifecycle-relevant. The exact number
    of terminal observations is 2**r, where r is the number of pair gadgets on
    the queried branch. This product oracle is exact because cross-pair effects
    commute by disjoint support and each local pair is validated to expose two
    distinct teardown outcomes.
    """
    if n_pairs < 2:
        raise ValueError("formal challenge size must contain at least two pairs")
    width = 2 * n_pairs + 1
    n_components = 4 + 2 * n_pairs
    modulus = rng.choice(PRIMES)
    names = _challenge_names(rng, n_components)
    keys = dict(zip(names, _random_keys(rng, n_components)))
    root_a, relay_a, root_b, relay_b = names[:4]
    leaves = names[4:]
    pad = width - 1

    pair_names = []
    effects = {
        root_a: (Op("add", pad, value=rng.randrange(1, modulus)),),
        relay_a: (Op("add", pad, value=rng.randrange(1, modulus)),),
        root_b: (Op("add", pad, value=rng.randrange(1, modulus)),),
        relay_b: (Op("add", pad, value=rng.randrange(1, modulus)),),
    }
    families = []
    for index in range(n_pairs):
        left, right = leaves[2 * index : 2 * index + 2]
        left_effect, right_effect, family = _pair_effects(
            rng, modulus, width, (2 * index, 2 * index + 1)
        )
        pair_names.append((left, right))
        effects[left] = left_effect
        effects[right] = right_effect
        families.append(family)

    low = max(1, n_pairs // 3)
    high = max(low, (2 * n_pairs) // 3)
    relevant_count = rng.randint(low, high)
    relevant_indices = frozenset(rng.sample(range(n_pairs), relevant_count))
    terminal_a = keys[relay_a]
    terminal_b = keys[relay_b]

    components = {
        root_a: Component(root_a, frozenset(), frozenset({keys[root_a]}), effects[root_a]),
        relay_a: Component(relay_a, frozenset({keys[root_a]}), frozenset({terminal_a}), effects[relay_a]),
        root_b: Component(root_b, frozenset(), frozenset({keys[root_b]}), effects[root_b]),
        relay_b: Component(relay_b, frozenset({keys[root_b]}), frozenset({terminal_b}), effects[relay_b]),
    }
    for index, (left, right) in enumerate(pair_names):
        required = terminal_a if index in relevant_indices else terminal_b
        for name in (left, right):
            components[name] = Component(
                name,
                frozenset({required}),
                frozenset({keys[name]}),
                effects[name],
            )

    order = list(names)
    rng.shuffle(order)
    world = World(modulus, width, tuple(components[name] for name in order))

    for _ in range(500):
        values = tuple(rng.randrange(modulus) for _ in range(width))
        start = RuntimeState(values, frozenset(names), frozenset(names))
        if all(_formal_pair_distinct(world, start, *pair) for pair in pair_names):
            break
    else:
        raise RuntimeError("could not find nondegenerate V1.11 formal challenge state")

    world_id = stable_id({
        "schema_version": "1.11-formal-challenge",
        "n_pairs": n_pairs,
        "modulus": modulus,
        "components": [
            {
                "name": c.name,
                "requires": sorted(c.requires),
                "provides": sorted(c.provides),
                "effect": [op.render() for op in c.effect],
            }
            for c in world.components
        ],
        "values": list(start.values),
        "relevant": sorted(relevant_indices),
    })
    return {
        "world_id": world_id,
        "world": world,
        "start": start,
        "directive": Action("disable", root_a),
        "pair_names": tuple(pair_names),
        "families": tuple(families),
        "relevant_indices": relevant_indices,
        "relevant_count": relevant_count,
        "terminal_outcomes": 2 ** relevant_count,
        "schedule_count": factorial(2 * relevant_count),
        "n_pairs": n_pairs,
    }


def _alpha_rename_formal(world, start, directive, rng):
    names = [component.name for component in world.components]
    all_keys = sorted({key for component in world.components for key in component.requires | component.provides})
    name_values = [f"unit_{rng.randrange(10**8):08d}" for _ in names]
    while len(set(name_values)) != len(name_values):
        name_values = [f"unit_{rng.randrange(10**8):08d}" for _ in names]
    key_values = [f"cap_{rng.randrange(10**8):08d}" for _ in all_keys]
    while len(set(key_values)) != len(key_values):
        key_values = [f"cap_{rng.randrange(10**8):08d}" for _ in all_keys]
    name_map = dict(zip(names, name_values))
    key_map = dict(zip(all_keys, key_values))
    components = [
        Component(
            name_map[component.name],
            frozenset(key_map[key] for key in component.requires),
            frozenset(key_map[key] for key in component.provides),
            component.effect,
        )
        for component in world.components
    ]
    rng.shuffle(components)
    renamed = World(world.modulus, world.width, tuple(components))
    renamed_start = RuntimeState(
        start.values,
        frozenset(name_map[name] for name in start.enabled),
        frozenset(name_map[name] for name in start.active),
    )
    return renamed, renamed_start, Action(directive.kind, name_map[directive.component])


def _formal_metadata(case, task_family, pair_id, perturbation):
    r = case["relevant_count"]
    metadata = {
        "benchmark_suite": "1.11-challenge",
        "benchmark_component": V111_FORMAL_COMPONENT,
        "component": V111_FORMAL_COMPONENT,
        "benchmark_subcomponent": f"formal_challenge.{task_family}",
        "challenge_track": True,
        "challenge_family": "formal_product_composition",
        "challenge_size": case["n_pairs"],
        "challenge_size_unit": "noncommuting_pairs",
        "challenge_world_id": case["world_id"],
        "challenge_pair_id": pair_id,
        "perturbation_variant": perturbation,
        "shortcut_resistant": True,
        "alpha_renaming": perturbation != "base",
        "component_reordering": perturbation != "base",
        "problem_size": 4 + 2 * case["n_pairs"],
        "state_width": 2 * case["n_pairs"] + 1,
        "total_interfering_pairs": case["n_pairs"],
        "relevant_interfering_pairs": r,
        "irrelevant_interfering_pairs": case["n_pairs"] - r,
        "relevant_fraction": r / case["n_pairs"],
        "schedule_count": case["schedule_count"],
        "terminal_observations": case["terminal_outcomes"],
        "task_family": task_family,
        "capability": "global",
        "realization": "finite_modular_micro_system",
        "interaction_topology": "disjoint_noncommuting_pairs_two_dependency_branches",
        "oracle_type": "exact_product_decomposition",
        "oracle_factorization_validated": True,
        "query_quantifier": "all_legal_schedules",
    }
    metadata["semantic_dimensions"] = {
        key: metadata[key]
        for key in (
            "benchmark_component", "benchmark_subcomponent", "challenge_track",
            "challenge_family", "challenge_size", "challenge_size_unit",
            "perturbation_variant", "shortcut_resistant", "alpha_renaming",
            "component_reordering", "problem_size", "state_width",
            "total_interfering_pairs", "relevant_interfering_pairs",
            "irrelevant_interfering_pairs", "relevant_fraction", "schedule_count",
            "terminal_observations", "task_family", "capability", "realization",
            "interaction_topology", "oracle_type", "query_quantifier",
        )
    }
    return metadata


def _relevance_prompt(world, start, directive, choices):
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    return f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(start)}

The orchestrator executes:
  {directive.render()}

A noncommuting pair is lifecycle-relevant iff both members become unsupported as a consequence of this directive. How many noncommuting pairs are lifecycle-relevant?
{options}

Answer with only the option letter."""


def _formal_records(rng, case):
    records = []
    renderings = {
        "base": (case["world"], case["start"], case["directive"]),
        "alpha_renamed_reordered": _alpha_rename_formal(
            case["world"], case["start"], case["directive"], rng
        ),
    }
    task_specs = (
        ("formal_challenge_relevance_count", "relevance_count", case["relevant_count"]),
        ("formal_challenge_outcome_count", "schedule_outcome_enumeration", case["terminal_outcomes"]),
    )
    for task_type, family, gold in task_specs:
        choices = _count_choices(rng, gold)
        answer = _count_answer(choices, gold)
        pair_id = stable_id({
            "schema_version": "1.11-challenge-pair",
            "world_id": case["world_id"],
            "task_type": task_type,
            "choices": choices,
        })
        for perturbation, (world, start, directive) in renderings.items():
            prompt = (
                _relevance_prompt(world, start, directive, choices)
                if task_type == "formal_challenge_relevance_count"
                else _outcome_count_prompt(world, start, directive, choices)
            )
            records.append({
                "id": stable_id({
                    "schema_version": "1.11-challenge",
                    "pair_id": pair_id,
                    "perturbation": perturbation,
                }),
                "schema_version": "1.11",
                "task_type": task_type,
                "prompt": prompt,
                "choices": choices,
                "answer": answer,
                "metadata": _formal_metadata(case, family, pair_id, perturbation),
            })
    return records


def generate_v111_formal_dataset(sizes=V111_FORMAL_SIZES, worlds_per_size=V111_FORMAL_WORLDS_PER_SIZE, seed=0):
    rng = random.Random(seed + 11_100_101)
    records = []
    for size in sizes:
        for _ in range(worlds_per_size):
            records.extend(_formal_records(rng, _build_formal_world(rng, int(size))))
    return records


def _native_schedule_orders(rng, names, limit=8):
    names = list(names)
    candidates = [
        names,
        list(reversed(names)),
        names[1:] + names[:1],
        names[::2] + names[1::2],
        names[1::2] + names[::2],
    ]
    zigzag = []
    lo, hi = 0, len(names) - 1
    while lo <= hi:
        zigzag.append(names[lo])
        lo += 1
        if lo <= hi:
            zigzag.append(names[hi])
            hi -= 1
    candidates.append(zigzag)
    while len(candidates) < limit + 6:
        item = names[:]
        rng.shuffle(item)
        candidates.append(item)
    unique = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
        if len(unique) == limit:
            break
    return {f"schedule_{index + 1}": order for index, order in enumerate(unique)}


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


def _random_label(rng, prefix):
    return f"{prefix}_{rng.randrange(10**9):09d}"


def _build_native_spec(rng, size, noncommuting, query_relevant):
    critical_provider = _random_label(rng, "provider")
    control_provider = _random_label(rng, "provider")
    critical_service = _random_label(rng, "service")
    control_service = _random_label(rng, "service")
    decoy = _random_label(rng, "plugin")
    critical_names = [_random_label(rng, "plugin") for _ in range(size)]
    control_count = max(4, size // 2)
    control_names = [_random_label(rng, "plugin") for _ in range(control_count)]

    leaves = []
    base_slots = {}
    for index, name in enumerate(critical_names):
        slot_index = index if not noncommuting else index // 2
        slot = f"critical.slot.{slot_index}"
        base_slots.setdefault(slot, f"base-c-{slot_index}")
        leaves.append({
            "name": name,
            "service": critical_service,
            "slot": slot,
            "value": f"critical-{index}-{rng.randrange(10000)}",
        })
    for index, name in enumerate(control_names):
        slot = f"control.slot.{index}"
        base_slots[slot] = f"base-q-{index}"
        leaves.append({
            "name": name,
            "service": control_service,
            "slot": slot,
            "value": f"control-{index}-{rng.randrange(10000)}",
        })

    startup_order = critical_names[:] + control_names[:]
    rng.shuffle(startup_order)
    affected = critical_names if query_relevant else control_names
    schedules = _native_schedule_orders(rng, affected, limit=8)
    query_provider = critical_provider if query_relevant else control_provider
    spec = {
        "native_case_id": stable_id({
            "schema_version": "1.11-native-challenge",
            "size": size,
            "noncommuting": noncommuting,
            "query_relevant": query_relevant,
            "providers": (critical_provider, control_provider),
            "startup": startup_order,
            "leaves": leaves,
            "schedules": schedules,
        }),
        "base_slots": base_slots,
        "providers": [
            {"name": critical_provider, "service": critical_service},
            {"name": control_provider, "service": control_service},
        ],
        "decoy": decoy,
        "leaves": leaves,
        "startup_order": startup_order,
        "query_provider": query_provider,
        "schedules": schedules,
        "target": {"slots": base_slots, "active": []},
        "options": {},
        "challenge_size": size,
        "noncommuting": noncommuting,
        "query_relevant": query_relevant,
    }
    expected, count = _native_reference(spec)
    if noncommuting and query_relevant and count <= 1:
        raise RuntimeError("noncommuting relevant native challenge did not expose multiple outcomes")
    spec["abstract_schedule_observations"] = expected
    spec["abstract_outcome_count"] = count
    spec["abstract_confluent"] = count == 1
    return spec


def _normalize_observation(value):
    return {
        "slots": dict(sorted(value["slots"].items())),
        "active": sorted(value["active"]),
    }


def _validate_native_runtime(spec, runtime):
    runtime_obs = {
        run["label"]: _normalize_observation(run["observation"])
        for run in runtime["schedule_runs"]
    }
    expected = {
        label: _normalize_observation(value)
        for label, value in spec["abstract_schedule_observations"].items()
    }
    schedule_agreement = runtime_obs == expected
    distinct = len({json.dumps(value, sort_keys=True) for value in runtime_obs.values()})
    confluence_agreement = runtime["confluent"] == spec["abstract_confluent"]
    outcome_count_agreement = distinct == spec["abstract_outcome_count"]
    if not (schedule_agreement and confluence_agreement and outcome_count_agreement):
        raise RuntimeError(
            f"V1.11 Cordis challenge differential disagreement for {spec['native_case_id']}"
        )
    return {
        "oracle_agreement": True,
        "schedule_agreement": schedule_agreement,
        "confluence_agreement": confluence_agreement,
        "outcome_count_agreement": outcome_count_agreement,
        "runtime_outcome_count": distinct,
    }


def _rename_native_for_render(spec, rng):
    all_names = [provider["name"] for provider in spec["providers"]] + [spec["decoy"]] + [leaf["name"] for leaf in spec["leaves"]]
    name_map = {name: _random_label(rng, "unit") for name in all_names}
    services = sorted({provider["service"] for provider in spec["providers"]})
    service_map = {service: _random_label(rng, "cap") for service in services}
    slots = sorted(spec["base_slots"])
    slot_map = {slot: _random_label(rng, "slot") for slot in slots}
    renamed = dict(spec)
    renamed["base_slots"] = {slot_map[key]: value for key, value in spec["base_slots"].items()}
    renamed["providers"] = [
        {"name": name_map[item["name"]], "service": service_map[item["service"]]}
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
    return renamed


def _native_prompt(spec, task_type, choices):
    schedules = "\n".join(
        f"  {label}: " + " -> ".join(order)
        for label, order in spec["schedules"].items()
    )
    code = _render_native_code(spec)
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    question = (
        "Across every controlled native Cordis teardown schedule above, is the final observable configuration (all harness slot values and active plugin fibers) identical?"
        if task_type == "cordis_challenge_schedule_invariance"
        else "Across the controlled native Cordis teardown schedules above, how many distinct final observable configurations (all harness slot values and active plugin fibers) occur?"
    )
    return f"""The following program is executed against Cordis {CORDIS_VERSION}. It uses real `ctx.provide`, `inject`, and `ctx.effect` APIs. Every effect records its activation-time previous value in its returned disposer. `gate(name)` changes no state; it only realizes the listed dependent-disposer completion order.

Controlled teardown schedules:
{schedules}

```js
{code}```

Maintenance action:
  dispose({spec['query_provider']})

{question}
{options}

Answer with only the option letter."""


def _native_metadata(spec, validation, task_family, pair_id, perturbation, runtime):
    present = bool(spec["noncommuting"])
    relevant = bool(spec["query_relevant"])
    metadata = {
        "benchmark_suite": "1.11-challenge",
        "benchmark_component": V111_NATIVE_COMPONENT,
        "component": V111_NATIVE_COMPONENT,
        "benchmark_subcomponent": f"cordis_native_challenge.{task_family}",
        "challenge_track": True,
        "challenge_family": "actual_cordis_multi_leaf_teardown",
        "challenge_size": spec["challenge_size"],
        "challenge_size_unit": "critical_dependent_leaves",
        "challenge_world_id": spec["native_case_id"],
        "challenge_pair_id": pair_id,
        "perturbation_variant": perturbation,
        "shortcut_resistant": True,
        "alpha_renaming": perturbation != "base",
        "component_reordering": perturbation != "base",
        "native_execution": True,
        "differential_oracle": True,
        "cordis_version": runtime["cordis_version"],
        "cordis_source_commit": CORDIS_SOURCE_COMMIT,
        "node_version": runtime.get("node_version"),
        **validation,
        "sampled_native_schedules": len(spec["schedules"]),
        "interference_present": present,
        "query_interaction_relevant": relevant,
        "interference_relevant": present and relevant,
        "factorial_cell": (
            ("noncommuting" if present else "commuting")
            + "_"
            + ("relevant" if relevant else "irrelevant")
        ),
        "terminal_observations": spec["abstract_outcome_count"],
        "task_family": task_family,
        "capability": "global",
        "realization": "actual_cordis_runtime",
        "oracle_type": "differential_finite_plus_cordis_runtime",
        "query_quantifier": "listed_controlled_native_interleavings",
        "problem_size": len(spec["providers"]) + 1 + len(spec["leaves"]),
    }
    metadata["semantic_dimensions"] = {
        key: metadata[key]
        for key in (
            "benchmark_component", "benchmark_subcomponent", "challenge_track",
            "challenge_family", "challenge_size", "challenge_size_unit",
            "perturbation_variant", "shortcut_resistant", "alpha_renaming",
            "component_reordering", "native_execution", "differential_oracle",
            "cordis_version", "sampled_native_schedules", "interference_present",
            "query_interaction_relevant", "interference_relevant", "factorial_cell",
            "terminal_observations", "task_family", "capability", "realization",
            "oracle_type", "query_quantifier", "problem_size",
        )
    }
    return metadata


def _native_records(rng, spec, runtime):
    validation = _validate_native_runtime(spec, runtime)
    renderings = {
        "base": spec,
        "alpha_renamed_reordered": _rename_native_for_render(spec, rng),
    }
    tasks = (
        (
            "cordis_challenge_schedule_invariance",
            "schedule_invariance",
            _binary_choices(rng),
            spec["abstract_confluent"],
        ),
        (
            "cordis_challenge_outcome_count",
            "sampled_schedule_outcome_enumeration",
            _count_choices(rng, spec["abstract_outcome_count"]),
            spec["abstract_outcome_count"],
        ),
    )
    records = []
    for task_type, family, choices, gold_value in tasks:
        answer = (
            _binary_answer(choices, gold_value)
            if task_type == "cordis_challenge_schedule_invariance"
            else _count_answer(choices, gold_value)
        )
        pair_id = stable_id({
            "schema_version": "1.11-native-challenge-pair",
            "case": spec["native_case_id"],
            "task": task_type,
            "choices": choices,
        })
        for perturbation, rendered_spec in renderings.items():
            records.append({
                "id": stable_id({
                    "schema_version": "1.11-native-challenge",
                    "pair_id": pair_id,
                    "perturbation": perturbation,
                }),
                "schema_version": "1.11",
                "task_type": task_type,
                "prompt": _native_prompt(rendered_spec, task_type, choices),
                "choices": choices,
                "answer": answer,
                "metadata": _native_metadata(
                    spec, validation, family, pair_id, perturbation, runtime
                ),
            })
    return records


def generate_v111_native_dataset(
    sizes=V111_NATIVE_SIZES,
    worlds_per_size=V111_NATIVE_WORLDS_PER_SIZE,
    seed=0,
    runner_path=None,
    node="node",
):
    rng = random.Random(seed + 11_100_701)
    specs = []
    for size in sizes:
        for _ in range(worlds_per_size):
            for noncommuting in (False, True):
                for query_relevant in (False, True):
                    specs.append(
                        _build_native_spec(
                            rng, int(size), noncommuting, query_relevant
                        )
                    )
    runtimes = execute_native_specs(specs, runner_path=runner_path, node=node)
    records = []
    for spec in specs:
        records.extend(_native_records(rng, spec, runtimes[spec["native_case_id"]]))
    return records


def generate_v111_dataset(
    n=None,
    seed=0,
    task_types=V111_TASKS,
    formal_sizes=V111_FORMAL_SIZES,
    native_sizes=V111_NATIVE_SIZES,
    formal_worlds=V111_FORMAL_WORLDS_PER_SIZE,
    native_worlds=V111_NATIVE_WORLDS_PER_SIZE,
    runner_path=None,
    node="node",
):
    if tuple(task_types) != V111_TASKS:
        raise ValueError("V1.11 challenge is a fixed task family; custom --tasks is not supported")
    formal_sizes = tuple(int(value) for value in formal_sizes)
    native_sizes = tuple(int(value) for value in native_sizes)
    if not formal_sizes or not native_sizes:
        raise ValueError("V1.11 requires at least one formal and one native challenge size")
    formal = generate_v111_formal_dataset(
        sizes=formal_sizes,
        worlds_per_size=formal_worlds,
        seed=seed,
    )
    native = generate_v111_native_dataset(
        sizes=native_sizes,
        worlds_per_size=native_worlds,
        seed=seed,
        runner_path=runner_path,
        node=node,
    )
    records = formal + native
    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("V1.11 challenge generated duplicate item IDs")
    if n is not None and n != len(records):
        raise ValueError(f"requested n={n}, but selected V1.11 ladder emits {len(records)} records")
    return records
