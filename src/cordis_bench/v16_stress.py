"""Optional boundary-search extension of the frozen V1.6 core.

This is deliberately *not* V1.7. The released V1.6 seed-0 dataset and its
none/minimal-vs-low result remain frozen. This module keeps the same V1.6
semantics and tasks but permits a modest 14/16-component stress search to find
where correctness eventually fails.

Use the 14-component profile first. The 16-component profile exists only as a
last small extension if 14 components still saturate. These strata are for
capability-boundary measurement, not the paper's main causal manipulation.
"""

from math import factorial
import random

from .core import Action, Component, Op, RuntimeState, World
from .generate import PRIMES
from .semantics import count_quiescent_paths, explore, observations, orchestrate, quiescent
from .tasks import serialize_state, serialize_world, stable_id
from .v15 import _pair_effects, _random_keys, _random_names
from .v16 import V16_TASKS, composition_curve_records


V16_STRESS_COMPONENT_COUNTS = (14, 16)
STRESS_MAX_STATES = 200_000


def stress_relevance_levels(n_components):
    if n_components == 14:
        return (0, 1, 2, 3, 4, 6)
    if n_components == 16:
        return (0, 1, 2, 3, 5, 7)
    raise ValueError("V1.6 stress supports --components 14 or 16")


def records_per_curve(n_components):
    return len(stress_relevance_levels(n_components)) * len(V16_TASKS)


def _build_static_curve(rng, n_components, width):
    if n_components not in V16_STRESS_COMPONENT_COUNTS:
        raise ValueError("V1.6 stress supports exactly 14 or 16 components")
    n_pairs = (n_components - 2) // 2
    minimum_width = 2 * n_pairs + 1
    if width < minimum_width:
        raise ValueError(
            f"V1.6 stress with {n_components} components requires width >= {minimum_width}"
        )

    modulus = rng.choice(PRIMES)
    names = _random_names(rng, n_components)
    keys = _random_keys(rng, n_components)
    key_by_name = dict(zip(names, keys))
    hub, relay = names[:2]
    pad = width - 1

    pair_names = []
    effects = {
        hub: (Op("add", pad, value=rng.randrange(1, modulus)),),
        relay: (Op("add", pad, value=rng.randrange(1, modulus)),),
    }
    families = []
    for pair_index in range(n_pairs):
        left = names[2 + 2 * pair_index]
        right = names[3 + 2 * pair_index]
        pair_names.append((left, right))
        left_effect, right_effect, family = _pair_effects(
            rng,
            modulus,
            width,
            (2 * pair_index, 2 * pair_index + 1),
        )
        effects[left] = left_effect
        effects[right] = right_effect
        families.append(family)

    relevance_order = list(range(n_pairs))
    rng.shuffle(relevance_order)
    preferred_leaf = {
        pair_index: rng.choice(pair_names[pair_index]) for pair_index in range(n_pairs)
    }
    component_order = list(names)
    rng.shuffle(component_order)
    plan_order = list(names[2:])
    rng.shuffle(plan_order)

    return {
        "modulus": modulus,
        "width": width,
        "n_components": n_components,
        "n_pairs": n_pairs,
        "names": tuple(names),
        "keys": key_by_name,
        "hub": hub,
        "relay": relay,
        "pair_names": tuple(pair_names),
        "effects": effects,
        "families": tuple(families),
        "relevance_order": tuple(relevance_order),
        "preferred_leaf": preferred_leaf,
        "component_order": tuple(component_order),
        "plan_order": tuple(plan_order),
    }


def _world_for_count(static, relevant_count):
    relevant_indices = frozenset(static["relevance_order"][:relevant_count])
    relay_key = static["keys"][static["relay"]]
    hub_key = static["keys"][static["hub"]]
    pair_by_name = {}
    for pair_index, pair in enumerate(static["pair_names"]):
        for name in pair:
            pair_by_name[name] = pair_index

    components = {}
    for name in static["names"]:
        if name == static["hub"]:
            requires = frozenset()
        elif name == static["relay"]:
            requires = frozenset({hub_key})
        else:
            pair_index = pair_by_name[name]
            requires = frozenset({relay_key}) if pair_index in relevant_indices else frozenset()
        components[name] = Component(
            name,
            requires,
            frozenset({static["keys"][name]}),
            static["effects"][name],
        )

    world = World(
        static["modulus"],
        static["width"],
        tuple(components[name] for name in static["component_order"]),
    )
    return world, relevant_indices


def _stress_variant_stats(world, post, expected_schedules):
    states, _ = explore(world, post, max_states=STRESS_MAX_STATES)
    quiet = {state for state in states if quiescent(world, state)}
    terminal = observations(quiet)
    schedule_count = count_quiescent_paths(
        world,
        post,
        cap=expected_schedules + 1,
    )
    return {
        "reachable_states": len(states),
        "quiescent_states": len(quiet),
        "quiescent_observations": len(terminal),
        "schedule_count": schedule_count,
    }, terminal


def sample_stress_curve(rng, attempts=1000, n_components=14, width=13):
    levels = stress_relevance_levels(n_components)
    for _ in range(attempts):
        static = _build_static_curve(rng, n_components=n_components, width=width)
        for _value_attempt in range(200):
            values = tuple(rng.randrange(static["modulus"]) for _ in range(width))
            names = frozenset(static["names"])
            start = RuntimeState(values, names, names)
            variants = {}
            valid = True
            for relevant_count in levels:
                world, relevant_indices = _world_for_count(static, relevant_count)
                directive = Action("disable", static["hub"])
                post = orchestrate(world, start, directive)
                expected_terminal = 2 ** relevant_count
                expected_schedules = factorial(2 * relevant_count) if relevant_count else 1
                stats, terminal = _stress_variant_stats(world, post, expected_schedules)
                if len(terminal) != expected_terminal:
                    valid = False
                    break
                if stats["schedule_count"] != expected_schedules:
                    valid = False
                    break
                variants[relevant_count] = {
                    "world": world,
                    "relevant_indices": relevant_indices,
                    "directive": directive,
                    "post": post,
                    "stats": stats,
                    "terminal": terminal,
                }
            if not valid:
                continue

            curve_id = stable_id(
                {
                    "schema_version": "1.6-stress",
                    "world_max": serialize_world(variants[max(levels)]["world"]),
                    "start": serialize_state(start),
                    "relevance_order": list(static["relevance_order"]),
                    "families": list(static["families"]),
                    "n_components": n_components,
                }
            )
            return {
                "curve_id": curve_id,
                "static": static,
                "start": start,
                "variants": variants,
                "levels": levels,
            }
    raise RuntimeError("could not sample a V1.6 stress curve")


def generate_v16_stress_dataset(
    n,
    seed=0,
    task_types=V16_TASKS,
    n_components=14,
    width=None,
):
    if tuple(task_types) != V16_TASKS:
        raise ValueError("V1.6 stress is a matched curve suite; custom --tasks is not supported")
    if n_components not in V16_STRESS_COMPONENT_COUNTS:
        raise ValueError("V1.6 stress supports --components 14 or 16")
    if width is None:
        width = n_components - 1

    per_curve = records_per_curve(n_components)
    if n % per_curve:
        raise ValueError(
            f"V1.6 stress with {n_components} components emits {per_curve} records per curve; "
            f"n must be a multiple of {per_curve}"
        )

    rng = random.Random(seed)
    records = []
    for _ in range(n // per_curve):
        curve = sample_stress_curve(
            rng,
            n_components=n_components,
            width=width,
        )
        for record in composition_curve_records(rng, curve):
            record["metadata"]["benchmark_suite_variant"] = "1.6-stress"
            record["metadata"]["boundary_search"] = True
            record["metadata"]["stress_max_states"] = STRESS_MAX_STATES
            records.append(record)
    return records
