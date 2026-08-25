"""Hardened formal implementation for the V1.12 challenge track.

This module keeps the V1.12 task surface and oracle from ``v112_formal`` but
removes generator-position leakage: local outcome multiplicities are sampled
from semantic content, not derived from the world/profile index.
"""

from math import prod
import random

from .core import Action, Component, Op, RuntimeState, World
from .semantics import apply_ops
from .tasks import stable_id
from . import v112_formal as _base


V112_FORMAL_COMPONENT = _base.V112_FORMAL_COMPONENT
V112_FORMAL_TASKS = _base.V112_FORMAL_TASKS
V112_FORMAL_SIZES = _base.V112_FORMAL_SIZES
V112_FORMAL_WORLDS_PER_SIZE = _base.V112_FORMAL_WORLDS_PER_SIZE
V112_FORMAL_RECORDS = _base.V112_FORMAL_RECORDS


def _sample_relevant_counts(rng, n_gadgets):
    counts = [rng.randrange(2, 7) for _ in range(n_gadgets)]
    while n_gadgets > 1 and len(set(counts)) == 1:
        counts = [rng.randrange(2, 7) for _ in range(n_gadgets)]
    return counts


def _build_formal_world(rng, n_gadgets, profile=None):
    if n_gadgets < 2:
        raise ValueError("formal challenge requires at least two gadgets")
    n_decoys = max(2, n_gadgets // 2)
    n_total_gadgets = n_gadgets + n_decoys
    n_components = 4 + 3 * n_total_gadgets
    width = 2 * n_total_gadgets + 1
    modulus = rng.choice(_base.PRIMES)
    names = _base._challenge_names(rng, n_components)
    keys = dict(zip(names, _base._random_keys(rng, n_components)))
    root_a, relay_a, root_b, relay_b = names[:4]
    leaves = names[4:]
    pad = width - 1

    values = [rng.randrange(modulus) for _ in range(width)]
    effects = {
        root_a: (Op("add", pad, value=rng.randrange(1, modulus)),),
        relay_a: (Op("add", pad, value=rng.randrange(1, modulus)),),
        root_b: (Op("add", pad, value=rng.randrange(1, modulus)),),
        relay_b: (Op("add", pad, value=rng.randrange(1, modulus)),),
    }

    desired = _sample_relevant_counts(rng, n_gadgets)
    local_counts = []
    gadget_names = []
    for gadget in range(n_total_gadgets):
        names3 = tuple(leaves[3 * gadget : 3 * gadget + 3])
        target = desired[gadget] if gadget < n_gadgets else rng.randrange(1, 7)
        local_effects, local_values = _base._sample_local_gadget(rng, modulus, target)
        offset = 2 * gadget
        values[offset : offset + 2] = local_values
        for name, effect in zip(names3, local_effects):
            effects[name] = _base._shift_effect(effect, offset)
        gadget_names.append(names3)
        if gadget < n_gadgets:
            local_counts.append(target)

    terminal_a = keys[relay_a]
    terminal_b = keys[relay_b]
    components = {
        root_a: Component(root_a, frozenset(), frozenset({keys[root_a]}), effects[root_a]),
        relay_a: Component(
            relay_a,
            frozenset({keys[root_a]}),
            frozenset({terminal_a}),
            effects[relay_a],
        ),
        root_b: Component(root_b, frozenset(), frozenset({keys[root_b]}), effects[root_b]),
        relay_b: Component(
            relay_b,
            frozenset({keys[root_b]}),
            frozenset({terminal_b}),
            effects[relay_b],
        ),
    }
    for gadget, names3 in enumerate(gadget_names):
        required = terminal_a if gadget < n_gadgets else terminal_b
        for name in names3:
            components[name] = Component(
                name,
                frozenset({required}),
                frozenset({keys[name]}),
                effects[name],
            )

    component_order = list(names)
    rng.shuffle(component_order)
    world = World(
        modulus,
        width,
        tuple(components[name] for name in component_order),
    )
    start = RuntimeState(tuple(values), frozenset(names), frozenset(names))
    directive = Action("disable", root_a)

    schedule = []
    for names3 in gadget_names[:n_gadgets]:
        order = list(names3)
        rng.shuffle(order)
        schedule.extend(order)
    schedule.extend((relay_a, root_a))

    final_values = start.values
    for name in schedule:
        final_values = apply_ops(
            final_values,
            world.by_name[name].inverse_effect(modulus),
            modulus,
        )

    world_id = stable_id(
        {
            "schema_version": "1.12-formal-hardened",
            "world": [
                (
                    component.name,
                    sorted(component.requires),
                    sorted(component.provides),
                    [op.render() for op in component.effect],
                )
                for component in world.components
            ],
            "values": list(start.values),
            "directive": directive.render(),
        }
    )
    return {
        "world_id": world_id,
        "world": world,
        "start": start,
        "directive": directive,
        "schedule": tuple(schedule),
        "schedule_checksum": _base.checksum(final_values),
        "outcome_count": prod(local_counts),
        "local_counts": tuple(local_counts),
        "n_gadgets": n_gadgets,
        "n_decoys": n_decoys,
    }


def generate_v112_formal_dataset(
    sizes=V112_FORMAL_SIZES,
    worlds_per_size=V112_FORMAL_WORLDS_PER_SIZE,
    seed=0,
):
    rng = random.Random(seed + 12_100_101)
    records = []
    for size in sizes:
        cases = []
        outcome_counts = set()
        attempts = 0
        while len(cases) < worlds_per_size:
            attempts += 1
            if attempts > max(100, worlds_per_size * 50):
                raise RuntimeError(f"could not sample varied formal worlds for size={size}")
            case = _build_formal_world(rng, int(size))
            if (
                worlds_per_size > 1
                and len(cases) == worlds_per_size - 1
                and len(outcome_counts) == 1
                and case["outcome_count"] in outcome_counts
            ):
                continue
            cases.append(case)
            outcome_counts.add(case["outcome_count"])
        for case in cases:
            records.extend(_base._records(rng, case))
    return records
