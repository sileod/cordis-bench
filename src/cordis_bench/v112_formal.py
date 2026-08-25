"""Scalable formal V1.12 challenge."""

from itertools import permutations
from math import prod
import random

from .core import Action, Component, DEFAULT_THEORY, Op, RuntimeState, World
from .generate import PRIMES
from .render import render_state, render_world, v1_rules
from .semantics import apply_ops
from .tasks import stable_id
from .v112_common import CHECKSUM_MODULUS, checksum, metadata, record


V112_FORMAL_COMPONENT = "formal_challenge"
V112_FORMAL_TASKS = (
    "formal_challenge_schedule_checksum",
    "formal_challenge_outcome_count",
)
V112_FORMAL_SIZES = (8, 16, 24, 32)
V112_FORMAL_WORLDS_PER_SIZE = 8
V112_FORMAL_RECORDS = (
    len(V112_FORMAL_SIZES) * V112_FORMAL_WORLDS_PER_SIZE * 2 * 2
)


def _rules():
    return v1_rules(DEFAULT_THEORY) + (
        "- Lifecycle priority: if any deactivation is legal, no activation step is legal "
        "until the required withdrawals are cleared.\n"
    )


def _challenge_names(rng, n):
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    rng.shuffle(alphabet)
    names = alphabet[: min(n, len(alphabet))]
    suffix = 0
    while len(names) < n:
        candidate = f"C{suffix}"
        suffix += 1
        if candidate not in names:
            names.append(candidate)
    return names


def _random_keys(rng, n):
    return [f"q{value}" for value in rng.sample(range(100_000, 999_999), n)]


def _random_local_op(rng, modulus):
    kind = rng.choice(("add", "scale", "shear", "swap"))
    i = rng.randrange(2)
    if kind == "add":
        return Op(kind, i, value=rng.randrange(1, modulus))
    if kind == "scale":
        return Op(kind, i, value=rng.randrange(2, modulus))
    j = 1 - i
    if kind == "swap":
        return Op(kind, i, j)
    return Op(kind, i, j, rng.randrange(1, modulus))


def _random_local_effect(rng, modulus):
    return tuple(_random_local_op(rng, modulus) for _ in range(rng.choice((1, 1, 2))))


def _inverse(effect, modulus):
    return tuple(op.inverse(modulus) for op in reversed(effect))


def _local_outcomes(effects, values, modulus):
    outcomes = set()
    for order in permutations(range(3)):
        current = values
        for index in order:
            current = apply_ops(current, _inverse(effects[index], modulus), modulus)
        outcomes.add(current)
    return outcomes


def _sample_local_gadget(rng, modulus, desired, attempts=20_000):
    for _ in range(attempts):
        effects = tuple(_random_local_effect(rng, modulus) for _ in range(3))
        values = (rng.randrange(modulus), rng.randrange(modulus))
        if len(_local_outcomes(effects, values, modulus)) == desired:
            return effects, values
    raise RuntimeError(f"could not sample local gadget with {desired} outcomes")


def _shift_op(op, offset):
    return Op(
        op.kind,
        op.i + offset,
        None if op.j is None else op.j + offset,
        op.value,
    )


def _shift_effect(effect, offset):
    return tuple(_shift_op(op, offset) for op in effect)


def _build_formal_world(rng, n_gadgets, profile=0):
    if n_gadgets < 2:
        raise ValueError("formal challenge requires at least two gadgets")
    n_decoys = max(2, n_gadgets // 2)
    n_total_gadgets = n_gadgets + n_decoys
    n_components = 4 + 3 * n_total_gadgets
    width = 2 * n_total_gadgets + 1
    modulus = rng.choice(PRIMES)
    names = _challenge_names(rng, n_components)
    keys = dict(zip(names, _random_keys(rng, n_components)))
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

    desired = [2 + ((index + profile) % 5) for index in range(n_gadgets)]
    rng.shuffle(desired)
    local_counts = []
    gadget_names = []

    for gadget in range(n_total_gadgets):
        names3 = tuple(leaves[3 * gadget : 3 * gadget + 3])
        target = desired[gadget] if gadget < n_gadgets else rng.randrange(1, 7)
        local_effects, local_values = _sample_local_gadget(rng, modulus, target)
        offset = 2 * gadget
        values[offset : offset + 2] = local_values
        for name, effect in zip(names3, local_effects):
            effects[name] = _shift_effect(effect, offset)
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
        component = world.by_name[name]
        final_values = apply_ops(
            final_values,
            component.inverse_effect(modulus),
            modulus,
        )

    world_id = stable_id(
        {
            "schema_version": "1.12-formal",
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
        "schedule_checksum": checksum(final_values),
        "outcome_count": prod(local_counts),
        "local_counts": tuple(local_counts),
        "n_gadgets": n_gadgets,
        "n_decoys": n_decoys,
    }


def _alpha_rename(case, rng):
    world = case["world"]
    names = [component.name for component in world.components]
    keys = sorted(
        {
            key
            for component in world.components
            for key in component.requires | component.provides
        }
    )
    renamed_names = [f"unit_{rng.randrange(10**9):09d}" for _ in names]
    while len(set(renamed_names)) != len(renamed_names):
        renamed_names = [f"unit_{rng.randrange(10**9):09d}" for _ in names]
    renamed_keys = [f"cap_{rng.randrange(10**9):09d}" for _ in keys]
    while len(set(renamed_keys)) != len(renamed_keys):
        renamed_keys = [f"cap_{rng.randrange(10**9):09d}" for _ in keys]
    name_map = dict(zip(names, renamed_names))
    key_map = dict(zip(keys, renamed_keys))
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
    return {
        **case,
        "world": World(world.modulus, world.width, tuple(components)),
        "start": RuntimeState(
            case["start"].values,
            frozenset(name_map[name] for name in case["start"].enabled),
            frozenset(name_map[name] for name in case["start"].active),
        ),
        "directive": Action(
            case["directive"].kind,
            name_map[case["directive"].component],
        ),
        "schedule": tuple(name_map[name] for name in case["schedule"]),
    }


def _prompt(case, task_type):
    prefix = f"""{_rules()}{render_world(case['world'])}

The system starts quiescent:
{render_state(case['start'])}

The orchestrator executes:
  {case['directive'].render()}
"""
    if task_type == "formal_challenge_schedule_checksum":
        schedule = " -> ".join(case["schedule"])
        return prefix + f"""
The runtime then follows this legal complete withdrawal order:
  {schedule}

After the listed withdrawals, compute
  sum((i+1) * x_i) mod {CHECKSUM_MODULUS}
over the final shared-state vector. Return only the integer checksum."""
    return prefix + """
The runtime may take lifecycle steps in any legal order until quiescence.
Across all legal schedules, how many distinct final observable outcomes
(shared state and active component set) are reachable?
Return only the integer count."""


def _records(rng, case):
    renderings = {
        "base": case,
        "alpha_renamed_reordered": _alpha_rename(case, rng),
    }
    specs = (
        (
            "formal_challenge_schedule_checksum",
            "schedule_checksum",
            case["schedule_checksum"],
        ),
        (
            "formal_challenge_outcome_count",
            "outcome_count",
            case["outcome_count"],
        ),
    )
    records = []
    for task_type, family, gold in specs:
        pair_id = stable_id(
            {
                "schema_version": "1.12-formal-pair",
                "world": case["world_id"],
                "task": task_type,
            }
        )
        for perturbation, rendered in renderings.items():
            md = metadata(
                V112_FORMAL_COMPONENT,
                family,
                case["n_gadgets"],
                "relevant_three_effect_gadgets",
                case["world_id"],
                pair_id,
                perturbation,
                realization="finite_modular_micro_system",
                oracle_type="validated_local_product",
                query_quantifier=(
                    "named_legal_schedule"
                    if task_type == "formal_challenge_schedule_checksum"
                    else "all_legal_schedules"
                ),
                problem_size=len(case["world"].components),
                state_width=case["world"].width,
                relevant_gadgets=case["n_gadgets"],
                irrelevant_gadgets=case["n_decoys"],
                dependency_depth=2,
                effect_ops_total=sum(
                    len(component.effect) for component in case["world"].components
                ),
            )
            records.append(
                record(
                    task_type,
                    _prompt(rendered, task_type),
                    gold,
                    md,
                    pair_id,
                    perturbation,
                )
            )
    return records


def generate_v112_formal_dataset(
    sizes=V112_FORMAL_SIZES,
    worlds_per_size=V112_FORMAL_WORLDS_PER_SIZE,
    seed=0,
):
    rng = random.Random(seed + 12_100_101)
    records = []
    for size in sizes:
        for profile in range(worlds_per_size):
            records.extend(_records(rng, _build_formal_world(rng, int(size), profile)))
    return records
