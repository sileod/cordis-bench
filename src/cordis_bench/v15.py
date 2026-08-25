"""V1.5 composition-structure curves for spatiotemporal composability.

V1.5 holds the *total* number of noncommuting component pairs fixed inside a
matched curve and varies how many of those pairs become lifecycle-relevant when
a provider is withdrawn.  This makes the main difficulty axis composition
structure rather than arithmetic size.

A curve uses one fixed set of component effects, names, keys, component order,
and initial state.  Across curve points only dependency edges change: 0/1/2/M
noncommuting pairs are placed two dependency hops below the provider being
removed.  The remaining noncommuting pairs stay active and are irrelevant to
the queried lifecycle.

Each curve point yields:

    composition_confluence -- do all legal unload schedules agree?
    safe_intervention      -- which preparatory plan makes the provider removal
                              both target-preserving and schedule-robust?

Supported component counts are 6, 8, and 10, corresponding to 2, 3, and 4
noncommuting pairs.  State width is kept fixed by default so component-count
scaling is a secondary stressor rather than extra arithmetic dimensionality.
"""

from itertools import combinations, product
from math import factorial
import random

from .core import Action, Component, DEFAULT_THEORY, Op, RuntimeState, World
from .generate import PRIMES
from .render import render_state, render_world, v1_rules
from .semantics import observations, orchestrate, quiescent_states, settle_plan, settling_stats
from .tasks import LETTERS, serialize_state, serialize_world, stable_id, v1_metadata
from .v13 import dependency_depth


V15_TASKS = ("composition_confluence", "safe_intervention")
V15_COMPONENT_COUNTS = (6, 8, 10)
PAIR_FAMILIES = ("shear_cycle", "scale_add", "swap_add", "add_shear")


def relevance_levels(n_components):
    if n_components not in V15_COMPONENT_COUNTS:
        raise ValueError("V1.5 supports --components 6, 8, or 10")
    n_pairs = (n_components - 2) // 2
    return tuple(sorted({0, 1, 2, n_pairs}))


def records_per_curve(n_components):
    return len(relevance_levels(n_components)) * len(V15_TASKS)


def _rules():
    return v1_rules(DEFAULT_THEORY) + (
        "- Lifecycle priority: if any deactivation is legal, no activation step is legal "
        "until the required withdrawals are cleared.\n"
    )


def _apply_effect(effect, values, modulus):
    for op in effect:
        values = op.apply(values, modulus)
    return values


def _effects_commute_on_support(effect_a, effect_b, modulus, width):
    """Exact commutativity check over coordinates touched by either effect."""
    touched = set()
    for op in (*effect_a, *effect_b):
        touched.add(op.i)
        if op.j is not None:
            touched.add(op.j)
    touched = sorted(touched)
    for local in product(range(modulus), repeat=len(touched)):
        values = [0] * width
        for index, value in zip(touched, local):
            values[index] = value
        values = tuple(values)
        ab = _apply_effect(effect_b, _apply_effect(effect_a, values, modulus), modulus)
        ba = _apply_effect(effect_a, _apply_effect(effect_b, values, modulus), modulus)
        if ab != ba:
            return False
    return True


def _pair_effects(rng, modulus, width, coordinates, family=None):
    a, b = coordinates
    family = family or rng.choice(PAIR_FAMILIES)
    if family == "shear_cycle":
        left = (Op("shear", a, b, rng.randrange(1, modulus)),)
        right = (Op("shear", b, a, rng.randrange(1, modulus)),)
    elif family == "scale_add":
        factor = rng.randrange(2, modulus)
        left = (Op("scale", a, value=factor),)
        right = (Op("add", a, value=rng.randrange(1, modulus)),)
    elif family == "swap_add":
        left = (Op("swap", a, b),)
        right = (Op("add", a, value=rng.randrange(1, modulus)),)
    elif family == "add_shear":
        left = (Op("add", a, value=rng.randrange(1, modulus)),)
        right = (Op("shear", b, a, rng.randrange(1, modulus)),)
    else:
        raise ValueError(f"unknown pair family: {family}")

    assert not _effects_commute_on_support(left, right, modulus, width)
    return left, right, family


def _random_names(rng, n):
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    rng.shuffle(alphabet)
    return alphabet[:n]


def _random_keys(rng, n):
    return [f"q{value}" for value in rng.sample(range(100, 999), n)]


def _build_static_curve(rng, n_components=10, width=9):
    if n_components not in V15_COMPONENT_COUNTS:
        raise ValueError("V1.5 supports exactly 6, 8, or 10 components")
    n_pairs = (n_components - 2) // 2
    minimum_width = 2 * n_pairs + 1
    if width < minimum_width:
        raise ValueError(
            f"V1.5 with {n_components} components requires width >= {minimum_width}"
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


def _terminal_observations(world, post):
    quiet, _ = quiescent_states(world, post)
    return observations(quiet)


def sample_composition_curve(rng, attempts=1000, n_components=10, width=9):
    """Sample one matched 0/1/2/M relevance curve with exact oracle separation."""
    levels = relevance_levels(n_components)
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
                stats = settling_stats(world, post)
                terminal = _terminal_observations(world, post)
                expected_terminal = 2 ** relevant_count
                expected_schedules = factorial(2 * relevant_count) if relevant_count else 1
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
                    "schema_version": "1.5",
                    "world_max": serialize_world(variants[max(levels)]["world"]),
                    "start": serialize_state(start),
                    "relevance_order": list(static["relevance_order"]),
                    "families": list(static["families"]),
                }
            )
            return {
                "curve_id": curve_id,
                "static": static,
                "start": start,
                "variants": variants,
                "levels": levels,
            }
    raise RuntimeError("could not sample a V1.5 composition curve")


def _metadata(curve, relevant_count, capability, **extra):
    static = curve["static"]
    variant = curve["variants"][relevant_count]
    world = variant["world"]
    stats = variant["stats"]
    return v1_metadata(
        world,
        curve["start"],
        benchmark_suite="1.5",
        capability=capability,
        curve_id=curve["curve_id"],
        problem_size=static["n_components"],
        total_interfering_pairs=static["n_pairs"],
        n_noncommuting_pairs=static["n_pairs"],
        relevant_interfering_pairs=relevant_count,
        irrelevant_interfering_pairs=static["n_pairs"] - relevant_count,
        relevant_fraction=(relevant_count / static["n_pairs"] if static["n_pairs"] else 0.0),
        dependency_depth=dependency_depth(world),
        schedule_count=stats["schedule_count"],
        reachable_states=stats["reachable_states"],
        terminal_observations=len(variant["terminal"]),
        effect_ops_total=sum(len(component.effect) for component in world.components),
        effect_families=list(static["families"]),
        pair_names=[list(pair) for pair in static["pair_names"]],
        relevance_order=list(static["relevance_order"]),
        queried_root=static["hub"],
        **extra,
    )


def _binary_choices(rng):
    if rng.random() < 0.5:
        return {"A": "YES", "B": "NO"}
    return {"A": "NO", "B": "YES"}


def _confluence_prompt(world, start, directive, choices):
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    return f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(start)}

The orchestrator executes:
  {directive.render()}

The runtime then takes lifecycle steps in any legal order until quiescence. Is the final observable outcome (shared state x and active component set) the same for every legal lifecycle schedule?
{options}

Answer with only the option letter."""


def composition_confluence_record(rng, curve, relevant_count):
    variant = curve["variants"][relevant_count]
    choices = _binary_choices(rng)
    confluent = len(variant["terminal"]) == 1
    desired = "YES" if confluent else "NO"
    answer = next(letter for letter, text in choices.items() if text == desired)
    payload = {
        "schema_version": "1.5",
        "task_type": "composition_confluence",
        "curve_id": curve["curve_id"],
        "relevant_interfering_pairs": relevant_count,
        "choices": choices,
    }
    return {
        "id": stable_id(payload),
        "schema_version": "1.5",
        "task_type": "composition_confluence",
        "prompt": _confluence_prompt(
            variant["world"], curve["start"], variant["directive"], choices
        ),
        "choices": choices,
        "answer": answer,
        "metadata": _metadata(
            curve,
            relevant_count,
            "global",
            confluent=confluent,
        ),
    }


def _prep_order(static, prep_set):
    rank = {name: index for index, name in enumerate(static["plan_order"])}
    return tuple(sorted(prep_set, key=lambda name: rank[name]))


def _plan_directives(static, prep_set):
    prep = _prep_order(static, prep_set)
    return tuple(Action("disable", name) for name in prep) + (
        Action("disable", static["hub"]),
    )


def _plan_success(world, start, directives, target_active):
    frontier = settle_plan(world, start, directives)
    obs = observations(frontier)
    return len(obs) == 1 and all(state.active == target_active for state in frontier)


def _correct_prep_set(curve, relevant_count):
    static = curve["static"]
    relevant = static["relevance_order"][:relevant_count]
    return frozenset(static["preferred_leaf"][pair_index] for pair_index in relevant)


def _candidate_prep_sets(curve, relevant_count):
    """Construct likely-hard distractors; fall back to small exhaustive subsets."""
    static = curve["static"]
    correct = _correct_prep_set(curve, relevant_count)
    relevant_indices = tuple(static["relevance_order"][:relevant_count])
    irrelevant_indices = tuple(static["relevance_order"][relevant_count:])
    candidates = []

    if relevant_count == 0:
        for pair_index in irrelevant_indices:
            candidates.extend(frozenset({leaf}) for leaf in static["pair_names"][pair_index])
    else:
        for pair_index in relevant_indices:
            candidates.append(correct - {static["preferred_leaf"][pair_index]})

        if irrelevant_indices:
            irrelevant_leaf = static["pair_names"][irrelevant_indices[0]][0]
            candidates.append(correct | {irrelevant_leaf})
            omitted = relevant_indices[0]
            candidates.append(
                (correct - {static["preferred_leaf"][omitted]}) | {irrelevant_leaf}
            )
        elif relevant_count >= 2:
            omitted = relevant_indices[0]
            doubled = relevant_indices[1]
            sibling = next(
                leaf
                for leaf in static["pair_names"][doubled]
                if leaf != static["preferred_leaf"][doubled]
            )
            candidates.append(
                (correct - {static["preferred_leaf"][omitted]}) | {sibling}
            )

    # Preserve order while removing duplicates and the correct set.
    unique = []
    seen = {correct}
    for candidate in candidates:
        candidate = frozenset(candidate)
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)

    # At most eight leaves exist, so this fallback is cheap and deterministic.
    leaves = [leaf for pair in static["pair_names"] for leaf in pair]
    for size in range(len(leaves) + 1):
        for subset in combinations(leaves, size):
            candidate = frozenset(subset)
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
    return correct, unique


def _plan_text(directives):
    return "; ".join(f"{action.render()}; settle to quiescence" for action in directives)


def safe_intervention_record(rng, curve, relevant_count):
    static = curve["static"]
    variant = curve["variants"][relevant_count]
    world = variant["world"]
    relevant_indices = set(static["relevance_order"][:relevant_count])
    target_active = frozenset(
        leaf
        for pair_index, pair in enumerate(static["pair_names"])
        if pair_index not in relevant_indices
        for leaf in pair
    )

    correct_prep, candidate_preps = _candidate_prep_sets(curve, relevant_count)
    correct_directives = _plan_directives(static, correct_prep)
    if not _plan_success(world, curve["start"], correct_directives, target_active):
        raise RuntimeError("constructed V1.5 safe plan did not satisfy its oracle")

    distractors = []
    for prep_set in candidate_preps:
        directives = _plan_directives(static, prep_set)
        if not _plan_success(world, curve["start"], directives, target_active):
            distractors.append(directives)
            if len(distractors) == 3:
                break
    if len(distractors) < 3:
        raise RuntimeError("could not construct three V1.5 intervention distractors")

    plans = [correct_directives, *distractors]
    rng.shuffle(plans)
    choices = {LETTERS[index]: _plan_text(plan) for index, plan in enumerate(plans)}
    answer = next(
        letter for letter, plan in zip(choices, plans) if plan == correct_directives
    )
    option_success = {
        letter: _plan_success(world, curve["start"], plan, target_active)
        for letter, plan in zip(choices, plans)
    }
    if sum(option_success.values()) != 1 or not option_success[answer]:
        raise RuntimeError("V1.5 intervention options are not uniquely scored")

    options = "\n".join(f"  {letter}. {text}" for letter, text in choices.items())
    target_text = "{" + ", ".join(sorted(target_active)) + "}"
    prompt = f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(curve['start'])}

Maintenance goal: remove component {static['hub']}. After every external disable directive shown in a plan, the runtime settles to quiescence before the next directive.

A plan is acceptable only if BOTH conditions hold:
1. the final active component set is exactly {target_text}; and
2. every legal lifecycle schedule allowed during the plan yields the same final shared state x.

Which plan is guaranteed to satisfy both conditions?
{options}

Answer with only the option letter."""

    payload = {
        "schema_version": "1.5",
        "task_type": "safe_intervention",
        "curve_id": curve["curve_id"],
        "relevant_interfering_pairs": relevant_count,
        "choices": choices,
    }
    return {
        "id": stable_id(payload),
        "schema_version": "1.5",
        "task_type": "safe_intervention",
        "prompt": prompt,
        "choices": choices,
        "answer": answer,
        "metadata": _metadata(
            curve,
            relevant_count,
            "act",
            target_active=sorted(target_active),
            option_success=option_success,
            option_plan_lengths={
                letter: len(plan) for letter, plan in zip(choices, plans)
            },
        ),
    }


def composition_curve_records(rng, curve):
    records = []
    for relevant_count in curve["levels"]:
        records.append(composition_confluence_record(rng, curve, relevant_count))
        records.append(safe_intervention_record(rng, curve, relevant_count))
    return records


def generate_v15_dataset(
    n,
    seed=0,
    task_types=V15_TASKS,
    n_components=10,
    width=9,
):
    if tuple(task_types) != V15_TASKS:
        raise ValueError("V1.5 is a matched curve suite; custom --tasks is not supported")
    per_curve = records_per_curve(n_components)
    if n % per_curve:
        raise ValueError(
            f"V1.5 with {n_components} components emits {per_curve} records per curve; "
            f"n must be a multiple of {per_curve}"
        )

    rng = random.Random(seed)
    records = []
    for _ in range(n // per_curve):
        curve = sample_composition_curve(
            rng,
            n_components=n_components,
            width=width,
        )
        records.extend(composition_curve_records(rng, curve))
    return records
