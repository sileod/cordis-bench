"""V1.6 effort-first boundary probes for spatiotemporal composability.

V1.6 keeps the V1.5 construct fixed and adds two deliberately modest changes:

1. a quantitative global probe that counts distinct terminal observations, so
   difficulty cannot collapse to finding one counterexample to confluence; and
2. a small 10 -> 12 component scale increase, with state width fixed at 11
   across the recommended size comparison.

Within a matched curve, effects, names, keys, initial state, and total
noncommuting-pair count remain fixed. Only dependency edges change to make
0/1/2/... pairs lifecycle-relevant. The intended primary experiment is the
same frozen dataset under low versus none/minimal reasoning effort.
"""

from math import factorial
import random

from .core import Action, Component, Op, RuntimeState, World
from .generate import PRIMES
from .render import render_state, render_world
from .semantics import count_quiescent_paths, explore, observations, orchestrate, quiescent
from .tasks import LETTERS, serialize_state, serialize_world, stable_id, v1_metadata
from .v13 import dependency_depth
from .v15 import (
    _binary_choices,
    _pair_effects,
    _plan_directives,
    _plan_success,
    _plan_text,
    _random_keys,
    _random_names,
    _rules,
)


V16_TASKS = (
    "composition_confluence",
    "composition_outcome_count",
    "safe_intervention",
)
V16_COMPONENT_COUNTS = (10, 12)


def relevance_levels(n_components):
    if n_components not in V16_COMPONENT_COUNTS:
        raise ValueError("V1.6 supports --components 10 or 12")
    n_pairs = (n_components - 2) // 2
    if n_pairs == 4:
        return (0, 1, 2, 4)
    return (0, 1, 2, 3, 5)


def records_per_curve(n_components):
    return len(relevance_levels(n_components)) * len(V16_TASKS)


def _build_static_curve(rng, n_components=12, width=11):
    if n_components not in V16_COMPONENT_COUNTS:
        raise ValueError("V1.6 supports exactly 10 or 12 components")
    n_pairs = (n_components - 2) // 2
    minimum_width = 2 * n_pairs + 1
    if width < minimum_width:
        raise ValueError(
            f"V1.6 with {n_components} components requires width >= {minimum_width}"
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


def _variant_stats(world, post, expected_schedules):
    states, _ = explore(world, post)
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


def sample_composition_curve(rng, attempts=1000, n_components=12, width=11):
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
                expected_terminal = 2 ** relevant_count
                expected_schedules = factorial(2 * relevant_count) if relevant_count else 1
                stats, terminal = _variant_stats(world, post, expected_schedules)
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
                    "schema_version": "1.6",
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
    raise RuntimeError("could not sample a V1.6 composition curve")


def _metadata(curve, relevant_count, capability, **extra):
    static = curve["static"]
    variant = curve["variants"][relevant_count]
    world = variant["world"]
    stats = variant["stats"]
    return v1_metadata(
        world,
        curve["start"],
        benchmark_suite="1.6",
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
        terminal_outcome_count=len(variant["terminal"]),
        effect_ops_total=sum(len(component.effect) for component in world.components),
        effect_families=list(static["families"]),
        pair_names=[list(pair) for pair in static["pair_names"]],
        relevance_order=list(static["relevance_order"]),
        queried_root=static["hub"],
        **extra,
    )


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
    return {
        "id": stable_id({
            "schema_version": "1.6",
            "task_type": "composition_confluence",
            "curve_id": curve["curve_id"],
            "relevant_interfering_pairs": relevant_count,
            "choices": choices,
        }),
        "schema_version": "1.6",
        "task_type": "composition_confluence",
        "prompt": _confluence_prompt(
            variant["world"], curve["start"], variant["directive"], choices
        ),
        "choices": choices,
        "answer": answer,
        "metadata": _metadata(curve, relevant_count, "global", confluent=confluent),
    }


def _count_choices(rng, gold):
    candidates = []
    raw = [
        gold + 1,
        gold + 2,
        gold * 2,
        max(1, gold // 2),
        gold * 4,
        max(1, gold - 2),
        gold + 4,
    ]
    for value in raw:
        if value != gold and value not in candidates:
            candidates.append(value)
    rng.shuffle(candidates)
    values = [gold, *candidates[:3]]
    rng.shuffle(values)
    return {LETTERS[index]: str(value) for index, value in enumerate(values)}


def _outcome_count_prompt(world, start, directive, choices):
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    return f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(start)}

The orchestrator executes:
  {directive.render()}

The runtime then takes lifecycle steps in any legal order until quiescence. Across all legal lifecycle schedules, how many distinct final observable outcomes (shared state x and active component set) are reachable?
{options}

Answer with only the option letter."""


def composition_outcome_count_record(rng, curve, relevant_count):
    variant = curve["variants"][relevant_count]
    gold = len(variant["terminal"])
    choices = _count_choices(rng, gold)
    answer = next(letter for letter, text in choices.items() if int(text) == gold)
    return {
        "id": stable_id({
            "schema_version": "1.6",
            "task_type": "composition_outcome_count",
            "curve_id": curve["curve_id"],
            "relevant_interfering_pairs": relevant_count,
            "choices": choices,
        }),
        "schema_version": "1.6",
        "task_type": "composition_outcome_count",
        "prompt": _outcome_count_prompt(
            variant["world"], curve["start"], variant["directive"], choices
        ),
        "choices": choices,
        "answer": answer,
        "metadata": _metadata(
            curve,
            relevant_count,
            "global",
            outcome_count_gold=gold,
        ),
    }


def _correct_prep_set(curve, relevant_count):
    static = curve["static"]
    relevant = static["relevance_order"][:relevant_count]
    return frozenset(static["preferred_leaf"][pair_index] for pair_index in relevant)


def _other_leaf(static, pair_index):
    preferred = static["preferred_leaf"][pair_index]
    return next(leaf for leaf in static["pair_names"][pair_index] if leaf != preferred)


def _balanced_candidate_prep_sets(curve, relevant_count):
    """Return the correct prep set and plan-length-matched failing candidates.

    For every positive k, distractors contain exactly k preparatory disables,
    matching the correct plan. They fail because they leave at least one
    relevant pair uncovered, while spending the displaced disable either on an
    irrelevant leaf or on the second leaf of another relevant pair.
    """
    static = curve["static"]
    correct = _correct_prep_set(curve, relevant_count)
    relevant = tuple(static["relevance_order"][:relevant_count])
    irrelevant = tuple(static["relevance_order"][relevant_count:])

    if relevant_count == 0:
        return correct, [
            frozenset({leaf})
            for pair_index in irrelevant
            for leaf in static["pair_names"][pair_index]
        ]

    candidates = []
    if irrelevant:
        for omitted in relevant:
            base = correct - {static["preferred_leaf"][omitted]}
            for extra_pair in irrelevant:
                for extra_leaf in static["pair_names"][extra_pair]:
                    candidates.append(base | {extra_leaf})
    else:
        for omitted in relevant:
            base = correct - {static["preferred_leaf"][omitted]}
            for donor in relevant:
                if donor == omitted:
                    continue
                candidates.append(base | {_other_leaf(static, donor)})

    unique = []
    seen = {correct}
    for candidate in candidates:
        candidate = frozenset(candidate)
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return correct, unique


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

    correct_prep, candidate_preps = _balanced_candidate_prep_sets(curve, relevant_count)
    correct_directives = _plan_directives(static, correct_prep)
    if not _plan_success(world, curve["start"], correct_directives, target_active):
        raise RuntimeError("constructed V1.6 safe plan did not satisfy its oracle")

    distractors = []
    for prep_set in candidate_preps:
        directives = _plan_directives(static, prep_set)
        if not _plan_success(world, curve["start"], directives, target_active):
            distractors.append(directives)
            if len(distractors) == 3:
                break
    if len(distractors) < 3:
        raise RuntimeError("could not construct three V1.6 intervention distractors")

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
        raise RuntimeError("V1.6 intervention options are not uniquely scored")

    option_plan_lengths = {
        letter: len(plan) for letter, plan in zip(choices, plans)
    }
    if relevant_count > 0 and len(set(option_plan_lengths.values())) != 1:
        raise RuntimeError("V1.6 positive-k intervention options leaked plan length")

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

    return {
        "id": stable_id({
            "schema_version": "1.6",
            "task_type": "safe_intervention",
            "curve_id": curve["curve_id"],
            "relevant_interfering_pairs": relevant_count,
            "choices": choices,
        }),
        "schema_version": "1.6",
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
            option_plan_lengths=option_plan_lengths,
            plan_length_balanced=(len(set(option_plan_lengths.values())) == 1),
        ),
    }


def composition_curve_records(rng, curve):
    records = []
    for relevant_count in curve["levels"]:
        records.append(composition_confluence_record(rng, curve, relevant_count))
        records.append(composition_outcome_count_record(rng, curve, relevant_count))
        records.append(safe_intervention_record(rng, curve, relevant_count))
    return records


def generate_v16_dataset(
    n,
    seed=0,
    task_types=V16_TASKS,
    n_components=12,
    width=11,
):
    if tuple(task_types) != V16_TASKS:
        raise ValueError("V1.6 is a matched curve suite; custom --tasks is not supported")
    per_curve = records_per_curve(n_components)
    if n % per_curve:
        raise ValueError(
            f"V1.6 with {n_components} components emits {per_curve} records per curve; "
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
