"""V1.4 relevance-factorial probes for spatiotemporal composability.

V1.4 keeps the paper-level construct broader than algebraic commutativity.  Each
episode has two matched lifecycle branches.  One branch contains a critical pair
whose effects are either commuting or noncommuting; the other contains a
commuting control pair.  We query both branches.

This yields a 2x2 semantic factorial:

    critical pair commutes / does not commute
        x
    critical pair is lifecycle-relevant / lifecycle-irrelevant

The noncommuting mutation changes the queried behavior only when the directive
actually causes the critical pair to participate in competing unload orders.
Component names, dependency keys, component listing order, state coordinates,
and the operation family are randomized across episodes.
"""

import itertools
import random

from .core import Action, Component, DEFAULT_THEORY, Op, RuntimeState, World
from .generate import PRIMES
from .render import render_state, render_world, v1_rules
from .semantics import orchestrate, quiescent_states, replay, settling_stats
from .tasks import distract_vectors, serialize_state, serialize_world, stable_id, v1_metadata
from .v13 import dependency_depth


V14_TASKS = ("relevance_trace", "relevance_confluence")
EFFECT_FAMILIES = ("shear_shear", "swap_swap", "scale_add", "swap_add")


def _rules():
    return v1_rules(DEFAULT_THEORY) + (
        "- Lifecycle priority: if any deactivation is legal, no activation step is legal "
        "until the required withdrawals are cleared.\n"
    )


def _apply_effect(effect, values, modulus):
    for op in effect:
        values = op.apply(values, modulus)
    return values


def effects_commute_exact(effect_a, effect_b, modulus, width):
    """Check commutativity over the complete finite state domain."""
    for values in itertools.product(range(modulus), repeat=width):
        ab = _apply_effect(effect_b, _apply_effect(effect_a, values, modulus), modulus)
        ba = _apply_effect(effect_a, _apply_effect(effect_b, values, modulus), modulus)
        if ab != ba:
            return False
    return True


def _effect_triplet(rng, modulus, width, family=None):
    """Return (left, commuting_right, noncommuting_right, family)."""
    allowed = list(EFFECT_FAMILIES)
    if width < 4:
        allowed.remove("swap_swap")
    family = family or rng.choice(allowed)

    if family == "shear_shear":
        target_a, target_b, source = rng.sample(range(width), 3)
        a = rng.randrange(1, modulus)
        b = rng.randrange(1, modulus)
        left = (Op("shear", target_a, source, a),)
        commuting = (Op("shear", target_b, source, b),)
        noncommuting = (Op("shear", target_b, target_a, b),)
    elif family == "swap_swap":
        a, b, c, d = rng.sample(range(width), 4)
        left = (Op("swap", a, b),)
        commuting = (Op("swap", c, d),)
        noncommuting = (Op("swap", b, c),)
    elif family == "scale_add":
        scaled, other = rng.sample(range(width), 2)
        factor = rng.randrange(2, modulus)
        delta = rng.randrange(1, modulus)
        left = (Op("scale", scaled, value=factor),)
        commuting = (Op("add", other, value=delta),)
        noncommuting = (Op("add", scaled, value=delta),)
    elif family == "swap_add":
        a, b, other = rng.sample(range(width), 3)
        delta = rng.randrange(1, modulus)
        left = (Op("swap", a, b),)
        commuting = (Op("add", other, value=delta),)
        noncommuting = (Op("add", a, value=delta),)
    else:
        raise ValueError(f"unknown effect family: {family}")

    assert effects_commute_exact(left, commuting, modulus, width)
    assert not effects_commute_exact(left, noncommuting, modulus, width)
    return left, commuting, noncommuting, family


def _random_names(rng, n):
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    rng.shuffle(alphabet)
    return alphabet[:n]


def _random_keys(rng, n):
    pool = [f"q{value}" for value in rng.sample(range(100, 999), n)]
    rng.shuffle(pool)
    return pool


def _root_effect(rng, modulus, width):
    return (Op("add", rng.randrange(width), value=rng.randrange(1, modulus)),)


def _build_worlds(rng, n_components=6, width=4):
    if n_components != 6:
        raise ValueError("V1.4 currently uses exactly 6 components to keep the 2x2 branches matched")
    if width < 3:
        raise ValueError("V1.4 requires state width >= 3")

    modulus = rng.choice(PRIMES)
    names = _random_names(rng, 6)
    keys = _random_keys(rng, 6)
    critical_root, critical_left, critical_right, neutral_root, neutral_left, neutral_right = names
    key_by_name = dict(zip(names, keys))

    left_effect, critical_commuting, critical_noncommuting, family = _effect_triplet(
        rng, modulus, width
    )
    neutral_left_effect, neutral_right_effect, _, neutral_family = _effect_triplet(
        rng, modulus, width
    )

    def components(right_effect):
        specs = [
            Component(critical_root, frozenset(), frozenset({key_by_name[critical_root]}), _root_effect(rng, modulus, width)),
            Component(critical_left, frozenset({key_by_name[critical_root]}), frozenset({key_by_name[critical_left]}), left_effect),
            Component(critical_right, frozenset({key_by_name[critical_root]}), frozenset({key_by_name[critical_right]}), right_effect),
            Component(neutral_root, frozenset(), frozenset({key_by_name[neutral_root]}), _root_effect(rng, modulus, width)),
            Component(neutral_left, frozenset({key_by_name[neutral_root]}), frozenset({key_by_name[neutral_left]}), neutral_left_effect),
            Component(neutral_right, frozenset({key_by_name[neutral_root]}), frozenset({key_by_name[neutral_right]}), neutral_right_effect),
        ]
        return specs

    # Root effects must be identical across the semantic counterfactual.  Build
    # the commuting world first, then replace only the critical-right effect.
    base_components = components(critical_commuting)
    variant_components = [
        Component(c.name, c.requires, c.provides, critical_noncommuting if c.name == critical_right else c.effect)
        for c in base_components
    ]

    order = list(range(6))
    rng.shuffle(order)
    base = World(modulus, width, tuple(base_components[i] for i in order))
    variant = World(modulus, width, tuple(variant_components[i] for i in order))
    return {
        "base": base,
        "variant": variant,
        "critical_root": critical_root,
        "critical_pair": tuple(sorted((critical_left, critical_right))),
        "neutral_root": neutral_root,
        "neutral_pair": tuple(sorted((neutral_left, neutral_right))),
        "family": family,
        "neutral_family": neutral_family,
    }


def _terminal_observations(world, post):
    quiet, _ = quiescent_states(world, post)
    return {(state.values, state.active) for state in quiet}


def _branch_trace(pair, root, rng):
    order = list(pair)
    rng.shuffle(order)
    return (
        Action("deactivate", order[0]),
        Action("deactivate", order[1]),
        Action("deactivate", root),
    )


def sample_relevance_episode(rng, attempts=1000, n_components=6, width=4):
    for _ in range(attempts):
        worlds = _build_worlds(rng, n_components=n_components, width=width)
        base = worlds["base"]
        variant = worlds["variant"]
        names = frozenset(component.name for component in base.components)
        start = RuntimeState(
            tuple(rng.randrange(base.modulus) for _ in range(width)),
            names,
            names,
        )
        directives = {
            "relevant": Action("disable", worlds["critical_root"]),
            "irrelevant": Action("disable", worlds["neutral_root"]),
        }
        lifecycles = {
            "relevant": _branch_trace(worlds["critical_pair"], worlds["critical_root"], rng),
            "irrelevant": _branch_trace(worlds["neutral_pair"], worlds["neutral_root"], rng),
        }

        stats = {}
        finals = {}
        terminals = {}
        valid = True
        for relevance in ("relevant", "irrelevant"):
            directive = directives[relevance]
            trace = (directive, *lifecycles[relevance])
            for variant_name, world in (("commuting", base), ("noncommuting", variant)):
                post = orchestrate(world, start, directive)
                key = (variant_name, relevance)
                stats[key] = settling_stats(world, post)
                terminals[key] = _terminal_observations(world, post)
                finals[key] = replay(world, start, trace)
                if stats[key]["schedule_count"] != 2:
                    valid = False
        if not valid:
            continue

        # The critical mutation matters only when its branch is withdrawn.
        if len(terminals[("commuting", "relevant")]) != 1:
            continue
        if len(terminals[("noncommuting", "relevant")]) < 2:
            continue
        if len(terminals[("commuting", "irrelevant")]) != 1:
            continue
        if len(terminals[("noncommuting", "irrelevant")]) != 1:
            continue
        if finals[("commuting", "relevant")].values == finals[("noncommuting", "relevant")].values:
            continue
        if finals[("commuting", "irrelevant")].values != finals[("noncommuting", "irrelevant")].values:
            continue

        episode_id = stable_id({
            "schema_version": "1.4",
            "base_world": serialize_world(base),
            "variant_world": serialize_world(variant),
            "start": serialize_state(start),
            "critical_root": worlds["critical_root"],
            "neutral_root": worlds["neutral_root"],
        })
        return {
            **worlds,
            "episode_id": episode_id,
            "start": start,
            "directives": directives,
            "lifecycles": lifecycles,
            "stats": stats,
            "finals": finals,
            "terminals": terminals,
        }
    raise RuntimeError("could not sample a V1.4 relevance episode")


def _metadata(episode, world, variant_name, relevance, pair_id, probe):
    key = (variant_name, relevance)
    noncommuting = variant_name == "noncommuting"
    is_relevant = relevance == "relevant"
    pair_type = "relevant_interference" if is_relevant else "irrelevant_interference"
    return v1_metadata(
        world,
        episode["start"],
        benchmark_suite="1.4",
        capability="global" if probe == "confluence" else "dynamics",
        relevance_episode_id=episode["episode_id"],
        pair_id=pair_id,
        pair_type=pair_type,
        variant=variant_name,
        interaction_probe=probe,
        interference_present=noncommuting,
        interference_relevant=is_relevant,
        factorial_cell=f"{'noncommuting' if noncommuting else 'commuting'}_{relevance}",
        critical_pair=list(episode["critical_pair"]),
        critical_pair_commutes=not noncommuting,
        queried_root=episode["critical_root"] if is_relevant else episode["neutral_root"],
        effect_family=episode["family"],
        neutral_effect_family=episode["neutral_family"],
        relevant_interfering_pairs=1 if (noncommuting and is_relevant) else 0,
        irrelevant_interfering_pairs=1 if (noncommuting and not is_relevant) else 0,
        dependency_depth=dependency_depth(world),
        effect_ops_total=sum(len(component.effect) for component in world.components),
        schedule_count=episode["stats"][key]["schedule_count"],
        reachable_states=episode["stats"][key]["reachable_states"],
        terminal_observations=len(episode["terminals"][key]),
        first_directive=episode["directives"][relevance].render(),
    )


def _trace_prompt(world, episode, relevance, choices):
    lifecycle = "\n".join(
        f"  {i + 1}. {action.render()}" for i, action in enumerate(episode["lifecycles"][relevance])
    )
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    return f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(episode['start'])}

The orchestrator executes:
  {episode['directives'][relevance].render()}

The following exact legal lifecycle sequence then occurs:
{lifecycle}

What is the final shared state x?
{options}

Answer with only the option letter."""


def _confluence_choices(rng):
    if rng.random() < 0.5:
        return {"A": "YES", "B": "NO"}
    return {"A": "NO", "B": "YES"}


def _confluence_prompt(world, episode, relevance, choices):
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    return f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(episode['start'])}

The orchestrator executes:
  {episode['directives'][relevance].render()}

The runtime then takes lifecycle steps in any legal order until quiescence. Is the final observable outcome (shared state x and active component set) the same for every legal lifecycle schedule?
{options}

Answer with only the option letter."""


def relevance_episode_records(rng, episode):
    records = []
    for relevance in ("relevant", "irrelevant"):
        base_final = episode["finals"][("commuting", relevance)].values
        variant_final = episode["finals"][("noncommuting", relevance)].values
        include = [variant_final] if variant_final != base_final else []
        trace_choices0 = distract_vectors(rng, base_final, episode["base"].modulus, include=include)
        trace_choices = {letter: list(value) for letter, value in trace_choices0.items()}
        trace_pair_id = stable_id({
            "episode": episode["episode_id"], "probe": "trace", "relevance": relevance
        })
        confluence_choices = _confluence_choices(rng)
        confluence_pair_id = stable_id({
            "episode": episode["episode_id"], "probe": "confluence", "relevance": relevance
        })

        for variant_name, world in (("commuting", episode["base"]), ("noncommuting", episode["variant"])):
            final_values = episode["finals"][(variant_name, relevance)].values
            trace_answer = next(letter for letter, value in trace_choices0.items() if value == final_values)
            trace_pair_type = "relevant_interference" if relevance == "relevant" else "irrelevant_interference"
            records.append({
                "id": stable_id({"pair_id": trace_pair_id, "variant": variant_name}),
                "schema_version": "1.4",
                "task_type": "relevance_trace",
                "prompt": _trace_prompt(world, episode, relevance, trace_choices0),
                "choices": trace_choices,
                "answer": trace_answer,
                "metadata": _metadata(episode, world, variant_name, relevance, trace_pair_id, "trace"),
            })

            same = len(episode["terminals"][(variant_name, relevance)]) == 1
            desired = "YES" if same else "NO"
            confluence_answer = next(letter for letter, value in confluence_choices.items() if value == desired)
            records.append({
                "id": stable_id({"pair_id": confluence_pair_id, "variant": variant_name}),
                "schema_version": "1.4",
                "task_type": "relevance_confluence",
                "prompt": _confluence_prompt(world, episode, relevance, confluence_choices),
                "choices": dict(confluence_choices),
                "answer": confluence_answer,
                "metadata": _metadata(episode, world, variant_name, relevance, confluence_pair_id, "confluence"),
            })

        # Relevant mutations must change both queried answers; irrelevant ones must not.
        pair_records = [r for r in records if r["metadata"]["pair_id"] == trace_pair_id]
        if relevance == "relevant":
            assert pair_records[0]["answer"] != pair_records[1]["answer"]
        else:
            assert pair_records[0]["answer"] == pair_records[1]["answer"]

    return records


def generate_v14_dataset(n, seed=0, task_types=V14_TASKS, n_components=6, width=4):
    if tuple(task_types) != V14_TASKS:
        raise ValueError("V1.4 is a matched relevance-factorial suite; custom --tasks is not supported")
    if n % 8:
        raise ValueError("V1.4 n must be a multiple of 8 to keep factorial episodes complete")

    rng = random.Random(seed)
    records = []
    for _ in range(n // 8):
        episode = sample_relevance_episode(rng, n_components=n_components, width=width)
        records.extend(relevance_episode_records(rng, episode))
    return records
