"""V1.3 matched interaction counterfactuals.

The V1.3 suite makes semantic interference the manipulated variable.  Each
interaction episode contains two worlds that are identical except for one
source/target swap in a shear operation.  In the control world the two critical
leaf effects commute; in the counterfactual world they do not.  The dependency
graph, component count, operation count/types, initial state, legal schedules,
and answer format are held fixed.

Every episode yields two paired probes under both variants:

    interaction_trace       -- execute one fixed legal unload order
    interaction_confluence  -- decide whether all legal unload orders agree

Pairs are retained only when the semantic mutation changes the oracle answer.
"""

import itertools
import random

from .core import Action, Component, DEFAULT_THEORY, Op, RuntimeState, World
from .generate import PRIMES
from .render import render_state, render_world, v1_rules
from .semantics import orchestrate, quiescent_states, replay, settling_stats
from .tasks import (
    LETTERS,
    distract_vectors,
    serialize_state,
    serialize_world,
    stable_id,
    v1_metadata,
)


V13_TASKS = ("interaction_trace", "interaction_confluence")


def _rules():
    return v1_rules(DEFAULT_THEORY) + (
        "- Lifecycle priority: if any deactivation is legal, no activation step is legal "
        "until the required withdrawals are cleared.\n"
    )


def _apply_effect(component, values, modulus):
    for op in component.effect:
        values = op.apply(values, modulus)
    return values


def effects_commute_exact(world, left, right):
    """Exact finite-domain commutativity check for two component effects."""
    a = world.by_name[left]
    b = world.by_name[right]
    for values in itertools.product(range(world.modulus), repeat=world.width):
        ab = _apply_effect(b, _apply_effect(a, values, world.modulus), world.modulus)
        ba = _apply_effect(a, _apply_effect(b, values, world.modulus), world.modulus)
        if ab != ba:
            return False
    return True


def dependency_depth(world):
    """Longest provider -> dependent path in the generated dependency DAG."""
    provider = world.provider_by_key
    parents = {
        component.name: {provider[key] for key in component.requires}
        for component in world.components
    }
    memo = {}

    def depth(name):
        if name in memo:
            return memo[name]
        if not parents[name]:
            memo[name] = 0
        else:
            memo[name] = 1 + max(depth(parent) for parent in parents[name])
        return memo[name]

    return max((depth(component.name) for component in world.components), default=0)


def _effect_ops_total(world):
    return sum(len(component.effect) for component in world.components)


def _terminal_values(world, post):
    quiet, _ = quiescent_states(world, post)
    return {state.values for state in quiet}


def _interaction_worlds(rng, n_components=6, width=4):
    if n_components < 3:
        raise ValueError("V1.3 requires at least 3 components")
    if n_components > 26:
        raise ValueError("V1.3 currently supports at most 26 components")
    if width < 3:
        raise ValueError("V1.3 requires state width >= 3")

    modulus = rng.choice(PRIMES)
    names = [chr(ord("A") + i) for i in range(n_components)]
    root, left, right = rng.sample(names, 3)
    key_by_name = {name: f"k{i}" for i, name in enumerate(names)}
    root_key = key_by_name[root]

    left_coeff = rng.randrange(1, modulus)
    right_coeff = rng.randrange(1, modulus)
    pad = 2

    base_components = []
    variant_components = []
    mutation = None

    for name in names:
        provides = frozenset({key_by_name[name]})
        if name == root:
            requires = frozenset()
            effect = (
                Op("add", pad, value=rng.randrange(1, modulus)),
                Op("add", pad, value=rng.randrange(1, modulus)),
            )
            base_component = variant_component = Component(name, requires, provides, effect)
        elif name == left:
            requires = frozenset({root_key})
            effect = (
                Op("shear", 0, 1, left_coeff),
                Op("add", pad, value=rng.randrange(1, modulus)),
            )
            base_component = variant_component = Component(name, requires, provides, effect)
        elif name == right:
            requires = frozenset({root_key})
            pad_add = rng.randrange(1, modulus)
            base_effect = (
                Op("shear", 0, 1, right_coeff),
                Op("add", pad, value=pad_add),
            )
            variant_effect = (
                Op("shear", 1, 0, right_coeff),
                Op("add", pad, value=pad_add),
            )
            base_component = Component(name, requires, provides, base_effect)
            variant_component = Component(name, requires, provides, variant_effect)
            mutation = (
                f"{name}: x0 += {right_coeff}*x1 -> "
                f"x1 += {right_coeff}*x0"
            )
        else:
            requires = frozenset()
            effect = (
                Op("add", pad, value=rng.randrange(1, modulus)),
                Op("add", pad, value=rng.randrange(1, modulus)),
            )
            base_component = variant_component = Component(name, requires, provides, effect)

        base_components.append(base_component)
        variant_components.append(variant_component)

    return (
        World(modulus, width, tuple(base_components)),
        World(modulus, width, tuple(variant_components)),
        root,
        left,
        right,
        mutation,
    )


def sample_interaction_episode(rng, attempts=2000, n_components=6, width=4):
    """Sample a matched commute/noncommute episode with a verified answer flip."""
    for _ in range(attempts):
        base, variant, root, left, right, mutation = _interaction_worlds(
            rng, n_components=n_components, width=width
        )
        values = tuple(rng.randrange(base.modulus) for _ in range(width))
        names = frozenset(component.name for component in base.components)
        start = RuntimeState(values, names, names)
        directive = Action("disable", root)
        base_post = orchestrate(base, start, directive)
        variant_post = orchestrate(variant, start, directive)

        base_stats = settling_stats(base, base_post)
        variant_stats = settling_stats(variant, variant_post)
        if base_stats["schedule_count"] != variant_stats["schedule_count"]:
            continue
        if base_stats["schedule_count"] < 2:
            continue
        if base_stats["quiescent_observations"] != 1:
            continue
        if variant_stats["quiescent_observations"] < 2:
            continue
        if not effects_commute_exact(base, left, right):
            continue
        if effects_commute_exact(variant, left, right):
            continue

        critical_order = [left, right]
        rng.shuffle(critical_order)
        lifecycle = tuple(
            [Action("deactivate", critical_order[0]), Action("deactivate", critical_order[1]), Action("deactivate", root)]
        )
        full_trace = (directive, *lifecycle)
        base_final = replay(base, start, full_trace)
        variant_final = replay(variant, start, full_trace)
        if base_final.values == variant_final.values:
            continue

        base_terminal = _terminal_values(base, base_post)
        variant_terminal = _terminal_values(variant, variant_post)
        if len(base_terminal) != 1 or len(variant_terminal) < 2:
            continue

        episode_id = stable_id(
            {
                "schema_version": "1.3",
                "base_world": serialize_world(base),
                "variant_world": serialize_world(variant),
                "start": serialize_state(start),
                "directive": directive.render(),
            }
        )
        return {
            "episode_id": episode_id,
            "base": base,
            "variant": variant,
            "start": start,
            "directive": directive,
            "lifecycle": lifecycle,
            "base_final": base_final,
            "variant_final": variant_final,
            "base_stats": base_stats,
            "variant_stats": variant_stats,
            "base_terminal": base_terminal,
            "variant_terminal": variant_terminal,
            "root": root,
            "critical_pair": tuple(sorted((left, right))),
            "mutation": mutation,
        }
    raise RuntimeError("could not sample a V1.3 interaction episode")


def _metadata(episode, world, variant_name, pair_id, probe):
    is_counterfactual = variant_name == "noncommuting"
    stats = episode["variant_stats"] if is_counterfactual else episode["base_stats"]
    terminal = episode["variant_terminal"] if is_counterfactual else episode["base_terminal"]
    critical_pair = list(episode["critical_pair"])
    noncommuting = [critical_pair] if is_counterfactual else []
    return v1_metadata(
        world,
        episode["start"],
        benchmark_suite="1.3",
        capability="global" if probe == "confluence" else "dynamics",
        interaction_episode_id=episode["episode_id"],
        pair_id=pair_id,
        pair_type="interaction",
        variant=variant_name,
        interaction_probe=probe,
        critical_pair=critical_pair,
        critical_pair_commutes=not is_counterfactual,
        noncommuting_pairs=noncommuting,
        n_noncommuting_pairs=len(noncommuting),
        conflict_density=(
            len(noncommuting) / (len(world.components) * (len(world.components) - 1) / 2)
            if len(world.components) > 1 else 0.0
        ),
        dependency_depth=dependency_depth(world),
        effect_ops_total=_effect_ops_total(world),
        schedule_count=stats["schedule_count"],
        reachable_states=stats["reachable_states"],
        terminal_observations=len(terminal),
        mutation=episode["mutation"],
        first_directive=episode["directive"].render(),
    )


def _trace_prompt(world, episode, choices):
    trace = "\n".join(
        f"  {i + 1}. {action.render()}" for i, action in enumerate(episode["lifecycle"])
    )
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    return f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(episode['start'])}

The orchestrator executes:
  {episode['directive'].render()}

The following exact legal lifecycle sequence then occurs:
{trace}

What is the final shared state x?
{options}

Answer with only the option letter."""


def interaction_trace_pair(rng, episode):
    base = episode["base"]
    variant = episode["variant"]
    choices0 = distract_vectors(
        rng,
        episode["base_final"].values,
        base.modulus,
        include=[episode["variant_final"].values],
    )
    choices = {letter: list(value) for letter, value in choices0.items()}
    pair_id = stable_id({"interaction_episode_id": episode["episode_id"], "probe": "trace"})
    records = []
    for variant_name, world, final in (
        ("commuting", base, episode["base_final"]),
        ("noncommuting", variant, episode["variant_final"]),
    ):
        answer = next(letter for letter, value in choices0.items() if value == final.values)
        records.append(
            {
                "id": stable_id({"pair_id": pair_id, "variant": variant_name}),
                "schema_version": "1.3",
                "task_type": "interaction_trace",
                "prompt": _trace_prompt(world, episode, choices0),
                "choices": choices,
                "answer": answer,
                "metadata": _metadata(episode, world, variant_name, pair_id, "trace"),
            }
        )
    assert records[0]["answer"] != records[1]["answer"]
    return records


def _confluence_prompt(world, episode):
    return f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(episode['start'])}

The orchestrator executes:
  {episode['directive'].render()}

The runtime then takes lifecycle steps in any legal order until quiescence. Is the final observable outcome (shared state x and active component set) the same for every legal lifecycle schedule?
  A. YES
  B. NO

Answer with only A or B."""


def interaction_confluence_pair(episode):
    pair_id = stable_id({"interaction_episode_id": episode["episode_id"], "probe": "confluence"})
    records = []
    for variant_name, world, answer in (
        ("commuting", episode["base"], "A"),
        ("noncommuting", episode["variant"], "B"),
    ):
        records.append(
            {
                "id": stable_id({"pair_id": pair_id, "variant": variant_name}),
                "schema_version": "1.3",
                "task_type": "interaction_confluence",
                "prompt": _confluence_prompt(world, episode),
                "choices": {"A": "YES", "B": "NO"},
                "answer": answer,
                "metadata": _metadata(episode, world, variant_name, pair_id, "confluence"),
            }
        )
    return records


def interaction_episode_records(rng, episode):
    return [
        *interaction_trace_pair(rng, episode),
        *interaction_confluence_pair(episode),
    ]


def generate_v13_dataset(n, seed=0, task_types=V13_TASKS, n_components=6, width=4):
    if tuple(task_types) != V13_TASKS:
        raise ValueError("V1.3 is a matched interaction suite; custom --tasks is not supported")
    if n % 4:
        raise ValueError("V1.3 n must be a multiple of 4 to keep interaction episodes complete")

    rng = random.Random(seed)
    records = []
    for _ in range(n // 4):
        episode = sample_interaction_episode(rng, n_components=n_components, width=width)
        records.extend(interaction_episode_records(rng, episode))
    return records
