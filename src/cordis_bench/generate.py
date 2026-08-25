import itertools
import random

from .core import Action, Component, DEFAULT_THEORY, Op, RuntimeState, World
from .semantics import (
    canonical_quiescent,
    is_confluent,
    observations,
    orchestrate,
    quiescent_states,
    random_quiescent,
    replay,
    settle_plan,
    settling_stats,
    successors,
)


PRIMES = (5, 7, 11)


def random_op(rng, width, modulus):
    kind = rng.choice(("add", "shear", "scale", "swap"))
    i = rng.randrange(width)
    if kind == "add":
        return Op(kind, i, value=rng.randrange(1, modulus))
    if kind == "scale":
        return Op(kind, i, value=rng.randrange(1, modulus))
    j = rng.randrange(width - 1)
    if j >= i:
        j += 1
    if kind == "swap":
        return Op(kind, i, j)
    return Op(kind, i, j, rng.randrange(1, modulus))


def random_effect(rng, width, modulus, length):
    return tuple(random_op(rng, width, modulus) for _ in range(length))


def sample_world(rng, n_components=4, width=3, effect_length=(1, 3), dependency_prob=0.35):
    modulus = rng.choice(PRIMES)
    components = []
    keys = []
    for i in range(n_components):
        name = chr(ord("A") + i)
        key = f"k{i}"
        requires = {k for k in keys if rng.random() < dependency_prob}
        length = rng.randint(*effect_length)
        components.append(
            Component(
                name=name,
                requires=frozenset(requires),
                provides=frozenset({key}),
                effect=random_effect(rng, width, modulus, length),
            )
        )
        keys.append(key)
    return World(modulus, width, tuple(components))


def random_values(rng, width, modulus):
    return tuple(rng.randrange(modulus) for _ in range(width))


def all_enabled_start(world, values):
    names = frozenset(component.name for component in world.components)
    return RuntimeState(values, names, frozenset())


def all_active_start(world, values, theory=DEFAULT_THEORY):
    start = all_enabled_start(world, values)
    quiet, _ = canonical_quiescent(world, start, theory=theory)
    return quiet


def sample_confluence_case(rng, desired=None, attempts=500, **world_kwargs):
    for _ in range(attempts):
        world = sample_world(rng, **world_kwargs)
        start = all_enabled_start(world, random_values(rng, world.width, world.modulus))
        label = is_confluent(world, start)
        if desired is None or label == desired:
            return world, start, label
    raise RuntimeError(f"could not sample confluence={desired} after {attempts} attempts")


def sample_intervention_case(rng, attempts=1000, **world_kwargs):
    for _ in range(attempts):
        world = sample_world(rng, **world_kwargs)
        start0 = all_enabled_start(world, random_values(rng, world.width, world.modulus))
        start, _ = canonical_quiescent(world, start0)
        if len(start.active) < 3:
            continue

        outcomes = {}
        for name in sorted(start.enabled):
            candidate = RuntimeState(start.values, start.enabled - {name}, start.active)
            quiet, _ = quiescent_states(world, candidate)
            obs = observations(quiet)
            if len(obs) == 1:
                outcomes[name] = next(iter(obs))[0]

        unique = {}
        for name, values in outcomes.items():
            unique.setdefault(values, []).append(name)
        targets = [(values, names[0]) for values, names in unique.items() if len(names) == 1]
        if not targets:
            continue
        target_values, answer = rng.choice(targets)
        return world, start, target_values, answer
    raise RuntimeError(f"could not sample intervention case after {attempts} attempts")


def v1_world(rng, n_components=6, width=4):
    return sample_world(
        rng,
        n_components=n_components,
        width=width,
        effect_length=(2, 4),
        dependency_prob=0.45,
    )


def disable_case(world, start, name, theory=DEFAULT_THEORY):
    post = orchestrate(world, start, Action("disable", name))
    quiet, parent = quiescent_states(world, post, theory=theory)
    return post, quiet, parent


def sample_dynamic_trace_case(rng, attempts=2000, n_components=6, width=4):
    for _ in range(attempts):
        world = v1_world(rng, n_components=n_components, width=width)
        full = all_active_start(world, random_values(rng, width, world.modulus))
        seed_disabled = list(full.active)
        rng.shuffle(seed_disabled)

        for to_enable in seed_disabled:
            partial = orchestrate(world, full, Action("disable", to_enable))
            start, _ = canonical_quiescent(world, partial)
            if len(start.active) < 3:
                continue

            to_disable_candidates = list(start.active)
            rng.shuffle(to_disable_candidates)
            for to_disable in to_disable_candidates:
                trace = [Action("disable", to_disable)]
                state = orchestrate(world, start, trace[-1])
                state, unload = random_quiescent(world, state, rng)
                trace.extend(unload)
                trace.append(Action("enable", to_enable))
                state = orchestrate(world, state, trace[-1])
                state, load = random_quiescent(world, state, rng)
                trace.extend(load)

                n_deactivate = sum(action.kind == "deactivate" for action in trace)
                n_activate = sum(action.kind == "activate" for action in trace)
                if n_deactivate < 2 or n_activate < 1:
                    continue
                if state.values == start.values:
                    continue
                return world, start, tuple(trace), state
    raise RuntimeError("could not sample a dynamic trace case")


def sample_dynamic_confluence_case(rng, desired=None, attempts=5000, n_components=6, width=4):
    for _ in range(attempts):
        world = v1_world(rng, n_components=n_components, width=width)
        start = all_active_start(world, random_values(rng, width, world.modulus))
        names = list(start.active)
        rng.shuffle(names)
        for name in names:
            post = orchestrate(world, start, Action("disable", name))
            stats = settling_stats(world, post)
            if stats["schedule_count"] < 2:
                continue
            label = stats["quiescent_observations"] == 1
            if desired is None or label == desired:
                return world, start, Action("disable", name), post, label, stats
    raise RuntimeError(f"could not sample dynamic confluence={desired}")


def sample_plan_case(rng, attempts=2000, n_components=6, width=4, option_count=4):
    for _ in range(attempts):
        world = v1_world(rng, n_components=n_components, width=width)
        start = all_active_start(world, random_values(rng, width, world.modulus))
        names = sorted(start.enabled)
        outcomes = {}
        for first, second in itertools.permutations(names, 2):
            directives = (Action("disable", first), Action("disable", second))
            quiet = settle_plan(world, start, directives)
            obs = observations(quiet)
            if len(obs) == 1:
                outcomes[(first, second)] = next(iter(obs))[0]
        if len(outcomes) < option_count:
            continue

        by_values = {}
        for plan, values in outcomes.items():
            by_values.setdefault(values, []).append(plan)
        unique = [(values, plans[0]) for values, plans in by_values.items() if len(plans) == 1]
        if not unique:
            continue

        target, answer_plan = rng.choice(unique)
        distractors = [plan for plan in outcomes if plan != answer_plan]
        if len(distractors) < option_count - 1:
            continue
        reversed_plan = tuple(reversed(answer_plan))
        selected = []
        if reversed_plan in distractors:
            selected.append(reversed_plan)
        rest = [plan for plan in distractors if plan not in selected]
        rng.shuffle(rest)
        selected.extend(rest[: option_count - 1 - len(selected)])
        options = [answer_plan, *selected]
        rng.shuffle(options)
        return world, start, target, answer_plan, tuple(options)
    raise RuntimeError("could not sample a two-step plan case")


def replay_under_undo_orders(world, start, trace):
    from .core import Theory

    return {
        order: replay(world, start, trace, theory=Theory(order))
        for order in ("lifo", "fifo")
    }
