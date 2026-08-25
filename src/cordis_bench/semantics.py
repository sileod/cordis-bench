from collections import deque
from functools import lru_cache

from .core import Action, DEFAULT_THEORY, RuntimeState


def apply_ops(values, ops, modulus):
    for op in ops:
        values = op.apply(values, modulus)
    return values


def active_providers(world, state):
    providers = {}
    for name in state.active:
        component = world.by_name[name]
        for key in component.provides:
            providers[key] = name
    return providers


def supported_components(world, state):
    providers = world.provider_by_key
    supported = set()
    changed = True
    while changed:
        changed = False
        for component in world.components:
            if component.name not in state.enabled or component.name in supported:
                continue
            provider_names = [providers.get(key) for key in component.requires]
            if all(name is not None and name in supported for name in provider_names):
                supported.add(component.name)
                changed = True
    return frozenset(supported)


def activation_ready(world, state, name):
    if name not in state.enabled or name in state.active:
        return False
    supported = supported_components(world, state)
    if name not in supported:
        return False
    component = world.by_name[name]
    providers = active_providers(world, state)
    return all(key in providers for key in component.requires)


def must_deactivate(world, state, name):
    return name in state.active and name not in supported_components(world, state)


def has_active_dependent(world, state, provider_name):
    provided = world.by_name[provider_name].provides
    for name in state.active:
        if name == provider_name:
            continue
        if world.by_name[name].requires & provided:
            return True
    return False


def deactivation_ready(world, state, name):
    return must_deactivate(world, state, name) and not has_active_dependent(world, state, name)


def step(world, state, action, theory=DEFAULT_THEORY):
    component = world.by_name[action.component]
    if action.kind == "activate":
        if not activation_ready(world, state, action.component):
            raise ValueError(f"illegal action: {action.render()}")
        values = apply_ops(state.values, component.effect, world.modulus)
        return RuntimeState(values, state.enabled, state.active | {action.component})
    if action.kind == "deactivate":
        if not deactivation_ready(world, state, action.component):
            raise ValueError(f"illegal action: {action.render()}")
        inverse = component.inverse_effect(world.modulus, theory.undo_order)
        values = apply_ops(state.values, inverse, world.modulus)
        return RuntimeState(values, state.enabled, state.active - {action.component})
    raise ValueError(f"unknown lifecycle action: {action.kind}")


def orchestrate(world, state, action):
    if action.component not in world.by_name:
        raise ValueError(f"unknown component: {action.component}")
    if action.kind == "enable":
        if action.component in state.enabled:
            raise ValueError(f"component already enabled: {action.component}")
        return RuntimeState(state.values, state.enabled | {action.component}, state.active)
    if action.kind == "disable":
        if action.component not in state.enabled:
            raise ValueError(f"component already disabled: {action.component}")
        return RuntimeState(state.values, state.enabled - {action.component}, state.active)
    raise ValueError(f"unknown orchestration action: {action.kind}")


def apply_action(world, state, action, theory=DEFAULT_THEORY):
    if action.kind in {"enable", "disable"}:
        return orchestrate(world, state, action)
    return step(world, state, action, theory=theory)


def replay(world, start, actions, theory=DEFAULT_THEORY):
    state = start
    for action in actions:
        state = apply_action(world, state, action, theory=theory)
    return state


def successors(world, state, theory=DEFAULT_THEORY):
    actions = []
    for component in world.components:
        name = component.name
        if deactivation_ready(world, state, name):
            actions.append(Action("deactivate", name))
    if actions:
        return [(action, step(world, state, action, theory=theory)) for action in actions]
    for component in world.components:
        name = component.name
        if activation_ready(world, state, name):
            actions.append(Action("activate", name))
    return [(action, step(world, state, action, theory=theory)) for action in actions]


def quiescent(world, state, theory=DEFAULT_THEORY):
    return not successors(world, state, theory=theory)


def explore(world, start, theory=DEFAULT_THEORY, max_states=50_000):
    seen = {start}
    todo = deque([start])
    parent = {start: None}

    while todo:
        state = todo.popleft()
        for action, nxt in successors(world, state, theory=theory):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = state, action
            todo.append(nxt)
            if len(seen) > max_states:
                raise RuntimeError(f"state-space exceeded {max_states} states")
    return seen, parent


def quiescent_states(world, start, theory=DEFAULT_THEORY, max_states=50_000):
    states, parent = explore(world, start, theory=theory, max_states=max_states)
    return {state for state in states if quiescent(world, state, theory=theory)}, parent


def observations(states):
    return {(state.values, state.active) for state in states}


def is_confluent(world, start, theory=DEFAULT_THEORY):
    states, _ = quiescent_states(world, start, theory=theory)
    return len(observations(states)) == 1


def shortest_trace(parent, target):
    trace = []
    state = target
    while parent[state] is not None:
        prev, action = parent[state]
        trace.append(action)
        state = prev
    return list(reversed(trace))


def canonical_quiescent(world, start, theory=DEFAULT_THEORY):
    state = start
    trace = []
    while True:
        next_steps = successors(world, state, theory=theory)
        if not next_steps:
            return state, trace
        action, state = min(next_steps, key=lambda item: item[0].render())
        trace.append(action)


def random_quiescent(world, start, rng, theory=DEFAULT_THEORY):
    state = start
    trace = []
    while True:
        next_steps = successors(world, state, theory=theory)
        if not next_steps:
            return state, trace
        action, state = rng.choice(next_steps)
        trace.append(action)


def settle_plan(world, start, directives, theory=DEFAULT_THEORY, max_states=50_000):
    frontier = {start}
    for directive in directives:
        next_frontier = set()
        for state in frontier:
            orchestrated = orchestrate(world, state, directive)
            quiet, _ = quiescent_states(world, orchestrated, theory=theory, max_states=max_states)
            next_frontier.update(quiet)
        frontier = next_frontier
    return frontier


def count_quiescent_paths(world, start, theory=DEFAULT_THEORY, cap=1_000_000):
    @lru_cache(None)
    def count(state):
        next_steps = successors(world, state, theory=theory)
        if not next_steps:
            return 1
        total = 0
        for _, nxt in next_steps:
            total += count(nxt)
            if total >= cap:
                return cap
        return total

    return count(start)


def settling_stats(world, start, theory=DEFAULT_THEORY):
    states, _ = explore(world, start, theory=theory)
    quiet = {state for state in states if quiescent(world, state, theory=theory)}
    return {
        "reachable_states": len(states),
        "quiescent_states": len(quiet),
        "quiescent_observations": len(observations(quiet)),
        "schedule_count": count_quiescent_paths(world, start, theory=theory),
    }
