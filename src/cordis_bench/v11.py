"""V1.1 diagnostic probes for state comprehension and lifecycle permission."""

import random

from .core import Action, DEFAULT_THEORY
from .generate import all_active_start, random_values, v1_world
from .render import render_state, render_world, v1_rules
from .semantics import orchestrate, quiescent_states, successors, supported_components
from .tasks import (
    LETTERS,
    counterfactual_pair,
    dynamic_confluence_task,
    dynamic_trace_task,
    isomorphism_pair,
    plan_task,
    serialize_state,
    serialize_world,
    stable_id,
    v1_metadata,
)


V11_TASKS = (
    "support_set",
    "legal_actions",
    "quiescence",
    "must_unload",
    "possible_next",
    "dynamic_trace",
    "dynamic_confluence",
    "plan",
    "counterfactual",
    "isomorphism",
)


def _rules():
    return v1_rules(DEFAULT_THEORY) + (
        "- Lifecycle priority: if any deactivation is legal, no activation step is legal "
        "until the required withdrawals are cleared.\n"
    )


def _component_set(names):
    return "{" + ", ".join(sorted(names)) + "}" if names else "{}"


def _action_set(actions):
    return "{" + ", ".join(sorted(actions)) + "}" if actions else "{}"


def _set_choices(rng, gold, universe, option_count=4):
    gold = frozenset(gold)
    universe = tuple(sorted(universe))
    candidates = {gold}
    for item in universe:
        if item in gold:
            candidates.add(gold - {item})
        else:
            candidates.add(gold | {item})
    while len(candidates) < option_count:
        candidates.add(frozenset(item for item in universe if rng.random() < 0.5))
    distractors = list(candidates - {gold})
    rng.shuffle(distractors)
    selected = [gold, *distractors[: option_count - 1]]
    rng.shuffle(selected)
    return {LETTERS[i]: value for i, value in enumerate(selected)}


def _record(task_type, prompt, choices, answer, metadata, payload):
    return {
        "id": stable_id({"schema_version": "1.1", "task_type": task_type, **payload}),
        "schema_version": "1.1",
        "task_type": task_type,
        "prompt": prompt,
        "choices": choices,
        "answer": answer,
        "metadata": {**metadata, "benchmark_suite": "1.1"},
    }


def sample_permission_case(rng, attempts=5000, n_components=6, width=4, min_affected=2):
    """Post-disable state with a nontrivial recursive support cascade."""
    for _ in range(attempts):
        world = v1_world(rng, n_components=n_components, width=width)
        start = all_active_start(world, random_values(rng, width, world.modulus))
        names = list(start.active)
        rng.shuffle(names)
        for name in names:
            directive = Action("disable", name)
            post = orchestrate(world, start, directive)
            supported = supported_components(world, post)
            affected = frozenset(start.active - supported)
            legal = tuple(action for action, _ in successors(world, post))
            if len(affected) >= min_affected and legal:
                return world, start, directive, post, supported, affected, legal
    raise RuntimeError("could not sample a nontrivial permission case")


def sample_quiescence_case(rng, desired=None, attempts=5000, n_components=6, width=4):
    for _ in range(attempts):
        world, _, _, post, _, _, _ = sample_permission_case(
            rng, attempts=200, n_components=n_components, width=width
        )
        quiet, _ = quiescent_states(world, post)
        if desired is True:
            return world, rng.choice(tuple(quiet)), True, 0
        state = post
        steps = 0
        for _ in range(rng.randint(0, 3)):
            nxt = successors(world, state)
            if not nxt:
                break
            _, state = rng.choice(nxt)
            steps += 1
        label = not successors(world, state)
        if desired is None or label == desired:
            return world, state, label, steps
    raise RuntimeError(f"could not sample quiescence={desired}")


def support_set_task(rng, n_components=6, width=4):
    world, _, directive, state, supported, affected, legal = sample_permission_case(
        rng, n_components=n_components, width=width
    )
    choices = _set_choices(rng, supported, state.enabled)
    answer = next(k for k, value in choices.items() if value == supported)
    options = "\n".join(f"  {k}. {_component_set(v)}" for k, v in choices.items())
    prompt = f"""{_rules()}{render_world(world)}

Current runtime configuration:
{render_state(state)}

Ignoring lifecycle steps that may happen next, which enabled components are semantically SUPPORTED in this configuration? Support is recursive through providers.
{options}

Answer with only the option letter."""
    return _record(
        "support_set",
        prompt,
        {k: sorted(v) for k, v in choices.items()},
        answer,
        v1_metadata(
            world, state, capability="state_comprehension",
            originating_directive=directive.render(), supported=sorted(supported),
            affected=sorted(affected), n_legal_actions=len(legal),
        ),
        {"world": serialize_world(world), "state": serialize_state(state), "choices": {k: sorted(v) for k, v in choices.items()}},
    )


def legal_actions_task(rng, n_components=6, width=4):
    world, _, directive, state, supported, affected, legal = sample_permission_case(
        rng, n_components=n_components, width=width
    )
    gold = frozenset(action.render() for action in legal)
    universe = [f"activate({c.name})" for c in world.components] + [f"deactivate({c.name})" for c in world.components]
    choices = _set_choices(rng, gold, universe)
    answer = next(k for k, value in choices.items() if value == gold)
    options = "\n".join(f"  {k}. {_action_set(v)}" for k, v in choices.items())
    prompt = f"""{_rules()}{render_world(world)}

Current runtime configuration:
{render_state(state)}

Which set is EXACTLY the legal lifecycle action set right now? External enable/disable directives are not lifecycle actions.
{options}

Answer with only the option letter."""
    return _record(
        "legal_actions", prompt, {k: sorted(v) for k, v in choices.items()}, answer,
        v1_metadata(
            world, state, capability="permission", originating_directive=directive.render(),
            supported=sorted(supported), affected=sorted(affected),
            legal_actions=sorted(gold), n_legal_actions=len(gold),
        ),
        {"world": serialize_world(world), "state": serialize_state(state), "choices": {k: sorted(v) for k, v in choices.items()}},
    )


def quiescence_task(rng, desired=None, n_components=6, width=4):
    world, state, label, steps = sample_quiescence_case(
        rng, desired=desired, n_components=n_components, width=width
    )
    answer = "A" if label else "B"
    prompt = f"""{_rules()}{render_world(world)}

Current runtime configuration:
{render_state(state)}

Is this runtime configuration quiescent, meaning that no lifecycle activation or deactivation action is currently legal?
  A. YES
  B. NO

Answer with only A or B."""
    return _record(
        "quiescence", prompt, {"A": "YES", "B": "NO"}, answer,
        v1_metadata(
            world, state, capability="permission", quiescent=label,
            sampled_lifecycle_steps=steps, n_legal_actions=len(successors(world, state)),
        ),
        {"world": serialize_world(world), "state": serialize_state(state)},
    )


def must_unload_task(rng, n_components=6, width=4):
    for _ in range(1000):
        world, start, directive, post, _, affected, legal = sample_permission_case(
            rng, n_components=n_components, width=width
        )
        if len(legal) >= 2:
            break
    else:
        raise RuntimeError("could not sample a branching must-unload case")
    quiet, _ = quiescent_states(world, post)
    guaranteed = frozenset(
        name for name in start.active if all(name not in state.active for state in quiet)
    )
    choices = _set_choices(rng, guaranteed, start.active)
    answer = next(k for k, value in choices.items() if value == guaranteed)
    options = "\n".join(f"  {k}. {_component_set(v)}" for k, v in choices.items())
    prompt = f"""{_rules()}{render_world(world)}

The system starts quiescent:
{render_state(start)}

The orchestrator now executes:
  {directive.render()}

The runtime may then take lifecycle steps in any legal order until quiescence. Which set contains EXACTLY the components that are active now but are guaranteed to be inactive in every final quiescent state?
{options}

Answer with only the option letter."""
    return _record(
        "must_unload", prompt, {k: sorted(v) for k, v in choices.items()}, answer,
        v1_metadata(
            world, start, capability="global_state", directive=directive.render(),
            affected=sorted(affected), guaranteed_unload=sorted(guaranteed),
            n_initial_legal_actions=len(legal), quiescent_states=len(quiet),
        ),
        {"world": serialize_world(world), "start": serialize_state(start), "directive": directive.render(), "choices": {k: sorted(v) for k, v in choices.items()}},
    )


def possible_next_task(rng, desired=None, n_components=6, width=4):
    world, _, directive, state, _, _, legal = sample_permission_case(
        rng, n_components=n_components, width=width
    )
    legal_set = {action.render() for action in legal}
    candidates = [Action("deactivate", name) for name in sorted(state.active)]
    yes = [a for a in candidates if a.render() in legal_set]
    no = [a for a in candidates if a.render() not in legal_set]
    if desired is None:
        desired = rng.choice((True, False)) if no else True
    pool = yes if desired else no
    if not pool:
        return possible_next_task(rng, desired=not desired, n_components=n_components, width=width)
    action = rng.choice(pool)
    answer = "A" if desired else "B"
    prompt = f"""{_rules()}{render_world(world)}

Current runtime configuration:
{render_state(state)}

Can the following lifecycle action be the very next lifecycle step under the supplied semantics?
  {action.render()}

  A. YES
  B. NO

Answer with only A or B."""
    return _record(
        "possible_next", prompt, {"A": "YES", "B": "NO"}, answer,
        v1_metadata(
            world, state, capability="permission", originating_directive=directive.render(),
            candidate_action=action.render(), possible_next=desired,
            legal_actions=sorted(legal_set),
        ),
        {"world": serialize_world(world), "state": serialize_state(state), "candidate": action.render()},
    )


def _v1_batch(rng, task_type, remaining, n_components, width, confluence_label):
    if task_type == "dynamic_trace":
        return [dynamic_trace_task(rng, n_components=n_components, width=width)], confluence_label
    if task_type == "dynamic_confluence":
        confluence_label = not confluence_label
        return [dynamic_confluence_task(rng, desired=confluence_label, n_components=n_components, width=width)], confluence_label
    if task_type == "plan":
        return [plan_task(rng, n_components=n_components, width=width)], confluence_label
    if task_type == "counterfactual":
        return (counterfactual_pair(rng, n_components=n_components, width=width) if remaining >= 2 else []), confluence_label
    if task_type == "isomorphism":
        return (isomorphism_pair(rng, n_components=n_components, width=width) if remaining >= 2 else []), confluence_label
    raise ValueError(f"unknown V1 task type: {task_type}")


def generate_v11_dataset(n, seed=0, task_types=V11_TASKS, n_components=6, width=4):
    rng = random.Random(seed)
    records = []
    i = 0
    confluence_label = False
    binary = {"quiescence": False, "possible_next": False}
    while len(records) < n:
        task_type = task_types[i % len(task_types)]
        i += 1
        remaining = n - len(records)
        if task_type == "support_set":
            batch = [support_set_task(rng, n_components=n_components, width=width)]
        elif task_type == "legal_actions":
            batch = [legal_actions_task(rng, n_components=n_components, width=width)]
        elif task_type == "quiescence":
            binary[task_type] = not binary[task_type]
            batch = [quiescence_task(rng, desired=binary[task_type], n_components=n_components, width=width)]
        elif task_type == "must_unload":
            batch = [must_unload_task(rng, n_components=n_components, width=width)]
        elif task_type == "possible_next":
            binary[task_type] = not binary[task_type]
            batch = [possible_next_task(rng, desired=binary[task_type], n_components=n_components, width=width)]
        else:
            batch, confluence_label = _v1_batch(
                rng, task_type, remaining, n_components, width, confluence_label
            )
            if not batch:
                continue
        for record in batch:
            record.setdefault("metadata", {})["benchmark_suite"] = "1.1"
        records.extend(batch[:remaining])
    return records
