import hashlib
import json
import random

from .core import Action, Component, DEFAULT_THEORY, RuntimeState, Theory, World
from .generate import (
    all_enabled_start,
    random_values,
    replay_under_undo_orders,
    sample_confluence_case,
    sample_dynamic_confluence_case,
    sample_dynamic_trace_case,
    sample_intervention_case,
    sample_plan_case,
    sample_world,
)
from .render import (
    render_confluence_task,
    render_dynamic_confluence_task,
    render_dynamic_trace_task,
    render_intervention_task,
    render_plan_task,
    render_trace_task,
)
from .semantics import replay, successors


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
V0_TASKS = ("trace", "confluence", "intervention")
V1_TASKS = ("dynamic_trace", "dynamic_confluence", "plan", "counterfactual", "isomorphism")


def stable_id(payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def distract_vectors(rng, answer, modulus, n=3, include=()):
    choices = {tuple(answer), *(tuple(value) for value in include)}
    width = len(answer)
    while len(choices) < n + 1:
        candidate = list(answer)
        i = rng.randrange(width)
        candidate[i] = (candidate[i] + rng.randrange(1, modulus)) % modulus
        choices.add(tuple(candidate))
    choices = list(choices)
    rng.shuffle(choices)
    return {LETTERS[i]: value for i, value in enumerate(choices)}


def trace_task(rng, n_components=4, width=3):
    for _ in range(200):
        world = sample_world(rng, n_components=n_components, width=width)
        values = random_values(rng, world.width, world.modulus)
        start = all_enabled_start(world, values)
        state = start
        trace = []
        while successors(world, state):
            action, nxt = rng.choice(successors(world, state))
            trace.append(action)
            state = nxt
        if len(trace) < 2:
            continue
        choices = distract_vectors(rng, state.values, world.modulus)
        answer = next(letter for letter, value in choices.items() if value == state.values)
        payload = {
            "task_type": "trace",
            "modulus": world.modulus,
            "values": values,
            "trace": [action.render() for action in trace],
        }
        return {
            "id": stable_id(payload),
            "schema_version": "0.1",
            "task_type": "trace",
            "prompt": render_trace_task(world, start, trace, choices),
            "choices": {k: list(v) for k, v in choices.items()},
            "answer": answer,
            "metadata": {
                "suite_version": 0,
                "n_components": len(world.components),
                "width": world.width,
                "modulus": world.modulus,
                "trace_length": len(trace),
            },
        }
    raise RuntimeError("failed to generate trace task")


def confluence_task(rng, desired=None, n_components=4, width=3):
    world, start, label = sample_confluence_case(
        rng,
        desired=desired,
        n_components=n_components,
        width=width,
    )
    answer = "A" if label else "B"
    payload = {"task_type": "confluence", "world": repr(world), "start": repr(start)}
    return {
        "id": stable_id(payload),
        "schema_version": "0.1",
        "task_type": "confluence",
        "prompt": render_confluence_task(world, start),
        "choices": {"A": "YES", "B": "NO"},
        "answer": answer,
        "metadata": {
            "suite_version": 0,
            "n_components": len(world.components),
            "width": world.width,
            "modulus": world.modulus,
            "confluent": label,
        },
    }


def intervention_task(rng, n_components=4, width=3):
    world, start, target_values, answer_name = sample_intervention_case(
        rng, n_components=n_components, width=width
    )
    names = sorted(start.enabled)
    rng.shuffle(names)
    options = {LETTERS[i]: name for i, name in enumerate(names)}
    answer = next(letter for letter, name in options.items() if name == answer_name)
    payload = {"task_type": "intervention", "world": repr(world), "start": repr(start), "target": target_values}
    return {
        "id": stable_id(payload),
        "schema_version": "0.1",
        "task_type": "intervention",
        "prompt": render_intervention_task(world, start, target_values, options),
        "choices": options,
        "answer": answer,
        "metadata": {
            "suite_version": 0,
            "n_components": len(world.components),
            "width": world.width,
            "modulus": world.modulus,
            "target_values": list(target_values),
        },
    }


def serialize_world(world):
    return {
        "modulus": world.modulus,
        "width": world.width,
        "components": [
            {
                "name": component.name,
                "requires": sorted(component.requires),
                "provides": sorted(component.provides),
                "effect": [op.render() for op in component.effect],
            }
            for component in world.components
        ],
    }


def serialize_state(state):
    return {
        "values": list(state.values),
        "enabled": sorted(state.enabled),
        "active": sorted(state.active),
    }


def v1_metadata(world, start, trace=None, **extra):
    metadata = {
        "suite_version": 1,
        "n_components": len(world.components),
        "width": world.width,
        "modulus": world.modulus,
        "world": serialize_world(world),
        "start": serialize_state(start),
    }
    if trace is not None:
        metadata.update(
            trace_length=len(trace),
            n_enable=sum(action.kind == "enable" for action in trace),
            n_disable=sum(action.kind == "disable" for action in trace),
            n_activate=sum(action.kind == "activate" for action in trace),
            n_deactivate=sum(action.kind == "deactivate" for action in trace),
        )
    metadata.update(extra)
    return metadata


def make_dynamic_trace_record(rng, world, start, trace, theory=DEFAULT_THEORY, choices=None, metadata=None):
    final = replay(world, start, trace, theory=theory)
    choices = choices or distract_vectors(rng, final.values, world.modulus)
    answer = next(letter for letter, value in choices.items() if value == final.values)
    payload = {
        "schema_version": "1.0",
        "task_type": "dynamic_trace",
        "world": serialize_world(world),
        "start": serialize_state(start),
        "trace": [action.render() for action in trace],
        "theory": theory.undo_order,
        "choices": {k: list(v) for k, v in choices.items()},
    }
    return {
        "id": stable_id(payload),
        "schema_version": "1.0",
        "task_type": "dynamic_trace",
        "prompt": render_dynamic_trace_task(world, start, trace, choices, theory),
        "choices": {k: list(v) for k, v in choices.items()},
        "answer": answer,
        "metadata": v1_metadata(
            world,
            start,
            trace,
            theory={"undo_order": theory.undo_order},
            **(metadata or {}),
        ),
    }


def dynamic_trace_task(rng, n_components=6, width=4):
    world, start, trace, _ = sample_dynamic_trace_case(rng, n_components=n_components, width=width)
    return make_dynamic_trace_record(rng, world, start, trace)


def dynamic_confluence_task(rng, desired=None, n_components=6, width=4):
    world, start, directive, _, label, stats = sample_dynamic_confluence_case(
        rng, desired=desired, n_components=n_components, width=width
    )
    answer = "A" if label else "B"
    payload = {
        "schema_version": "1.0",
        "task_type": "dynamic_confluence",
        "world": serialize_world(world),
        "start": serialize_state(start),
        "directive": directive.render(),
    }
    return {
        "id": stable_id(payload),
        "schema_version": "1.0",
        "task_type": "dynamic_confluence",
        "prompt": render_dynamic_confluence_task(world, start, directive, DEFAULT_THEORY),
        "choices": {"A": "YES", "B": "NO"},
        "answer": answer,
        "metadata": v1_metadata(
            world,
            start,
            directive=directive.render(),
            confluent=label,
            **stats,
        ),
    }


def plan_task(rng, n_components=6, width=4):
    world, start, target, answer_plan, plans = sample_plan_case(
        rng, n_components=n_components, width=width
    )
    options = {LETTERS[i]: plan for i, plan in enumerate(plans)}
    answer = next(letter for letter, plan in options.items() if plan == answer_plan)
    payload = {
        "schema_version": "1.0",
        "task_type": "plan",
        "world": serialize_world(world),
        "start": serialize_state(start),
        "target": list(target),
        "options": options,
    }
    return {
        "id": stable_id(payload),
        "schema_version": "1.0",
        "task_type": "plan",
        "prompt": render_plan_task(world, start, target, options, DEFAULT_THEORY),
        "choices": {letter: list(plan) for letter, plan in options.items()},
        "answer": answer,
        "metadata": v1_metadata(
            world,
            start,
            target_values=list(target),
            answer_plan=list(answer_plan),
        ),
    }


def counterfactual_pair(rng, n_components=6, width=4, attempts=1000):
    for _ in range(attempts):
        world, start, trace, _ = sample_dynamic_trace_case(rng, n_components=n_components, width=width)
        finals = replay_under_undo_orders(world, start, trace)
        if finals["lifo"].values == finals["fifo"].values:
            continue
        choices = distract_vectors(
            rng,
            finals["lifo"].values,
            world.modulus,
            include=[finals["fifo"].values],
        )
        pair_payload = {
            "type": "semantic",
            "world": serialize_world(world),
            "start": serialize_state(start),
            "trace": [action.render() for action in trace],
        }
        pair_id = stable_id(pair_payload)
        records = []
        for order in ("lifo", "fifo"):
            record = make_dynamic_trace_record(
                rng,
                world,
                start,
                trace,
                theory=Theory(order),
                choices=choices,
                metadata={"pair_id": pair_id, "pair_type": "semantic", "variant": order},
            )
            record["task_type"] = "counterfactual_trace"
            record["id"] = stable_id({"pair_id": pair_id, "variant": order})
            records.append(record)
        if records[0]["answer"] != records[1]["answer"]:
            return records
    raise RuntimeError("could not sample semantic counterfactual pair")


def rename_case(rng, world, start, trace):
    labels = [f"u{value}" for value in rng.sample(range(10, 99), len(world.components))]
    name_map = {component.name: labels[i] for i, component in enumerate(world.components)}
    all_keys = sorted({key for component in world.components for key in component.provides})
    key_labels = [f"r{value}" for value in rng.sample(range(100, 999), len(all_keys))]
    key_map = dict(zip(all_keys, key_labels))
    components = tuple(
        Component(
            name_map[component.name],
            frozenset(key_map[key] for key in component.requires),
            frozenset(key_map[key] for key in component.provides),
            component.effect,
        )
        for component in world.components
    )
    renamed_world = World(world.modulus, world.width, components)
    renamed_start = RuntimeState(
        start.values,
        frozenset(name_map[name] for name in start.enabled),
        frozenset(name_map[name] for name in start.active),
    )
    renamed_trace = tuple(Action(action.kind, name_map[action.component]) for action in trace)
    return renamed_world, renamed_start, renamed_trace


def isomorphism_pair(rng, n_components=6, width=4):
    world, start, trace, _ = sample_dynamic_trace_case(rng, n_components=n_components, width=width)
    final = replay(world, start, trace)
    choices = distract_vectors(rng, final.values, world.modulus)
    pair_id = stable_id({
        "type": "isomorphism",
        "world": serialize_world(world),
        "start": serialize_state(start),
        "trace": [action.render() for action in trace],
    })
    base = make_dynamic_trace_record(
        rng,
        world,
        start,
        trace,
        choices=choices,
        metadata={"pair_id": pair_id, "pair_type": "isomorphism", "variant": "base"},
    )
    renamed_world, renamed_start, renamed_trace = rename_case(rng, world, start, trace)
    renamed = make_dynamic_trace_record(
        rng,
        renamed_world,
        renamed_start,
        renamed_trace,
        choices=choices,
        metadata={"pair_id": pair_id, "pair_type": "isomorphism", "variant": "renamed"},
    )
    for variant, record in (("base", base), ("renamed", renamed)):
        record["task_type"] = "isomorphic_trace"
        record["id"] = stable_id({"pair_id": pair_id, "variant": variant})
    assert base["answer"] == renamed["answer"]
    return [base, renamed]


def generate_dataset(n, seed=0, task_types=V0_TASKS, n_components=4, width=3):
    rng = random.Random(seed)
    records = []
    confluence_label = False
    for i in range(n):
        task_type = task_types[i % len(task_types)]
        if task_type == "trace":
            record = trace_task(rng, n_components=n_components, width=width)
        elif task_type == "confluence":
            confluence_label = not confluence_label
            record = confluence_task(rng, desired=confluence_label, n_components=n_components, width=width)
        elif task_type == "intervention":
            record = intervention_task(rng, n_components=n_components, width=width)
        else:
            raise ValueError(f"unknown v0 task type: {task_type}")
        records.append(record)
    return records


def generate_v1_dataset(n, seed=0, task_types=V1_TASKS, n_components=6, width=4):
    rng = random.Random(seed)
    records = []
    i = 0
    confluence_label = False
    while len(records) < n:
        task_type = task_types[i % len(task_types)]
        i += 1
        remaining = n - len(records)
        if task_type == "dynamic_trace":
            batch = [dynamic_trace_task(rng, n_components=n_components, width=width)]
        elif task_type == "dynamic_confluence":
            confluence_label = not confluence_label
            batch = [dynamic_confluence_task(
                rng, desired=confluence_label, n_components=n_components, width=width
            )]
        elif task_type == "plan":
            batch = [plan_task(rng, n_components=n_components, width=width)]
        elif task_type == "counterfactual":
            if remaining < 2:
                continue
            batch = counterfactual_pair(rng, n_components=n_components, width=width)
        elif task_type == "isomorphism":
            if remaining < 2:
                continue
            batch = isomorphism_pair(rng, n_components=n_components, width=width)
        else:
            raise ValueError(f"unknown v1 task type: {task_type}")
        records.extend(batch)
    return records
