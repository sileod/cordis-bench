"""V1.2 matched-world capability bundles.

Each bundle shares one world, one quiescent start state, and one initial disable
directive. The six questions form a diagnostic ladder:

    support -> permission -> local dynamics -> global state -> confluence -> act

This lets us ask whether downstream failures remain after the model has shown
that it understands the exact same configuration.
"""

import random

from .core import Action, DEFAULT_THEORY
from .generate import all_active_start, random_values, v1_world
from .render import render_state, render_world, v1_rules
from .semantics import (
    orchestrate,
    quiescent_states,
    settle_plan,
    settling_stats,
    step,
    successors,
    supported_components,
)
from .tasks import LETTERS, distract_vectors, serialize_state, serialize_world, stable_id, v1_metadata
from .v11 import _action_set, _component_set, _set_choices


V12_TASKS = (
    "support_set",
    "legal_actions",
    "next_state",
    "must_unload",
    "dynamic_confluence",
    "robust_intervention",
)


def _rules():
    return v1_rules(DEFAULT_THEORY) + (
        "- Lifecycle priority: if any deactivation is legal, no activation step is legal "
        "until the required withdrawals are cleared.\n"
    )


def _record(bundle, task_type, prompt, choices, answer, capability, **metadata):
    payload = {
        "schema_version": "1.2",
        "bundle_id": bundle["bundle_id"],
        "task_type": task_type,
        "choices": choices,
    }
    return {
        "id": stable_id(payload),
        "schema_version": "1.2",
        "task_type": task_type,
        "prompt": prompt,
        "choices": choices,
        "answer": answer,
        "metadata": v1_metadata(
            bundle["world"],
            bundle["start"],
            benchmark_suite="1.2",
            capability=capability,
            bundle_id=bundle["bundle_id"],
            bundle_stage=V12_TASKS.index(task_type),
            first_directive=bundle["directive"].render(),
            affected=sorted(bundle["affected"]),
            supported_after_directive=sorted(bundle["supported"]),
            n_initial_legal_actions=len(bundle["legal"]),
            **bundle["stats"],
            **metadata,
        ),
    }


def _fixed_first_active_plan(world, start, first, option_count=4):
    """Find second disables with schedule-invariant final active sets.

    The target is structural rather than arithmetic so it remains meaningful on
    both confluent and non-confluent bundles. The harder arithmetic plan task is
    retained in V1/V1.1 as a separate temporal stress test.
    """
    outcomes = {}
    for second in sorted(start.enabled - {first}):
        quiet = settle_plan(
            world,
            start,
            (Action("disable", first), Action("disable", second)),
        )
        active_sets = {state.active for state in quiet}
        if len(active_sets) == 1:
            outcomes[second] = next(iter(active_sets))
    if len(outcomes) < option_count:
        return None

    by_active = {}
    for second, active in outcomes.items():
        by_active.setdefault(active, []).append(second)
    unique = [(active, names[0]) for active, names in by_active.items() if len(names) == 1]
    if not unique:
        return None
    return outcomes, unique


def sample_bundle(rng, desired_confluent=None, attempts=8000, n_components=6, width=4):
    """Sample one episode that supports all six matched V1.2 questions."""
    for _ in range(attempts):
        world = v1_world(rng, n_components=n_components, width=width)
        start = all_active_start(world, random_values(rng, width, world.modulus))
        names = list(start.active)
        rng.shuffle(names)
        for first in names:
            directive = Action("disable", first)
            post = orchestrate(world, start, directive)
            supported = supported_components(world, post)
            affected = frozenset(start.active - supported)
            legal = tuple(action for action, _ in successors(world, post))
            if len(affected) < 2 or len(legal) < 2:
                continue

            stats = settling_stats(world, post)
            if stats["schedule_count"] < 2:
                continue
            confluent = stats["quiescent_observations"] == 1
            if desired_confluent is not None and confluent != desired_confluent:
                continue

            fixed_plan = _fixed_first_active_plan(world, start, first)
            if fixed_plan is None:
                continue
            outcomes, unique_targets = fixed_plan
            target_active, answer_second = rng.choice(unique_targets)
            distractors = [name for name in outcomes if name != answer_second]
            if len(distractors) < 3:
                continue
            rng.shuffle(distractors)
            second_options = [answer_second, *distractors[:3]]
            rng.shuffle(second_options)

            local_action = rng.choice(legal)
            local_next = step(world, post, local_action)
            quiet, _ = quiescent_states(world, post)
            guaranteed = frozenset(
                name for name in start.active if all(name not in state.active for state in quiet)
            )

            bundle_id = stable_id(
                {
                    "schema_version": "1.2",
                    "world": serialize_world(world),
                    "start": serialize_state(start),
                    "directive": directive.render(),
                }
            )
            return {
                "bundle_id": bundle_id,
                "world": world,
                "start": start,
                "directive": directive,
                "post": post,
                "supported": supported,
                "affected": affected,
                "legal": legal,
                "local_action": local_action,
                "local_next": local_next,
                "quiet": quiet,
                "guaranteed": guaranteed,
                "confluent": confluent,
                "stats": stats,
                "target_active": target_active,
                "answer_second": answer_second,
                "second_options": tuple(second_options),
            }
    raise RuntimeError(f"could not sample V1.2 bundle with confluence={desired_confluent}")


def support_record(rng, bundle):
    choices0 = _set_choices(rng, bundle["supported"], bundle["post"].enabled)
    answer = next(letter for letter, value in choices0.items() if value == bundle["supported"])
    choices = {letter: sorted(value) for letter, value in choices0.items()}
    options = "\n".join(f"  {letter}. {_component_set(value)}" for letter, value in choices0.items())
    prompt = f"""{_rules()}{render_world(bundle['world'])}

The system starts quiescent:
{render_state(bundle['start'])}

The orchestrator executes:
  {bundle['directive'].render()}

Current runtime configuration immediately after that directive, before any lifecycle step:
{render_state(bundle['post'])}

Which enabled components are semantically SUPPORTED right now? Support is recursive through providers.
{options}

Answer with only the option letter."""
    return _record(bundle, "support_set", prompt, choices, answer, "state_comprehension")


def legal_actions_record(rng, bundle):
    gold = frozenset(action.render() for action in bundle["legal"])
    universe = [
        f"{kind}({component.name})"
        for kind in ("activate", "deactivate")
        for component in bundle["world"].components
    ]
    choices0 = _set_choices(rng, gold, universe)
    answer = next(letter for letter, value in choices0.items() if value == gold)
    choices = {letter: sorted(value) for letter, value in choices0.items()}
    options = "\n".join(f"  {letter}. {_action_set(value)}" for letter, value in choices0.items())
    prompt = f"""{_rules()}{render_world(bundle['world'])}

The system starts quiescent:
{render_state(bundle['start'])}

The orchestrator executes:
  {bundle['directive'].render()}

Current runtime configuration immediately after that directive:
{render_state(bundle['post'])}

Which set is EXACTLY the legal lifecycle action set right now?
{options}

Answer with only the option letter."""
    return _record(bundle, "legal_actions", prompt, choices, answer, "permission")


def next_state_record(rng, bundle):
    action = bundle["local_action"]
    nxt = bundle["local_next"]
    choices0 = distract_vectors(rng, nxt.values, bundle["world"].modulus)
    answer = next(letter for letter, value in choices0.items() if value == nxt.values)
    choices = {letter: list(value) for letter, value in choices0.items()}
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices0.items())
    prompt = f"""{_rules()}{render_world(bundle['world'])}

The system starts quiescent:
{render_state(bundle['start'])}

The orchestrator executes:
  {bundle['directive'].render()}

Immediately afterward the runtime takes this legal lifecycle step:
  {action.render()}

What is the shared state x immediately after that lifecycle step?
{options}

Answer with only the option letter."""
    return _record(
        bundle,
        "next_state",
        prompt,
        choices,
        answer,
        "dynamics",
        local_action=action.render(),
    )


def must_unload_record(rng, bundle):
    choices0 = _set_choices(rng, bundle["guaranteed"], bundle["start"].active)
    answer = next(letter for letter, value in choices0.items() if value == bundle["guaranteed"])
    choices = {letter: sorted(value) for letter, value in choices0.items()}
    options = "\n".join(f"  {letter}. {_component_set(value)}" for letter, value in choices0.items())
    prompt = f"""{_rules()}{render_world(bundle['world'])}

The system starts quiescent:
{render_state(bundle['start'])}

The orchestrator executes:
  {bundle['directive'].render()}

The runtime may now take lifecycle steps in any legal order until quiescence. Which set contains EXACTLY the components that are active initially but are guaranteed to be inactive in every final quiescent state?
{options}

Answer with only the option letter."""
    return _record(bundle, "must_unload", prompt, choices, answer, "global_state")


def confluence_record(bundle):
    answer = "A" if bundle["confluent"] else "B"
    prompt = f"""{_rules()}{render_world(bundle['world'])}

The system starts quiescent:
{render_state(bundle['start'])}

The orchestrator executes:
  {bundle['directive'].render()}

The runtime then takes lifecycle steps in any legal order until quiescence. Is the final observable outcome (shared state x and active component set) the same for every legal lifecycle schedule?
  A. YES
  B. NO

Answer with only A or B."""
    return _record(
        bundle,
        "dynamic_confluence",
        prompt,
        {"A": "YES", "B": "NO"},
        answer,
        "global",
        confluent=bundle["confluent"],
    )


def robust_intervention_record(bundle):
    options = {LETTERS[i]: name for i, name in enumerate(bundle["second_options"])}
    answer = next(letter for letter, name in options.items() if name == bundle["answer_second"])
    choice_text = "\n".join(f"  {letter}. disable({name})" for letter, name in options.items())
    target = _component_set(bundle["target_active"])
    prompt = f"""{_rules()}{render_world(bundle['world'])}

The system starts quiescent:
{render_state(bundle['start'])}

First, the orchestrator executes:
  {bundle['directive'].render()}

The runtime is allowed to settle to quiescence in ANY legal order. The orchestrator then executes exactly one second directive from the choices below, after which the runtime again settles in any legal order.

Which second directive guarantees that EVERY complete two-stage schedule ends with active component set exactly {target}?
{choice_text}

Answer with only the option letter."""
    return _record(
        bundle,
        "robust_intervention",
        prompt,
        options,
        answer,
        "act",
        target_active=sorted(bundle["target_active"]),
        answer_second=bundle["answer_second"],
    )


def bundle_records(rng, bundle):
    return [
        support_record(rng, bundle),
        legal_actions_record(rng, bundle),
        next_state_record(rng, bundle),
        must_unload_record(rng, bundle),
        confluence_record(bundle),
        robust_intervention_record(bundle),
    ]


def generate_v12_dataset(n, seed=0, task_types=V12_TASKS, n_components=6, width=4):
    if tuple(task_types) != V12_TASKS:
        raise ValueError("V1.2 is a matched-bundle suite; custom --tasks is not supported")
    if n % len(V12_TASKS):
        raise ValueError(f"V1.2 n must be a multiple of {len(V12_TASKS)} to keep bundles complete")

    rng = random.Random(seed)
    records = []
    n_bundles = n // len(V12_TASKS)
    for i in range(n_bundles):
        desired = (i % 2) == 0
        bundle = sample_bundle(
            rng,
            desired_confluent=desired,
            n_components=n_components,
            width=width,
        )
        records.extend(bundle_records(rng, bundle))
    return records
