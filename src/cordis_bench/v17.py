"""V1.7 Cordis/harness realization bridge.

V1.7 is intentionally not another synthetic difficulty escalation.  It realizes
one central Cordis mechanism more literally than the modular-vector suites:
context-dependent witnessed effects.  When a package starts, each `set` effect
records the previous harness value as a private rollback witness; when the
package stops, that recorded value is restored.

The bridge combines those witnessed inverses with reactive package dependencies
and nondeterministic teardown.  A full 2x2 design crosses whether a critical
pair writes independent or overlapping harness slots with whether that pair is
actually affected by the queried provider removal.

The surface uses Creator-style lifecycle verbs (`inspect`, `undefine`) but the
oracle remains this repository's small executable model.  V1.7 therefore
provides construct validation for Cordis/harness reasoning without claiming to
execute the production DeepSeek Harness runtime itself.
"""

from dataclasses import dataclass
from functools import lru_cache
import random

from .tasks import LETTERS, stable_id


V17_TASKS = (
    "harness_reconfiguration_confluence",
    "harness_safe_reconfiguration",
)
V17_COMPONENTS = 7
V17_WIDTH = 5
RECORDS_PER_EPISODE = 8


SLOT_VALUES = {
    "memory.mode": ("raw", ("ranked", "summarized", "episodic")),
    "tool.route": ("direct", ("guarded", "sandboxed", "parallel")),
    "context.profile": ("full", ("compact", "filtered", "priority")),
    "skill.policy": ("default", ("strict", "adaptive", "pinned")),
    "trace.sink": ("stdout", ("audit", "buffered", "remote")),
}

DOMAINS = (
    ("memory-core", "memory.api", ("memory-ranker", "memory-pruner")),
    ("tool-core", "tool.api", ("tool-policy", "tool-alias")),
    ("context-core", "context.api", ("context-compactor", "context-filter")),
    ("skill-core", "skill.api", ("skill-router", "skill-guard")),
    ("trace-core", "trace.api", ("trace-buffer", "trace-auditor")),
    ("prompt-core", "prompt.api", ("prompt-overlay", "prompt-sanitizer")),
)


@dataclass(frozen=True)
class HarnessPackage:
    name: str
    requires: frozenset[str]
    provides: frozenset[str]
    writes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class HarnessState:
    slots: tuple[tuple[str, str], ...]
    enabled: frozenset[str]
    active: frozenset[str]
    # package -> ((slot, previous_value), ...)
    witnesses: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


def _slot_dict(state):
    return dict(state.slots)


def _witness_dict(state):
    return {name: dict(entries) for name, entries in state.witnesses}


def _package_map(packages):
    return {package.name: package for package in packages}


def _provider_map(packages):
    providers = {}
    for package in packages:
        for key in package.provides:
            if key in providers:
                raise ValueError(f"V1.7 requires a unique provider for key {key}")
            providers[key] = package.name
    return providers


def _supported(packages, enabled):
    by_name = _package_map(packages)
    providers = _provider_map(packages)

    @lru_cache(None)
    def supported(name):
        if name not in enabled:
            return False
        return all(
            key in providers and supported(providers[key])
            for key in by_name[name].requires
        )

    return frozenset(name for name in enabled if supported(name))


def _legal_stops(packages, state):
    by_name = _package_map(packages)
    supported = _supported(packages, state.enabled)
    actions = []
    for name in state.active:
        if name in supported:
            continue
        provided = by_name[name].provides
        if any(
            by_name[dependent].requires & provided
            for dependent in state.active
            if dependent != name
        ):
            continue
        actions.append(name)
    return tuple(sorted(actions))


def _stop(packages, state, name):
    by_name = _package_map(packages)
    slots = _slot_dict(state)
    witnesses = _witness_dict(state)
    witness = witnesses.pop(name)

    # A package's own effects are rolled back in reverse order.  Every write is
    # restored to the value captured when this package originally started.
    for slot, _value in reversed(by_name[name].writes):
        slots[slot] = witness[slot]

    return HarnessState(
        tuple(sorted(slots.items())),
        state.enabled,
        state.active - {name},
        tuple(
            sorted(
                (package, tuple(sorted(entries.items())))
                for package, entries in witnesses.items()
            )
        ),
    )


def _undefine(state, name):
    return HarnessState(
        state.slots,
        state.enabled - {name},
        state.active,
        state.witnesses,
    )


def _settle(packages, state):
    """Return quiet states, reachable-state count, and exact teardown paths."""

    seen = {state}
    stack = [state]
    quiet = set()

    @lru_cache(None)
    def path_count(current):
        actions = _legal_stops(packages, current)
        if not actions:
            return 1
        return sum(path_count(_stop(packages, current, action)) for action in actions)

    while stack:
        current = stack.pop()
        actions = _legal_stops(packages, current)
        if not actions:
            quiet.add(current)
            continue
        for action in actions:
            successor = _stop(packages, current, action)
            if successor not in seen:
                seen.add(successor)
                stack.append(successor)

    return quiet, len(seen), path_count(state)


def _settle_plan(packages, start, commands):
    """Execute Creator-style commands, settling after every undefine."""

    frontier = {start}
    for kind, name in commands:
        if kind == "inspect":
            continue
        if kind != "undefine":
            raise ValueError(f"unknown V1.7 command: {kind}")
        next_frontier = set()
        for state in frontier:
            quiet, _reachable, _paths = _settle(packages, _undefine(state, name))
            next_frontier.update(quiet)
        frontier = next_frontier
    return frontier


def _observation(state):
    return state.slots, state.active


def _activate(packages, base_slots, startup_order):
    """Construct a quiescent running harness and its witnessed inverses."""

    by_name = _package_map(packages)
    slots = dict(base_slots)
    witnesses = {}
    active = set()

    for name in startup_order:
        package = by_name[name]
        missing = package.requires - frozenset(
            key
            for active_name in active
            for key in by_name[active_name].provides
        )
        if missing:
            raise RuntimeError(f"invalid V1.7 startup order for {name}: {sorted(missing)}")

        previous = {}
        for slot, value in package.writes:
            previous[slot] = slots[slot]
            slots[slot] = value
        witnesses[name] = previous
        active.add(name)

    names = frozenset(by_name)
    return HarnessState(
        tuple(sorted(slots.items())),
        names,
        frozenset(active),
        tuple(
            sorted(
                (name, tuple(sorted(previous.items())))
                for name, previous in witnesses.items()
            )
        ),
    )


def _choose_alt(rng, slot, excluded=()):
    values = [value for value in SLOT_VALUES[slot][1] if value not in excluded]
    return rng.choice(values)


def _serialize_package(package):
    return {
        "name": package.name,
        "requires": sorted(package.requires),
        "provides": sorted(package.provides),
        "writes": [{"slot": slot, "value": value} for slot, value in package.writes],
    }


def _serialize_state(state):
    return {
        "slots": dict(state.slots),
        "enabled": sorted(state.enabled),
        "active": sorted(state.active),
        "witnesses": {
            name: dict(entries) for name, entries in state.witnesses
        },
    }


def _build_episode(rng):
    critical_domain, control_domain, decoy_domain = rng.sample(DOMAINS, 3)
    critical_provider, critical_key, critical_leaves = critical_domain
    control_provider, control_key, control_leaves = control_domain
    decoy = decoy_domain[2][0]

    critical_slot, commuting_slot, control_slot_a, control_slot_b = rng.sample(
        list(SLOT_VALUES), 4
    )
    critical_value_a = _choose_alt(rng, critical_slot)
    critical_value_b_overlap = _choose_alt(
        rng, critical_slot, excluded={critical_value_a}
    )
    critical_value_b_commuting = _choose_alt(rng, commuting_slot)
    control_value_a = _choose_alt(rng, control_slot_a)
    control_value_b = _choose_alt(rng, control_slot_b)

    critical_order = list(critical_leaves)
    control_order = list(control_leaves)
    rng.shuffle(critical_order)
    rng.shuffle(control_order)
    startup_order = (
        critical_provider,
        control_provider,
        decoy,
        *critical_order,
        *control_order,
    )

    base_slots = {slot: baseline for slot, (baseline, _values) in SLOT_VALUES.items()}

    def packages(noncommuting):
        second_critical_write = (
            (critical_slot, critical_value_b_overlap)
            if noncommuting
            else (commuting_slot, critical_value_b_commuting)
        )
        return (
            HarnessPackage(
                critical_provider,
                frozenset(),
                frozenset({critical_key}),
                (),
            ),
            HarnessPackage(
                critical_leaves[0],
                frozenset({critical_key}),
                frozenset({f"{critical_leaves[0]}.ready"}),
                ((critical_slot, critical_value_a),),
            ),
            HarnessPackage(
                critical_leaves[1],
                frozenset({critical_key}),
                frozenset({f"{critical_leaves[1]}.ready"}),
                (second_critical_write,),
            ),
            HarnessPackage(
                control_provider,
                frozenset(),
                frozenset({control_key}),
                (),
            ),
            HarnessPackage(
                control_leaves[0],
                frozenset({control_key}),
                frozenset({f"{control_leaves[0]}.ready"}),
                ((control_slot_a, control_value_a),),
            ),
            HarnessPackage(
                control_leaves[1],
                frozenset({control_key}),
                frozenset({f"{control_leaves[1]}.ready"}),
                ((control_slot_b, control_value_b),),
            ),
            HarnessPackage(
                decoy,
                frozenset(),
                frozenset({f"{decoy}.ready"}),
                (),
            ),
        )

    worlds = {}
    for noncommuting in (False, True):
        world = packages(noncommuting)
        worlds[noncommuting] = {
            "packages": world,
            "start": _activate(world, base_slots, startup_order),
        }

    episode_id = stable_id(
        {
            "schema_version": "1.7",
            "critical_domain": critical_domain,
            "control_domain": control_domain,
            "decoy": decoy,
            "startup_order": startup_order,
            "slots": (
                critical_slot,
                commuting_slot,
                control_slot_a,
                control_slot_b,
            ),
            "values": (
                critical_value_a,
                critical_value_b_overlap,
                critical_value_b_commuting,
                control_value_a,
                control_value_b,
            ),
        }
    )

    return {
        "episode_id": episode_id,
        "critical_domain": critical_domain,
        "control_domain": control_domain,
        "decoy": decoy,
        "startup_order": startup_order,
        "base_slots": base_slots,
        "worlds": worlds,
    }


def _affected_packages(episode, relevant):
    domain = episode["critical_domain"] if relevant else episode["control_domain"]
    provider, _key, leaves = domain
    return frozenset({provider, *leaves})


def _query_provider(episode, relevant):
    domain = episode["critical_domain"] if relevant else episode["control_domain"]
    return domain[0]


def _target_observation(episode, packages, start, relevant):
    affected = _affected_packages(episode, relevant)
    by_name = _package_map(packages)
    target_slots = _slot_dict(start)
    for name in affected:
        for slot, _value in by_name[name].writes:
            target_slots[slot] = episode["base_slots"][slot]
    return tuple(sorted(target_slots.items())), start.active - affected


def _plan_success(packages, start, commands, target):
    frontier = _settle_plan(packages, start, commands)
    observations = {_observation(state) for state in frontier}
    return len(observations) == 1 and next(iter(observations)) == target


def _cell_name(noncommuting, relevant):
    interaction = "noncommuting" if noncommuting else "commuting"
    relevance = "relevant" if relevant else "irrelevant"
    return f"{interaction}_{relevance}"


def _bridge_rules():
    return """Cordis-style harness package semantics (finite realization):
- A package declares required and provided keys.
- A running package is supported only while every required key has a supported provider.
- `set(slot=value)` is a witnessed effect: when the package starts, the runtime records that slot's previous value in that package's private rollback witness, then writes the new value.
- When a package stops, its writes are undone in reverse order by restoring the values recorded in its own rollback witness.
- `undefine(P)` removes P from the defined package set. If P is running, it becomes unsupported and must eventually stop.
- If a provider disappears, running dependents that require it become unsupported recursively.
- A provider cannot stop while a running dependent still requires one of its keys; dependents stop first.
- If several stop actions are legal, the runtime may choose any order.
- `inspect(P)` is read-only and changes nothing.
- After every `undefine` command in a maintenance plan, the runtime settles to quiescence before the next command.
"""


def _render_packages(packages, start):
    witnesses = _witness_dict(start)
    blocks = []
    for package in packages:
        requires = ", ".join(sorted(package.requires)) or "none"
        provides = ", ".join(sorted(package.provides)) or "none"
        lines = [
            f"  {package.name}: requires {{{requires}}}; provides {{{provides}}}"
        ]
        if package.writes:
            for slot, value in package.writes:
                previous = witnesses[package.name][slot]
                lines.append(
                    f"    on start: set({slot}={value}); rollback witness records {slot}={previous}"
                )
        else:
            lines.append("    on start: no harness-state write")
        blocks.append("\n".join(lines))
    return "Packages:\n" + "\n".join(blocks)


def _render_config(state):
    slots = ", ".join(f"{slot}={value}" for slot, value in state.slots)
    active = ", ".join(sorted(state.active))
    return f"Current harness state:\n  slots: {slots}\n  running packages: {{{active}}}"


def _common_metadata(episode, packages, start, noncommuting, relevant, stats):
    cell = _cell_name(noncommuting, relevant)
    return {
        "benchmark_suite": "1.7",
        "capability": None,
        "bridge_episode_id": episode["episode_id"],
        "bridge_mechanism": "context_dependent_witnessed_restore",
        "bridge_surface": "creator_style_package_lifecycle",
        "factorial_cell": cell,
        "interference_present": noncommuting,
        "interference_relevant": relevant,
        "interaction_variant": "noncommuting" if noncommuting else "commuting",
        "query_relevance": "relevant" if relevant else "irrelevant",
        "queried_provider": _query_provider(episode, relevant),
        "startup_order": list(episode["startup_order"]),
        "packages": [_serialize_package(package) for package in packages],
        "start": _serialize_state(start),
        "schedule_count": stats["schedule_count"],
        "reachable_states": stats["reachable_states"],
        "terminal_observations": stats["terminal_observations"],
        "n_components": len(packages),
        "problem_size": len(packages),
        "effect_ops_total": sum(len(package.writes) for package in packages),
    }


def _cell_stats(episode, noncommuting, relevant):
    world = episode["worlds"][noncommuting]
    packages = world["packages"]
    start = world["start"]
    provider = _query_provider(episode, relevant)
    quiet, reachable_states, schedule_count = _settle(
        packages, _undefine(start, provider)
    )
    observations = {_observation(state) for state in quiet}
    return {
        "packages": packages,
        "start": start,
        "provider": provider,
        "quiet": quiet,
        "observations": observations,
        "stats": {
            "schedule_count": schedule_count,
            "reachable_states": reachable_states,
            "terminal_observations": len(observations),
        },
    }


def _binary_choices(rng):
    return {"A": "YES", "B": "NO"} if rng.random() < 0.5 else {"A": "NO", "B": "YES"}


def _confluence_record(rng, episode, noncommuting, relevant):
    cell = _cell_stats(episode, noncommuting, relevant)
    packages = cell["packages"]
    start = cell["start"]
    provider = cell["provider"]
    confluent = len(cell["observations"]) == 1

    expected = not (noncommuting and relevant)
    if confluent != expected:
        raise RuntimeError("V1.7 bridge factorial did not produce the expected confluence cell")
    if cell["stats"]["schedule_count"] != 2:
        raise RuntimeError("V1.7 bridge cells should expose exactly two teardown orders")

    choices = _binary_choices(rng)
    desired = "YES" if confluent else "NO"
    answer = next(letter for letter, text in choices.items() if text == desired)
    prompt = f"""{_bridge_rules()}
{_render_packages(packages, start)}

{_render_config(start)}

Creator-style maintenance command:
  undefine({provider})

The runtime now settles using any legal package-stop order. Is the final observable harness configuration (all slot values and the running-package set) the same for every legal teardown schedule?
  A. {choices['A']}
  B. {choices['B']}

Answer with only the option letter."""

    metadata = _common_metadata(
        episode, packages, start, noncommuting, relevant, cell["stats"]
    )
    metadata.update({"capability": "global", "confluent": confluent})
    return {
        "id": stable_id(
            {
                "schema_version": "1.7",
                "task_type": "harness_reconfiguration_confluence",
                "episode": episode["episode_id"],
                "cell": _cell_name(noncommuting, relevant),
                "choices": choices,
            }
        ),
        "schema_version": "1.7",
        "task_type": "harness_reconfiguration_confluence",
        "prompt": prompt,
        "choices": choices,
        "answer": answer,
        "metadata": metadata,
    }


def _plan_text(commands):
    return "; ".join(f"{kind}({name})" for kind, name in commands)


def _safe_record(rng, episode, noncommuting, relevant):
    cell = _cell_stats(episode, noncommuting, relevant)
    packages = cell["packages"]
    start = cell["start"]
    provider = cell["provider"]
    target = _target_observation(episode, packages, start, relevant)

    direct = (("inspect", provider), ("undefine", provider))
    candidates = [direct]
    for package in sorted(start.enabled - {provider}):
        candidates.append((("undefine", package), ("undefine", provider)))

    successes = [
        plan for plan in candidates if _plan_success(packages, start, plan, target)
    ]
    if noncommuting and relevant:
        critical_leaves = set(episode["critical_domain"][2])
        preferred = [
            plan
            for plan in successes
            if plan != direct and plan[0][1] in critical_leaves
        ]
        if len(preferred) != 1 or direct in successes:
            raise RuntimeError("V1.7 noncommuting-relevant ACT cell is not uniquely prepared")
        correct = preferred[0]
    else:
        if direct not in successes:
            raise RuntimeError("V1.7 benign ACT cell unexpectedly rejects direct withdrawal")
        correct = direct

    distractors = [plan for plan in candidates if plan not in successes]
    if len(distractors) < 3:
        raise RuntimeError("V1.7 could not construct three oracle-failing ACT distractors")
    rng.shuffle(distractors)
    plans = [correct, *distractors[:3]]
    rng.shuffle(plans)

    choices = {LETTERS[index]: _plan_text(plan) for index, plan in enumerate(plans)}
    answer = next(letter for letter, plan in zip(choices, plans) if plan == correct)
    option_success = {
        letter: _plan_success(packages, start, plan, target)
        for letter, plan in zip(choices, plans)
    }
    if sum(option_success.values()) != 1 or not option_success[answer]:
        raise RuntimeError("V1.7 ACT options are not uniquely scored")
    if len({len(plan) for plan in plans}) != 1:
        raise RuntimeError("V1.7 ACT options leaked command count")

    target_slots, target_active = target
    target_slot_text = ", ".join(
        f"{slot}={value}" for slot, value in target_slots
    )
    target_active_text = ", ".join(sorted(target_active))
    option_text = "\n".join(
        f"  {letter}. {text}" for letter, text in choices.items()
    )
    prompt = f"""{_bridge_rules()}
{_render_packages(packages, start)}

{_render_config(start)}

Maintenance goal: remove provider {provider} cleanly. The final observable configuration must be exactly:
  slots: {target_slot_text}
  running packages: {{{target_active_text}}}

Which Creator-style plan is guaranteed to reach that exact configuration for every legal teardown schedule? Every option contains two commands; `inspect` is the read-only padding command described above.
{option_text}

Answer with only the option letter."""

    metadata = _common_metadata(
        episode, packages, start, noncommuting, relevant, cell["stats"]
    )
    metadata.update(
        {
            "capability": "act",
            "target_slots": dict(target_slots),
            "target_active": sorted(target_active),
            "option_success": option_success,
            "option_command_counts": {
                letter: len(plan) for letter, plan in zip(choices, plans)
            },
        }
    )
    return {
        "id": stable_id(
            {
                "schema_version": "1.7",
                "task_type": "harness_safe_reconfiguration",
                "episode": episode["episode_id"],
                "cell": _cell_name(noncommuting, relevant),
                "choices": choices,
            }
        ),
        "schema_version": "1.7",
        "task_type": "harness_safe_reconfiguration",
        "prompt": prompt,
        "choices": choices,
        "answer": answer,
        "metadata": metadata,
    }


def bridge_episode_records(rng, episode):
    records = []
    for noncommuting in (False, True):
        for relevant in (False, True):
            records.append(_confluence_record(rng, episode, noncommuting, relevant))
            records.append(_safe_record(rng, episode, noncommuting, relevant))
    return records


def generate_v17_dataset(
    n,
    seed=0,
    task_types=V17_TASKS,
    n_components=V17_COMPONENTS,
    width=V17_WIDTH,
):
    if tuple(task_types) != V17_TASKS:
        raise ValueError("V1.7 is a matched bridge suite; custom --tasks is not supported")
    if n_components != V17_COMPONENTS or width != V17_WIDTH:
        raise ValueError("V1.7 uses a fixed seven-package, five-slot bridge realization")
    if n % RECORDS_PER_EPISODE:
        raise ValueError(
            f"V1.7 emits {RECORDS_PER_EPISODE} records per bridge episode; "
            f"n must be a multiple of {RECORDS_PER_EPISODE}"
        )

    rng = random.Random(seed)
    records = []
    for _ in range(n // RECORDS_PER_EPISODE):
        records.extend(bridge_episode_records(rng, _build_episode(rng)))
    return records
