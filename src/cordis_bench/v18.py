"""V1.8: paper-facing composite with an actual Cordis-native validation slice.

V1.8 preserves the exact V1.7 controlled core and harness bridge and adds a
third component, ``cordis_native``. Native cases compile a small witnessed-
effect/dependency scenario to real Cordis plugins, execute two controlled
teardown interleavings on the pinned Cordis runtime, and retain a case only
when the runtime observations agree with an independent finite reference
semantics.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import subprocess
import tempfile

from .tasks import LETTERS, stable_id
from .v17_suite import V17_DEFAULT_N, V17_TASKS, generate_v17_dataset

CORDIS_PACKAGE = "cordis"
CORDIS_VERSION = "4.0.0-rc.7"
CORDIS_SOURCE_COMMIT = "56b3d4f725681cf4556c1a8695a709cc3b6eed74"
V18_NATIVE_COMPONENT = "cordis_native"
V18_NATIVE_TASKS = (
    "cordis_native_reconfiguration_confluence",
    "cordis_native_safe_reconfiguration",
)
V18_TASKS = V17_TASKS + V18_NATIVE_TASKS
V18_NATIVE_EPISODES = 8
V18_NATIVE_RECORDS = V18_NATIVE_EPISODES * 8
V18_DEFAULT_N = V17_DEFAULT_N + V18_NATIVE_RECORDS

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


def _choose_alt(rng, slot, excluded=()):
    values = [value for value in SLOT_VALUES[slot][1] if value not in excluded]
    return rng.choice(values)


def _canonical_observation(slots, active):
    return {
        "slots": dict(sorted(slots.items())),
        "active": sorted(active),
    }


def _cell_name(noncommuting, query_relevant):
    return (
        ("noncommuting" if noncommuting else "commuting")
        + "_"
        + ("relevant" if query_relevant else "irrelevant")
    )


def _build_episode(rng):
    critical_domain, control_domain, decoy_domain = rng.sample(DOMAINS, 3)
    critical_provider, critical_service, critical_leaves = critical_domain
    control_provider, control_service, control_leaves = control_domain
    decoy = decoy_domain[2][0]

    critical_slot, commuting_slot, control_slot_a, control_slot_b = rng.sample(
        list(SLOT_VALUES), 4
    )
    critical_value_a = _choose_alt(rng, critical_slot)
    critical_value_b_overlap = _choose_alt(rng, critical_slot, {critical_value_a})
    critical_value_b_commuting = _choose_alt(rng, commuting_slot)
    control_value_a = _choose_alt(rng, control_slot_a)
    control_value_b = _choose_alt(rng, control_slot_b)

    critical_order = list(critical_leaves)
    control_order = list(control_leaves)
    rng.shuffle(critical_order)
    rng.shuffle(control_order)
    startup_order = tuple((*critical_order, *control_order))
    base_slots = {
        slot: baseline for slot, (baseline, _alternatives) in SLOT_VALUES.items()
    }

    episode_id = stable_id(
        {
            "schema_version": "1.8-native",
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
        "critical_slot": critical_slot,
        "commuting_slot": commuting_slot,
        "control_slots": (control_slot_a, control_slot_b),
        "values": {
            "critical_a": critical_value_a,
            "critical_b_overlap": critical_value_b_overlap,
            "critical_b_commuting": critical_value_b_commuting,
            "control_a": control_value_a,
            "control_b": control_value_b,
        },
    }


def _leaf_defs(episode, noncommuting):
    _cp, cs, critical = episode["critical_domain"]
    _qp, qs, control = episode["control_domain"]
    return [
        {
            "name": critical[0],
            "service": cs,
            "slot": episode["critical_slot"],
            "value": episode["values"]["critical_a"],
            "branch": "critical",
        },
        {
            "name": critical[1],
            "service": cs,
            "slot": (
                episode["critical_slot"]
                if noncommuting
                else episode["commuting_slot"]
            ),
            "value": (
                episode["values"]["critical_b_overlap"]
                if noncommuting
                else episode["values"]["critical_b_commuting"]
            ),
            "branch": "critical",
        },
        {
            "name": control[0],
            "service": qs,
            "slot": episode["control_slots"][0],
            "value": episode["values"]["control_a"],
            "branch": "control",
        },
        {
            "name": control[1],
            "service": qs,
            "slot": episode["control_slots"][1],
            "value": episode["values"]["control_b"],
            "branch": "control",
        },
    ]


def _startup(episode, leaves):
    slots = dict(episode["base_slots"])
    witnesses = {}
    by_name = {leaf["name"]: leaf for leaf in leaves}
    for name in episode["startup_order"]:
        leaf = by_name[name]
        witnesses[name] = {leaf["slot"]: slots[leaf["slot"]]}
        slots[leaf["slot"]] = leaf["value"]
    active = {
        episode["critical_domain"][0],
        episode["control_domain"][0],
        episode["decoy"],
        *(leaf["name"] for leaf in leaves),
    }
    return slots, witnesses, active


def _restore(slots, witnesses, leaves_by_name, name):
    leaf = leaves_by_name[name]
    slots[leaf["slot"]] = witnesses[name][leaf["slot"]]


def _simulate_plan(episode, leaves, plan, schedule_order):
    slots, witnesses, active = _startup(episode, leaves)
    leaves_by_name = {leaf["name"]: leaf for leaf in leaves}
    provider_to_branch = {
        episode["critical_domain"][0]: "critical",
        episode["control_domain"][0]: "control",
    }

    for command in plan:
        kind = command["kind"]
        name = command["name"]
        if kind == "inspect":
            continue
        if kind != "dispose":
            raise ValueError(f"unknown plan command: {kind}")
        if name not in active:
            continue
        if name in leaves_by_name:
            _restore(slots, witnesses, leaves_by_name, name)
            active.remove(name)
            continue
        if name in provider_to_branch:
            branch = provider_to_branch[name]
            affected = [
                leaf_name
                for leaf_name in schedule_order
                if leaf_name in active and leaves_by_name[leaf_name]["branch"] == branch
            ]
            for leaf_name in affected:
                _restore(slots, witnesses, leaves_by_name, leaf_name)
                active.remove(leaf_name)
            active.remove(name)
            continue
        active.remove(name)

    return _canonical_observation(slots, active)


def _target(episode, leaves, query_relevant):
    slots, _witnesses, active = _startup(episode, leaves)
    branch = "critical" if query_relevant else "control"
    provider = (
        episode["critical_domain"][0]
        if query_relevant
        else episode["control_domain"][0]
    )
    affected = [leaf for leaf in leaves if leaf["branch"] == branch]
    for leaf in affected:
        slots[leaf["slot"]] = episode["base_slots"][leaf["slot"]]
        active.remove(leaf["name"])
    active.remove(provider)
    return _canonical_observation(slots, active)


def _plans(episode, noncommuting, query_relevant, rng):
    critical = episode["critical_domain"][2]
    control = episode["control_domain"][2]
    provider = (
        episode["critical_domain"][0]
        if query_relevant
        else episode["control_domain"][0]
    )
    direct = (
        {"kind": "inspect", "name": provider},
        {"kind": "dispose", "name": provider},
    )

    if noncommuting and query_relevant:
        critical_start = [name for name in episode["startup_order"] if name in critical]
        early, late = critical_start[0], critical_start[1]
        candidates = [
            (
                {"kind": "dispose", "name": late},
                {"kind": "dispose", "name": provider},
            ),
            direct,
            (
                {"kind": "dispose", "name": early},
                {"kind": "dispose", "name": provider},
            ),
            (
                {"kind": "dispose", "name": episode["decoy"]},
                {"kind": "dispose", "name": provider},
            ),
        ]
    else:
        other_branch = control if query_relevant else critical
        candidates = [
            direct,
            (
                {"kind": "dispose", "name": episode["decoy"]},
                {"kind": "dispose", "name": provider},
            ),
            (
                {"kind": "dispose", "name": other_branch[0]},
                {"kind": "dispose", "name": provider},
            ),
            (
                {"kind": "dispose", "name": other_branch[1]},
                {"kind": "dispose", "name": provider},
            ),
        ]
    rng.shuffle(candidates)
    return {LETTERS[index]: list(plan) for index, plan in enumerate(candidates)}


def _build_native_spec(rng, episode, noncommuting, query_relevant):
    leaves = _leaf_defs(episode, noncommuting)
    branch = "critical" if query_relevant else "control"
    query_provider = (
        episode["critical_domain"][0]
        if query_relevant
        else episode["control_domain"][0]
    )
    affected = [leaf["name"] for leaf in leaves if leaf["branch"] == branch]
    schedules = {
        "left_then_right": affected,
        "right_then_left": list(reversed(affected)),
    }
    target = _target(episode, leaves, query_relevant)
    options = _plans(episode, noncommuting, query_relevant, rng)

    schedule_observations = {
        label: _simulate_plan(
            episode,
            leaves,
            [{"kind": "dispose", "name": query_provider}],
            order,
        )
        for label, order in schedules.items()
    }
    first = next(iter(schedule_observations.values()))
    abstract_confluent = all(value == first for value in schedule_observations.values())
    abstract_option_success = {
        letter: all(
            _simulate_plan(episode, leaves, plan, order) == target
            for order in schedules.values()
        )
        for letter, plan in options.items()
    }
    successes = [letter for letter, success in abstract_option_success.items() if success]
    if len(successes) != 1:
        raise RuntimeError(f"native V1.8 abstract ACT is not unique: {successes}")

    expected_confluent = not (noncommuting and query_relevant)
    if abstract_confluent != expected_confluent:
        raise RuntimeError("native V1.8 abstract factorial did not match expected cell")

    native_case_id = stable_id(
        {
            "schema_version": "1.8-native",
            "episode_id": episode["episode_id"],
            "cell": _cell_name(noncommuting, query_relevant),
            "options": options,
        }
    )
    return {
        "native_case_id": native_case_id,
        "episode_id": episode["episode_id"],
        "cell": _cell_name(noncommuting, query_relevant),
        "noncommuting": noncommuting,
        "query_relevant": query_relevant,
        "base_slots": episode["base_slots"],
        "providers": [
            {
                "name": episode["critical_domain"][0],
                "service": episode["critical_domain"][1],
            },
            {
                "name": episode["control_domain"][0],
                "service": episode["control_domain"][1],
            },
        ],
        "decoy": episode["decoy"],
        "leaves": leaves,
        "startup_order": list(episode["startup_order"]),
        "query_provider": query_provider,
        "schedules": schedules,
        "target": target,
        "options": options,
        "abstract_schedule_observations": schedule_observations,
        "abstract_confluent": abstract_confluent,
        "abstract_option_success": abstract_option_success,
    }


def generate_native_specs(episodes=V18_NATIVE_EPISODES, seed=0):
    rng = random.Random(seed)
    specs = []
    for _ in range(episodes):
        episode = _build_episode(rng)
        for noncommuting in (False, True):
            for query_relevant in (False, True):
                specs.append(_build_native_spec(rng, episode, noncommuting, query_relevant))
    return specs


def _default_runner_path():
    return Path(__file__).resolve().parents[2] / "native" / "cordis" / "run.mjs"


def _execute_native_batch(specs, runner, node):
    payload = "".join(json.dumps(spec, sort_keys=True) + "\n" for spec in specs)
    try:
        # Reading a large JSONL payload from fd 0 can return EAGAIN when the
        # caller's stdin is non-blocking.  The runner supports a file path as
        # its first argument, so use a temporary file for deterministic native
        # execution across shells, PTYs, and CI workers.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", encoding="utf-8", delete=True
        ) as input_file:
            input_file.write(payload)
            input_file.flush()
            completed = subprocess.run(
                [node, str(runner), input_file.name],
                text=True,
                capture_output=True,
                check=True,
            )
    except FileNotFoundError as error:
        raise RuntimeError("Node.js is required for V1.8 native Cordis execution") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip()
        raise RuntimeError(
            "native Cordis execution failed; run `npm install --prefix native/cordis` "
            f"and retry. stderr: {stderr}"
        ) from error

    results = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(results) != len(specs):
        raise RuntimeError(
            f"native Cordis runner returned {len(results)} results for {len(specs)} specs"
        )
    return results


def execute_native_specs(specs, runner_path=None, node="node"):
    specs = list(specs)
    runner = Path(runner_path) if runner_path else _default_runner_path()
    if not runner.exists():
        raise RuntimeError(f"Cordis native runner not found: {runner}")
    if not specs:
        return {}

    worker_count = min(4, len(specs))
    batches = [specs[index::worker_count] for index in range(worker_count)]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_execute_native_batch, batch, runner, node)
            for batch in batches
        ]
        results = [result for future in futures for result in future.result()]
    return {result["native_case_id"]: result for result in results}


def _normalize_observation(observation):
    return {
        "slots": dict(sorted(observation["slots"].items())),
        "active": sorted(observation["active"]),
    }


def _validate_runtime(spec, runtime):
    if runtime.get("cordis_version") != CORDIS_VERSION:
        raise RuntimeError(
            f"Cordis runtime version mismatch: {runtime.get('cordis_version')} != {CORDIS_VERSION}"
        )
    runtime_schedules = {
        run["label"]: _normalize_observation(run["observation"])
        for run in runtime["schedule_runs"]
    }
    expected_schedules = {
        label: _normalize_observation(observation)
        for label, observation in spec["abstract_schedule_observations"].items()
    }
    schedule_agreement = runtime_schedules == expected_schedules
    confluence_agreement = runtime["confluent"] == spec["abstract_confluent"]
    option_agreement = runtime["option_success"] == spec["abstract_option_success"]
    agreement = schedule_agreement and confluence_agreement and option_agreement
    if not agreement:
        raise RuntimeError(
            "Cordis-native differential oracle disagreement for "
            f"{spec['native_case_id']}: schedules={schedule_agreement}, "
            f"confluence={confluence_agreement}, options={option_agreement}"
        )
    return {
        "oracle_agreement": True,
        "schedule_agreement": schedule_agreement,
        "confluence_agreement": confluence_agreement,
        "option_agreement": option_agreement,
        "runtime_schedule_observations": runtime_schedules,
    }


def _render_native_code(spec):
    provider_lines = [
        f"const {provider['name'].replace('-', '_')} = provider({provider['name']!r}, {provider['service']!r})"
        for provider in spec["providers"]
    ]
    leaf_lines = [
        f"const {leaf['name'].replace('-', '_')} = leaf({leaf['name']!r}, {leaf['service']!r}, {leaf['slot']!r}, {leaf['value']!r})"
        for leaf in spec["leaves"]
    ]
    startup = "\n".join(
        f"await root.plugin({name.replace('-', '_')})" for name in spec["startup_order"]
    )
    providers = "\n".join(
        f"const f_{provider['name'].replace('-', '_')} = await root.plugin({provider['name'].replace('-', '_')})"
        for provider in spec["providers"]
    )
    return f"""import {{ Context }} from 'cordis'

const root = new Context()
const harness = {json.dumps(spec['base_slots'], sort_keys=True)}

const provider = (name, service) => ({{
  name,
  apply(ctx) {{ ctx.provide(service, {{ name }}) }},
}})

const leaf = (name, service, slot, value) => ({{
  name,
  inject: [service],
  apply(ctx) {{
    ctx.effect(() => {{
      const previous = harness[slot]
      harness[slot] = value
      return async () => {{
        await gate(name)  // benchmark schedule controller
        harness[slot] = previous
      }}
    }})
  }},
}})

{chr(10).join(provider_lines)}
{chr(10).join(leaf_lines)}
{providers}
const f_{spec['decoy'].replace('-', '_')} = await root.plugin({{ name: {spec['decoy']!r}, apply() {{}} }})
{startup}
"""


def _plan_text(plan):
    return "; ".join(f"{command['kind']}({command['name']})" for command in plan)


def _common_metadata(spec, runtime, validation, capability, family):
    present = bool(spec["noncommuting"])
    query_relevant = bool(spec["query_relevant"])
    actual_relevant = present and query_relevant
    runtime_observations = validation["runtime_schedule_observations"]
    terminal_observations = len({
        json.dumps(value, sort_keys=True) for value in runtime_observations.values()
    })
    metadata = {
        "benchmark_suite": "1.8",
        "source_suite": "1.8-native",
        "benchmark_component": V18_NATIVE_COMPONENT,
        "component": V18_NATIVE_COMPONENT,
        "benchmark_subcomponent": f"native.{family}",
        "native_case_id": spec["native_case_id"],
        "native_episode_id": spec["episode_id"],
        "native_execution": True,
        "native_schedule_control": "async_disposer_delay_permutation",
        "cordis_package": CORDIS_PACKAGE,
        "cordis_version": runtime["cordis_version"],
        "cordis_source_commit": CORDIS_SOURCE_COMMIT,
        "node_version": runtime.get("node_version"),
        "runtime_oracle": f"{CORDIS_PACKAGE}@{CORDIS_VERSION}",
        "differential_oracle": True,
        **validation,
        "abstract_confluent": spec["abstract_confluent"],
        "abstract_option_success": spec["abstract_option_success"],
        "runtime_confluent": runtime["confluent"],
        "runtime_option_success": runtime["option_success"],
        "runtime_schedule_traces": {
            run["label"]: run["trace"] for run in runtime["schedule_runs"]
        },
        "factorial_cell": spec["cell"],
        "interference_present": present,
        "query_interaction_relevant": query_relevant,
        "interference_relevant": actual_relevant,
        "relevance_regime": "all" if query_relevant else "none",
        "problem_size": 7,
        "state_width": len(spec["base_slots"]),
        "total_interfering_pairs": 1 if present else 0,
        "relevant_interfering_pairs": 1 if actual_relevant else 0,
        "irrelevant_interfering_pairs": 1 if present and not query_relevant else 0,
        "relevant_fraction": 1.0 if actual_relevant else 0.0,
        "dependency_depth": 1,
        "schedule_count": len(spec["schedules"]),
        "terminal_observations": terminal_observations,
        "effect_ops_total": 4,
        "realization": "actual_cordis_runtime",
        "surface": "native_cordis_javascript",
        "state_representation": "named_harness_slots",
        "effect_semantics": "cordis_ctx_effect_activation_witness",
        "inverse_semantics": "context_dependent",
        "effect_context_dependence": True,
        "dependency_semantics": "cordis_provide_inject",
        "lifecycle_semantics": "cordis_provider_disposal_dependent_unload",
        "action_surface": "fiber_dispose",
        "query_quantifier": "both_controlled_native_interleavings",
        "task_family": family,
        "answer_type": "binary_choice" if family == "schedule_invariance" else "plan_choice",
        "capability": capability,
        "harness_surface": True,
        "construct_role": "cordis_native_construct_validation",
        "oracle_type": "differential_finite_plus_cordis_runtime",
        "interaction_topology": "single_critical_pair_plus_control_branch",
    }
    metadata["semantic_dimensions"] = {
        key: metadata[key]
        for key in (
            "source_suite",
            "benchmark_component",
            "benchmark_subcomponent",
            "realization",
            "surface",
            "state_representation",
            "effect_semantics",
            "inverse_semantics",
            "effect_context_dependence",
            "dependency_semantics",
            "lifecycle_semantics",
            "action_surface",
            "query_quantifier",
            "task_family",
            "answer_type",
            "capability",
            "harness_surface",
            "construct_role",
            "oracle_type",
            "interaction_topology",
            "factorial_cell",
            "interference_present",
            "interference_relevant",
            "query_interaction_relevant",
            "relevance_regime",
            "problem_size",
            "state_width",
            "total_interfering_pairs",
            "relevant_interfering_pairs",
            "irrelevant_interfering_pairs",
            "relevant_fraction",
            "dependency_depth",
            "schedule_count",
            "terminal_observations",
            "effect_ops_total",
            "native_execution",
            "native_schedule_control",
            "cordis_version",
            "cordis_source_commit",
            "differential_oracle",
            "oracle_agreement",
        )
    }
    return metadata


def native_records(specs, runtimes, rng):
    records = []
    for spec in specs:
        runtime = runtimes[spec["native_case_id"]]
        validation = _validate_runtime(spec, runtime)
        code = _render_native_code(spec)

        choices = {"A": "YES", "B": "NO"}
        if rng.random() < 0.5:
            choices = {"A": "NO", "B": "YES"}
        desired = "YES" if runtime["confluent"] else "NO"
        answer = next(letter for letter, text in choices.items() if text == desired)
        schedules = "\n".join(
            f"  {label}: " + " -> ".join(order)
            for label, order in spec["schedules"].items()
        )
        prompt = f"""The following program is executed against Cordis {CORDIS_VERSION}. It uses real `ctx.provide`, `inject`, and `ctx.effect` APIs. Each effect records its activation-time previous value in the JavaScript closure returned as its disposer.

`gate(name)` does not change state. It only forces the concurrently triggered dependent disposers to complete in one of these two orders:
{schedules}

```js
{code}```

The maintenance action is:
  dispose({spec['query_provider']})

Across both native Cordis teardown interleavings above, is the final observable configuration (all `harness` slot values and active plugin fibers) the same?
  A. {choices['A']}
  B. {choices['B']}

Answer with only the option letter."""
        records.append(
            {
                "id": stable_id({
                    "schema_version": "1.8",
                    "task_type": "cordis_native_reconfiguration_confluence",
                    "native_case_id": spec["native_case_id"],
                    "choices": choices,
                }),
                "schema_version": "1.8",
                "task_type": "cordis_native_reconfiguration_confluence",
                "prompt": prompt,
                "choices": choices,
                "answer": answer,
                "metadata": _common_metadata(
                    spec, runtime, validation, "global", "schedule_invariance"
                ),
            }
        )

        option_text = "\n".join(
            f"  {letter}. {_plan_text(plan)}" for letter, plan in spec["options"].items()
        )
        target = json.dumps(spec["target"], sort_keys=True)
        successful = [
            letter for letter, success in runtime["option_success"].items() if success
        ]
        if len(successful) != 1:
            raise RuntimeError(
                f"Cordis native ACT is not unique for {spec['native_case_id']}: {successful}"
            )
        answer = successful[0]
        prompt = f"""The following program is executed against Cordis {CORDIS_VERSION}. It uses real `ctx.provide`, `inject`, and `ctx.effect` APIs. `gate(name)` only selects either native dependent-disposer completion order shown below.

{schedules}

```js
{code}```

Maintenance target after removing provider {spec['query_provider']}:
  {target}

After each `dispose(...)` command, Cordis is allowed to settle before the next command. `inspect(...)` is read-only padding. Which two-command plan reaches the exact target under BOTH native teardown interleavings?
{option_text}

Answer with only the option letter."""
        records.append(
            {
                "id": stable_id({
                    "schema_version": "1.8",
                    "task_type": "cordis_native_safe_reconfiguration",
                    "native_case_id": spec["native_case_id"],
                    "options": spec["options"],
                }),
                "schema_version": "1.8",
                "task_type": "cordis_native_safe_reconfiguration",
                "prompt": prompt,
                "choices": {
                    letter: _plan_text(plan) for letter, plan in spec["options"].items()
                },
                "answer": answer,
                "metadata": _common_metadata(
                    spec, runtime, validation, "act", "robust_intervention"
                ),
            }
        )
    return records


def generate_v18_native_dataset(
    n=V18_NATIVE_RECORDS,
    seed=0,
    runner_path=None,
    node="node",
):
    if n % 8:
        raise ValueError("V1.8 native slice emits 8 records per matched episode")
    specs = generate_native_specs(n // 8, seed=seed)
    runtimes = execute_native_specs(specs, runner_path=runner_path, node=node)
    return native_records(specs, runtimes, random.Random(seed + 700001))


def _annotate_v17_parent(record):
    # Preserve V1.7 IDs so existing V1.7 predictions can be reused verbatim in
    # the V1.8 composite. Provenance remains in source_suite/source_id fields.
    metadata = record.setdefault("metadata", {})
    metadata["parent_benchmark_suite"] = "1.8"
    metadata["benchmark_suite"] = "1.8"
    dimensions = metadata.setdefault("semantic_dimensions", {})
    dimensions["parent_benchmark_suite"] = "1.8"
    return record


def generate_v18_dataset(
    n=V18_DEFAULT_N,
    seed=0,
    task_types=V18_TASKS,
    n_components=None,
    width=None,
    runner_path=None,
    node="node",
):
    if n != V18_DEFAULT_N:
        raise ValueError(
            f"V1.8 is a fixed composite of {V17_DEFAULT_N} V1.7 records + "
            f"{V18_NATIVE_RECORDS} native Cordis records; use n={V18_DEFAULT_N}"
        )
    if tuple(task_types) != V18_TASKS:
        raise ValueError("V1.8 is a fixed composite suite; custom --tasks is not supported")
    if n_components is not None or width is not None:
        raise ValueError("V1.8 has per-component sizes; do not pass --components or --width")

    inherited = [
        _annotate_v17_parent(record)
        for record in generate_v17_dataset(V17_DEFAULT_N, seed=seed)
    ]
    native = generate_v18_native_dataset(
        V18_NATIVE_RECORDS,
        seed=seed + 180081,
        runner_path=runner_path,
        node=node,
    )
    return inherited + native


def write_native_artifacts(specs, runtimes, directory):
    """Persist native specs and actual runtime traces for paper auditability."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec_path = directory / "v1.8-native-spec.jsonl"
    runtime_path = directory / "v1.8-native-runtime.jsonl"
    spec_path.write_text(
        "".join(json.dumps(spec, sort_keys=True) + "\n" for spec in specs),
        encoding="utf-8",
    )
    runtime_path.write_text(
        "".join(
            json.dumps(runtimes[spec["native_case_id"]], sort_keys=True) + "\n"
            for spec in specs
        ),
        encoding="utf-8",
    )
    return spec_path, runtime_path
