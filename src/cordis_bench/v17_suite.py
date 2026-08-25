"""Composite V1.7 benchmark.

V1.7 is the paper-facing suite, not a replacement for the frozen standalone
V1.6 artifact. It combines:

- `harnessless_core`: the exact V1.6 composability construction, preserving its
  global/outcome-count/intervention tasks; and
- `harness_bridge`: the Cordis/harness realization with activation-time
  rollback witnesses and Creator-style package maintenance.

Every record is annotated with a coarse benchmark component, a task-level
subcomponent, and a scalar `semantic_dimensions` dictionary. The scorer uses
those fields to report component and dimension slices automatically.
"""

from .tasks import stable_id
from .v16 import V16_TASKS, generate_v16_dataset
from .v17 import (
    V17_TASKS as V17_BRIDGE_TASKS,
    generate_v17_dataset as generate_v17_bridge_dataset,
)


V17_TASKS = V16_TASKS + V17_BRIDGE_TASKS
V17_DEFAULT_N = 228
V17_CORE_COMPONENT = "harnessless_core"
V17_BRIDGE_COMPONENT = "harness_bridge"


TASK_FAMILY = {
    "composition_confluence": "schedule_invariance",
    "composition_outcome_count": "schedule_outcome_enumeration",
    "safe_intervention": "robust_intervention",
    "harness_reconfiguration_confluence": "schedule_invariance",
    "harness_safe_reconfiguration": "robust_intervention",
}

ANSWER_TYPE = {
    "composition_confluence": "binary_choice",
    "composition_outcome_count": "count_choice",
    "safe_intervention": "plan_choice",
    "harness_reconfiguration_confluence": "binary_choice",
    "harness_safe_reconfiguration": "plan_choice",
}

CAPABILITY = {
    "composition_confluence": "global",
    "composition_outcome_count": "global",
    "safe_intervention": "act",
    "harness_reconfiguration_confluence": "global",
    "harness_safe_reconfiguration": "act",
}


def _schedule_bucket(value):
    if value is None:
        return None
    if value <= 1:
        return "1"
    if value <= 10:
        return "2-10"
    if value <= 1000:
        return "11-1k"
    if value <= 100000:
        return "1k-100k"
    return ">100k"


def _terminal_bucket(value):
    if value is None:
        return None
    if value <= 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 16:
        return "5-16"
    return ">16"


def _relevance_regime(relevant, total):
    if not relevant:
        return "none"
    if total is not None and relevant == total:
        return "all"
    return "partial"


def _rekey_for_v17(record, component, source_suite):
    source_id = record["id"]
    record["id"] = stable_id(
        {
            "schema_version": "1.7",
            "benchmark_component": component,
            "source_suite": source_suite,
            "source_id": source_id,
        }
    )
    record["schema_version"] = "1.7"
    metadata = record.setdefault("metadata", {})
    metadata["source_id"] = source_id
    metadata["source_suite"] = source_suite
    metadata["benchmark_suite"] = "1.7"
    metadata["benchmark_component"] = component
    metadata["component"] = component
    return metadata


def _annotate_core(record):
    metadata = _rekey_for_v17(record, V17_CORE_COMPONENT, "1.6")
    task = record["task_type"]
    family = TASK_FAMILY[task]
    capability = metadata.get("capability") or CAPABILITY[task]
    total = metadata.get("total_interfering_pairs")
    relevant = metadata.get("relevant_interfering_pairs") or 0
    irrelevant = metadata.get("irrelevant_interfering_pairs")
    relevance_regime = _relevance_regime(relevant, total)

    metadata.update(
        {
            "benchmark_subcomponent": f"core.{family}",
            "realization": "finite_modular_micro_system",
            "surface": "formal_component_system",
            "state_representation": "modular_vector",
            "effect_semantics": "explicit_reversible_modular_ops",
            "inverse_semantics": "context_independent",
            "effect_context_dependence": False,
            "dependency_semantics": "reactive_required_provided_keys",
            "lifecycle_semantics": "dependent_first_nondeterministic_unload",
            "action_surface": "disable_component",
            "query_quantifier": "all_legal_schedules",
            "task_family": family,
            "answer_type": ANSWER_TYPE[task],
            "capability": capability,
            "harness_surface": False,
            "construct_role": "controlled_core",
            "oracle_type": "exact_reachable_state_graph",
            "interaction_topology": "disjoint_interfering_pairs",
            "interference_present": bool(total),
            "interference_relevant": bool(relevant),
            "relevance_regime": relevance_regime,
            "state_width": metadata.get("width", 11),
        }
    )

    metadata["semantic_dimensions"] = {
        "source_suite": "1.6",
        "benchmark_component": V17_CORE_COMPONENT,
        "benchmark_subcomponent": metadata["benchmark_subcomponent"],
        "realization": metadata["realization"],
        "surface": metadata["surface"],
        "state_representation": metadata["state_representation"],
        "effect_semantics": metadata["effect_semantics"],
        "inverse_semantics": metadata["inverse_semantics"],
        "effect_context_dependence": False,
        "dependency_semantics": metadata["dependency_semantics"],
        "lifecycle_semantics": metadata["lifecycle_semantics"],
        "action_surface": metadata["action_surface"],
        "query_quantifier": metadata["query_quantifier"],
        "task_family": family,
        "answer_type": ANSWER_TYPE[task],
        "capability": capability,
        "harness_surface": False,
        "construct_role": metadata["construct_role"],
        "oracle_type": metadata["oracle_type"],
        "interaction_topology": metadata["interaction_topology"],
        "interference_present": bool(total),
        "interference_relevant": bool(relevant),
        "relevance_regime": relevance_regime,
        "problem_size": metadata.get("problem_size"),
        "state_width": metadata.get("state_width"),
        "total_interfering_pairs": total,
        "relevant_interfering_pairs": relevant,
        "irrelevant_interfering_pairs": irrelevant,
        "relevant_fraction": metadata.get("relevant_fraction"),
        "dependency_depth": metadata.get("dependency_depth"),
        "schedule_count_bucket": _schedule_bucket(metadata.get("schedule_count")),
        "terminal_observation_bucket": _terminal_bucket(metadata.get("terminal_observations")),
        "effect_ops_total": metadata.get("effect_ops_total"),
    }
    return record


def _annotate_bridge(record):
    metadata = _rekey_for_v17(record, V17_BRIDGE_COMPONENT, "1.7-bridge")
    task = record["task_type"]
    family = TASK_FAMILY[task]
    capability = metadata.get("capability") or CAPABILITY[task]
    present = bool(metadata.get("interference_present"))
    # The bridge generator's historical field means that the queried lifecycle
    # reaches the critical pair, even in the commuting control. Preserve that
    # concept under an unambiguous name before deriving actual interference.
    query_relevant = bool(metadata.get("interference_relevant"))
    actual_relevant = present and query_relevant
    relevant_interfering = 1 if actual_relevant else 0
    irrelevant_interfering = 1 if present and not query_relevant else 0
    relevance_regime = "all" if query_relevant else "none"

    metadata.update(
        {
            "benchmark_subcomponent": f"bridge.{family}",
            "realization": "witnessed_harness_packages",
            "surface": "creator_style_package_lifecycle",
            "state_representation": "named_harness_slots",
            "effect_semantics": "activation_time_witnessed_restore",
            "inverse_semantics": "context_dependent",
            "effect_context_dependence": True,
            "dependency_semantics": "reactive_required_provided_keys",
            "lifecycle_semantics": "dependent_first_nondeterministic_package_stop",
            "action_surface": "creator_inspect_undefine",
            "query_quantifier": "all_legal_schedules",
            "task_family": family,
            "answer_type": ANSWER_TYPE[task],
            "capability": capability,
            "harness_surface": True,
            "construct_role": "cordis_harness_bridge",
            "oracle_type": "exact_reachable_state_graph",
            "interaction_topology": "single_critical_pair_plus_control_branch",
            "query_interaction_relevant": query_relevant,
            "interference_relevant": actual_relevant,
            "relevance_regime": relevance_regime,
            "problem_size": metadata.get("problem_size", 7),
            "state_width": 5,
            "total_interfering_pairs": 1 if present else 0,
            "relevant_interfering_pairs": relevant_interfering,
            "irrelevant_interfering_pairs": irrelevant_interfering,
            "relevant_fraction": 1.0 if relevant_interfering else 0.0,
            "dependency_depth": 1,
        }
    )

    metadata["semantic_dimensions"] = {
        "source_suite": "1.7-bridge",
        "benchmark_component": V17_BRIDGE_COMPONENT,
        "benchmark_subcomponent": metadata["benchmark_subcomponent"],
        "realization": metadata["realization"],
        "surface": metadata["surface"],
        "state_representation": metadata["state_representation"],
        "effect_semantics": metadata["effect_semantics"],
        "inverse_semantics": metadata["inverse_semantics"],
        "effect_context_dependence": True,
        "dependency_semantics": metadata["dependency_semantics"],
        "lifecycle_semantics": metadata["lifecycle_semantics"],
        "action_surface": metadata["action_surface"],
        "query_quantifier": metadata["query_quantifier"],
        "task_family": family,
        "answer_type": ANSWER_TYPE[task],
        "capability": capability,
        "harness_surface": True,
        "construct_role": metadata["construct_role"],
        "oracle_type": metadata["oracle_type"],
        "interaction_topology": metadata["interaction_topology"],
        "factorial_cell": metadata.get("factorial_cell"),
        "interference_present": present,
        "interference_relevant": actual_relevant,
        "query_interaction_relevant": query_relevant,
        "relevance_regime": relevance_regime,
        "problem_size": metadata.get("problem_size"),
        "state_width": metadata.get("state_width"),
        "total_interfering_pairs": metadata.get("total_interfering_pairs"),
        "relevant_interfering_pairs": relevant_interfering,
        "irrelevant_interfering_pairs": irrelevant_interfering,
        "relevant_fraction": metadata.get("relevant_fraction"),
        "dependency_depth": metadata.get("dependency_depth"),
        "schedule_count_bucket": _schedule_bucket(metadata.get("schedule_count")),
        "terminal_observation_bucket": _terminal_bucket(metadata.get("terminal_observations")),
        "effect_ops_total": metadata.get("effect_ops_total"),
    }
    return record


def component_sizes(n):
    """Split total V1.7 size while preserving the 15:4 core/bridge ratio.

    V1.6 emits 15 records per matched curve and the bridge emits 8 per episode.
    A 15:4 ratio is exactly compatible when n is a multiple of 38. The default
    n=228 therefore reproduces the frozen 180-item V1.6 core and adds 48 bridge
    items.
    """

    if n % 38:
        raise ValueError(
            "V1.7 composite size must be a multiple of 38; use the default 228 "
            "for 180 harnessless-core + 48 harness-bridge records"
        )
    units = n // 19
    core_n = 15 * units
    bridge_n = 4 * units
    if core_n % 15 or bridge_n % 8:
        raise RuntimeError("internal V1.7 component split is not generator-compatible")
    return core_n, bridge_n


def generate_v17_dataset(
    n=V17_DEFAULT_N,
    seed=0,
    task_types=V17_TASKS,
    n_components=None,
    width=None,
):
    if tuple(task_types) != V17_TASKS:
        raise ValueError("V1.7 is a composite matched suite; custom --tasks is not supported")
    if n_components is not None or width is not None:
        raise ValueError(
            "V1.7 has per-component sizes (core 12x11, bridge 7x5); "
            "do not pass --components or --width"
        )

    core_n, bridge_n = component_sizes(n)
    core = generate_v16_dataset(
        core_n,
        seed=seed,
        n_components=12,
        width=11,
    )
    bridge = generate_v17_bridge_dataset(
        bridge_n,
        seed=seed + 1000003,
        n_components=7,
        width=5,
    )

    records = [_annotate_core(record) for record in core]
    records.extend(_annotate_bridge(record) for record in bridge)
    return records
