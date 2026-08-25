"""V1.9: V1.8 plus a procedural formal capability-boundary component.

V1.9 is a strict item-ID superset of V1.8.  The first 292 records are emitted
by ``generate_v18_dataset`` unchanged.  A fourth component, ``formal_boundary``,
then extends the controlled formal semantics along deliberately interpretable
axes: modest component-count scaling, overlapping conflict graphs, dependency
depth, and reversible-effect program length.

The new component is a stress experiment, not a replacement for the validated
V1.8 construct.  Its purpose is to locate 95/90/75% capability frontiers while
retaining exact executable ground truth.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
import random

from .core import Action, Component, Op, RuntimeState, World
from .generate import PRIMES
from .semantics import count_quiescent_paths, explore, observations, orchestrate, quiescent
from .tasks import serialize_state, serialize_world, stable_id, v1_metadata
from .v13 import dependency_depth
from .v15 import _binary_choices, _effects_commute_on_support, _random_keys, _random_names
from .v16 import V16_TASKS, _confluence_prompt, _count_choices, _outcome_count_prompt
from .v16_stress import generate_v16_stress_dataset
from .v18 import V18_DEFAULT_N, V18_TASKS, generate_v18_dataset


V19_TASKS = V18_TASKS
V19_COMPONENT = "formal_boundary"
V19_BOUNDARY_LEVELS = (0, 2, 4, 6, 8)
V19_BOUNDARY_MAX_STATES = 150_000
V19_DISJOINT_CURVES = 2
V19_OVERLAP_CURVES = 2
V19_DISJOINT_RECORDS = 36 + 36
V19_OVERLAP_PROFILES = (
    ("path_d2_ops1", "path", 2, 1),
    ("path_d4_ops1", "path", 4, 1),
    ("cycle_d2_ops1", "cycle", 2, 1),
    ("cycle_d4_ops2", "cycle", 4, 2),
)
V19_OVERLAP_RECORDS = len(V19_OVERLAP_PROFILES) * V19_OVERLAP_CURVES * len(V19_BOUNDARY_LEVELS) * 2
V19_BOUNDARY_RECORDS = V19_DISJOINT_RECORDS + V19_OVERLAP_RECORDS
V19_DEFAULT_N = V18_DEFAULT_N + V19_BOUNDARY_RECORDS


def _bucket(value):
    if value <= 1:
        return "1"
    if value <= 10:
        return "2-10"
    if value <= 100:
        return "11-100"
    if value <= 1_000:
        return "101-1k"
    if value <= 100_000:
        return "1k-100k"
    if value <= 10_000_000:
        return "100k-10m"
    return ">10m"


def _annotate_disjoint(record, n_components):
    metadata = record.setdefault("metadata", {})
    relevant_pairs = int(metadata.get("relevant_interfering_pairs", 0))
    relevant_effects = 2 * relevant_pairs
    conflict_edges = relevant_pairs
    denom = relevant_effects * (relevant_effects - 1) / 2
    conflict_density = (conflict_edges / denom) if denom else 0.0
    profile = f"disjoint_n{n_components}"
    metadata.update(
        benchmark_component=V19_COMPONENT,
        component=V19_COMPONENT,
        benchmark_subcomponent=f"formal_boundary.{profile}",
        parent_benchmark_suite="1.9",
        source_suite="1.6-stress",
        construct_role="formal_failure_boundary",
        boundary_search=True,
        boundary_family="disjoint_scale",
        boundary_profile=profile,
        boundary_level=relevant_pairs,
        relevant_effects=relevant_effects,
        relevant_conflict_edges=conflict_edges,
        relevant_conflict_density=conflict_density,
        effect_ops_per_component=(metadata.get("effect_ops_total", 0) / n_components),
    )
    dimensions = metadata.setdefault("semantic_dimensions", {})
    dimensions.update({
        "source_suite": "1.6-stress",
        "benchmark_component": V19_COMPONENT,
        "benchmark_subcomponent": f"formal_boundary.{profile}",
        "construct_role": "formal_failure_boundary",
        "boundary_family": "disjoint_scale",
        "boundary_profile": profile,
        "boundary_level": relevant_pairs,
        "relevant_effects": relevant_effects,
        "relevant_conflict_edges": conflict_edges,
        "relevant_conflict_density": conflict_density,
        "effect_ops_per_component": metadata["effect_ops_per_component"],
        "parent_benchmark_suite": "1.9",
    })
    return record


def _leaf_effects(rng, modulus, leaf_count, topology, ops_per_leaf):
    effects = []
    for index in range(leaf_count):
        has_next = index < leaf_count - 1 or topology == "cycle"
        if has_next:
            nxt = (index + 1) % leaf_count
            shear = Op("shear", index, nxt, rng.randrange(1, modulus))
            if ops_per_leaf == 1:
                effect = (shear,)
            else:
                effect = (Op("add", index, value=rng.randrange(1, modulus)), shear)
        else:
            if ops_per_leaf == 1:
                effect = (Op("add", index, value=rng.randrange(1, modulus)),)
            else:
                effect = (
                    Op("scale", index, value=rng.randrange(2, modulus)),
                    Op("add", index, value=rng.randrange(1, modulus)),
                )
        effects.append(effect)
    return tuple(effects)


def _conflict_edges(effects, modulus, width):
    edges = []
    for left, right in combinations(range(len(effects)), 2):
        if not _effects_commute_on_support(effects[left], effects[right], modulus, width):
            edges.append((left, right))
    return tuple(edges)


def _build_overlap_static(rng, topology, dep_depth, ops_per_leaf, leaf_count=8):
    if topology not in {"path", "cycle"}:
        raise ValueError(f"unknown V1.9 topology: {topology}")
    if dep_depth < 1:
        raise ValueError("V1.9 dependency depth must be positive")
    n_components = dep_depth + leaf_count
    width = leaf_count + 1
    modulus = rng.choice(PRIMES)
    names = _random_names(rng, n_components)
    keys = dict(zip(names, _random_keys(rng, n_components)))
    infra = tuple(names[:dep_depth])
    leaves = tuple(names[dep_depth:])
    pad = width - 1
    leaf_effects = _leaf_effects(rng, modulus, leaf_count, topology, ops_per_leaf)
    effects = {
        name: (Op("add", pad, value=rng.randrange(1, modulus)),)
        for name in infra
    }
    effects.update({name: effect for name, effect in zip(leaves, leaf_effects)})
    conflict_edges = _conflict_edges(leaf_effects, modulus, width)
    component_order = list(names)
    rng.shuffle(component_order)
    return {
        "modulus": modulus,
        "width": width,
        "n_components": n_components,
        "names": tuple(names),
        "keys": keys,
        "infra": infra,
        "leaves": leaves,
        "effects": effects,
        "leaf_effects": leaf_effects,
        "component_order": tuple(component_order),
        "topology": topology,
        "dependency_depth_target": dep_depth,
        "ops_per_leaf": ops_per_leaf,
        "conflict_edges": conflict_edges,
    }


def _overlap_world(static, relevant_effects):
    relevant = frozenset(range(relevant_effects))
    infra = static["infra"]
    keys = static["keys"]
    components = {}
    for index, name in enumerate(infra):
        requires = frozenset() if index == 0 else frozenset({keys[infra[index - 1]]})
        components[name] = Component(
            name,
            requires,
            frozenset({keys[name]}),
            static["effects"][name],
        )
    terminal_key = keys[infra[-1]]
    for index, name in enumerate(static["leaves"]):
        requires = frozenset({terminal_key}) if index in relevant else frozenset()
        components[name] = Component(
            name,
            requires,
            frozenset({keys[name]}),
            static["effects"][name],
        )
    world = World(
        static["modulus"],
        static["width"],
        tuple(components[name] for name in static["component_order"]),
    )
    return world, relevant


def _variant_stats(world, post, expected_schedules):
    states, _ = explore(world, post, max_states=V19_BOUNDARY_MAX_STATES)
    quiet = {state for state in states if quiescent(world, state)}
    terminal = observations(quiet)
    schedule_count = count_quiescent_paths(world, post, cap=expected_schedules + 1)
    return {
        "reachable_states": len(states),
        "quiescent_states": len(quiet),
        "schedule_count": schedule_count,
        "terminal_observations": len(terminal),
    }, terminal


def _relevant_edges(static, relevant_effects):
    return tuple(
        edge for edge in static["conflict_edges"]
        if edge[0] < relevant_effects and edge[1] < relevant_effects
    )


def _sample_overlap_curve(rng, profile, topology, dep_depth, ops_per_leaf, attempts=500):
    for _ in range(attempts):
        static = _build_overlap_static(rng, topology, dep_depth, ops_per_leaf)
        for _values in range(200):
            values = tuple(rng.randrange(static["modulus"]) for _ in range(static["width"]))
            all_names = frozenset(static["names"])
            start = RuntimeState(values, all_names, all_names)
            variants = {}
            previous_outcomes = 0
            valid = True
            for level in V19_BOUNDARY_LEVELS:
                world, relevant = _overlap_world(static, level)
                post = orchestrate(world, start, Action("disable", static["infra"][0]))
                expected_schedules = factorial(level) if level else 1
                try:
                    stats, terminal = _variant_stats(world, post, expected_schedules)
                except RuntimeError:
                    valid = False
                    break
                edges = _relevant_edges(static, level)
                if stats["schedule_count"] != expected_schedules:
                    valid = False
                    break
                if edges and len(terminal) <= 1:
                    valid = False
                    break
                if len(terminal) < previous_outcomes:
                    valid = False
                    break
                previous_outcomes = len(terminal)
                variants[level] = {
                    "world": world,
                    "relevant": relevant,
                    "post": post,
                    "stats": stats,
                    "terminal": terminal,
                    "relevant_edges": edges,
                }
            if not valid:
                continue
            curve_id = stable_id({
                "schema_version": "1.9-boundary",
                "profile": profile,
                "world": serialize_world(variants[max(V19_BOUNDARY_LEVELS)]["world"]),
                "start": serialize_state(start),
                "topology": topology,
                "dep_depth": dep_depth,
                "ops_per_leaf": ops_per_leaf,
            })
            return {
                "curve_id": curve_id,
                "profile": profile,
                "static": static,
                "start": start,
                "variants": variants,
            }
    raise RuntimeError(f"could not sample V1.9 overlap profile {profile}")


def _overlap_metadata(curve, level, family):
    static = curve["static"]
    variant = curve["variants"][level]
    edges = variant["relevant_edges"]
    denom = level * (level - 1) / 2
    density = (len(edges) / denom) if denom else 0.0
    stats = variant["stats"]
    metadata = v1_metadata(
        variant["world"],
        curve["start"],
        benchmark_suite="1.9",
        parent_benchmark_suite="1.9",
        source_suite="1.9-boundary",
        benchmark_component=V19_COMPONENT,
        component=V19_COMPONENT,
        benchmark_subcomponent=f"formal_boundary.{curve['profile']}",
        construct_role="formal_failure_boundary",
        capability="global",
        task_family=family,
        boundary_search=True,
        boundary_family="overlap_structure",
        boundary_profile=curve["profile"],
        boundary_level=level,
        problem_size=static["n_components"],
        state_width=static["width"],
        total_interfering_pairs=len(static["conflict_edges"]),
        n_noncommuting_pairs=len(static["conflict_edges"]),
        relevant_interfering_pairs=len(edges),
        irrelevant_interfering_pairs=len(static["conflict_edges"]) - len(edges),
        relevant_fraction=(len(edges) / len(static["conflict_edges"]) if static["conflict_edges"] else 0.0),
        relevant_effects=level,
        relevant_conflict_edges=len(edges),
        relevant_conflict_density=density,
        dependency_depth=dependency_depth(variant["world"]),
        schedule_count=stats["schedule_count"],
        reachable_states=stats["reachable_states"],
        terminal_observations=stats["terminal_observations"],
        terminal_outcome_count=stats["terminal_observations"],
        effect_ops_total=sum(len(component.effect) for component in variant["world"].components),
        effect_ops_per_component=(sum(len(component.effect) for component in variant["world"].components) / static["n_components"]),
        interaction_topology=static["topology"],
        conflict_graph_edges=[list(edge) for edge in static["conflict_edges"]],
        realization="finite_modular_micro_system",
        surface="formal_component_system",
        state_representation="modular_vector",
        effect_semantics="overlapping_reversible_modular_ops",
        inverse_semantics="context_independent",
        effect_context_dependence=False,
        dependency_semantics="reactive_required_provided_keys",
        lifecycle_semantics="dependent_first_nondeterministic_unload",
        action_surface="disable_component",
        query_quantifier="all_legal_schedules",
        answer_type="binary_choice" if family == "schedule_invariance" else "count_choice",
        harness_surface=False,
        oracle_type="exact_reachable_state_graph",
        schedule_count_bucket=_bucket(stats["schedule_count"]),
        terminal_observation_bucket=_bucket(stats["terminal_observations"]),
    )
    metadata["semantic_dimensions"] = {
        key: metadata[key]
        for key in (
            "source_suite", "benchmark_component", "benchmark_subcomponent",
            "construct_role", "capability", "task_family", "boundary_family",
            "boundary_profile", "boundary_level", "problem_size", "state_width",
            "total_interfering_pairs", "relevant_interfering_pairs",
            "irrelevant_interfering_pairs", "relevant_fraction", "relevant_effects",
            "relevant_conflict_edges", "relevant_conflict_density", "dependency_depth",
            "schedule_count", "reachable_states", "terminal_observations",
            "effect_ops_total", "effect_ops_per_component", "interaction_topology",
            "realization", "surface", "state_representation", "effect_semantics",
            "inverse_semantics", "effect_context_dependence", "dependency_semantics",
            "lifecycle_semantics", "action_surface", "query_quantifier", "answer_type",
            "harness_surface", "oracle_type", "schedule_count_bucket",
            "terminal_observation_bucket", "parent_benchmark_suite",
        )
    }
    return metadata


def _overlap_records(rng, curve):
    records = []
    for level in V19_BOUNDARY_LEVELS:
        variant = curve["variants"][level]
        choices = _binary_choices(rng)
        confluent = len(variant["terminal"]) == 1
        desired = "YES" if confluent else "NO"
        answer = next(letter for letter, text in choices.items() if text == desired)
        records.append({
            "id": stable_id({
                "schema_version": "1.9-boundary",
                "task_type": "composition_confluence",
                "curve_id": curve["curve_id"],
                "level": level,
                "choices": choices,
            }),
            "schema_version": "1.9",
            "task_type": "composition_confluence",
            "prompt": _confluence_prompt(
                variant["world"], curve["start"],
                Action("disable", curve["static"]["infra"][0]), choices,
            ),
            "choices": choices,
            "answer": answer,
            "metadata": _overlap_metadata(curve, level, "schedule_invariance"),
        })

        gold = len(variant["terminal"])
        count_choices = _count_choices(rng, gold)
        count_answer = next(
            letter for letter, text in count_choices.items() if int(text) == gold
        )
        records.append({
            "id": stable_id({
                "schema_version": "1.9-boundary",
                "task_type": "composition_outcome_count",
                "curve_id": curve["curve_id"],
                "level": level,
                "choices": count_choices,
            }),
            "schema_version": "1.9",
            "task_type": "composition_outcome_count",
            "prompt": _outcome_count_prompt(
                variant["world"], curve["start"],
                Action("disable", curve["static"]["infra"][0]), count_choices,
            ),
            "choices": count_choices,
            "answer": count_answer,
            "metadata": _overlap_metadata(curve, level, "schedule_outcome_enumeration"),
        })
    return records


def generate_v19_boundary_dataset(seed=0):
    records = []
    for index, n_components in enumerate((14, 16)):
        stress = generate_v16_stress_dataset(
            36,
            seed=seed + 190100 + index,
            n_components=n_components,
            width=n_components - 1,
        )
        records.extend(_annotate_disjoint(record, n_components) for record in stress)

    rng = random.Random(seed + 190900)
    for profile, topology, dep_depth, ops_per_leaf in V19_OVERLAP_PROFILES:
        for _ in range(V19_OVERLAP_CURVES):
            curve = _sample_overlap_curve(
                rng,
                profile=profile,
                topology=topology,
                dep_depth=dep_depth,
                ops_per_leaf=ops_per_leaf,
            )
            records.extend(_overlap_records(rng, curve))

    if len(records) != V19_BOUNDARY_RECORDS:
        raise RuntimeError(
            f"V1.9 boundary emitted {len(records)} records, expected {V19_BOUNDARY_RECORDS}"
        )
    return records


def generate_v19_dataset(
    n=V19_DEFAULT_N,
    seed=0,
    task_types=V19_TASKS,
    n_components=None,
    width=None,
    runner_path=None,
    node="node",
):
    if n != V19_DEFAULT_N:
        raise ValueError(
            f"V1.9 is a fixed superset of {V18_DEFAULT_N} V1.8 records + "
            f"{V19_BOUNDARY_RECORDS} formal-boundary records; use n={V19_DEFAULT_N}"
        )
    if tuple(task_types) != V19_TASKS:
        raise ValueError("V1.9 is a fixed composite suite; custom --tasks is not supported")
    if n_components is not None or width is not None:
        raise ValueError("V1.9 has per-component sizes; do not pass --components or --width")

    inherited = generate_v18_dataset(
        V18_DEFAULT_N,
        seed=seed,
        runner_path=runner_path,
        node=node,
    )
    boundary = generate_v19_boundary_dataset(seed=seed)
    inherited_ids = {item["id"] for item in inherited}
    if any(item["id"] in inherited_ids for item in boundary):
        raise RuntimeError("V1.9 boundary ID collided with inherited V1.8 item")
    return inherited + boundary
