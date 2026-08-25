"""Dimension-aware reporting layered over the historical scorer.

The base evaluator remains backward compatible with V0-V1.6 artifacts. This
wrapper rejoins each scored row with dataset metadata and automatically
summarizes benchmark components and scalar semantic dimensions. V1.9 adds a
formal-boundary report with empirical 95/90/75% frontiers per stress profile
and task. V1.10 additionally derives an inherited-vs-replication partition at
report time so the frozen V1.9 rows remain byte-for-byte unchanged. V1.11 adds
challenge-size curves and paired isomorphism diagnostics. V1.12 uses answered
accuracy with a parse-rate gate for challenge frontiers.
"""

from .evaluate import _summary, score_predictions as _base_score_predictions


DEFAULT_DIMENSIONS = (
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
    "schedule_count_bucket",
    "reachable_states",
    "terminal_observations",
    "terminal_observation_bucket",
    "effect_ops_total",
    "boundary_family",
    "boundary_profile",
    "boundary_level",
    "relevant_effects",
    "relevant_conflict_edges",
    "relevant_conflict_density",
    "effect_ops_per_component",
    "release_added",
    "replication_release",
    "replication_block",
    "independent_semantic_world",
    "challenge_track",
    "challenge_family",
    "challenge_size",
    "challenge_size_unit",
    "perturbation_variant",
    "shortcut_resistant",
    "alpha_renaming",
    "component_reordering",
    "sampled_native_schedules",
)


def _dimension_value(metadata, key):
    dimensions = metadata.get("semantic_dimensions") or {}
    if key in dimensions:
        return dimensions[key]
    return metadata.get(key)


def _stable_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _group_summary(rows, key):
    values = sorted(
        {row.get(key) for row in rows if row.get(key) is not None},
        key=lambda value: str(value),
    )
    return {
        _stable_value(value): _summary([row for row in rows if row.get(key) == value])
        for value in values
    }


def _dimension_tables(rows, dimension_keys):
    tables = {}
    for key in sorted(dimension_keys):
        if any(row.get(key) is not None for row in rows):
            tables[key] = _group_summary(rows, key)
    if any(row.get("reasoning_effort") is not None for row in rows):
        tables["reasoning_effort"] = _group_summary(rows, "reasoning_effort")
    return tables


def _threshold_frontier(level_rows, threshold, answered=False, min_parse_rate=None):
    """Largest contiguous level satisfying an accuracy and optional parse gate."""
    frontier = None
    for level in sorted(level_rows):
        summary = _summary(level_rows[level])
        if min_parse_rate is not None and summary["parsed_rate"] < min_parse_rate:
            break
        value = summary["answered_accuracy"] if answered else summary["accuracy"]
        if value is None or value < threshold:
            break
        frontier = level
    return frontier


def _formal_boundary_report(rows):
    boundary = [row for row in rows if row.get("benchmark_component") == "formal_boundary"]
    if not boundary:
        return None

    report = {
        "n": len(boundary),
        "overall": _summary(boundary),
        "by_profile": {},
    }
    for profile in sorted({row.get("boundary_profile") for row in boundary if row.get("boundary_profile")}):
        profile_rows = [row for row in boundary if row.get("boundary_profile") == profile]
        profile_report = {
            "overall": _summary(profile_rows),
            "by_task": {},
        }
        for task in sorted({row["task_type"] for row in profile_rows}):
            task_rows = [row for row in profile_rows if row["task_type"] == task]
            level_rows = {}
            for row in task_rows:
                level = row.get("boundary_level")
                if level is not None:
                    level_rows.setdefault(level, []).append(row)
            profile_report["by_task"][task] = {
                "by_level": {
                    _stable_value(level): _summary(level_rows[level])
                    for level in sorted(level_rows)
                },
                "frontier_95": _threshold_frontier(level_rows, 0.95),
                "frontier_90": _threshold_frontier(level_rows, 0.90),
                "frontier_75": _threshold_frontier(level_rows, 0.75),
            }
        report["by_profile"][profile] = profile_report
    return report


def _challenge_pair_metrics(rows):
    pairs = {}
    for row in rows:
        pair_id = row.get("challenge_pair_id")
        if pair_id:
            pairs.setdefault(pair_id, []).append(row)
    complete = [pair for pair in pairs.values() if len(pair) == 2]
    parseable = [pair for pair in complete if all(row.get("prediction") is not None for row in pair)]
    both_correct = sum(all(row["correct"] for row in pair) for pair in complete)
    base_only = 0
    perturbed_only = 0
    both_wrong = 0
    for pair in complete:
        by_variant = {row.get("perturbation_variant"): row for row in pair}
        base = by_variant.get("base")
        perturbed = by_variant.get("alpha_renamed_reordered")
        if not base or not perturbed:
            continue
        if base["correct"] and not perturbed["correct"]:
            base_only += 1
        elif perturbed["correct"] and not base["correct"]:
            perturbed_only += 1
        elif not base["correct"] and not perturbed["correct"]:
            both_wrong += 1
    consistent = sum(pair[0]["prediction"] == pair[1]["prediction"] for pair in parseable)
    return {
        "n_pairs": len(complete),
        "parseable_pairs": len(parseable),
        "both_correct": both_correct,
        "both_correct_rate": both_correct / len(complete) if complete else None,
        "base_only_correct": base_only,
        "perturbed_only_correct": perturbed_only,
        "both_wrong": both_wrong,
        "parsed_answer_consistency": consistent / len(parseable) if parseable else None,
    }


def _challenge_report(rows):
    challenge = [row for row in rows if row.get("challenge_track")]
    if not challenge:
        return None
    report = {
        "n": len(challenge),
        "overall": _summary(challenge),
        "by_component": {},
        "perturbation_pairs": _challenge_pair_metrics(challenge),
    }
    for component in sorted({row["benchmark_component"] for row in challenge}):
        component_rows = [row for row in challenge if row["benchmark_component"] == component]
        sizes = sorted({row["challenge_size"] for row in component_rows})
        by_task_size = {}
        for task in sorted({row["task_type"] for row in component_rows}):
            task_rows = [row for row in component_rows if row["task_type"] == task]
            by_task_size[task] = {
                _stable_value(size): _summary(
                    [row for row in task_rows if row["challenge_size"] == size]
                )
                for size in sizes
                if any(row["challenge_size"] == size for row in task_rows)
            }
        size_rows = {
            size: [row for row in component_rows if row["challenge_size"] == size]
            for size in sizes
        }
        report["by_component"][component] = {
            "overall": _summary(component_rows),
            "by_size": {
                _stable_value(size): _summary(size_rows[size]) for size in sizes
            },
            "by_task_and_size": by_task_size,
            "frontier_95": _threshold_frontier(
                size_rows, 0.95, answered=True, min_parse_rate=0.95
            ),
            "frontier_90": _threshold_frontier(
                size_rows, 0.90, answered=True, min_parse_rate=0.95
            ),
            "frontier_75": _threshold_frontier(
                size_rows, 0.75, answered=True, min_parse_rate=0.95
            ),
            "frontier_60": _threshold_frontier(
                size_rows, 0.60, answered=True, min_parse_rate=0.95
            ),
            "perturbation_pairs": _challenge_pair_metrics(component_rows),
        }
    return report


def score_predictions(dataset, predictions):
    report = _base_score_predictions(dataset, predictions)
    metadata_by_id = {item["id"]: item.get("metadata", {}) for item in dataset}
    is_v110 = any(
        metadata.get("release_added") == "1.10"
        for metadata in metadata_by_id.values()
    )

    dimension_keys = set(DEFAULT_DIMENSIONS)
    for metadata in metadata_by_id.values():
        dimension_keys.update((metadata.get("semantic_dimensions") or {}).keys())
    if is_v110:
        dimension_keys.add("replication_partition")

    for row in report["rows"]:
        metadata = metadata_by_id.get(row["id"], {})
        row["benchmark_component"] = metadata.get("benchmark_component") or metadata.get("component")
        row["benchmark_subcomponent"] = metadata.get("benchmark_subcomponent")
        row["semantic_dimensions"] = metadata.get("semantic_dimensions") or {}
        row["challenge_pair_id"] = metadata.get("challenge_pair_id")
        for key in dimension_keys:
            if key == "replication_partition":
                row[key] = (
                    "v1.10_added"
                    if metadata.get("release_added") == "1.10"
                    else "inherited_v1.9"
                )
            else:
                row[key] = _dimension_value(metadata, key)

    rows = report["rows"]
    components = [row for row in rows if row.get("benchmark_component") is not None]
    if components:
        report["by_component"] = _group_summary(rows, "benchmark_component")
        report["by_subcomponent"] = _group_summary(rows, "benchmark_subcomponent")

        component_task = {}
        component_dimension = {}
        for component in sorted({row["benchmark_component"] for row in components}):
            component_rows = [row for row in rows if row["benchmark_component"] == component]
            component_task[component] = {
                task: _summary([row for row in component_rows if row["task_type"] == task])
                for task in sorted({row["task_type"] for row in component_rows})
            }
            component_dimension[component] = _dimension_tables(
                component_rows,
                dimension_keys,
            )
        report["by_component_and_task"] = component_task
        report["by_component_and_dimension"] = component_dimension

    by_dimension = _dimension_tables(rows, dimension_keys)
    if by_dimension:
        report["by_dimension"] = by_dimension
        report["dimension_manifest"] = sorted(by_dimension)

    if is_v110:
        report["by_replication_partition"] = _group_summary(rows, "replication_partition")

    if any(row.get("capability") is not None for row in rows):
        report["by_capability"] = _group_summary(rows, "capability")

    boundary_report = _formal_boundary_report(rows)
    if boundary_report is not None:
        report["formal_boundary"] = boundary_report

    challenge_report = _challenge_report(rows)
    if challenge_report is not None:
        report["challenge"] = challenge_report

    return report
