import re


ANSWER_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


CAPABILITY_BY_TASK = {
    "trace": "dynamics",
    "dynamic_trace": "dynamics",
    "next_state": "dynamics",
    "interaction_trace": "dynamics",
    "relevance_trace": "dynamics",
    "counterfactual_trace": "formal_conditioning",
    "isomorphic_trace": "formal_conditioning",
    "confluence": "global",
    "dynamic_confluence": "global",
    "interaction_confluence": "global",
    "relevance_confluence": "global",
    "composition_confluence": "global",
    "composition_outcome_count": "global",
    "intervention": "act",
    "plan": "act",
    "robust_intervention": "act",
    "safe_intervention": "act",
    "support_set": "state_comprehension",
    "legal_actions": "permission",
    "quiescence": "permission",
    "possible_next": "permission",
    "must_unload": "global_state",
}

BUNDLE_LADDER = (
    "support_set",
    "legal_actions",
    "next_state",
    "must_unload",
    "dynamic_confluence",
    "robust_intervention",
)


def parse_answer(text, valid=None):
    text = (text or "").strip().upper()
    if valid and text in valid:
        return text
    matches = ANSWER_RE.findall(text)
    if valid:
        matches = [match for match in matches if match in valid]
    return matches[-1] if matches else None


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _summary(rows):
    n = len(rows)
    parsed = [row for row in rows if row["prediction"] is not None]
    correct = [row for row in rows if row["correct"]]
    wrong = [row for row in rows if not row["correct"]]
    return {
        "n": n,
        "accuracy": sum(row["correct"] for row in rows) / n if n else 0.0,
        "parsed": len(parsed),
        "unparsed": n - len(parsed),
        "parsed_rate": len(parsed) / n if n else 0.0,
        "answered_accuracy": (
            sum(row["correct"] for row in parsed) / len(parsed) if parsed else None
        ),
        "mean_reasoning_tokens": _mean([row.get("reasoning_tokens") for row in rows]),
        "mean_reasoning_tokens_correct": _mean([row.get("reasoning_tokens") for row in correct]),
        "mean_reasoning_tokens_wrong": _mean([row.get("reasoning_tokens") for row in wrong]),
        "mean_completion_tokens": _mean([row.get("completion_tokens") for row in rows]),
        "mean_cost": _mean([row.get("cost") for row in rows]),
    }


def _bundle_metrics(rows):
    bundles = {}
    for row in rows:
        if row.get("bundle_id"):
            bundles.setdefault(row["bundle_id"], {})[row["task_type"]] = row
    if not bundles:
        return None

    complete = [bundle for bundle in bundles.values() if all(task in bundle for task in BUNDLE_LADDER)]
    foundations = ("support_set", "legal_actions")
    foundation_ok = [all(bundle[task]["correct"] for task in foundations) for bundle in complete]

    conditional = {}
    for task in BUNDLE_LADDER[2:]:
        eligible = [
            bundle for bundle in complete
            if all(bundle[name]["correct"] for name in foundations)
        ]
        conditional[task] = {
            "n": len(eligible),
            "accuracy": _mean([bundle[task]["correct"] for bundle in eligible]),
        }

    survival = {}
    for i, task in enumerate(BUNDLE_LADDER):
        prefix = BUNDLE_LADDER[: i + 1]
        survival[task] = _mean([
            all(bundle[name]["correct"] for name in prefix)
            for bundle in complete
        ])

    return {
        "n_bundles": len(bundles),
        "complete_bundles": len(complete),
        "foundations_all_correct": _mean(foundation_ok),
        "conditional_on_state_and_permission": conditional,
        "ladder_survival": survival,
    }


def _factorial_metrics(by_cell):
    required = {
        "commuting_relevant",
        "noncommuting_relevant",
        "commuting_irrelevant",
        "noncommuting_irrelevant",
    }
    if not required.issubset(by_cell):
        return None
    penalty_relevant = (
        by_cell["commuting_relevant"]["accuracy"]
        - by_cell["noncommuting_relevant"]["accuracy"]
    )
    penalty_irrelevant = (
        by_cell["commuting_irrelevant"]["accuracy"]
        - by_cell["noncommuting_irrelevant"]["accuracy"]
    )
    return {
        "noncommutativity_penalty_relevant": penalty_relevant,
        "noncommutativity_penalty_irrelevant": penalty_irrelevant,
        "relevance_difference_in_differences": penalty_relevant - penalty_irrelevant,
    }


def _contiguous_frontier(series, threshold=0.95):
    frontier = None
    for count in sorted(int(value) for value in series):
        summary = series[str(count)]
        if summary["accuracy"] < threshold:
            break
        frontier = count
    return frontier


def _composition_curve_metrics(rows):
    curve_rows = [row for row in rows if row.get("curve_id")]
    if not curve_rows:
        return None

    counts = sorted({row["relevant_interfering_pairs"] for row in curve_rows})
    sizes = sorted({row["n_components"] for row in curve_rows})
    by_count = {
        str(count): _summary(
            [row for row in curve_rows if row["relevant_interfering_pairs"] == count]
        )
        for count in counts
    }
    by_size = {
        str(size): _summary([row for row in curve_rows if row["n_components"] == size])
        for size in sizes
    }
    by_task_and_count = {}
    for task in sorted({row["task_type"] for row in curve_rows}):
        by_task_and_count[task] = {
            str(count): _summary([
                row for row in curve_rows
                if row["task_type"] == task
                and row["relevant_interfering_pairs"] == count
            ])
            for count in counts
            if any(
                row["task_type"] == task
                and row["relevant_interfering_pairs"] == count
                for row in curve_rows
            )
        }

    baseline = by_count.get("0", {}).get("accuracy")
    accuracy_drop = {
        str(count): (baseline - by_count[str(count)]["accuracy"] if baseline is not None else None)
        for count in counts
    }
    frontier_by_task = {
        task: _contiguous_frontier(series, threshold=0.95)
        for task, series in by_task_and_count.items()
    }
    return {
        "n_curves": len({row["curve_id"] for row in curve_rows}),
        "by_relevant_interfering_pairs": by_count,
        "by_problem_size": by_size,
        "by_task_and_relevant_interfering_pairs": by_task_and_count,
        "accuracy_drop_from_zero": accuracy_drop,
        "frontier_95_by_task": frontier_by_task,
    }


def score_predictions(dataset, predictions):
    by_id = {prediction["id"]: prediction for prediction in predictions}
    rows = []
    for item in dataset:
        prediction = by_id.get(item["id"], {})
        valid = set(item["choices"])
        parsed = prediction.get("parsed_answer") or parse_answer(prediction.get("response", ""), valid)
        metadata = item.get("metadata", {})
        usage = prediction.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        correct = parsed == item["answer"]
        rows.append(
            {
                "id": item["id"],
                "task_type": item["task_type"],
                "gold": item["answer"],
                "prediction": parsed,
                "correct": correct,
                "pair_id": metadata.get("pair_id"),
                "pair_type": metadata.get("pair_type"),
                "variant": metadata.get("variant"),
                "bundle_id": metadata.get("bundle_id"),
                "bundle_stage": metadata.get("bundle_stage"),
                "curve_id": metadata.get("curve_id"),
                "n_components": metadata.get("n_components"),
                "problem_size": metadata.get("problem_size"),
                "total_interfering_pairs": metadata.get("total_interfering_pairs"),
                "relevant_fraction": metadata.get("relevant_fraction"),
                "interaction_episode_id": metadata.get("interaction_episode_id"),
                "relevance_episode_id": metadata.get("relevance_episode_id"),
                "interaction_probe": metadata.get("interaction_probe"),
                "interaction_variant": metadata.get("variant") if metadata.get("pair_type") == "interaction" else None,
                "interference_present": metadata.get("interference_present"),
                "interference_relevant": metadata.get("interference_relevant"),
                "factorial_cell": metadata.get("factorial_cell"),
                "effect_family": metadata.get("effect_family"),
                "relevant_interfering_pairs": metadata.get("relevant_interfering_pairs"),
                "irrelevant_interfering_pairs": metadata.get("irrelevant_interfering_pairs"),
                "n_noncommuting_pairs": metadata.get("n_noncommuting_pairs"),
                "conflict_density": metadata.get("conflict_density"),
                "dependency_depth": metadata.get("dependency_depth"),
                "schedule_count": metadata.get("schedule_count"),
                "reachable_states": metadata.get("reachable_states"),
                "terminal_observations": metadata.get("terminal_observations"),
                "terminal_outcome_count": metadata.get("terminal_outcome_count"),
                "effect_ops_total": metadata.get("effect_ops_total"),
                "reasoning_effort": prediction.get("reasoning_effort"),
                "reasoning_tokens": details.get("reasoning_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost": usage.get("cost"),
            }
        )

    overall = _summary(rows)
    by_task = {}
    for task_type in sorted({row["task_type"] for row in rows}):
        by_task[task_type] = _summary([row for row in rows if row["task_type"] == task_type])

    by_capability = {}
    capabilities = {CAPABILITY_BY_TASK.get(row["task_type"], "other") for row in rows}
    for capability in sorted(capabilities):
        by_capability[capability] = _summary(
            [row for row in rows if CAPABILITY_BY_TASK.get(row["task_type"], "other") == capability]
        )

    by_interaction_variant = {}
    variants = sorted({row["interaction_variant"] for row in rows if row["interaction_variant"]})
    for variant in variants:
        by_interaction_variant[variant] = _summary(
            [row for row in rows if row["interaction_variant"] == variant]
        )

    by_factorial_cell = {}
    cells = sorted({row["factorial_cell"] for row in rows if row["factorial_cell"]})
    for cell in cells:
        by_factorial_cell[cell] = _summary(
            [row for row in rows if row["factorial_cell"] == cell]
        )

    groups = {}
    for row in rows:
        if row["pair_id"]:
            groups.setdefault((row["pair_type"], row["pair_id"]), []).append(row)

    pair_metrics = {}
    pair_types = (
        "semantic",
        "isomorphism",
        "interaction",
        "relevant_interference",
        "irrelevant_interference",
    )
    for pair_type in pair_types:
        pairs = [group for (kind, _), group in groups.items() if kind == pair_type and len(group) == 2]
        both_correct = [all(row["correct"] for row in pair) for pair in pairs]
        valid_pairs = [pair for pair in pairs if all(row["prediction"] is not None for row in pair)]
        common = {
            "n_pairs": len(pairs),
            "parseable_pairs": len(valid_pairs),
            "both_correct": _mean(both_correct),
        }
        if pair_type == "semantic":
            changed = [pair[0]["prediction"] != pair[1]["prediction"] for pair in valid_pairs]
            pair_metrics[pair_type] = {
                **common,
                "semantic_sensitivity": _mean(changed),
                "gold_changes": sum(pair[0]["gold"] != pair[1]["gold"] for pair in pairs),
            }
        elif pair_type == "interaction":
            changed = [pair[0]["prediction"] != pair[1]["prediction"] for pair in valid_pairs]
            pair_metrics[pair_type] = {
                **common,
                "interaction_sensitivity": _mean(changed),
                "gold_changes": sum(pair[0]["gold"] != pair[1]["gold"] for pair in pairs),
            }
        elif pair_type == "relevant_interference":
            changed = [pair[0]["prediction"] != pair[1]["prediction"] for pair in valid_pairs]
            pair_metrics[pair_type] = {
                **common,
                "relevance_sensitivity": _mean(changed),
                "gold_changes": sum(pair[0]["gold"] != pair[1]["gold"] for pair in pairs),
            }
        elif pair_type == "irrelevant_interference":
            same = [pair[0]["prediction"] == pair[1]["prediction"] for pair in valid_pairs]
            pair_metrics[pair_type] = {
                **common,
                "irrelevance_consistency": _mean(same),
                "gold_same": sum(pair[0]["gold"] == pair[1]["gold"] for pair in pairs),
            }
        else:
            same = [pair[0]["prediction"] == pair[1]["prediction"] for pair in valid_pairs]
            pair_metrics[pair_type] = {
                **common,
                "isomorphism_consistency": _mean(same),
            }

    result = {
        **overall,
        "by_task": by_task,
        "by_capability": by_capability,
        "pair_metrics": pair_metrics,
        "rows": rows,
    }
    if by_interaction_variant:
        result["by_interaction_variant"] = by_interaction_variant
    if by_factorial_cell:
        result["by_factorial_cell"] = by_factorial_cell
        factorial_metrics = _factorial_metrics(by_factorial_cell)
        if factorial_metrics is not None:
            result["relevance_factorial"] = factorial_metrics
    bundle_metrics = _bundle_metrics(rows)
    if bundle_metrics is not None:
        result["bundle_metrics"] = bundle_metrics
    composition_curve = _composition_curve_metrics(rows)
    if composition_curve is not None:
        result["composition_curve"] = composition_curve
    return result
