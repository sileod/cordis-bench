"""Answer-aware and executable scoring for CordisBench V2."""

from __future__ import annotations

from collections import Counter
import json

from .v18 import execute_native_specs
from .v2_answers import parse_item_answer


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def string_set_jaccard(gold, prediction):
    """Return Jaccard similarity for canonical JSON string-set answers.

    A missing or malformed prediction has zero overlap. Both empty sets are a
    perfect match. ``None`` is reserved for a malformed benchmark gold value.
    """
    try:
        gold_values = json.loads(gold)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(gold_values, list) or not all(
        isinstance(value, str) for value in gold_values
    ):
        return None
    if prediction is None:
        return 0.0
    try:
        predicted_values = json.loads(prediction)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if not isinstance(predicted_values, list) or not all(
        isinstance(value, str) for value in predicted_values
    ):
        return 0.0
    gold_set = set(gold_values)
    predicted_set = set(predicted_values)
    union = gold_set | predicted_set
    return len(gold_set & predicted_set) / len(union) if union else 1.0


def _summary(rows):
    n = len(rows)
    parsed = [row for row in rows if row["prediction"] is not None]
    return {
        "n": n,
        "accuracy": sum(row["correct"] for row in rows) / n if n else 0.0,
        "parsed": len(parsed),
        "unparsed": n - len(parsed),
        "parsed_rate": len(parsed) / n if n else 0.0,
        "answered_accuracy": (
            sum(row["correct"] for row in parsed) / len(parsed) if parsed else None
        ),
        "mean_jaccard": _mean(row.get("jaccard") for row in rows),
        "mean_reasoning_tokens": _mean(row.get("reasoning_tokens") for row in rows),
        "mean_completion_tokens": _mean(row.get("completion_tokens") for row in rows),
        "mean_cost": _mean(row.get("cost") for row in rows),
    }


def _group(rows, key):
    values = sorted({row.get(key) for row in rows if row.get(key) is not None}, key=str)
    return {
        str(value): _summary([row for row in rows if row.get(key) == value])
        for value in values
    }


def task_metric_summary(rows):
    """Summarize rows with the task-appropriate primary semantic metric.

    The rows must represent one capability. Scores are normalized to ``[0, 1]``.
    Sequence prediction is micro-averaged over observables; set tasks are
    macro-averaged over item-level Jaccard; executable actions use success rate.
    """
    if not rows:
        raise ValueError("cannot summarize an empty row collection")
    capabilities = {row.get("capability") for row in rows}
    if len(capabilities) != 1:
        raise ValueError(f"expected one capability, found {sorted(capabilities, key=str)}")
    capability = next(iter(capabilities))
    if capability in {"localization", "postcondition_guarantee", "reachability"}:
        return {
            "metric": "jaccard",
            "score": sum(float(row.get("jaccard") or 0.0) for row in rows) / len(rows),
            "n": len(rows),
        }
    if capability == "prediction":
        correct = total = 0
        for row in rows:
            try:
                gold = json.loads(row["gold"])
            except (json.JSONDecodeError, TypeError):
                raise ValueError(f"row {row.get('id')} has malformed scalar-sequence gold")
            try:
                prediction = json.loads(row["prediction"]) if row.get("prediction") else []
            except (json.JSONDecodeError, TypeError):
                prediction = []
            if not isinstance(gold, list):
                raise ValueError(f"row {row.get('id')} has non-sequence gold")
            if not isinstance(prediction, list):
                prediction = []
            correct += sum(
                index < len(prediction) and prediction[index] == value
                for index, value in enumerate(gold)
            )
            total += len(gold)
        return {
            "metric": "per_observable_accuracy",
            "score": correct / total if total else 0.0,
            "n": len(rows),
            "observables": total,
        }
    if capability == "act":
        return {
            "metric": "executed_success_rate",
            "score": sum(bool(row.get("correct")) for row in rows) / len(rows),
            "n": len(rows),
        }
    raise ValueError(f"no primary metric for capability {capability!r}")


def task_appropriate_metrics(rows):
    """Return task-appropriate summaries for primary evaluation rows."""
    primary = [row for row in rows if row.get("paper_role") == "primary"]
    return {
        task: task_metric_summary([row for row in primary if row["task_type"] == task])
        for task in sorted({row["task_type"] for row in primary})
    }


def _execution_scores(dataset, parsed_by_id, runner_path=None, node="node"):
    prepared = []
    statuses = {}
    for item in dataset:
        oracle = item.get("oracle") or {}
        if oracle.get("kind") != "cordis_reconfiguration":
            continue
        parsed = parsed_by_id.get(item["id"])
        if parsed is None:
            statuses[item["id"]] = {"status": "malformed", "success": False}
            continue
        try:
            names = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            statuses[item["id"]] = {"status": "malformed", "success": False}
            continue
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            statuses[item["id"]] = {"status": "malformed", "success": False}
            continue

        spec = dict(oracle["spec"])
        known = {leaf["name"] for leaf in spec["leaves"]} | {spec["decoy"]}
        if any(name not in known or name == spec["query_provider"] for name in names):
            statuses[item["id"]] = {"status": "invalid_action", "success": False}
            continue
        plan = [
            *({"kind": "dispose", "name": name} for name in names),
            {"kind": "dispose", "name": spec["query_provider"]},
        ]
        spec["options"] = {"MODEL": plan}
        prepared.append((item, names, spec))

    if prepared:
        try:
            runtimes = execute_native_specs(
                [spec for _item, _names, spec in prepared],
                runner_path=runner_path,
                node=node,
            )
        except Exception as error:
            for item, _names, _spec in prepared:
                statuses[item["id"]] = {
                    "status": "runtime_failure",
                    "success": False,
                    "runtime_error": str(error),
                }
        else:
            for item, names, spec in prepared:
                runtime = runtimes[spec["native_case_id"]]
                runs = runtime.get("option_runs", {}).get("MODEL", [])
                query_slots = item["oracle"]["query_slots"]
                target = spec["target"]["slots"]
                query_success = bool(runs) and all(
                    all(run["observation"]["slots"][slot] == target[slot] for slot in query_slots)
                    for run in runs
                )
                full_success = runtime.get("option_success", {}).get("MODEL") is True
                minimum = int(item["oracle"]["minimum_predisposals"])
                if full_success and len(names) == minimum:
                    status = "success"
                    success = True
                elif full_success:
                    status = "nonminimal_success"
                    success = False
                elif query_success:
                    status = "collateral_change"
                    success = False
                else:
                    status = "missed_target"
                    success = False
                statuses[item["id"]] = {
                    "status": status,
                    "success": success,
                    "query_target_reached": query_success,
                    "full_target_reached": full_success,
                    "proposed_predisposals": len(names),
                    "minimum_predisposals": minimum,
                }
    return statuses


def score_v2_predictions(dataset, predictions, runner_path=None, node="node"):
    if not dataset or any(item.get("schema_version") != "2.0" for item in dataset):
        raise ValueError("score_v2_predictions requires a pure V2 dataset")
    by_id = {prediction["id"]: prediction for prediction in predictions}
    parsed_by_id = {}
    for item in dataset:
        prediction = by_id.get(item["id"], {})
        parsed = prediction.get("parsed_answer")
        if parsed is None:
            parsed = parse_item_answer(item, prediction.get("response", ""))
        parsed_by_id[item["id"]] = parsed

    execution = _execution_scores(
        dataset,
        parsed_by_id,
        runner_path=runner_path,
        node=node,
    )

    rows = []
    for item in dataset:
        prediction = by_id.get(item["id"], {})
        parsed = parsed_by_id[item["id"]]
        metadata = item.get("metadata") or {}
        usage = prediction.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        exact_match = parsed == item["answer"]
        jaccard = (
            string_set_jaccard(item["answer"], parsed)
            if item.get("answer_type") == "string_set"
            else None
        )
        execution_result = execution.get(item["id"])
        correct = execution_result["success"] if execution_result else exact_match
        rows.append(
            {
                "id": item["id"],
                "task_type": item["task_type"],
                "gold": item["answer"],
                "prediction": parsed,
                "correct": correct,
                "exact_match": exact_match,
                "jaccard": jaccard,
                "execution_status": execution_result.get("status") if execution_result else None,
                "benchmark_track": metadata.get("benchmark_track"),
                "benchmark_component": metadata.get("benchmark_component"),
                "realization": (
                    "cordis_native" if metadata.get("harness_surface") else "formal"
                ),
                "capability": metadata.get("capability"),
                "answer_type": item.get("answer_type"),
                "paper_role": metadata.get("paper_role"),
                "semantic_size": metadata.get("semantic_size"),
                "challenge_size": metadata.get("challenge_size"),
                "reasoning_effort": prediction.get("reasoning_effort"),
                "reasoning_tokens": details.get("reasoning_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost": usage.get("cost"),
            }
        )

    result = {
        **_summary(rows),
        "by_track": _group(rows, "benchmark_track"),
        "by_realization": _group(rows, "realization"),
        "by_component": _group(rows, "benchmark_component"),
        "by_task": _group(rows, "task_type"),
        "by_capability": _group(rows, "capability"),
        "by_answer_type": _group(rows, "answer_type"),
        "by_paper_role": _group(rows, "paper_role"),
        "rows": rows,
    }
    result["primary_metrics"] = task_appropriate_metrics(rows)

    challenge = [row for row in rows if row["benchmark_track"] == "challenge"]
    by_task_size = {}
    for task in sorted({row["task_type"] for row in challenge}):
        task_rows = [row for row in challenge if row["task_type"] == task]
        by_task_size[task] = {
            str(size): _summary([row for row in task_rows if row["semantic_size"] == size])
            for size in sorted({row["semantic_size"] for row in task_rows})
        }
    result["challenge"] = {
        "overall": _summary(challenge),
        "by_task_and_size": by_task_size,
    }

    execution_rows = [row for row in rows if row["execution_status"] is not None]
    if execution_rows:
        status_counts = Counter(row["execution_status"] for row in execution_rows)
        result["reconfiguration_execution"] = {
            "n": len(execution_rows),
            "success_rate": _mean(row["correct"] for row in execution_rows),
            "status_counts": dict(sorted(status_counts.items())),
        }
    return result
