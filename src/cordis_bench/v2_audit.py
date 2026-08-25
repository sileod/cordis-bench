"""Prompt-blind release audit for CordisBench V2 artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import re

from .io import read_jsonl

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _fold(group_id):
    digest = hashlib.sha256(str(group_id).encode()).digest()
    return digest[0] & 1


def _tokens(prompt):
    return set(TOKEN_RE.findall((prompt or "").lower()))


def _mode(rows, key_fn):
    table = defaultdict(Counter)
    for row in rows:
        table[key_fn(row)][row["answer"]] += 1
    return {
        key: counts.most_common(1)[0][0]
        for key, counts in table.items()
        if counts
    }


def _accuracy(pairs):
    pairs = list(pairs)
    return sum(pred == gold for pred, gold in pairs) / len(pairs) if pairs else None


def _cross_validated(records, predictor):
    scored = []
    for fold in (0, 1):
        train = [row for row in records if _fold(row["metadata"]["latent_group_id"]) != fold]
        test = [row for row in records if _fold(row["metadata"]["latent_group_id"]) == fold]
        if not train or not test:
            continue
        for row, prediction in predictor(train, test):
            scored.append((row, prediction))
    return scored


def _mode_predictor(key_fn):
    def predict(train, test):
        table = _mode(train, key_fn)
        fallback = Counter(row["answer"] for row in train).most_common(1)[0][0]
        for row in test:
            yield row, table.get(key_fn(row), fallback)
    return predict


def _length_predictor(train, test):
    by_task = defaultdict(list)
    for row in train:
        by_task[row["task_type"]].append(row)
    for row in test:
        candidates = by_task.get(row["task_type"], train)
        target = len(row["prompt"])
        nearest = min(
            candidates,
            key=lambda candidate: (abs(len(candidate["prompt"]) - target), candidate["id"]),
        )
        yield row, nearest["answer"]


def _lexical_predictor(train, test):
    by_task = defaultdict(list)
    cached = {}
    for row in train:
        tokens = _tokens(row["prompt"])
        cached[row["id"]] = tokens
        by_task[row["task_type"]].append(row)
    for row in test:
        query = _tokens(row["prompt"])
        candidates = by_task.get(row["task_type"], train)

        def score(candidate):
            tokens = cached[candidate["id"]]
            union = len(query | tokens)
            similarity = len(query & tokens) / union if union else 0.0
            return (similarity, candidate["id"])

        nearest = max(candidates, key=score)
        yield row, nearest["answer"]


def _position_predictor(period):
    def predict(train, test):
        index = {row["id"]: i for i, row in enumerate(_POSITION_RECORDS)}
        table = _mode(train, lambda row: (row["task_type"], index[row["id"]] % period))
        fallback = _mode(train, lambda row: row["task_type"])
        global_fallback = Counter(row["answer"] for row in train).most_common(1)[0][0]
        for row in test:
            key = (row["task_type"], index[row["id"]] % period)
            yield row, table.get(key, fallback.get(row["task_type"], global_fallback))
    return predict


def _summarize(scored):
    if not scored:
        return {"n": 0, "accuracy": None, "by_task": {}, "by_track": {}}
    rows = [row for row, _prediction in scored]
    accuracy = _accuracy((prediction, row["answer"]) for row, prediction in scored)
    by_task = {}
    for task in sorted({row["task_type"] for row in rows}):
        pairs = [(row, prediction) for row, prediction in scored if row["task_type"] == task]
        by_task[task] = {
            "n": len(pairs),
            "accuracy": _accuracy((prediction, row["answer"]) for row, prediction in pairs),
        }
    by_track = {}
    for track in ("core", "challenge"):
        pairs = [
            (row, prediction)
            for row, prediction in scored
            if row["metadata"]["benchmark_track"] == track
        ]
        if pairs:
            by_track[track] = {
                "n": len(pairs),
                "accuracy": _accuracy((prediction, row["answer"]) for row, prediction in pairs),
            }
    return {"n": len(scored), "accuracy": accuracy, "by_task": by_task, "by_track": by_track}


_POSITION_RECORDS = []


def audit_v2_shortcuts(records):
    """Run deterministic prompt-blind baselines with latent-group held-out folds."""
    global _POSITION_RECORDS
    records = list(records)
    if not records:
        raise ValueError("V2 release audit requires records")
    if any(row.get("schema_version") != "2.0" for row in records):
        raise ValueError("V2 release audit requires a pure V2 artifact")
    _POSITION_RECORDS = records

    predictors = {
        "task_only": _mode_predictor(lambda row: row["task_type"]),
        "task_and_semantic_size": _mode_predictor(
            lambda row: (row["task_type"], row["metadata"]["semantic_size"])
        ),
        "prompt_length_nearest": _length_predictor,
        "lexical_1nn_group_held_out": _lexical_predictor,
    }
    results = {
        name: _summarize(_cross_validated(records, predictor))
        for name, predictor in predictors.items()
    }

    periodic = {}
    for period in range(2, 9):
        summary = _summarize(_cross_validated(records, _position_predictor(period)))
        periodic[str(period)] = summary
    best_period, best_summary = max(
        periodic.items(),
        key=lambda item: (-math.inf if item[1]["accuracy"] is None else item[1]["accuracy"]),
    )
    results["record_position_periodic"] = {
        "best_period": int(best_period),
        **best_summary,
        "all_periods": {period: value["accuracy"] for period, value in periodic.items()},
    }

    primary = [row for row in records if row["metadata"].get("paper_role") == "primary"]
    primary_task_only = _summarize(
        _cross_validated(primary, _mode_predictor(lambda row: row["task_type"]))
    )
    blind_max = max(
        summary["accuracy"] or 0.0
        for name, summary in results.items()
        if name != "record_position_periodic"
    )
    blind_max = max(blind_max, results["record_position_periodic"]["accuracy"] or 0.0)

    return {
        "schema": "cordisbench-v2-shortcut-audit-1",
        "n": len(records),
        "n_latent_groups": len({row["metadata"]["latent_group_id"] for row in records}),
        "primary_task_only": primary_task_only,
        "baselines": results,
        "max_prompt_blind_accuracy": blind_max,
        "notes": [
            "All reported predictors use two-fold latent-group-held-out evaluation.",
            "These controls are necessary but do not replace model-based dependency-only/effect-only/native-identifier ablations.",
            "No options-only baseline exists because V2 exposes no candidate answer options.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(prog="python -m cordis_bench.v2_audit")
    parser.add_argument("dataset")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_v2_shortcuts(read_jsonl(args.dataset))
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
