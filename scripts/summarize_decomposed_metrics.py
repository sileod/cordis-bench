#!/usr/bin/env python3
"""Summarize size-normalized structured-output metrics for release scores."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    "Gemini": ROOT / "results" / "gemini-v2.0.1-release-score.json",
    "Luna": ROOT / "results" / "luna-v2.0.1-release-score.json",
}
OUT = ROOT / "results" / "decomposed-v2.0.1-release-summary.json"
REALIZATIONS = ("formal", "cordis_native")
SET_CAPABILITIES = {"localization", "postcondition_guarantee", "reachability"}


def load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def primary_rows(payload):
    return [row for row in payload["rows"] if row.get("paper_role") == "primary"]


def parsed(rows):
    return [row for row in rows if row.get("prediction") is not None]


def exact_match(row):
    return bool(row.get("exact_match", row.get("correct")))


def structured(value):
    return value if isinstance(value, list) else json.loads(value)


def ratio(num, den):
    return num / den if den else None


def raw_exact_match(rows):
    return ratio(sum(exact_match(row) for row in rows), len(rows))


def parse_rate(rows):
    return ratio(len(parsed(rows)), len(rows))


def parsed_exact_match(rows):
    rows = parsed(rows)
    return ratio(sum(exact_match(row) for row in rows), len(rows))


def observable_accuracy(rows):
    rows = [row for row in parsed(rows) if row.get("answer_type") == "scalar_sequence"]
    correct = total = 0
    for row in rows:
        gold = structured(row["gold"])
        prediction = structured(row["prediction"])
        correct += sum(
            i < len(prediction) and prediction[i] == value
            for i, value in enumerate(gold)
        )
        total += len(gold)
    return ratio(correct, total)


def set_jaccard(rows):
    rows = [
        row
        for row in parsed(rows)
        if row.get("answer_type") == "string_set"
        and row.get("capability") in SET_CAPABILITIES
    ]
    scores = []
    for row in rows:
        gold = set(structured(row["gold"]))
        prediction = set(structured(row["prediction"]))
        union = gold | prediction
        scores.append(len(gold & prediction) / len(union) if union else 1.0)
    return sum(scores) / len(scores) if scores else None


def summarize(rows):
    sizes = sorted({int(row["semantic_size"]) for row in rows})
    result = {}
    for realization in REALIZATIONS:
        realization_rows = [row for row in rows if row.get("realization") == realization]
        result[realization] = {
            "overall": {
                "n": len(realization_rows),
                "raw_exact_match": raw_exact_match(realization_rows),
                "parse_rate": parse_rate(realization_rows),
                "parsed_exact_match": parsed_exact_match(realization_rows),
                "per_observable_accuracy": observable_accuracy(realization_rows),
                "set_jaccard": set_jaccard(realization_rows),
            },
            "by_semantic_size": {},
            "by_capability": {},
        }
        for size in sizes:
            subset = [
                row for row in realization_rows
                if int(row["semantic_size"]) == size
            ]
            if not subset:
                continue
            result[realization]["by_semantic_size"][str(size)] = {
                "n": len(subset),
                "raw_exact_match": raw_exact_match(subset),
                "parse_rate": parse_rate(subset),
                "parsed_exact_match": parsed_exact_match(subset),
                "per_observable_accuracy": observable_accuracy(subset),
                "set_jaccard": set_jaccard(subset),
            }
        for capability in sorted({row["capability"] for row in realization_rows}):
            capability_rows = [row for row in realization_rows if row["capability"] == capability]
            result[realization]["by_capability"][capability] = {
                str(size): raw_exact_match(
                    [row for row in capability_rows if int(row["semantic_size"]) == size]
                )
                for size in sizes
                if any(int(row["semantic_size"]) == size for row in capability_rows)
            }
    return result


def main():
    payload = {
        "release": "v2.0.1",
        "metrics": {
            "raw_exact_match": "whole-answer exact match over all primary rows",
            "parse_rate": "fraction of primary rows with a parsed prediction",
            "parsed_exact_match": "whole-answer exact match conditional on parsing",
            "per_observable_accuracy": "micro-averaged requested-position accuracy for parsed scalar_sequence rows; missing positions are errors",
            "set_jaccard": "macro-averaged Jaccard over parsed string_set localization, guarantee, and reachability rows",
        },
        "models": {},
    }
    for model, path in MODEL_FILES.items():
        payload["models"][model] = summarize(primary_rows(load(path)))
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
