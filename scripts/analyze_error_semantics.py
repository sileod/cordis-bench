"""Explain native schedule and reconfiguration errors with alternative semantics.

Each Cordis-native prompt is self-contained, so the structure is re-parsed from
the prompt text. Every queried slot is either *independent* (one dependent
writes it) or *interfering* (two dependents capture and restore the same slot).
Wrong answers are matched to the value a simpler, incorrect semantics predicts.

Usage: python scripts/analyze_error_semantics.py DATASET SCORE.json [...] --output out.json
"""

import argparse
import collections
import json
import re
from pathlib import Path

LEAF = re.compile(r"const (\w+) = leaf\('(\w+)', '(\w+)', '(\w+)', (\d+)\)")
PROVIDER = re.compile(r"const (\w+) = provider\('(\w+)', '(\w+)'\)")


def parse_native(prompt):
    harness = json.loads(re.search(r"const harness = (\{.*?\})", prompt).group(1))
    leaves = {m[1]: dict(service=m[2], slot=m[3], value=int(m[4])) for m in LEAF.findall(prompt)}
    services = {m[1]: m[2] for m in PROVIDER.findall(prompt)}
    started = [n for n in re.findall(r"await root\.plugin\((\w+)\)", prompt) if n in leaves]
    disposed = re.search(r"dispose\((\w+)\)", prompt.split("Maintenance action:")[1]).group(1)
    schedules = {k: v.split(" -> ") for k, v in re.findall(r"(schedule_\d+): (.+)", prompt)}
    affected = [n for n in started if leaves[n]["service"] == services[disposed]]
    by_slot = collections.defaultdict(list)
    for n in affected:
        by_slot[leaves[n]["slot"]].append(n)  # startup order within slot
    return dict(harness=harness, leaves=leaves, started=started, affected=affected,
                by_slot=dict(by_slot), schedules=schedules, prompt=prompt)


def simulate(w, order, predisposed=()):
    """True semantics: capture on start, restore captured value on dispose."""
    state, prev = dict(w["harness"]), {}
    for n in w["started"]:
        leaf = w["leaves"][n]
        prev[n], state[leaf["slot"]] = state[leaf["slot"]], leaf["value"]
    for n in [*predisposed, *[n for n in order if n not in predisposed]]:
        state[w["leaves"][n]["slot"]] = prev[n]
    return state


def classify_value(w, slot, schedule, pred):
    names = w["by_slot"][slot]
    base, truth = w["harness"][slot], simulate(w, w["schedules"][schedule])[slot]
    kind = "interfering" if len(names) > 1 else "independent"
    if pred == truth:
        return kind, "correct"
    values = [w["leaves"][n]["value"] for n in names]
    if pred == base:
        return kind, "assumed_nested_restore"  # as if lifetimes were LIFO-nested
    if kind == "interfering" and pred == values[0]:
        return kind, "spurious_leak"
    if pred in values:
        return kind, "ignored_cleanup"  # value set at startup survives
    return kind, "other"


def schedule_rows(item, pred):
    w = parse_native(item["prompt"])
    schedule = re.search(r"completion order `(schedule_\d+)`", item["prompt"]).group(1)
    slots = re.findall(r"\d+\. harness\['(\w+)'\]", item["prompt"])
    pred = pred if isinstance(pred, list) and len(pred) == len(slots) else [None] * len(slots)
    return [classify_value(w, s, schedule, p) for s, p in zip(slots, pred)]


def reconfiguration_row(item, pred, gold):
    w = parse_native(item["prompt"])
    pairs = [names for names in w["by_slot"].values() if len(names) > 1]
    pred = set(pred) if isinstance(pred, list) else None
    if pred is None:
        return {"malformed": 1}
    later = {names[-1] for names in pairs}
    earlier = {names[0] for names in pairs}
    counts = collections.Counter(
        extra_partner=len(pred & earlier & {n for p in pairs if p[-1] in pred for n in p}),
        wrong_member=sum(p[0] in pred and p[-1] not in pred for p in pairs),
        missed_pair=sum(not set(p) & pred for p in pairs),
        # independent dependents: the provider cascade would dispose them with the same result
        duplicated_cascade=len(pred & (set(w["affected"]) - later - earlier)),
        unrelated=len(pred - set(w["affected"])),
    )
    counts["exact"] = int(pred == set(gold))
    return counts


def load(path):
    return [json.loads(line) for line in open(path)]


def analyze(dataset, score_path):
    rows = json.load(open(score_path))["rows"]
    out = {"schedule": collections.defaultdict(collections.Counter),
           "reconfiguration": collections.defaultdict(collections.Counter)}
    for row in rows:
        item = dataset.get(row["id"])
        if item is None or not item["task_type"].startswith("cordis_"):
            continue
        size = str(item["metadata"]["semantic_size"])
        try:
            pred = json.loads(row["prediction"]) if row["prediction"] else None
        except (TypeError, ValueError):
            pred = None
        if item["task_type"] == "cordis_schedule_prediction":
            for kind, label in schedule_rows(item, pred):
                out["schedule"][f"{kind}|{size}"][label] += 1
        elif item["task_type"] == "cordis_reconfiguration_set":
            out["reconfiguration"][size].update(reconfiguration_row(item, pred, json.loads(row["gold"])))
    return {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("scores", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = {item["id"]: item for item in load(args.dataset)}
    report = {Path(p).name.removesuffix("-score.json"): analyze(dataset, p) for p in args.scores}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
