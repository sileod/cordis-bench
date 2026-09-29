"""Compare harness-doc and what-if runs with the one-shot release run on the same IDs.

Usage:
  python scripts/summarize_harness_experiments.py --baseline results/X-release-score.json \
      [--doc results/doc-X-score.json] [--whatif whatif.jsonl results/whatif-X-score.json] \
      --output results/harness-experiments-X.json
"""

import argparse
import collections
import json
import statistics


def rows(path):
    return {r["id"]: r for r in json.load(open(path))["rows"]}


def per_task(score_rows, ids):
    """Primary metric per native task on the given IDs (per-observable for prediction)."""
    out = collections.defaultdict(list)
    for i in ids:
        r = score_rows[i]
        if r["task_type"] == "cordis_schedule_prediction":
            gold, pred = json.loads(r["gold"]), _load(r["prediction"])
            ok = [p == g for p, g in zip(pred, gold)] if isinstance(pred, list) and len(pred) == len(gold) else [False] * len(gold)
            out[r["task_type"]].extend(ok)
        else:
            out[r["task_type"]].append(r["correct"] in (True, "True"))
    return {t: round(100 * statistics.mean(v), 1) for t, v in out.items()}


def _load(text):
    try:
        return json.loads(text) if text else None
    except (TypeError, ValueError):
        return None


def statuses(score_rows, ids):
    return dict(collections.Counter(score_rows[i]["execution_status"] for i in ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--doc")
    parser.add_argument("--whatif", nargs=2, metavar=("TRACE", "SCORE"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = rows(args.baseline)
    report = {}
    if args.doc:
        doc = rows(args.doc)
        ids = sorted(set(doc) & set(base))
        reconf = [i for i in ids if doc[i]["task_type"] == "cordis_reconfiguration_set"]
        report["harness_doc"] = {
            "n": len(ids),
            "baseline": per_task(base, ids), "with_doc": per_task(doc, ids),
            "reconfiguration_status": {"baseline": statuses(base, reconf), "with_doc": statuses(doc, reconf)},
        }
    if args.whatif:
        traces = {t["id"]: t for t in map(json.loads, open(args.whatif[0]))}
        loop = rows(args.whatif[1])
        ids = sorted(set(traces) & set(loop) & set(base))
        by_size = collections.defaultdict(list)
        for i in ids:
            by_size[traces[i]["semantic_size"]].append(i)
        summary = lambda sel: {
            "n": len(sel),
            "one_shot_success": round(100 * statistics.mean(base[i]["correct"] in (True, "True") for i in sel), 1),
            "loop_success": round(100 * statistics.mean(loop[i]["correct"] in (True, "True") for i in sel), 1),
            "used_dry_run": round(100 * statistics.mean(traces[i]["dry_runs"] > 0 for i in sel), 1),
            "mean_dry_runs": round(statistics.mean(traces[i]["dry_runs"] for i in sel), 2),
            "dry_run_verified_final": round(100 * statistics.mean(
                any(t["success"] and sorted(t["plan"]) == sorted(_load(traces[i]["response"]) or []) for t in traces[i]["trace"])
                for i in sel), 1),
            "mean_latency_s": round(statistics.mean(traces[i].get("latency_s", 0) for i in sel), 1),
            "loop_status": statuses(loop, sel), "one_shot_status": statuses(base, sel),
        }
        report["whatif"] = {"all": summary(ids), "by_size": {s: summary(v) for s, v in sorted(by_size.items())}}
    json.dump(report, open(args.output, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
