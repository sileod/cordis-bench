"""Reconfiguration with an exact harness dry-run in the loop.

The harness can execute any candidate plan in a sandbox (real Cordis, every
listed completion order) and report the outcome. The model may dry-run up to
``--budget`` plans before submitting. This measures what anticipation still buys
when verification is free: dry runs, tokens, and plan minimality.

Usage:
  python scripts/whatif_loop.py DATASET --model albert/deepseek-v4-flash-0731 \
      --output whatif.jsonl [--budget 6 --workers 8 --effort low]
Then score the submitted plans with the standard executor:
  cordis-bench score DATASET whatif.jsonl --output whatif-score.json
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litlm

from cordis_bench.openrouter import _plain_mapping
from cordis_bench.v18 import execute_native_specs

PROTOCOL = """

You have access to the harness's dry-run facility. To test a candidate set, reply
with a line of the form
DRY_RUN: ["plugin_a", "plugin_b"]
The harness then executes, in a sandbox, those explicit disposals (in the given
order) followed by the maintenance action under every listed completion order,
and reports the final target values. You may use at most {budget} dry runs.
Dry runs cost time, so use them only when useful. When ready, reply with
FINAL: <sorted JSON array>"""

ACTION = re.compile(r"(DRY_RUN|FINAL)\s*:\s*(\[[^\]]*\])", re.S)


def last_action(text):
    matches = ACTION.findall(text or "")
    if not matches:
        return None, None
    kind, payload = matches[-1]
    try:
        names = json.loads(payload)
    except ValueError:
        return kind, None
    return kind, names if all(isinstance(n, str) for n in names) else None


def dry_run(item, names):
    spec = dict(item["oracle"]["spec"])
    known = {leaf["name"] for leaf in spec["leaves"]} | {spec["decoy"]}
    unknown = [n for n in names if n not in known]
    if unknown:
        return f"Dry run rejected: unknown dependent plugin(s) {unknown}.", False
    spec["options"] = {"DRY": [*({"kind": "dispose", "name": n} for n in names),
                               {"kind": "dispose", "name": spec["query_provider"]}]}
    runtime = execute_native_specs([spec])[spec["native_case_id"]]
    target = {s: spec["target"]["slots"][s] for s in item["oracle"]["query_slots"]}
    lines, ok = [], 0
    for label, run in zip(spec["schedules"], runtime["option_runs"]["DRY"]):
        slots = run["observation"]["slots"]
        misses = [f"{s}={slots[s]} (target {v})" for s, v in target.items() if slots[s] != v]
        ok += not misses
        lines.append(f"  {label}: " + ("target reached" if not misses else "MISSED " + ", ".join(misses)))
    header = f"Dry run of {json.dumps(names)}: target reached under {ok}/{len(lines)} orders."
    return "\n".join([header, *lines]), ok == len(lines)


def run_item(item, args):
    messages = [{"role": "user", "content": item["prompt"] + PROTOCOL.format(budget=args.budget)}]
    trace, tokens, dry_runs, final = [], 0, 0, None
    start = time.monotonic()
    for _turn in range(args.budget + 2):
        result = litlm.complete(messages, model=args.model, max_tokens=args.max_tokens,
                                temperature=0.0, reasoning_effort=args.effort,
                                timeout=args.timeout, show_progress=False)
        if result.failed:
            raise result.error
        text = str(result)
        tokens += (_plain_mapping(getattr(result, "usage", None)) or {}).get("completion_tokens") or 0
        messages.append({"role": "assistant", "content": text})
        kind, names = last_action(text)
        if kind is None and _turn > 0:  # protocol ignored twice: take the last array
            arrays = re.findall(r"\[[^\[\]]*\]", text)
            kind, names = last_action(f"FINAL: {arrays[-1]}") if arrays else ("FINAL", None)
        if kind == "FINAL":
            final = names
            break
        if kind == "DRY_RUN" and names is not None and dry_runs < args.budget:
            dry_runs += 1
            feedback, success = dry_run(item, names)
            trace.append({"plan": names, "success": success})
            note = "" if dry_runs < args.budget else "\nNo dry runs left; reply with FINAL."
            messages.append({"role": "user", "content": feedback + note})
        else:
            messages.append({"role": "user", "content": "Reply with FINAL: <sorted JSON array>."})
    answer = json.dumps(sorted(final)) if final is not None else ""
    return {
        "id": item["id"], "model": args.model, "reasoning_effort": args.effort,
        "response": answer, "parsed_answer": answer or None,
        "dry_runs": dry_runs, "trace": trace, "completion_tokens": tokens,
        "turns": len(messages) // 2, "latency_s": time.monotonic() - start, "semantic_size": item["metadata"]["semantic_size"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.dataset)]
    items = [i for i in items if i["task_type"] == "cordis_reconfiguration_set"]
    done = set()
    if os.path.exists(args.output):
        done = {json.loads(line)["id"] for line in open(args.output)}
    pending = [i for i in items if i["id"] not in done]
    with open(args.output, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_item, item, args): item["id"] for item in pending}
        for future in as_completed(futures):
            try:
                out.write(json.dumps(future.result()) + "\n")
                out.flush()
            except Exception as error:  # resumable: rerun to retry failures
                print(f"{futures[future]} failed: {error}")


if __name__ == "__main__":
    main()
