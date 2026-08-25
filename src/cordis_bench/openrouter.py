"""Model probing through litlm with CordisBench-owned JSONL checkpoints."""

import json
import os

import litlm

from .evaluate import parse_answer
from .v2_answers import parse_item_answer as parse_typed_item_answer


DEFAULT_MODELS = [
    "google/gemini-3.7-flash",
    "openai/gpt-5.6-luna",
    "x-ai/grok-4.6",
]

EFFORT_ALIASES = {
    "instant": "none",
}
VALID_EFFORTS = {"none", "minimal", "low", "medium", "high"}


def normalize_effort(effort):
    if effort is None:
        return None
    effort = EFFORT_ALIASES.get(effort, effort)
    if effort not in VALID_EFFORTS:
        raise ValueError(f"unknown reasoning effort: {effort}")
    return effort


def parse_item_answer(item, text):
    if item.get("answer_type"):
        parsed = parse_typed_item_answer(item, text)
        if parsed is not None:
            return parsed
        if not item.get("choices"):
            return None
    return parse_answer(text, set(item.get("choices", {})))


def _plain_mapping(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    try:
        return dict(value)
    except (TypeError, ValueError):
        return None


def _prediction_from_result(item, model, effort, result):
    choice = result.choices[0]
    usage = _plain_mapping(getattr(result, "usage", None)) or {}
    if usage.get("cost") is None and result.cost is not None:
        usage["cost"] = result.cost
    response_ms = getattr(result, "_response_ms", None)
    return {
        "id": item["id"],
        # Keep the requested route stable for checkpoint matching and record
        # litlm's resolved route separately.
        "model": model,
        "provider_model": result.model_used,
        "reasoning_effort": effort,
        "response": str(result),
        "parsed_answer": parse_item_answer(item, str(result)),
        "finish_reason": getattr(choice, "finish_reason", None),
        "native_finish_reason": getattr(choice, "native_finish_reason", None),
        "latency_s": response_ms / 1000 if response_ms is not None else None,
        "usage": usage or None,
        "litlm_failed": False,
    }


def probe_one(
    item,
    model,
    api_key=None,
    temperature=0.0,
    max_tokens=4096,
    reasoning_effort="low",
    timeout=120.0,
    attempt_timeout=None,
    retries=3,
    fallbacks=None,
):
    effort = normalize_effort(reasoning_effort)
    request = dict(
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=effort,
        timeout=timeout,
        attempt_timeout=attempt_timeout,
        num_retries=retries,
        fallbacks=fallbacks,
        show_progress=False,
    )
    if api_key is not None:
        request["api_key"] = api_key
    result = litlm.complete(item["prompt"], model=model, **request)
    if result.failed:
        raise result.error
    return _prediction_from_result(item, model, effort, result)


def _load_checkpoint(path, model, effort):
    completed = {}
    if not path or not os.path.exists(path):
        return completed
    with open(path) as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if (
                row.get("model") == model
                and row.get("reasoning_effort") == effort
                and row.get("id")
                and row.get("finish_reason") != "error"
                and row.get("litlm_failed") is not True
            ):
                completed[row["id"]] = row
    return completed


def probe_dataset(
    dataset,
    model,
    api_key=None,
    workers=4,
    output_path=None,
    resume=True,
    temperature=0.0,
    max_tokens=4096,
    reasoning_effort="low",
    timeout=120.0,
    attempt_timeout=None,
    retries=3,
    rpm=None,
    fallbacks=None,
):
    effort = normalize_effort(reasoning_effort)
    completed = _load_checkpoint(output_path, model, effort) if resume else {}
    pending = [item for item in dataset if item["id"] not in completed]
    if not pending:
        return [completed[item["id"]] for item in dataset]

    checkpoint = open(output_path, "a") if output_path else None

    def persist(index, result):
        if result.failed:
            return
        item = pending[index]
        prediction = _prediction_from_result(item, model, effort, result)
        completed[item["id"]] = prediction
        if checkpoint:
            checkpoint.write(json.dumps(prediction, separators=(",", ":")) + "\n")
            checkpoint.flush()

    try:
        request = dict(
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=effort,
            timeout=timeout,
            attempt_timeout=attempt_timeout,
            num_retries=retries,
            max_concurrency=workers,
            rpm=rpm,
            on_result=persist,
            fallbacks=fallbacks,
            show_progress=True,
        )
        if api_key is not None:
            request["api_key"] = api_key
        results = litlm.complete(
            [item["prompt"] for item in pending], model=model, **request
        )
    finally:
        if checkpoint:
            checkpoint.close()

    failures = [result for result in results if result.failed]
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(pending)} probes failed; successful results "
            "were checkpointed, so rerun the same command to resume"
        ) from failures[0].error
    return [completed[item["id"]] for item in dataset]
