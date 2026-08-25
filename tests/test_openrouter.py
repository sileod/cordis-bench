import json
from types import SimpleNamespace

from cordis_bench import openrouter


class Result(str):
    def __new__(cls, text, *, failed=False, error=None):
        value = super().__new__(cls, text)
        value.failed = failed
        value.error = error
        value.model_used = "openrouter/test/model"
        value.cost = 0.01
        value.usage = SimpleNamespace(
            model_dump=lambda mode=None: {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            }
        )
        value.choices = [
            SimpleNamespace(finish_reason="stop", native_finish_reason="stop")
        ]
        value._response_ms = 1500
        return value


def test_probe_dataset_checkpoints_each_settled_result_and_resumes(monkeypatch, tmp_path):
    dataset = [
        {"id": "a", "prompt": "a", "choices": {"A": "yes"}},
        {"id": "b", "prompt": "b", "choices": {"B": "yes"}},
    ]
    output = tmp_path / "predictions.jsonl"
    calls = []

    def fake_complete(inputs, on_result, **kwargs):
        calls.append(list(inputs))
        results = [Result("A"), Result("B")]
        # Settlement order need not match input order.
        on_result(1, results[1])
        assert len(output.read_text().splitlines()) == 1
        on_result(0, results[0])
        return results

    monkeypatch.setattr(openrouter.litlm, "complete", fake_complete)
    first = openrouter.probe_dataset(
        dataset,
        "test/model",
        api_key="key",
        workers=2,
        output_path=output,
        reasoning_effort="low",
    )
    assert [row["id"] for row in first] == ["a", "b"]
    assert calls == [["a", "b"]]
    assert len(output.read_text().splitlines()) == 2
    assert first[0]["provider_model"] == "openrouter/test/model"
    assert first[0]["latency_s"] == 1.5

    calls.clear()
    second = openrouter.probe_dataset(
        dataset,
        "test/model",
        api_key="key",
        workers=2,
        output_path=output,
        reasoning_effort="low",
    )
    assert [row["id"] for row in second] == ["a", "b"]
    assert calls == []
    assert all(json.loads(line)["model"] == "test/model" for line in output.read_text().splitlines())


def test_probe_dataset_checkpoints_successes_but_not_failures(monkeypatch, tmp_path):
    dataset = [
        {"id": "a", "prompt": "a", "choices": {"A": "yes"}},
        {"id": "b", "prompt": "b", "choices": {"B": "yes"}},
    ]
    output = tmp_path / "predictions.jsonl"

    def fake_complete(inputs, on_result, **kwargs):
        results = [Result("A"), Result("", failed=True, error=TimeoutError("slow"))]
        for index, result in enumerate(results):
            on_result(index, result)
        return results

    monkeypatch.setattr(openrouter.litlm, "complete", fake_complete)
    try:
        openrouter.probe_dataset(
            dataset, "test/model", api_key="key", output_path=output
        )
    except RuntimeError as error:
        assert "1 of 2 probes failed" in str(error)
    else:
        raise AssertionError("probe_dataset should report incomplete batches")

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["a"]


def test_checkpoint_loader_retries_error_rows_but_keeps_truncations(tmp_path):
    output = tmp_path / "predictions.jsonl"
    rows = [
        {
            "id": "error",
            "model": "test/model",
            "reasoning_effort": "low",
            "finish_reason": "error",
        },
        {
            "id": "length",
            "model": "test/model",
            "reasoning_effort": "low",
            "finish_reason": "length",
        },
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))

    completed = openrouter._load_checkpoint(output, "test/model", "low")

    assert set(completed) == {"length"}


def test_probe_one_uses_litlm(monkeypatch):
    calls = []

    def fake_complete(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return Result("A")

    monkeypatch.setattr(openrouter.litlm, "complete", fake_complete)
    result = openrouter.probe_one(
        {"id": "a", "prompt": "question", "choices": {"A": "yes"}},
        "test/model",
        "key",
        retries=1,
    )
    assert result["parsed_answer"] == "A"
    assert calls[0][0] == "question"
    assert calls[0][1]["num_retries"] == 1


def test_probe_one_forwards_fallback_controls(monkeypatch):
    captured = {}

    def fake_complete(prompt, **kwargs):
        captured.update(kwargs)
        return Result("A")

    monkeypatch.setattr(openrouter.litlm, "complete", fake_complete)
    openrouter.probe_one(
        {"id": "a", "prompt": "question", "choices": {"A": "yes"}},
        "gemini-3.7-flash",
        attempt_timeout=45,
        fallbacks=[
            "direct/gemini/gemini-3.7-flash",
            "openrouter/google/gemini-3.7-flash",
        ],
    )

    assert captured["attempt_timeout"] == 45
    assert captured["fallbacks"][-1] == "openrouter/google/gemini-3.7-flash"


def test_probe_one_leaves_provider_key_resolution_to_litlm(monkeypatch):
    captured = {}

    def fake_complete(prompt, **kwargs):
        captured.update(kwargs)
        return Result("A")

    monkeypatch.setattr(openrouter.litlm, "complete", fake_complete)
    openrouter.probe_one(
        {"id": "x", "prompt": "prompt", "choices": {"A": "answer"}},
        "albert/deepseek-v4-flash",
    )

    assert "api_key" not in captured
