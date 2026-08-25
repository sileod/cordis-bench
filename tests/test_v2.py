import json
import random

from cordis_bench import v2
from cordis_bench.guards import audit_v2_dataset
from cordis_bench.openrouter import parse_item_answer
from cordis_bench.v2_audit import audit_v2_shortcuts
from cordis_bench.v2_scoring import score_v2_predictions, task_metric_summary
from cordis_bench.v2_native import CORDIS_SOURCE_COMMIT, CORDIS_VERSION


def _fake_execute(specs, runner_path=None, node="node"):
    runtimes = {}
    for spec in specs:
        runs = [
            {"label": label, "observation": observation, "trace": []}
            for label, observation in spec["abstract_schedule_observations"].items()
        ]
        option_success = {}
        option_runs = {}
        for label in spec.get("options", {}):
            option_success[label] = True
            option_runs[label] = [
                {"label": schedule, "observation": spec["target"], "trace": []}
                for schedule in spec["schedules"]
            ]
        runtimes[spec["native_case_id"]] = {
            "native_case_id": spec["native_case_id"],
            "cordis_package": "cordis",
            "cordis_version": CORDIS_VERSION,
            "cordis_source_commit": CORDIS_SOURCE_COMMIT,
            "node_version": "v22-test",
            "schedule_runs": runs,
            "confluent": spec["abstract_confluent"],
            "option_success": option_success,
            "option_runs": option_runs,
        }
    return runtimes


def test_v2_default_layout():
    assert v2.V2_FORMAL_CHALLENGE_SIZES == (8, 16, 24, 32)
    assert v2.V2_NATIVE_CHALLENGE_SIZES == (8, 16, 24, 32)
    assert v2.V2_FORMAL_CORE_RECORDS == 64
    assert v2.V2_FORMAL_CHALLENGE_RECORDS == 160
    assert v2.V2_NATIVE_CORE_RECORDS == 80
    assert v2.V2_NATIVE_CHALLENGE_RECORDS == 96
    assert v2.V2_DEFAULT_N == 400


def test_v2_formal_core_challenge_share_task_contracts():
    core = v2.generate_v2_formal_core_dataset(sizes=(2,), worlds_per_size=4, seed=3)
    challenge = v2.generate_v2_formal_challenge_dataset(sizes=(8,), worlds_per_size=4, seed=4)
    primary_core = {item["task_type"] for item in core}
    primary_challenge = {
        item["task_type"]
        for item in challenge
        if item["metadata"]["paper_role"] == "primary"
    }
    assert primary_core == primary_challenge == set(v2.V2_FORMAL_CORE_TASKS)
    assert all(item["choices"] == {} for item in core + challenge)
    assert "scalar_sequence" in {item["answer_type"] for item in core + challenge}
    assert all(
        len(json.loads(item["answer"])) >= 2
        for item in core + challenge
        if item["task_type"] in {
            "formal_guaranteed_conditions",
            "formal_reachable_conditions",
        }
    )


def test_v2_native_core_challenge_share_generator_and_contracts(monkeypatch):
    import cordis_bench.v2_native as native

    monkeypatch.setattr(native, "execute_native_specs", _fake_execute)
    core = native.generate_v2_native_core_dataset(sizes=(2,), worlds_per_size=4, seed=7)
    challenge = native.generate_v2_native_challenge_dataset(sizes=(8,), worlds_per_size=4, seed=8)
    primary_core = {item["task_type"] for item in core}
    primary_challenge = {
        item["task_type"]
        for item in challenge
        if item["metadata"]["paper_role"] == "primary"
    }
    assert primary_core == primary_challenge == set(native.V2_NATIVE_CORE_TASKS)
    assert all(item["choices"] == {} for item in core + challenge)
    assert all(item["metadata"]["prompt_contract"] for item in core + challenge)
    reconfig = [item for item in challenge if item["task_type"] == "cordis_reconfiguration_set"]
    assert reconfig and all(item.get("oracle", {}).get("kind") == "cordis_reconfiguration" for item in reconfig)


def test_v2_full_small_audit_and_provenance(monkeypatch):
    import cordis_bench.v2_native as native

    monkeypatch.setattr(native, "execute_native_specs", _fake_execute)
    records = v2.generate_v2_dataset(
        seed=11,
        formal_sizes=(8,),
        native_sizes=(8,),
        formal_worlds=4,
        native_worlds=4,
        formal_core_sizes=(2,),
        formal_core_worlds=4,
        native_core_sizes=(2,),
        native_core_worlds=4,
    )
    audit = audit_v2_dataset(records)
    assert audit["n"] == len(records)
    assert audit["mcq_records"] == 0
    assert audit["primary_contracts"] == 9
    assert audit["anchored_primary_contracts"] == 9
    assert all(item["metadata"]["generation_seed"] == 11 for item in records)
    assert all(item["metadata"]["hard_guard_passed"] is True for item in records)


def test_v2_formal_conditions_are_defined_across_seeds():
    for seed in range(6):
        records = v2.generate_v2_formal_core_dataset(
            sizes=(2, 4), worlds_per_size=2, seed=seed
        )
        assert records
        assert all(
            item["answer"]
            for item in records
            if item["task_type"] in {
                "formal_guaranteed_conditions",
                "formal_reachable_conditions",
            }
        )


def test_v2_native_core_size4_does_not_require_count_variation(monkeypatch):
    import cordis_bench.v2_native as native

    monkeypatch.setattr(native, "execute_native_specs", _fake_execute)
    records = native.generate_v2_native_core_dataset(
        sizes=(4,), worlds_per_size=4, seed=29
    )
    assert len(records) == 4 * len(native.V2_NATIVE_CORE_TASKS)
    assert {item["metadata"]["semantic_size"] for item in records} == {4}


def test_v2_native_interference_is_mixed_and_required():
    import cordis_bench.v2_native as native

    rng = random.Random(19)
    spec = native._sample_native_spec(rng, 8)
    spec = native._augment_reconfiguration(spec)
    units = spec["v2_pairs"]
    assert any(unit["interfering"] for unit in units)
    assert any(not unit["interfering"] for unit in units)
    positions = {name: index for index, name in enumerate(spec["startup_order"])}
    expected = sorted(
        max(unit["names"], key=positions.__getitem__)
        for unit in units
        if unit["interfering"]
    )
    assert spec["v2_predispose"] == expected
    for unit in units:
        pair = tuple(unit["names"])
        orientations = {
            tuple(name for name in order if name in pair)
            for order in spec["schedules"].values()
        }
        assert orientations == {pair, tuple(reversed(pair))}


def test_v2_answer_parser():
    assert parse_item_answer({"answer_type": "integer", "choices": {}}, "Answer: 12,345") == "12345"
    assert parse_item_answer(
        {"answer_type": "string_set", "choices": {}}, '["b", "a", "a"]'
    ) == '["a","b"]'
    assert parse_item_answer(
        {"answer_type": "scalar_sequence", "choices": {}}, '[3, "guarded", 7]'
    ) == '[3,"guarded",7]'


def test_v2_scorer_executes_reconfiguration(monkeypatch):
    import cordis_bench.v2_native as native
    import cordis_bench.v2_scoring as scoring

    monkeypatch.setattr(native, "execute_native_specs", _fake_execute)
    records = native.generate_v2_native_core_dataset(sizes=(2,), worlds_per_size=2, seed=17)
    item = next(row for row in records if row["task_type"] == "cordis_reconfiguration_set")

    def fake_score_execute(specs, runner_path=None, node="node"):
        result = {}
        for spec in specs:
            option_runs = {
                "MODEL": [
                    {"label": label, "observation": spec["target"], "trace": []}
                    for label in spec["schedules"]
                ]
            }
            result[spec["native_case_id"]] = {
                "option_success": {"MODEL": True},
                "option_runs": option_runs,
            }
        return result

    monkeypatch.setattr(scoring, "execute_native_specs", fake_score_execute)
    report = score_v2_predictions([item], [{"id": item["id"], "response": item["answer"]}])
    assert report["accuracy"] == 1.0
    assert report["reconfiguration_execution"]["status_counts"] == {"success": 1}


def test_v2_string_set_jaccard():
    item = {
        "id": "set-item",
        "schema_version": "2.0",
        "task_type": "withdrawal_cone",
        "answer_type": "string_set",
        "answer": '["a","b"]',
        "metadata": {},
    }
    report = score_v2_predictions(
        [item], [{"id": item["id"], "response": '["b","c"]'}]
    )
    assert report["accuracy"] == 0.0
    assert report["mean_jaccard"] == 1 / 3
    assert report["rows"][0]["jaccard"] == 1 / 3
    assert report["by_answer_type"]["string_set"]["mean_jaccard"] == 1 / 3


def test_v2_string_set_accepts_unambiguous_set_literal():
    item = {
        "id": "set-item",
        "answer_type": "string_set",
        "choices": {},
    }
    assert parse_item_answer(item, '{"b", "a", "b"}') == '["a","b"]'
    assert (
        parse_item_answer(item, 'Reasoning. Final answer: {"b", "a"}')
        == '["a","b"]'
    )


def test_v2_string_set_rejects_object_and_ordered_sequence_stays_strict():
    set_item = {"id": "set-item", "answer_type": "string_set", "choices": {}}
    sequence_item = {
        "id": "sequence-item",
        "answer_type": "identifier_sequence",
        "choices": {},
    }
    assert parse_item_answer(set_item, '{"a": 1}') is None
    assert parse_item_answer(sequence_item, '{"a", "b"}') is None


def test_v2_scalar_sequence_accepts_exact_observable_mapping():
    item = {
        "id": "sequence-item",
        "answer_type": "scalar_sequence",
        "choices": {},
        "prompt": """Report values, in this exact order:
  1. x2
  2. x0
Return only a JSON array of values.""",
    }
    assert parse_item_answer(item, '{"x0":1,"x2":2}') == "[2,1]"
    assert parse_item_answer(item, '{"x0":1,"x2":2,"x3":3}') is None
    assert parse_item_answer(item, '{"x0":1}') is None


def test_v2_missing_string_set_prediction_has_zero_jaccard():
    item = {
        "id": "empty-set-item",
        "schema_version": "2.0",
        "task_type": "withdrawal_cone",
        "answer_type": "string_set",
        "answer": "[]",
        "metadata": {},
    }
    report = score_v2_predictions([item], [])
    assert report["mean_jaccard"] == 0.0


def test_v2_task_metric_summary_uses_semantic_units():
    set_rows = [
        {"capability": "localization", "jaccard": 0.5},
        {"capability": "localization", "jaccard": 1.0},
    ]
    assert task_metric_summary(set_rows) == {
        "metric": "jaccard",
        "score": 0.75,
        "n": 2,
    }

    sequence_rows = [
        {
            "id": "a",
            "capability": "prediction",
            "gold": '[1,2,3]',
            "prediction": '[1,9,3]',
        },
        {
            "id": "b",
            "capability": "prediction",
            "gold": '[4,5]',
            "prediction": None,
        },
    ]
    assert task_metric_summary(sequence_rows) == {
        "metric": "per_observable_accuracy",
        "score": 0.4,
        "n": 2,
        "observables": 5,
    }


def test_v2_shortcut_audit_runs(monkeypatch):
    import cordis_bench.v2_native as native

    monkeypatch.setattr(native, "execute_native_specs", _fake_execute)
    records = v2.generate_v2_dataset(
        seed=23,
        formal_sizes=(8,), native_sizes=(8,),
        formal_worlds=4, native_worlds=4,
        formal_core_sizes=(2,), formal_core_worlds=4,
        native_core_sizes=(2,), native_core_worlds=4,
    )
    report = audit_v2_shortcuts(records)
    assert report["n"] == len(records)
    assert "lexical_1nn_group_held_out" in report["baselines"]
    assert "record_position_periodic" in report["baselines"]
