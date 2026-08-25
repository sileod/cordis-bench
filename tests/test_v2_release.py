from cordis_bench import v2
from cordis_bench.v2_release import (
    V2_BLOCK_N,
    V2_RELEASE_N,
    V2_RELEASE_REPLICATES,
    generate_v2_release_dataset,
)


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
            "cordis_version": "4.0.0-rc.7",
            "cordis_source_commit": "56b3d4f725681cf4556c1a8695a709cc3b6eed74",
            "node_version": "v22-test",
            "schedule_runs": runs,
            "confluent": spec["abstract_confluent"],
            "option_success": option_success,
            "option_runs": option_runs,
        }
    return runtimes


def test_v2_release_constants():
    assert V2_BLOCK_N == v2.V2_DEFAULT_N == 400
    assert V2_RELEASE_REPLICATES == 3
    assert V2_RELEASE_N == 1200


def test_v2_release_assembles_independent_seed_blocks(monkeypatch):
    import cordis_bench.v2_native as native

    monkeypatch.setattr(native, "execute_native_specs", _fake_execute)
    records = generate_v2_release_dataset(
        seed=7,
        replicates=3,
        formal_sizes=(8,),
        native_sizes=(8,),
        formal_worlds=2,
        native_worlds=2,
        formal_core_sizes=(2,),
        formal_core_worlds=2,
        native_core_sizes=(2,),
        native_core_worlds=2,
    )
    assert len(records) == 120
    assert len({item["id"] for item in records}) == len(records)
    assert {item["metadata"]["generation_seed"] for item in records} == {7, 8, 9}
    assert {item["metadata"]["release_replicate_index"] for item in records} == {0, 1, 2}
    assert all(
        item["metadata"]["release_status"] == "v2.0.1_frozen"
        for item in records
    )


def test_v2_release_single_block_preserves_seed_ids(monkeypatch):
    import cordis_bench.v2_native as native

    monkeypatch.setattr(native, "execute_native_specs", _fake_execute)
    kwargs = dict(
        formal_sizes=(8,),
        native_sizes=(8,),
        formal_worlds=2,
        native_worlds=2,
        formal_core_sizes=(2,),
        formal_core_worlds=2,
        native_core_sizes=(2,),
        native_core_worlds=2,
    )
    block = v2.generate_v2_dataset(seed=13, **kwargs)
    release = generate_v2_release_dataset(seed=13, replicates=1, **kwargs)
    assert {item["id"] for item in block} == {item["id"] for item in release}
