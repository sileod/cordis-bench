import random
import re

from cordis_bench import v2_release
from cordis_bench import v2_formal
from cordis_bench import v2_native
from cordis_bench.v201_surface import _short_formal_keys, _short_native_label


SHORT_NATIVE = re.compile(r"^(provider|service|plugin|slot)_[a-z]{1,2}$")


def test_v201_release_size_is_unchanged():
    assert v2_release.V2_RELEASE_VERSION == "2.0.1"
    assert v2_release.V2_RELEASE_REPLICATES == 3
    assert v2_release.V2_RELEASE_N == 1200
    assert v2_release.V2_RELEASE_SEEDS == (0, 1, 2)


def test_v201_native_identifiers_are_short_and_unique_per_world():
    rng = random.Random(17)
    spec = v2_native._sample_native_spec(rng, 32)
    identifiers = [
        spec["query_provider"],
        spec["decoy"],
        *(provider["name"] for provider in spec["providers"]),
        *(provider["service"] for provider in spec["providers"]),
        *(leaf["name"] for leaf in spec["leaves"]),
        *(leaf["service"] for leaf in spec["leaves"]),
        *(leaf["slot"] for leaf in spec["leaves"]),
        *spec["base_slots"],
    ]
    assert all(SHORT_NATIVE.fullmatch(identifier) for identifier in identifiers)

    plugin_names = {spec["decoy"], *(leaf["name"] for leaf in spec["leaves"])}
    provider_names = {provider["name"] for provider in spec["providers"]}
    services = {provider["service"] for provider in spec["providers"]}
    slots = set(spec["base_slots"])
    assert len(plugin_names) == 1 + 2 * spec["v2_semantic_size"]
    assert len(provider_names) == 2
    assert len(services) == 2
    assert all(name.startswith("slot_") for name in slots)


def test_v201_formal_capability_keys_are_short():
    records = v2_formal.generate_v2_formal_core_dataset(
        sizes=(2,), worlds_per_size=1, seed=23
    )
    prompts = "\n".join(item["prompt"] for item in records)
    assert "cap_" in prompts
    assert re.search(r"\bq\d{6}\b", prompts) is None


def test_v201_native_label_consumes_same_rng_draw_as_v20():
    short_rng = random.Random(101)
    setattr(short_rng, "_cordis_v201_identifier_state", {})
    label = _short_native_label(short_rng, "plugin")
    short_tail = [short_rng.random() for _ in range(5)]

    old_rng = random.Random(101)
    old_rng.randrange(10**9)
    old_tail = [old_rng.random() for _ in range(5)]

    assert SHORT_NATIVE.fullmatch(label)
    assert short_tail == old_tail


def test_v201_formal_keys_consume_same_rng_draws_as_v20():
    short_rng = random.Random(202)
    keys = _short_formal_keys(short_rng, 40)
    short_tail = [short_rng.random() for _ in range(5)]

    old_rng = random.Random(202)
    old_rng.sample(range(100_000, 999_999), 40)
    old_tail = [old_rng.random() for _ in range(5)]

    assert len(keys) == len(set(keys)) == 40
    assert all(re.fullmatch(r"cap_[a-z]{1,2}", key) for key in keys)
    assert short_tail == old_tail


def test_v201_release_annotation_is_explicit():
    item = {
        "metadata": {
            "semantic_dimensions": {},
            "benchmark_suite": "2.0",
            "generator_version": "2.0-rc2",
        }
    }
    v2_release._annotate_release_block([item], block_index=1, block_seed=2)
    metadata = item["metadata"]
    assert metadata["benchmark_suite"] == "2.0.1"
    assert metadata["generator_version"] == "2.0.1"
    assert metadata["identifier_surface"] == "short_opaque"
    assert metadata["identifier_surface_version"] == "2.0.1"
    assert metadata["release_replicate_index"] == 1
    assert metadata["release_replicate_seed"] == 2
