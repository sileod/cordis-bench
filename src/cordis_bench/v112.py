"""V1.12 challenge track.

V1.12 leaves the frozen paper releases untouched and replaces the invalid V1.11
challenge with free-form, shortcut-guarded formal and Cordis-native families.
"""

from .guards import assert_challenge_guards
from .v112_formal_challenge import (
    V112_FORMAL_COMPONENT,
    V112_FORMAL_RECORDS,
    V112_FORMAL_SIZES,
    V112_FORMAL_TASKS,
    V112_FORMAL_WORLDS_PER_SIZE,
    _build_formal_world,
    generate_v112_formal_dataset,
)
from .v112_native_challenge import (
    CORDIS_VERSION,
    V112_NATIVE_COMPONENT,
    V112_NATIVE_RECORDS,
    V112_NATIVE_SIZES,
    V112_NATIVE_TASKS,
    V112_NATIVE_WORLDS_PER_SIZE,
    _native_reference,
    _sample_native_spec,
    generate_v112_native_dataset,
)


V112_TASKS = V112_FORMAL_TASKS + V112_NATIVE_TASKS
V112_PERTURBATIONS = ("base", "alpha_renamed_reordered")
V112_DEFAULT_N = V112_FORMAL_RECORDS + V112_NATIVE_RECORDS


def generate_v112_dataset(
    n=None,
    seed=0,
    task_types=V112_TASKS,
    formal_sizes=V112_FORMAL_SIZES,
    native_sizes=V112_NATIVE_SIZES,
    formal_worlds=V112_FORMAL_WORLDS_PER_SIZE,
    native_worlds=V112_NATIVE_WORLDS_PER_SIZE,
    runner_path=None,
    node="node",
):
    if tuple(task_types) != V112_TASKS:
        raise ValueError("V1.12 has a fixed task family")
    formal_sizes = tuple(map(int, formal_sizes))
    native_sizes = tuple(map(int, native_sizes))
    if not formal_sizes or not native_sizes:
        raise ValueError("V1.12 requires formal and native sizes")
    if formal_worlds <= 0 or native_worlds <= 0:
        raise ValueError("worlds-per-size must be positive")

    records = generate_v112_formal_dataset(
        formal_sizes,
        formal_worlds,
        seed,
    ) + generate_v112_native_dataset(
        native_sizes,
        native_worlds,
        seed,
        runner_path,
        node,
    )
    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate V1.12 IDs")
    if n is not None and n != len(records):
        raise ValueError(f"requested n={n}, selected ladder emits {len(records)}")
    assert_challenge_guards(records)
    return records
