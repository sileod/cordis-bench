"""Shared helpers for the V1.12 challenge track."""

from .tasks import stable_id

CHECKSUM_MODULUS = 1_000_003


def checksum(values):
    return sum((i + 1) * value for i, value in enumerate(values)) % CHECKSUM_MODULUS


def metadata(component, family, size, unit, world_id, pair_id, perturbation, **extra):
    data = {
        "benchmark_suite": "1.12-challenge",
        "benchmark_component": component,
        "component": component,
        "benchmark_subcomponent": f"{component}.{family}",
        "challenge_track": True,
        "challenge_family": family,
        "challenge_size": size,
        "challenge_size_unit": unit,
        "challenge_world_id": world_id,
        "challenge_pair_id": pair_id,
        "perturbation_variant": perturbation,
        "shortcut_resistant": True,
        "alpha_renaming": perturbation != "base",
        "component_reordering": perturbation != "base",
        "answer_type": "integer",
        "capability": "global",
        **extra,
    }
    data["semantic_dimensions"] = {
        key: value for key, value in data.items()
        if key not in {"challenge_world_id", "challenge_pair_id"}
        and not isinstance(value, (dict, list))
    }
    return data


def record(task, prompt, gold, metadata, pair_id, perturbation):
    return {
        "id": stable_id({"schema_version": "1.12", "pair": pair_id, "v": perturbation}),
        "schema_version": "1.12",
        "task_type": task,
        "prompt": prompt,
        "choices": {},
        "answer_type": "integer",
        "answer": str(gold),
        "metadata": metadata,
    }
