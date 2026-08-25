"""Shared schema helpers for the V2 benchmark reset."""

from __future__ import annotations

import json

from .tasks import stable_id

V2_SCHEMA_VERSION = "2.0"
V2_GENERATOR_VERSION = "2.0-rc2"
V2_ORACLE_VERSION = "2.0-rc2"


def canonical_list(values, *, sort=False):
    values = list(values)
    if sort:
        values = sorted(values)
    return json.dumps(values, separators=(",", ":"), ensure_ascii=False)


def canonical_scalar_sequence(values):
    return json.dumps(list(values), separators=(",", ":"), ensure_ascii=False)


def make_metadata(
    *,
    track,
    realization,
    family,
    capability,
    answer_type,
    paper_role="primary",
    semantic_size=None,
    semantic_size_unit=None,
    latent_group_id=None,
    prompt_contract=None,
    **extra,
):
    if track not in {"core", "challenge"}:
        raise ValueError(f"unknown V2 track: {track}")
    if realization not in {"formal", "cordis_native"}:
        raise ValueError(f"unknown V2 realization: {realization}")
    component = f"v2_{realization}_{track}"
    data = {
        "benchmark_suite": "2.0",
        "benchmark_track": track,
        "benchmark_component": component,
        "component": component,
        "benchmark_subcomponent": f"{component}.{family}",
        "realization": (
            "formal_micro_system" if realization == "formal" else "actual_cordis_runtime"
        ),
        "task_family": family,
        "capability": capability,
        "answer_type": answer_type,
        "paper_role": paper_role,
        "challenge_track": track == "challenge",
        "challenge_family": family if track == "challenge" else None,
        "semantic_size": semantic_size,
        "semantic_size_unit": semantic_size_unit,
        "challenge_size": semantic_size if track == "challenge" else None,
        "challenge_size_unit": semantic_size_unit if track == "challenge" else None,
        "latent_group_id": latent_group_id,
        "prompt_contract": prompt_contract,
        "generator_version": V2_GENERATOR_VERSION,
        "oracle_version": V2_ORACLE_VERSION,
        "harness_surface": realization == "cordis_native",
        "mcq": False,
        "exact_oracle": True,
        **extra,
    }
    data["semantic_dimensions"] = {
        key: value
        for key, value in data.items()
        if value is not None
        and key not in {"semantic_dimensions"}
        and not isinstance(value, (dict, list, tuple))
    }
    return data


def make_record(
    *,
    task_type,
    prompt,
    answer,
    answer_type,
    metadata,
    identity,
    oracle=None,
):
    record = {
        "id": stable_id(
            {
                "schema_version": V2_SCHEMA_VERSION,
                "task_type": task_type,
                "identity": identity,
            }
        ),
        "schema_version": V2_SCHEMA_VERSION,
        "task_type": task_type,
        "prompt": prompt,
        "choices": {},
        "answer_type": answer_type,
        "answer": str(answer),
        "metadata": metadata,
    }
    if oracle is not None:
        record["oracle"] = oracle
    return record
