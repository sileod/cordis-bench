"""Generation-time shortcut audits for challenge releases and V2."""

from __future__ import annotations

from collections import defaultdict
import re


_FORBIDDEN_NATIVE_LABELS = re.compile(
    r"(critical(?:[._-]|$)|control(?:[._-]|$)|base-[cq]-)",
    re.IGNORECASE,
)
_FORBIDDEN_GOLD_KEYS = {
    "answer",
    "gold",
    "outcome_count_gold",
    "terminal_outcome_count",
    "terminal_observations",
    "runtime_outcome_count",
    "abstract_outcome_count",
    "local_counts",
    "local_multiplicities",
    "schedule_checksum",
    "checksum_gold",
    "profile",
    "world_index",
}


def _nested_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _nested_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_keys(item)


def audit_challenge_dataset(records):
    if not records:
        raise ValueError("challenge audit requires at least one record")

    non_integer = [
        item["id"]
        for item in records
        if item.get("answer_type") != "integer" or item.get("choices")
    ]
    if non_integer:
        raise ValueError(
            "challenge counting/checksum tasks must eliminate MCQ option leakage; "
            f"bad records: {non_integer[:3]}"
        )

    leaked_labels = [
        item["id"]
        for item in records
        if _FORBIDDEN_NATIVE_LABELS.search(item.get("prompt", ""))
    ]
    if leaked_labels:
        raise ValueError(
            "challenge prompt contains semantic partition labels; "
            f"bad records: {leaked_labels[:3]}"
        )

    leaked_metadata = []
    for item in records:
        metadata = item.get("metadata") or {}
        bad = _FORBIDDEN_GOLD_KEYS & set(_nested_keys(metadata))
        if bad:
            leaked_metadata.append((item["id"], sorted(bad)))
    if leaked_metadata:
        raise ValueError(
            "challenge metadata exposes gold or generator-position information; "
            f"bad records: {leaked_metadata[:3]}"
        )

    pairs = defaultdict(list)
    for item in records:
        pair_id = item.get("metadata", {}).get("challenge_pair_id")
        if not pair_id:
            raise ValueError(f"challenge record {item['id']} has no challenge_pair_id")
        pairs[pair_id].append(item)

    malformed_pairs = [
        pair_id
        for pair_id, pair in pairs.items()
        if len(pair) != 2
        or {item["metadata"].get("perturbation_variant") for item in pair}
        != {"base", "alpha_renamed_reordered"}
        or len({item["answer"] for item in pair}) != 1
        or len({item["prompt"] for item in pair}) != 2
    ]
    if malformed_pairs:
        raise ValueError(
            "challenge perturbation pairs must be semantic isomorphs with identical gold; "
            f"bad pairs: {malformed_pairs[:3]}"
        )

    by_cell = defaultdict(list)
    for item in records:
        if item["metadata"].get("perturbation_variant") != "base":
            continue
        key = (
            item["metadata"].get("benchmark_component"),
            item["task_type"],
            item["metadata"].get("challenge_size"),
        )
        by_cell[key].append(item["answer"])

    constant_cells = [
        key
        for key, answers in by_cell.items()
        if len(answers) >= 2 and len(set(answers)) == 1
    ]
    if constant_cells:
        raise ValueError(
            "challenge gold is constant within task x size cell; "
            f"size-only shortcut exists in {constant_cells[:3]}"
        )

    return {
        "n": len(records),
        "n_pairs": len(pairs),
        "free_form_integer_rate": sum(
            item.get("answer_type") == "integer" and not item.get("choices")
            for item in records
        )
        / len(records),
        "constant_task_size_cells": len(constant_cells),
        "forbidden_label_hits": len(leaked_labels),
        "forbidden_metadata_hits": len(leaked_metadata),
    }


def assert_challenge_guards(records):
    return audit_challenge_dataset(records)


_V2_ALLOWED_ANSWER_TYPES = {
    "boolean",
    "integer",
    "scalar",
    "scalar_sequence",
    "string_set",
    "identifier_sequence",
}


def audit_v2_dataset(records):
    """Reject static answer-surface and core/challenge anchoring failures."""
    if not records:
        raise ValueError("V2 audit requires at least one record")

    mcq = [item["id"] for item in records if item.get("choices")]
    if mcq:
        raise ValueError(f"V2 must not rely on MCQ choices; bad records: {mcq[:3]}")

    bad_schema = [
        item["id"]
        for item in records
        if item.get("schema_version") != "2.0"
        or item.get("answer_type") not in _V2_ALLOWED_ANSWER_TYPES
    ]
    if bad_schema:
        raise ValueError(f"V2 answer schema violation: {bad_schema[:3]}")

    leaked_labels = [
        item["id"]
        for item in records
        if item.get("metadata", {}).get("realization") == "actual_cordis_runtime"
        and _FORBIDDEN_NATIVE_LABELS.search(item.get("prompt", ""))
    ]
    if leaked_labels:
        raise ValueError(
            "V2 native prompt contains semantic partition labels; "
            f"bad records: {leaked_labels[:3]}"
        )

    leaked_metadata = []
    for item in records:
        metadata = item.get("metadata") or {}
        bad = _FORBIDDEN_GOLD_KEYS & set(_nested_keys(metadata))
        if bad:
            leaked_metadata.append((item["id"], sorted(bad)))
    if leaked_metadata:
        raise ValueError(
            "V2 metadata exposes gold or generator-position information; "
            f"bad records: {leaked_metadata[:3]}"
        )

    malformed = []
    for item in records:
        metadata = item.get("metadata") or {}
        if (
            metadata.get("benchmark_suite") not in {"2.0", "2.0.1"}
            or metadata.get("benchmark_track") not in {"core", "challenge"}
            or metadata.get("paper_role") not in {"primary", "diagnostic"}
            or metadata.get("mcq") is not False
            or metadata.get("semantic_size") is None
            or not metadata.get("latent_group_id")
            or not metadata.get("prompt_contract")
        ):
            malformed.append(item["id"])
    if malformed:
        raise ValueError(f"V2 metadata contract violation: {malformed[:3]}")

    contracts = defaultdict(
        lambda: {
            "tracks": set(),
            "answer_types": set(),
            "prompts": set(),
            "sizes": defaultdict(set),
        }
    )
    for item in records:
        metadata = item["metadata"]
        if metadata.get("paper_role") != "primary":
            continue
        key = (metadata.get("realization"), item["task_type"])
        entry = contracts[key]
        track = metadata["benchmark_track"]
        entry["tracks"].add(track)
        entry["answer_types"].add(item["answer_type"])
        entry["prompts"].add(metadata.get("prompt_contract"))
        entry["sizes"][track].add(int(metadata["semantic_size"]))

    bad_anchor = []
    for key, entry in contracts.items():
        if (
            entry["tracks"] != {"core", "challenge"}
            or len(entry["answer_types"]) != 1
            or len(entry["prompts"]) != 1
            or max(entry["sizes"]["core"]) >= min(entry["sizes"]["challenge"])
        ):
            bad_anchor.append(key)
    if bad_anchor:
        raise ValueError(
            "V2 primary task lacks a clean core/challenge size anchor; "
            f"bad contracts: {bad_anchor[:3]}"
        )

    diagnostic_core = [
        item["id"]
        for item in records
        if item["metadata"].get("paper_role") == "diagnostic"
        and item["metadata"].get("benchmark_track") != "challenge"
    ]
    if diagnostic_core:
        raise ValueError(f"V2 diagnostics must stay out of core; bad records: {diagnostic_core[:3]}")

    count_cells = defaultdict(list)
    for item in records:
        metadata = item["metadata"]
        if metadata.get("paper_role") == "diagnostic":
            key = (
                metadata.get("realization"),
                item["task_type"],
                metadata.get("semantic_size"),
            )
            count_cells[key].append(item["answer"])
    constant_count_cells = [
        key
        for key, answers in count_cells.items()
        if len(answers) >= 2 and len(set(answers)) == 1
    ]
    if constant_count_cells:
        raise ValueError(
            "V2 diagnostic answer is constant within task x size cell; "
            f"bad cells: {constant_count_cells[:3]}"
        )

    return {
        "n": len(records),
        "mcq_rate": 0.0,
        "mcq_records": 0,
        "answer_types": sorted({item["answer_type"] for item in records}),
        "answer_type_count": len({item["answer_type"] for item in records}),
        "primary_contracts": len(contracts),
        "anchored_primary_contracts": len(contracts) - len(bad_anchor),
        "constant_diagnostic_cells": len(constant_count_cells),
        "forbidden_label_hits": len(leaked_labels),
        "forbidden_metadata_hits": len(leaked_metadata),
    }


def assert_v2_guards(records):
    return audit_v2_dataset(records)
