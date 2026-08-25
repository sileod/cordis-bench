"""Formal CordisBench V2 core/challenge from one size-scaled task generator."""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations
import random

from .core import DEFAULT_THEORY
from .render import render_state, render_world, v1_rules
from .semantics import apply_ops
from .v112_formal_challenge import _build_formal_world
from .v2_common import canonical_list, canonical_scalar_sequence, make_metadata, make_record


V2_FORMAL_CORE_SIZES = (2, 4)
V2_FORMAL_CORE_WORLDS_PER_SIZE = 8
V2_FORMAL_CHALLENGE_SIZES = (8, 16, 24, 32)
V2_FORMAL_CHALLENGE_WORLDS_PER_SIZE = 8

V2_FORMAL_PRIMARY_TASKS = (
    "formal_withdrawal_cone",
    "formal_schedule_prediction",
    "formal_guaranteed_conditions",
    "formal_reachable_conditions",
)
V2_FORMAL_DIAGNOSTIC_TASKS = ("formal_outcome_count_diagnostic",)
V2_FORMAL_CORE_TASKS = V2_FORMAL_PRIMARY_TASKS
V2_FORMAL_CHALLENGE_TASKS = V2_FORMAL_PRIMARY_TASKS + V2_FORMAL_DIAGNOSTIC_TASKS
V2_FORMAL_TASKS = V2_FORMAL_CHALLENGE_TASKS

V2_FORMAL_CORE_RECORDS = (
    len(V2_FORMAL_CORE_SIZES)
    * V2_FORMAL_CORE_WORLDS_PER_SIZE
    * len(V2_FORMAL_CORE_TASKS)
)
V2_FORMAL_CHALLENGE_RECORDS = (
    len(V2_FORMAL_CHALLENGE_SIZES)
    * V2_FORMAL_CHALLENGE_WORLDS_PER_SIZE
    * len(V2_FORMAL_CHALLENGE_TASKS)
)


def _formal_prefix(case):
    return f"""{v1_rules(DEFAULT_THEORY)}
{render_world(case['world'])}

The system starts quiescent:
{render_state(case['start'])}

The orchestrator executes:
  {case['directive'].render()}
"""


def _root_relay_terminal(case):
    world = case["world"]
    root = world.by_name[case["directive"].component]
    if len(root.provides) != 1:
        raise RuntimeError("V2 formal root must provide exactly one capability")
    root_key = next(iter(root.provides))
    relays = [component for component in world.components if root_key in component.requires]
    if len(relays) != 1 or len(relays[0].provides) != 1:
        raise RuntimeError("V2 formal queried branch must have exactly one relay")
    return root.name, relays[0].name, next(iter(relays[0].provides))


def _relevant_groups(case):
    world = case["world"]
    _root, _relay, terminal = _root_relay_terminal(case)
    leaves = [component for component in world.components if terminal in component.requires]
    by_coords = defaultdict(list)
    pad = world.width - 1
    for component in leaves:
        coords = sorted(
            {
                coord
                for op in component.effect
                for coord in (op.i, op.j)
                if coord is not None and coord != pad
            }
        )
        if not coords:
            raise RuntimeError(f"formal V2 leaf {component.name} has no local state coordinates")
        offset = 2 * (min(coords) // 2)
        by_coords[offset].append(component.name)
    groups = []
    for offset, names in sorted(by_coords.items()):
        if len(names) != 3:
            raise RuntimeError(f"formal V2 expected three-effect group at {offset}, got {len(names)}")
        groups.append((offset, tuple(sorted(names))))
    if len(groups) != case["n_gadgets"]:
        raise RuntimeError(
            f"formal V2 found {len(groups)} relevant groups, expected {case['n_gadgets']}"
        )
    return groups


def _local_outcomes(case, offset, names):
    world = case["world"]
    initial = tuple(case["start"].values[offset : offset + 2])
    outcomes = set()
    for order in permutations(names):
        values = initial
        for name in order:
            local_ops = []
            for op in world.by_name[name].inverse_effect(world.modulus):
                local_ops.append(
                    type(op)(
                        op.kind,
                        op.i - offset,
                        None if op.j is None else op.j - offset,
                        op.value,
                    )
                )
            values = apply_ops(values, local_ops, world.modulus)
        outcomes.add(values)
    return outcomes


def _groups_with_outcomes(case):
    return [
        (offset, names, _local_outcomes(case, offset, names))
        for offset, names in _relevant_groups(case)
    ]


def _final_values(case, schedule):
    values = case["start"].values
    for name in schedule:
        values = apply_ops(
            values,
            case["world"].by_name[name].inverse_effect(case["world"].modulus),
            case["world"].modulus,
        )
    return values


def _metadata(case, track, family, capability, answer_type, prompt_contract, **extra):
    return make_metadata(
        track=track,
        realization="formal",
        family=family,
        capability=capability,
        answer_type=answer_type,
        semantic_size=case["n_gadgets"],
        semantic_size_unit="relevant_effect_groups",
        latent_group_id=case["world_id"],
        prompt_contract=prompt_contract,
        problem_size=len(case["world"].components),
        state_width=case["world"].width,
        relevant_effect_groups=case["n_gadgets"],
        **extra,
    )


def _affected_record(case, track):
    affected = sorted(set(case["schedule"]))
    prompt = _formal_prefix(case) + """
Which components belong to the withdrawal cone of this disable operation and
therefore must be withdrawn before the system can become quiescent?
Exclude components on unrelated dependency branches.
Return a sorted JSON array of component names and nothing else."""
    return make_record(
        task_type="formal_withdrawal_cone",
        prompt=prompt,
        answer=canonical_list(affected, sort=True),
        answer_type="string_set",
        metadata=_metadata(
            case, track, "withdrawal_cone", "localization", "string_set",
            "formal.withdrawal_cone.v2",
            query_quantifier="dependency_closure",
            oracle_type="exact_dependency_closure",
            operational_relevance="identify_affected_harness_components",
        ),
        identity={"world": case["world_id"], "task": "withdrawal_cone"},
    )


def _prediction_record(case, track, rng):
    groups = _relevant_groups(case)
    coords = [offset + rng.randrange(2) for offset, _names in groups]
    values = _final_values(case, case["schedule"])
    requested = "\n".join(f"  {index + 1}. x{coord}" for index, coord in enumerate(coords))
    schedule = " -> ".join(case["schedule"])
    prompt = _formal_prefix(case) + f"""
Suppose withdrawals complete in this legal order:
  {schedule}

Report the final values of these application observables, in this exact order:
{requested}

Return only a JSON array of values."""
    return make_record(
        task_type="formal_schedule_prediction",
        prompt=prompt,
        answer=canonical_scalar_sequence(values[coord] for coord in coords),
        answer_type="scalar_sequence",
        metadata=_metadata(
            case, track, "schedule_prediction", "prediction", "scalar_sequence",
            "formal.schedule_prediction.v2",
            queried_observables=len(coords),
            query_quantifier="named_legal_schedule",
            oracle_type="exact_transition_execution",
            operational_relevance="predict_application_observables",
        ),
        identity={"world": case["world_id"], "task": "prediction", "coords": coords},
    )


def _absent_pair(outcomes, modulus, rng):
    outcomes = set(outcomes)
    start = rng.randrange(modulus * modulus)
    for delta in range(modulus * modulus):
        value = (start + delta) % (modulus * modulus)
        candidate = (value // modulus, value % modulus)
        if candidate not in outcomes:
            return candidate
    raise RuntimeError("formal V2 local outcome set covers the full pair space")


def _condition_record(case, track, rng, *, reachable):
    groups = _groups_with_outcomes(case)
    probes = []
    for offset, _names, outcomes in groups:
        present = rng.choice(sorted(outcomes))
        absent = _absent_pair(outcomes, case["world"].modulus, rng)
        if reachable:
            probes.extend(((offset, "==", present, True), (offset, "==", absent, False)))
        else:
            probes.extend(((offset, "!=", absent, True), (offset, "!=", present, False)))
    rng.shuffle(probes)

    conditions = []
    true_labels = []
    width = max(2, len(str(len(probes))))
    for index, (offset, operator, value, truth) in enumerate(probes, start=1):
        label = f"c{index:0{width}d}"
        conditions.append((label, offset, operator, value))
        if truth:
            true_labels.append(label)

    lines = "\n".join(
        f"  {label}: (x{offset}, x{offset + 1}) {operator} ({value[0]}, {value[1]})"
        for label, offset, operator, value in conditions
    )
    if reachable:
        question = """For each condition independently, include its label if AT LEAST ONE legal
withdrawal schedule reaches a final state satisfying that condition."""
        family = "reachable_conditions"
        capability = "reachability"
        quantifier = "exists_legal_withdrawal_schedule_per_condition"
        relevance = "identify_possible_application_conditions"
    else:
        question = """For each condition independently, include its label if EVERY legal withdrawal
schedule ends in a final state satisfying that condition."""
        family = "guaranteed_conditions"
        capability = "postcondition_guarantee"
        quantifier = "all_legal_withdrawal_schedules_per_condition"
        relevance = "identify_guaranteed_application_conditions"

    prompt = _formal_prefix(case) + f"""
Named application conditions:
{lines}

{question}
Return a sorted JSON array of condition labels and nothing else."""
    task_type = f"formal_{family}"
    return make_record(
        task_type=task_type,
        prompt=prompt,
        answer=canonical_list(true_labels, sort=True),
        answer_type="string_set",
        metadata=_metadata(
            case, track, family, capability, "string_set", f"formal.{family}.v2",
            queried_conditions=len(conditions),
            query_quantifier=quantifier,
            oracle_type="factorized_exact_condition_set",
            operational_relevance=relevance,
        ),
        identity={"world": case["world_id"], "task": family, "conditions": conditions},
    )


def _count_record(case):
    prompt = _formal_prefix(case) + """
Diagnostic only: across all legal complete withdrawal schedules, how many
distinct final observable outcomes (shared state plus active component set)
are reachable?
Return only the integer count."""
    return make_record(
        task_type="formal_outcome_count_diagnostic",
        prompt=prompt,
        answer=case["outcome_count"],
        answer_type="integer",
        metadata=_metadata(
            case, "challenge", "outcome_count", "global_cardinality", "integer",
            "formal.outcome_count.v2", paper_role="diagnostic",
            query_quantifier="all_legal_withdrawal_schedules",
            oracle_type="validated_factorized_cardinality",
            operational_relevance="diagnostic_not_primary",
        ),
        identity={"world": case["world_id"], "task": "outcome_count"},
    )


def _sample_cases(rng, size, worlds_per_size):
    cases = []
    counts = set()
    attempts = 0
    while len(cases) < worlds_per_size:
        attempts += 1
        if attempts > max(100, worlds_per_size * 50):
            raise RuntimeError(f"could not sample varied V2 formal worlds for size={size}")
        case = _build_formal_world(rng, int(size))
        if (
            worlds_per_size > 1
            and len(cases) == worlds_per_size - 1
            and len(counts) == 1
            and case["outcome_count"] in counts
        ):
            continue
        cases.append(case)
        counts.add(case["outcome_count"])
    return cases


def _records_for_case(case, track, rng):
    records = [
        _affected_record(case, track),
        _prediction_record(case, track, rng),
        _condition_record(case, track, rng, reachable=False),
        _condition_record(case, track, rng, reachable=True),
    ]
    if track == "challenge":
        records.append(_count_record(case))
    return records


def generate_v2_formal_core_dataset(
    sizes=V2_FORMAL_CORE_SIZES,
    worlds_per_size=V2_FORMAL_CORE_WORLDS_PER_SIZE,
    seed=0,
):
    rng = random.Random(seed + 20_100_101)
    return [
        record
        for size in map(int, sizes)
        for case in _sample_cases(rng, size, worlds_per_size)
        for record in _records_for_case(case, "core", rng)
    ]


def generate_v2_formal_challenge_dataset(
    sizes=V2_FORMAL_CHALLENGE_SIZES,
    worlds_per_size=V2_FORMAL_CHALLENGE_WORLDS_PER_SIZE,
    seed=0,
):
    rng = random.Random(seed + 20_200_101)
    return [
        record
        for size in map(int, sizes)
        for case in _sample_cases(rng, size, worlds_per_size)
        for record in _records_for_case(case, "challenge", rng)
    ]
