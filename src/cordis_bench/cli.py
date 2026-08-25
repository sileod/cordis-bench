import argparse
import json

from .reporting import score_predictions
from .v2_scoring import score_v2_predictions
from .io import read_jsonl, write_jsonl
from .openrouter import DEFAULT_MODELS, probe_dataset
from .tasks import V0_TASKS, V1_TASKS, generate_dataset, generate_v1_dataset
from .v11 import V11_TASKS, generate_v11_dataset
from .v12 import V12_TASKS, generate_v12_dataset
from .v13 import V13_TASKS, generate_v13_dataset
from .v14 import V14_TASKS, generate_v14_dataset
from .v15 import V15_TASKS, generate_v15_dataset
from .v16 import V16_TASKS, generate_v16_dataset
from .v16_stress import generate_v16_stress_dataset
from .v17_suite import V17_DEFAULT_N, V17_TASKS, generate_v17_dataset
from .v18 import V18_DEFAULT_N, V18_TASKS, generate_v18_dataset
from .v19 import V19_DEFAULT_N, V19_TASKS, generate_v19_dataset
from .v110 import V110_DEFAULT_N, V110_TASKS, generate_v110_dataset
from .v111 import (
    V111_FORMAL_SIZES,
    V111_FORMAL_WORLDS_PER_SIZE,
    V111_NATIVE_SIZES,
    V111_NATIVE_WORLDS_PER_SIZE,
    V111_TASKS,
    generate_v111_dataset,
)
from .v112 import (
    V112_FORMAL_SIZES,
    V112_FORMAL_WORLDS_PER_SIZE,
    V112_NATIVE_SIZES,
    V112_NATIVE_WORLDS_PER_SIZE,
    V112_TASKS,
    generate_v112_dataset,
)
from .v2 import (
    V2_FORMAL_CHALLENGE_SIZES,
    V2_FORMAL_CHALLENGE_WORLDS_PER_SIZE,
    V2_NATIVE_CHALLENGE_SIZES,
    V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE,
    V2_TASKS,
)
from .v2_release import V2_RELEASE_REPLICATES, generate_v2_release_dataset


def _parse_int_list(value, default):
    if not value:
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def cmd_generate(args):
    if args.tasks:
        task_types = tuple(part.strip() for part in args.tasks.split(",") if part.strip())
    else:
        task_types = {
            "0": V0_TASKS,
            "1": V1_TASKS,
            "1.1": V11_TASKS,
            "1.2": V12_TASKS,
            "1.3": V13_TASKS,
            "1.4": V14_TASKS,
            "1.5": V15_TASKS,
            "1.6": V16_TASKS,
            "1.6-stress": V16_TASKS,
            "1.7": V17_TASKS,
            "1.8": V18_TASKS,
            "1.9": V19_TASKS,
            "1.10": V110_TASKS,
            "1.11": V111_TASKS,
            "1.12": V112_TASKS,
            "2": V2_TASKS,
        }[args.version]

    if args.version in {"1.11", "1.12", "2"}:
        if args.components is not None or args.width is not None:
            raise ValueError(
                f"V{args.version} uses fixed component-specific sizes; "
                "use --formal-challenge-sizes / --native-challenge-sizes for the challenge ladder"
            )
        if args.version == "1.11":
            formal_default = V111_FORMAL_SIZES
            native_default = V111_NATIVE_SIZES
            formal_worlds_default = V111_FORMAL_WORLDS_PER_SIZE
            native_worlds_default = V111_NATIVE_WORLDS_PER_SIZE
            generator = generate_v111_dataset
        elif args.version == "1.12":
            formal_default = V112_FORMAL_SIZES
            native_default = V112_NATIVE_SIZES
            formal_worlds_default = V112_FORMAL_WORLDS_PER_SIZE
            native_worlds_default = V112_NATIVE_WORLDS_PER_SIZE
            generator = generate_v112_dataset
        else:
            formal_default = V2_FORMAL_CHALLENGE_SIZES
            native_default = V2_NATIVE_CHALLENGE_SIZES
            formal_worlds_default = V2_FORMAL_CHALLENGE_WORLDS_PER_SIZE
            native_worlds_default = V2_NATIVE_CHALLENGE_WORLDS_PER_SIZE
            generator = generate_v2_release_dataset

        formal_sizes = _parse_int_list(args.formal_challenge_sizes, formal_default)
        native_sizes = _parse_int_list(args.native_challenge_sizes, native_default)
        formal_worlds = (
            args.formal_challenge_worlds
            if args.formal_challenge_worlds is not None
            else formal_worlds_default
        )
        native_worlds = (
            args.native_challenge_worlds
            if args.native_challenge_worlds is not None
            else native_worlds_default
        )
        kwargs = dict(
            n=args.n,
            seed=args.seed,
            task_types=task_types,
            formal_sizes=formal_sizes,
            native_sizes=native_sizes,
            formal_worlds=formal_worlds,
            native_worlds=native_worlds,
            runner_path=args.native_runner,
            node=args.node,
        )
        if args.version == "2":
            kwargs["replicates"] = args.v2_replicates
        records = generator(**kwargs)

        if args.version == "2":
            output = args.output or "data/generated/v2.jsonl"
            write_jsonl(output, records)
            print(
                f"wrote {len(records)} V2 tasks from "
                f"{args.v2_replicates} independent replicate block(s) to {output}"
            )
        else:
            output = args.output or f"data/generated/v{args.version}-challenge.jsonl"
            write_jsonl(output, records)
            print(f"wrote {len(records)} v{args.version} challenge tasks to {output}")
        return

    if args.version == "0":
        n_components = args.components or 4
        width = args.width or 3
    elif args.version == "1.5":
        n_components = args.components or 10
        width = args.width or 9
    elif args.version == "1.6":
        n_components = args.components or 12
        width = args.width or 11
    elif args.version == "1.6-stress":
        n_components = args.components or 14
        width = args.width or (n_components - 1)
    elif args.version in {"1.7", "1.8", "1.9", "1.10"}:
        if args.components is not None or args.width is not None:
            raise ValueError(
                f"V{args.version} is composite with per-component sizes; "
                "do not pass --components or --width"
            )
        n_components = None
        width = None
    else:
        n_components = args.components or 6
        width = args.width or 4

    generator = {
        "0": generate_dataset,
        "1": generate_v1_dataset,
        "1.1": generate_v11_dataset,
        "1.2": generate_v12_dataset,
        "1.3": generate_v13_dataset,
        "1.4": generate_v14_dataset,
        "1.5": generate_v15_dataset,
        "1.6": generate_v16_dataset,
        "1.6-stress": generate_v16_stress_dataset,
        "1.7": generate_v17_dataset,
        "1.8": generate_v18_dataset,
        "1.9": generate_v19_dataset,
        "1.10": generate_v110_dataset,
    }[args.version]

    if args.n is not None:
        n = args.n
    elif args.version == "1.10":
        n = V110_DEFAULT_N
    elif args.version == "1.9":
        n = V19_DEFAULT_N
    elif args.version == "1.8":
        n = V18_DEFAULT_N
    elif args.version == "1.7":
        n = V17_DEFAULT_N
    elif args.version == "1.6-stress":
        n = 108
    elif args.version == "1.6":
        n = 180
    elif args.version in {"1.4", "1.5"}:
        n = 160
    else:
        n = 120

    output = args.output or f"data/generated/v{args.version}-dev.jsonl"
    kwargs = {
        "seed": args.seed,
        "task_types": task_types,
        "n_components": n_components,
        "width": width,
    }
    if args.version in {"1.8", "1.9", "1.10"}:
        kwargs["runner_path"] = args.native_runner
        kwargs["node"] = args.node
    records = generator(n, **kwargs)
    write_jsonl(output, records)
    print(f"wrote {len(records)} v{args.version} tasks to {output}")


def cmd_probe(args):
    dataset = read_jsonl(args.dataset)
    results = probe_dataset(
        dataset,
        args.model,
        workers=args.workers,
        output_path=args.output,
        resume=not args.no_resume,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.effort,
        timeout=args.timeout,
        attempt_timeout=args.attempt_timeout,
        retries=args.retries,
        rpm=args.rpm,
        fallbacks=args.fallback,
    )
    write_jsonl(args.output, results)
    print(f"wrote {len(results)} predictions to {args.output}")


def cmd_score(args):
    dataset = read_jsonl(args.dataset)
    predictions = read_jsonl(args.predictions)
    is_v2 = bool(dataset) and all(item.get("schema_version") == "2.0" for item in dataset)
    report = (
        score_v2_predictions(dataset, predictions)
        if is_v2
        else score_predictions(dataset, predictions)
    )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(prog="cordis-bench")
    sub = parser.add_subparsers(required=True)

    generate = sub.add_parser("generate", help="generate a procedural JSONL dataset")
    generate.add_argument("--output")
    generate.add_argument("-n", type=int)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument(
        "--version",
        choices=(
            "0", "1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6",
            "1.6-stress", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12", "2"
        ),
        default="1.10",
    )
    generate.add_argument("--tasks")
    generate.add_argument("--components", type=int)
    generate.add_argument("--width", type=int)
    generate.add_argument(
        "--formal-challenge-sizes",
        help="comma-separated formal challenge sizes (V1.11/V1.12/V2)",
    )
    generate.add_argument(
        "--native-challenge-sizes",
        help="comma-separated native challenge sizes (V1.11/V1.12/V2)",
    )
    generate.add_argument("--formal-challenge-worlds", type=int)
    generate.add_argument("--native-challenge-worlds", type=int)
    generate.add_argument(
        "--v2-replicates",
        type=int,
        default=V2_RELEASE_REPLICATES,
        help=(
            "number of independent V2 400-item blocks; default 3 gives the "
            "1,200-item paper release, use 1 to reproduce one seed block"
        ),
    )
    generate.add_argument(
        "--native-runner",
        help="override native/cordis/run.mjs for V1.8+ Cordis execution",
    )
    generate.add_argument(
        "--node",
        default="node",
        help="Node.js executable used by the V1.8+ native Cordis runner",
    )
    generate.set_defaults(func=cmd_generate)

    probe = sub.add_parser("probe", help="probe a model through litlm")
    probe.add_argument("dataset")
    probe.add_argument("--model", default=DEFAULT_MODELS[0])
    probe.add_argument("--output", default="results/predictions.jsonl")
    probe.add_argument("--workers", type=int, default=4)
    probe.add_argument(
        "--retries",
        type=int,
        default=3,
        help="retries for transient transport/5xx/429 failures",
    )
    probe.add_argument(
        "--rpm",
        type=float,
        help="maximum request starts per minute; concurrency still applies",
    )
    # Kept so commands created before the litlm migration remain resumable;
    # LiteLLM now owns retry timing.
    probe.add_argument(
        "--retry-backoff",
        type=float,
        help=argparse.SUPPRESS,
    )
    probe.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any existing checkpoint at the output path",
    )
    probe.add_argument("--temperature", type=float, default=0.0)
    probe.add_argument("--max-tokens", type=int, default=8192)
    probe.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-operation HTTP timeout in seconds (not a total generation deadline)",
    )
    probe.add_argument(
        "--attempt-timeout",
        type=float,
        help="hard wall-clock timeout for each provider before trying a fallback",
    )
    probe.add_argument(
        "--fallback",
        action="append",
        help="exact fallback route; repeat to define an ordered hierarchy",
    )
    probe.add_argument(
        "--effort",
        choices=("instant", "none", "minimal", "low", "medium", "high"),
        default="low",
        help="provider reasoning-effort setting; 'instant' is an alias for 'none'",
    )
    probe.set_defaults(func=cmd_probe)

    score = sub.add_parser("score", help="score predictions against a generated dataset")
    score.add_argument("dataset")
    score.add_argument("predictions")
    score.add_argument("--output")
    score.set_defaults(func=cmd_score)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)
