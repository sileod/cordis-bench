# CordisBench

CordisBench evaluates whether language models can reason about the consequences
of component lifecycle changes in dynamic agent harnesses. It combines a
controlled formal setting with executable programs run by Cordis.

The frozen dataset is available on
[Hugging Face](https://huggingface.co/datasets/sileod/cordis-bench). This
repository contains the generator, deterministic scorers, Cordis runner, and
aggregate reports used for the paper.

## Tasks and metrics

| Task | Primary metric |
| --- | --- |
| Localization | Jaccard similarity |
| Schedule prediction | Per-observable accuracy |
| Guaranteed conditions | Jaccard similarity |
| Reachable conditions | Jaccard similarity |
| Reconfiguration | Executed success rate |

Whole-answer exact match and parse rate are retained as strict diagnostics.
Reconfiguration answers are executed against the pinned Cordis runtime rather
than judged as text.

## Install

CordisBench requires Python 3.11 or later and Node.js 20.

```bash
git clone https://github.com/sileod/cordis-bench.git
cd cordis-bench
python -m pip install -e '.[dev]'
npm ci --prefix native/cordis
node native/cordis/run.mjs --self-test
```

## Generate the frozen release

The paper release contains 1,200 questions generated from seeds 0, 1, and 2.

```bash
cordis-bench generate --version 2 --seed 0 \
  --output v2.0.1-release.jsonl
```

Generation includes differential checks between the finite reference semantics
and Cordis execution for every native instance.

## Evaluate a model

Model calls use [`litlm`](https://pypi.org/project/litlm/) for provider
selection, retries, throttling, resumable output, and failure reporting. Supply
the credentials required by the provider selected through `--model`.

```bash
cordis-bench probe v2.0.1-release.jsonl \
  --model provider/model-name \
  --effort low --max-tokens 8192 --workers 8 --rpm 35 \
  --output predictions.jsonl

cordis-bench score v2.0.1-release.jsonl predictions.jsonl \
  --output score.json
```

Each completed response is checkpointed immediately. Repeating the same probe
command resumes unfinished questions. Explicit fallback routes can be supplied
with repeated `--fallback` options.

## Reproduce the paper diagnostics

The scripts directory contains the deterministic transformations used for the
reasoning-effort, fixed-schedule, and completion-cap diagnostics. The aggregate
reports used to construct the paper figures are in [`results/`](results/).

Run the benchmark checks with:

```bash
pytest -q
```

The public Hugging Face release also includes a standalone verifier for scoring
prediction files without installing the generator package.
