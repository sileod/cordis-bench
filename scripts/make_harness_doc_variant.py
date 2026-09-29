"""Add an explicit description of Cordis's automatic behavior to native prompts.

The baseline prompt shows the program and states capture/restore; this variant
also documents what the harness does on the model's behalf (cascade disposal,
cleanup order), testing whether anticipating the harness is the bottleneck.

Usage: python scripts/make_harness_doc_variant.py DATASET OUTPUT
"""

import json
import sys

HARNESS_DOC = """
What Cordis does automatically (documented behavior of this harness):
  - Disposing a provider automatically disposes every plugin that injects its
    service; plugins injecting other services stay active and untouched.
  - When a plugin is disposed, its effect's disposer runs: it writes back the
    value `previous` that this effect captured at activation, which may be a
    value written by an earlier-activated plugin rather than the initial value.
  - Disposers of the automatically disposed plugins complete in the listed
    controlled completion order; a later write overwrites an earlier one.
  - A plugin disposed explicitly beforehand runs its disposer at that point and
    is not disposed again with the provider.
"""
ANCHOR = "\n```js\n"


def main(dataset, output):
    with open(dataset) as src, open(output, "w") as out:
        for line in src:
            item = json.loads(line)
            if not item["task_type"].startswith("cordis_"):
                continue
            item["prompt"] = item["prompt"].replace(ANCHOR, HARNESS_DOC + ANCHOR, 1)
            item["metadata"]["prompt_variant"] = "harness_doc"
            out.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
