# Paper results

This directory contains the final score reports used to construct the paper's
tables and figures. Reports retain task-level and per-question measurements so
the published aggregates can be checked without rerunning model inference.

The main release reports cover Gemini 3.7 Flash, GPT-5.6 Luna, and
DeepSeek V4 Flash 0731 under the paper's low-reasoning protocol. The
`effort-size16-*` reports contain the reasoning-effort diagnostic. The remaining
files contain the fixed-schedule, completion-cap, shortcut, and decomposed-score
diagnostics described in the paper.

Raw model responses are intentionally omitted from the code repository. The
dataset, gold answers, and standalone verifier are published on
[Hugging Face](https://huggingface.co/datasets/sileod/cordis-bench).
