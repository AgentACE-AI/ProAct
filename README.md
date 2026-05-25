# ProAct

This repository contains the public reproducibility release for ProAct. It is a focused code and benchmark subset intended for paper readers, artifact reviewers, and researchers who want to inspect or rerun the ProActEval experiments.

The arXiv link will be added here after the paper is posted.

## Contents

- Core runtime packages used by the evaluation path: `agents/`, `core/`, `memory/`, `server/`, `services/`, and `tools/`.
- ProActEval evaluation and generation code under `experiments/ProactiveBench/`.
- The 200 ProActEval scenario JSON files and `scenario_groups.json`.
- The MemBench adapter/evaluation subset under `experiments/MemBench/benchmark/`.
- Aggregate ProActEval result summaries under `results/` when available.
- Focused tests for command importability, ProactiveAgent adapter leakage, judge-labeled anticipation, public release contents, and leak scanning.

## Exclusions

This public release intentionally excludes private development history, local handoff notes, `.env`, local tooling metadata, raw traces, runtime user state, caches, logs, upstream MemBench datasets, checkpoints, and generated large experiment outputs. Raw ProActEval traces are omitted to keep the repository compact; aggregate summaries, bootstrap intervals, resource summaries, and provenance manifests can be added as small supplemental files when available.

## Cost And Data Notes

ProActEval reproduction uses OpenAI API calls for simulator, assistant, and judge components. Running the full 200-scenario matrix incurs external API cost. MemBench reproduction requires a local Qwen-compatible inference setup and upstream MemBench data/model access; those third-party assets are not redistributed in this repository.

## License And Citation

Code and original benchmark assets in this repository are released under the MIT License. See `LICENSE`, `LICENSES.md`, and `CITATION.cff` for reuse and citation details.
