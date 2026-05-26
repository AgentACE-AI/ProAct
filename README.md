# ProAct

This repository contains the public reproducibility release for **ProAct**, the proactive agent architecture introduced in:

- Paper: [Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents](https://arxiv.org/abs/2605.25971)
- Project showcase: https://agentace-ai.github.io/proact-showcase/
- Code: https://github.com/AgentACE-AI/ProAct

This is a focused code and benchmark subset intended for paper readers, artifact reviewers, and researchers who want to inspect or rerun the ProActEval experiments. It is not a full product checkout or hosted demo backend.

## Contents

- Core runtime packages used by the evaluation path: `agents/`, `core/`, `memory/`, `server/`, `services/`, and `tools/`.
- ProActEval evaluation and generation code under `experiments/ProactiveBench/`.
- The 200 ProActEval scenario JSON files and `scenario_groups.json`.
- The MemBench adapter/evaluation subset under `experiments/MemBench/benchmark/`.
- Aggregate ProActEval result summaries under `results/` when available.
- Focused tests for command importability, ProactiveAgent adapter leakage, judge-labeled anticipation, public release contents, and leak scanning.

## Environment Setup

Use Python 3.10 or newer. The release has been kept dependency-light, but full evaluation runs require paid LLM API access.

```bash
git clone https://github.com/AgentACE-AI/ProAct.git
cd ProAct

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file before running API-backed experiments:

```bash
cp .env.example .env
```

Required for ProActEval:

- `OPENAI_API_KEY`: used by the assistant, simulator, and judge calls.
- `OPENAI_BASE_URL`: optional OpenAI-compatible endpoint; defaults to `https://api.openai.com/v1`.
- `EMBEDDING_MODEL`: defaults to `text-embedding-3-small`.

Optional search/fetch configuration:

- `SERPER_API_KEY`: enables Serper-backed web search.
- `JINA_API_KEY`: enables Jina Reader-backed page fetching.
- `PREFER_SERPER`: set to `true` to prefer Serper when available.

Do not commit `.env`, runtime user data, generated traces, or local outputs.

## Offline Validation

These checks do not require an API key:

```bash
python -m experiments.ProactiveBench.generation.validate_scenario \
  --scenario-dir experiments/ProactiveBench/data/scenarios

python -m unittest discover -s tests -p 'test_*.py'
```

The scenario validator should report 200 passed scenarios.

## Reproducing ProActEval

For a quick API-backed smoke run, execute one scenario first:

```bash
python -m experiments.ProactiveBench.eval.runner \
  --scenario-dir experiments/ProactiveBench/data/scenarios \
  --output-dir outputs/proacteval_smoke \
  --conditions Baseline Blind Full-single-idle \
  --seed 42 \
  --judge-model gpt-4o-mini \
  --simulator-model gpt-4o \
  --max-queries-per-search 1 \
  --max-intents-per-idle 3 \
  --idle-trigger-seconds 5.0 \
  --max-total-searches 999 \
  --only finance_basic_01
```

To run the full 200-scenario ProActEval matrix:

```bash
python -m experiments.ProactiveBench.eval.runner \
  --scenario-dir experiments/ProactiveBench/data/scenarios \
  --output-dir outputs/proacteval_full200 \
  --conditions Baseline Blind Full-single-idle \
  --seed 42 \
  --judge-model gpt-4o-mini \
  --simulator-model gpt-4o \
  --max-queries-per-search 1 \
  --max-intents-per-idle 3 \
  --idle-trigger-seconds 5.0 \
  --max-total-searches 999
```

Full reproduction uses OpenAI API calls for simulator, assistant, and judge components and can incur meaningful external API cost.

## Baselines And Additional Experiments

Run the ProactiveAgent-style baseline:

```bash
python -m experiments.ProactiveBench.eval.run_proactive_agent \
  --scenario-dir experiments/ProactiveBench/data/scenarios \
  --output-dir outputs/proactive_agent_style_4o \
  --seed 42 \
  --judge-model gpt-4o-mini \
  --simulator-model gpt-4o \
  --adapter-model gpt-4o
```

Generate the judge-labeled anticipation report from a completed ProactiveAgent-style run:

```bash
python -m experiments.ProactiveBench.eval.judge_labeled_anticipation \
  --results-dir outputs/proactive_agent_style_4o
```

Run the MemBench reflective-memory comparison subset:

```bash
python -m experiments.MemBench.benchmark.comparison_experiment \
  --table 4 \
  --local-model Qwen/Qwen2.5-7B-Instruct
```

The MemBench command expects upstream MemBench data and a local Qwen-compatible inference setup. Those third-party assets are not redistributed in this repository.

For a compact command appendix, see `RUN_COMMANDS.md`.

## Outputs

Experiment outputs are written under the `outputs/` directory in the commands above. Generated outputs, raw traces, caches, and local runtime state are intentionally ignored and should not be committed.

## Exclusions

This public release intentionally excludes private development history, local handoff notes, `.env`, local tooling metadata, raw traces, runtime user state, caches, logs, upstream MemBench datasets, checkpoints, and generated large experiment outputs. Raw ProActEval traces are omitted to keep the repository compact; aggregate summaries, bootstrap intervals, resource summaries, and provenance manifests can be added as small supplemental files when available.

## Cost And Data Notes

ProActEval reproduction uses OpenAI API calls for simulator, assistant, and judge components. Running the full 200-scenario matrix incurs external API cost. MemBench reproduction requires a local Qwen-compatible inference setup and upstream MemBench data/model access; those third-party assets are not redistributed in this repository.

## License And Citation

Code and original benchmark assets in this repository are released under the MIT License. See `LICENSE`, `LICENSES.md`, and `CITATION.cff` for reuse and citation details.
