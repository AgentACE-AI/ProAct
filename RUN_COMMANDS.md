# Reproduction Commands

Run these commands from the repository root.

## Environment

```bash
pip install -r requirements.txt
```

Populate `.env` from `.env.example` before paid API runs. The reported ProActEval runs used OpenAI API models `gpt-4o` and `gpt-4o-mini`; the MemBench reflective memory runs used `Qwen/Qwen2.5-7B-Instruct` locally.

## Offline Checks

```bash
python -m experiments.ProactiveBench.generation.validate_scenario \
  --scenario-dir experiments/ProactiveBench/data/scenarios

python -m unittest \
  tests.test_proactive_agent_adapter \
  tests.test_proactivebench_judge_labeled_anticipation \
  tests.test_proactivebench_generation_module_imports \
  tests.test_membench_release_imports \
  tests.test_public_release_contents \
  tests.test_anonymity_scan
```

## ProActEval Main Evaluation

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

## ProactiveAgent-Style Baseline

```bash
python -m experiments.ProactiveBench.eval.run_proactive_agent \
  --scenario-dir experiments/ProactiveBench/data/scenarios \
  --output-dir outputs/proactive_agent_style_4o \
  --seed 42 \
  --judge-model gpt-4o-mini \
  --simulator-model gpt-4o \
  --adapter-model gpt-4o
```

## Judge-Labeled Anticipation Report

```bash
python -m experiments.ProactiveBench.eval.judge_labeled_anticipation \
  --results-dir outputs/proactive_agent_style_4o
```

## MemBench Reflective Memory

```bash
python -m experiments.MemBench.benchmark.comparison_experiment \
  --table 4 \
  --local-model Qwen/Qwen2.5-7B-Instruct
```

The MemBench command expects upstream MemBench data to be available externally. It is not redistributed here.
