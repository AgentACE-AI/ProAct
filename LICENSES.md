# Asset And License Notes

Audit date: 2026-05-25.

This public release does not redistribute MemBench datasets or source repositories, ProactiveAgent checkpoints, Qwen model weights, OpenAI model weights, or third-party package source trees. Those assets remain available from their original providers under their own terms.

| Asset | Use | Source | License or terms |
| --- | --- | --- | --- |
| ProActEval | Synthetic benchmark scenarios, generation code, and evaluation protocol | This repository | MIT for original code and benchmark assets in this repository. |
| MemBench | Reflective memory benchmark interface and comparison command | https://github.com/import-myself/Membench | Upstream repository reports MIT. Datasets are not redistributed here. |
| Public proactive-agent baseline | Prompting protocol baseline adapted for ProActEval | Public third-party implementation | Upstream repository reports Apache-2.0. Checkpoints are not redistributed here. |
| Qwen2.5-7B-Instruct | Local MemBench backbone | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Model card reports Apache-2.0. Weights are not redistributed here. |
| OpenAI API models | ProActEval simulator, assistant, and judge calls | https://platform.openai.com/ | Governed by OpenAI service terms; model weights are not redistributed. |
| ChromaDB and Python dependencies | Runtime dependencies | `requirements.txt` | Installed by the reproducer from package indexes under package-specific terms. |
