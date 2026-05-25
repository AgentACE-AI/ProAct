"""LLM usage snapshotting and module-level delta attribution."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator


USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")


def snapshot_usage(llm_client: Any) -> Dict[str, Dict[str, int]]:
    getter = getattr(llm_client, "get_usage_stats", None)
    if not callable(getter):
        return {}
    try:
        raw = getter()
    except Exception:
        return {}
    return {
        str(model): {
            key: int((stats or {}).get(key, 0) or 0)
            for key in USAGE_KEYS
        }
        for model, stats in (raw or {}).items()
        if isinstance(stats, dict)
    }


def usage_delta(
    before: Dict[str, Dict[str, int]],
    after: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    delta: Dict[str, Dict[str, int]] = {}
    for model in sorted(set(before) | set(after)):
        model_delta = {
            key: max(
                0,
                int(after.get(model, {}).get(key, 0))
                - int(before.get(model, {}).get(key, 0)),
            )
            for key in USAGE_KEYS
        }
        if any(value != 0 for value in model_delta.values()):
            delta[model] = model_delta
    return delta


def flatten_usage(
    usage_by_model: Dict[str, Dict[str, int]],
) -> Dict[str, int]:
    totals = {key: 0 for key in USAGE_KEYS}
    for stats in usage_by_model.values():
        if not isinstance(stats, dict):
            continue
        for key in USAGE_KEYS:
            totals[key] += int(stats.get(key, 0) or 0)
    return totals


class LLMCostTracker:
    """Accumulates usage deltas into named benchmark module buckets."""

    def __init__(
        self,
        llm_client: Any,
        initial_snapshot: Dict[str, Dict[str, int]] | None = None,
    ):
        self.llm_client = llm_client
        self.module_usage: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.latest_snapshot = initial_snapshot if initial_snapshot is not None else snapshot_usage(llm_client)

    @contextmanager
    def track(self, module: str) -> Iterator[None]:
        before = self.latest_snapshot
        try:
            yield
        finally:
            after = snapshot_usage(self.llm_client)
            if after:
                self.latest_snapshot = after
            self.add_delta(module, usage_delta(before, after))

    def add_delta(
        self,
        module: str,
        delta: Dict[str, Dict[str, int]],
    ) -> None:
        if not delta:
            return
        bucket = self.module_usage.setdefault(module, {})
        for model, stats in delta.items():
            model_bucket = bucket.setdefault(model, {key: 0 for key in USAGE_KEYS})
            for key in USAGE_KEYS:
                model_bucket[key] += int(stats.get(key, 0) or 0)

    @property
    def module_totals(self) -> Dict[str, Dict[str, int]]:
        return {
            module: flatten_usage(usage_by_model)
            for module, usage_by_model in self.module_usage.items()
        }
