"""
Duck-typed SearchTool and WebFetcher implementations that query ExternalFactStore.

These satisfy the same interfaces that SearchService expects from
tools/search_tool.py:SearchTool and tools/web_fetcher.py:WebFetcher,
redirecting all search traffic to the benchmark's external fact store.
"""

from __future__ import annotations

from typing import List, Optional

from core.models import SearchSource
from experiments.ProactiveBench.eval.external_fact_store import ExternalFactStore


class BenchmarkSearchTool:
    """Drop-in replacement for SearchTool that queries ExternalFactStore."""

    def __init__(self, fact_store: ExternalFactStore):
        self._fact_store = fact_store

    def search(self, query: str, num_results: int = 5) -> List[SearchSource]:
        return self._fact_store.search(query, n_results=num_results)

    @property
    def is_available(self) -> bool:
        return True


class BenchmarkWebFetcher:
    """Drop-in replacement for WebFetcher that resolves extfact:// URLs."""

    def __init__(self, fact_store: ExternalFactStore):
        self._fact_store = fact_store

    def fetch(self, url: str, **kwargs) -> Optional[str]:
        if url.startswith("extfact://"):
            fact_id = url[len("extfact://"):]
            return self._fact_store.get_fact_by_id(fact_id)
        return None
