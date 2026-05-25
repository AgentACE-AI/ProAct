"""
External fact store backed by an in-memory ChromaDB collection.

Simulates the "external world" that the system must actively search to
discover facts, rather than having them pre-loaded into system memory.
"""

from __future__ import annotations

import logging
from typing import List, TYPE_CHECKING

import chromadb

from core.models import SearchSource

if TYPE_CHECKING:
    from experiments.ProactiveBench.generation.models import Fact

logger = logging.getLogger(__name__)


class ExternalFactStore:

    def __init__(self, scenario_id: str):
        self._client = chromadb.Client()
        safe_name = scenario_id.replace("-", "_").replace(" ", "_")[:60]
        col_name = f"facts_{safe_name}"
        try:
            self._client.delete_collection(name=col_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(name=col_name)
        self._scenario_id = scenario_id

    def load_facts(self, facts: List["Fact"], domain: str) -> None:
        if not facts:
            return
        documents = []
        metadatas = []
        ids = []
        for fact in facts:
            documents.append(f"{fact.id}: {fact.fact}")
            metadatas.append({
                "fact_id": fact.id,
                "category": fact.category,
                "domain": domain,
            })
            ids.append(fact.id)
        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(
            "ExternalFactStore: loaded %d facts for scenario %s",
            len(facts), self._scenario_id,
        )

    def search(self, query: str, n_results: int = 5) -> List[SearchSource]:
        count = self._collection.count()
        if count == 0:
            return []
        n = min(n_results, count)
        results = self._collection.query(query_texts=[query], n_results=n)
        sources = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            fact_id = meta.get("fact_id", "")
            category = meta.get("category", "")
            fact_text = doc.split(": ", 1)[-1] if ": " in doc else doc
            sources.append(SearchSource(
                url=f"extfact://{fact_id}",
                title=f"[{category}] {fact_id}",
                snippet=fact_text,
                full_content=fact_text,
            ))
        return sources

    def get_fact_by_id(self, fact_id: str) -> str | None:
        try:
            result = self._collection.get(ids=[fact_id])
            docs = result.get("documents", [])
            if docs:
                doc = docs[0]
                return doc.split(": ", 1)[-1] if ": " in doc else doc
        except Exception:
            pass
        return None

    def get_fact_count(self) -> int:
        return self._collection.count()
