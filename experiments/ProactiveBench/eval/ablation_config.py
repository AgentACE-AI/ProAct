"""
Ablation experiment configuration, intent queue, and shared data structures.

This module defines the four experimental conditions (Full, Blind, Paralyzed,
Baseline) and the dedup priority queue used by FSP/ITE modules.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


VALID_CONDITIONS = frozenset({
    "Full",
    "Full-single-idle",
    "Full-idle-window",
    "Blind",
    "Paralyzed",
    "Baseline",
    # Legacy Route C/C1/C0 labels remain accepted so older trace tooling can
    # still deserialize and run compatibility tests against this adapter.
    "C0",
    "C1",
    "C2",
})

FILLER_TOPIC_TERMS = frozenset({
    "understanding",
    "overview",
    "strategies",
    "strategy",
    "basics",
    "basic",
    "general",
})

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class AblationConfig:
    fsp_enabled: bool
    ite_enabled: bool
    idle_window_mode: bool = False
    max_idle_ticks_per_window: int = 4
    max_direct_pushes_per_idle_window: int = 2
    min_idle_tick_interval_seconds: float = 3.0
    max_intents_per_idle_window: int = 6
    stop_on_consecutive_no_new_facts: int = 2
    cold_start_turns: int = 2
    cold_start_max_intents: int = 1
    min_anchor_tokens_for_fsp: int = 6
    cross_turn_duplicate_jaccard: float = 0.6
    max_total_searches: int = 999

    @classmethod
    def from_condition(cls, condition: str) -> AblationConfig:
        mapping = {
            "Full":             cls(fsp_enabled=True,  ite_enabled=True),
            "Full-single-idle": cls(fsp_enabled=True,  ite_enabled=True),
            "Full-idle-window": cls(
                fsp_enabled=True,
                ite_enabled=True,
                idle_window_mode=True,
            ),
            "Blind":            cls(fsp_enabled=False, ite_enabled=True),
            "Paralyzed":        cls(fsp_enabled=True,  ite_enabled=False),
            "Baseline":         cls(fsp_enabled=False, ite_enabled=False),
            "C0":               cls(fsp_enabled=False, ite_enabled=False),
            "C1":               cls(fsp_enabled=False, ite_enabled=False),
            "C2":               cls(fsp_enabled=False, ite_enabled=False),
        }
        if condition not in mapping:
            raise ValueError(
                f"Unknown condition {condition!r}. "
                f"Must be one of {sorted(VALID_CONDITIONS)}."
            )
        return mapping[condition]


def normalize_topic_key(topic: str) -> str:
    """Normalize topic text before exact-key and semantic queue dedup."""
    normalized = (topic or "").lower()
    normalized = normalized.replace("401(k)", "401k")
    normalized = normalized.replace("401 k", "401k")
    tokens = [
        token
        for token in TOKEN_RE.findall(normalized)
        if token not in FILLER_TOPIC_TERMS
    ]
    return " ".join(tokens)


def _make_dedup_key(topic: str) -> str:
    return hashlib.md5(normalize_topic_key(topic).encode()).hexdigest()


@dataclass
class ExplorationIntent:
    topic: str
    query: str
    confidence: float
    reason: str
    source: str           # "fsp_critic" | "fsp_predictor" | "blind_random"
    created_at_turn: int
    dedup_key: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.dedup_key:
            self.dedup_key = _make_dedup_key(self.topic)

    def to_dict(self) -> Dict:
        return asdict(self)


class DedupPriorityQueue:
    """Priority queue with exact and semantic topic deduplication."""

    MAX_SIZE = 20

    def __init__(self):
        self._items: Dict[str, ExplorationIntent] = {}
        self.drop_reasons: List[dict] = []

    def push(self, intent: ExplorationIntent) -> bool:
        key = intent.dedup_key
        existing = self._items.get(key)
        if existing is not None:
            if intent.confidence > existing.confidence:
                self._items[key] = intent
                self._record_drop(existing, intent, "queue_exact_duplicate")
                return True
            self._record_drop(intent, existing, "queue_exact_duplicate")
            return False

        semantic_key, semantic_existing = self._semantic_duplicate(intent)
        if semantic_existing is not None and semantic_key is not None:
            if intent.confidence > semantic_existing.confidence:
                del self._items[semantic_key]
                self._items[key] = intent
                self._record_drop(semantic_existing, intent, "queue_semantic_duplicate")
                return True
            self._record_drop(intent, semantic_existing, "queue_semantic_duplicate")
            return False

        if len(self._items) >= self.MAX_SIZE:
            worst_key = min(self._items, key=lambda k: self._items[k].confidence)
            if intent.confidence > self._items[worst_key].confidence:
                dropped = self._items[worst_key]
                del self._items[worst_key]
                self._record_drop(dropped, intent, "queue_capacity_replace")
            else:
                self._record_drop(intent, self._items[worst_key], "queue_capacity_drop")
                return False
        self._items[key] = intent
        return True

    def push_all(self, intents: List[ExplorationIntent]) -> int:
        accepted = 0
        for intent in intents:
            if self.push(intent):
                accepted += 1
        return accepted

    def pull(self, max_items: int = 3) -> List[ExplorationIntent]:
        sorted_items = sorted(
            self._items.values(),
            key=lambda i: i.confidence,
            reverse=True,
        )
        pulled = sorted_items[:max_items]
        for item in pulled:
            del self._items[item.dedup_key]
        return pulled

    def __len__(self) -> int:
        return len(self._items)

    @property
    def size(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def clear_drop_reasons(self) -> None:
        self.drop_reasons.clear()

    def _semantic_duplicate(
        self,
        intent: ExplorationIntent,
    ) -> tuple[Optional[str], Optional[ExplorationIntent]]:
        intent_tokens = _semantic_tokens(intent.topic)
        if not intent_tokens:
            return None, None
        for key, existing in self._items.items():
            existing_tokens = _semantic_tokens(existing.topic)
            if _topics_semantically_overlap(intent_tokens, existing_tokens):
                return key, existing
        return None, None

    def _record_drop(
        self,
        dropped: ExplorationIntent,
        kept: ExplorationIntent,
        reason: str,
    ) -> None:
        self.drop_reasons.append({
            "topic": dropped.topic,
            "query": dropped.query,
            "source": dropped.source,
            "reason": reason,
            "kept_topic": kept.topic,
            "kept_confidence": kept.confidence,
            "dropped_confidence": dropped.confidence,
        })


def _semantic_tokens(topic: str) -> set[str]:
    return set(normalize_topic_key(topic).split())


def _topics_semantically_overlap(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    union = left | right
    if union and len(left & right) / len(union) >= 0.6:
        return True
    digit_anchors = {
        token for token in left & right
        if any(ch.isdigit() for ch in token)
    }
    if digit_anchors and (len(left) <= 3 or len(right) <= 3):
        return True
    return False
