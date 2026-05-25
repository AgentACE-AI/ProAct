"""
Benchmark FSP (Forecasted Search Planning) module.

Wraps three production agents — MemoryCriticAgent, ResearchTopicPredictor,
ResearchValueEvaluator — to generate exploration intents that the ITE module
can act on.

Anti-leakage: this module NEVER imports or accesses user_needs, key_fact_ids,
predictable_after, or turn_order.
"""

from __future__ import annotations

import logging
import re
from typing import List, TYPE_CHECKING

from agents.memory_critic import MemoryCriticAgent
from agents.research_predictor import ResearchTopicPredictor
from agents.research_evaluator import ResearchValueEvaluator
from experiments.ProactiveBench.eval.ablation_config import (
    ExplorationIntent,
    normalize_topic_key,
)

if TYPE_CHECKING:
    from core.config import Config
    from memory.memory_system import MemorySystem
    from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)

EVALUATOR_THRESHOLD = 40
GENERIC_TOPIC_MARKERS = (
    "basics",
    "general",
    "overview",
    "best practices",
    "key entities",
    "steps in",
    "introduction",
)
STOPWORDS = {
    "about", "after", "and", "are", "because", "before", "being", "for",
    "from", "have", "how", "into", "just", "likely", "need", "needs",
    "not", "the", "their", "there", "this", "through", "user", "want",
    "wants", "with", "will", "would",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class BenchmarkFSP:

    def __init__(
        self,
        memory: "MemorySystem",
        llm_client: "LLMClient",
        config: "Config",
    ):
        self.memory = memory
        self.memory_critic = MemoryCriticAgent(
            config=config.agents.memory_critic,
            llm_client=llm_client,
        )
        self.research_predictor = ResearchTopicPredictor(
            config=config.agents.research_predictor,
            llm_client=llm_client,
        )
        self.research_evaluator = ResearchValueEvaluator(
            config=config.agents.research_evaluator,
            llm_client=llm_client,
        )
        self.last_filter_reasons: List[dict] = []
        self.last_generated_count = 0
        self.last_accepted_count = 0
        self.last_filtered_count = 0
        self.last_duplicate_count = 0
        self.cold_start_turns = 2
        self.cold_start_max_intents = 1
        self.min_anchor_tokens_for_fsp = 6
        self.cross_turn_duplicate_jaccard = 0.6
        self._fingerprint_history: List[set[str]] = []

    def generate_intents(
        self,
        conversation_history: str,
        user_profile_text: str,
        current_turn: int,
        queue_size: int = 0,
        max_queue_size: int = 20,
    ) -> List[ExplorationIntent]:
        self._reset_trace_counts()
        if queue_size >= max_queue_size:
            self.last_filter_reasons.append({
                "reason": "queue_saturated",
                "queue_size": queue_size,
                "max_queue_size": max_queue_size,
            })
            self.last_filtered_count = 1
            return []

        if self._cold_start_unanchored(current_turn, conversation_history):
            self.last_filter_reasons.append({
                "reason": "cold_start_unanchored_turn",
                "turn": current_turn,
                "min_anchor_tokens_for_fsp": self._min_anchor_tokens(),
            })
            self.last_filtered_count = 1
            return []

        candidates: List[ExplorationIntent] = []

        candidates.extend(self._critic_intents(current_turn))
        candidates.extend(
            self._predictor_intents(conversation_history, user_profile_text, current_turn)
        )
        self.last_generated_count = len(candidates)

        if not candidates:
            return []

        candidates = self._prefilter_candidates(
            candidates,
            conversation_history=conversation_history,
            user_profile_text=user_profile_text,
        )
        if not candidates:
            self._update_trace_counts(accepted=0)
            return []

        accepted = self._evaluate_and_filter(candidates, user_profile_text)
        accepted = self._apply_cold_start_cap(accepted, current_turn)
        self._remember_fingerprints(accepted)
        self._update_trace_counts(accepted=len(accepted))
        return accepted

    def _prefilter_candidates(
        self,
        candidates: List[ExplorationIntent],
        conversation_history: str,
        user_profile_text: str,
    ) -> List[ExplorationIntent]:
        anchors = self._tokens(f"{conversation_history}\n{user_profile_text}")
        filtered: List[ExplorationIntent] = []
        seen_fingerprints: List[set[str]] = []

        for intent in candidates:
            topic_text = (
                f"{normalize_topic_key(intent.topic)} "
                f"{normalize_topic_key(intent.query)} {intent.reason}"
            )
            topic_tokens = self._tokens(topic_text)
            topic_query_tokens = self._tokens(
                f"{normalize_topic_key(intent.topic)} {normalize_topic_key(intent.query)}"
            )
            topic = intent.topic.lower()

            if self._is_generic_topic(topic, topic_query_tokens) and not (topic_tokens & anchors):
                self._record_filter(intent, "generic_topic_penalty")
                continue

            fingerprint = self._fingerprint(intent)
            if self._is_duplicate_fingerprint(fingerprint, seen_fingerprints):
                self._record_filter(intent, "fsp_semantic_duplicate")
                continue

            if self._is_cross_turn_duplicate(fingerprint):
                self._record_filter(intent, "fsp_cross_turn_duplicate")
                continue

            seen_fingerprints.append(fingerprint)
            filtered.append(intent)

        return filtered

    @staticmethod
    def _is_generic_topic(topic: str, topic_tokens: set[str]) -> bool:
        if "key entities" in topic or "steps in" in topic:
            return True
        if "basics" in topic and len(topic_tokens) <= 3:
            return True
        if "general" in topic or "overview" in topic or "introduction" in topic:
            return len(topic_tokens) <= 4
        return any(marker in topic for marker in GENERIC_TOPIC_MARKERS) and len(topic_tokens) <= 3

    @staticmethod
    def _is_duplicate_fingerprint(
        fingerprint: set[str],
        seen_fingerprints: List[set[str]],
    ) -> bool:
        if not fingerprint:
            return False
        for seen in seen_fingerprints:
            if BenchmarkFSP._fingerprints_overlap(fingerprint, seen, threshold=0.8):
                return True
        return False

    def _record_filter(self, intent: ExplorationIntent, reason: str) -> None:
        self.last_filter_reasons.append({
            "topic": intent.topic,
            "query": intent.query,
            "source": intent.source,
            "reason": reason,
        })

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in TOKEN_RE.findall((text or "").lower())
            if len(token) > 1 and token not in STOPWORDS
        }

    @classmethod
    def _fingerprint(cls, intent: ExplorationIntent) -> set[str]:
        text = f"{normalize_topic_key(intent.topic)} {normalize_topic_key(intent.query)}"
        tokens = cls._tokens(text)
        return {
            token for token in tokens
            if len(token) >= 4 or any(ch.isdigit() for ch in token)
        }

    def _is_cross_turn_duplicate(self, fingerprint: set[str]) -> bool:
        if not fingerprint:
            return False
        threshold = self._cross_turn_duplicate_jaccard()
        for seen in getattr(self, "_fingerprint_history", []):
            if self._fingerprints_overlap(fingerprint, seen, threshold=threshold):
                return True
        return False

    @staticmethod
    def _fingerprints_overlap(
        left: set[str],
        right: set[str],
        threshold: float,
    ) -> bool:
        union = left | right
        if not union:
            return False
        intersection = left & right
        if len(intersection) / len(union) >= threshold:
            return True
        smaller = min(len(left), len(right))
        if smaller and len(intersection) / smaller >= threshold:
            return True
        return bool(
            intersection
            and any(any(ch.isdigit() for ch in token) for token in intersection)
            and len(intersection) >= 2
        )

    def _remember_fingerprints(self, intents: List[ExplorationIntent]) -> None:
        history = list(getattr(self, "_fingerprint_history", []))
        for intent in intents:
            fingerprint = self._fingerprint(intent)
            if fingerprint:
                history.append(fingerprint)
        self._fingerprint_history = history[-100:]

    def _cold_start_unanchored(
        self,
        current_turn: int,
        conversation_history: str,
    ) -> bool:
        return (
            current_turn == 1
            and len(self._tokens(conversation_history)) < self._min_anchor_tokens()
        )

    def _apply_cold_start_cap(
        self,
        intents: List[ExplorationIntent],
        current_turn: int,
    ) -> List[ExplorationIntent]:
        if current_turn > self._cold_start_turns():
            return intents
        cap = self._cold_start_max_intents()
        if cap < 0 or len(intents) <= cap:
            return intents
        for dropped in intents[cap:]:
            self._record_filter(dropped, "cold_start_cap")
        return intents[:cap]

    def _reset_trace_counts(self) -> None:
        self.last_filter_reasons = []
        self.last_generated_count = 0
        self.last_accepted_count = 0
        self.last_filtered_count = 0
        self.last_duplicate_count = 0
        if not hasattr(self, "_fingerprint_history"):
            self._fingerprint_history = []

    def _update_trace_counts(self, accepted: int) -> None:
        self.last_accepted_count = accepted
        self.last_filtered_count = len(getattr(self, "last_filter_reasons", []))
        self.last_duplicate_count = sum(
            1
            for item in getattr(self, "last_filter_reasons", [])
            if "duplicate" in str(item.get("reason", ""))
        )

    def _cold_start_turns(self) -> int:
        return int(getattr(self, "cold_start_turns", 2))

    def _cold_start_max_intents(self) -> int:
        return int(getattr(self, "cold_start_max_intents", 1))

    def _min_anchor_tokens(self) -> int:
        return int(getattr(self, "min_anchor_tokens_for_fsp", 6))

    def _cross_turn_duplicate_jaccard(self) -> float:
        return float(getattr(self, "cross_turn_duplicate_jaccard", 0.6))

    def _critic_intents(self, current_turn: int) -> List[ExplorationIntent]:
        topic = self.memory.current_topic or "general"
        memory_context = self.memory.format_context_for_query("")
        if not memory_context.strip():
            memory_context = "No knowledge stored yet."

        try:
            result = self.memory_critic.run(topic=topic, memory_content=memory_context)
        except Exception:
            logger.warning("FSP: memory_critic.run failed", exc_info=True)
            return []

        intents = []
        for gap in result.get("knowledge_gaps", []):
            query = gap.get("search_query", "")
            desc = gap.get("description", "")
            if not query:
                continue
            intents.append(ExplorationIntent(
                topic=query,
                query=query,
                confidence=0.6,
                reason=desc,
                source="fsp_critic",
                created_at_turn=current_turn,
            ))
        return intents[:3]

    def _predictor_intents(
        self,
        conversation_history: str,
        user_profile_text: str,
        current_turn: int,
    ) -> List[ExplorationIntent]:
        recent_topics = self.memory.get_recent_topics(limit=3)
        profile = self.memory.get_user_profile()
        research_history = getattr(profile, "research_history", None) or []

        try:
            result = self.research_predictor.run(
                user_profile_text=user_profile_text,
                recent_interactions=conversation_history,
                recent_topics=recent_topics,
                research_history=research_history,
            )
        except Exception:
            logger.warning("FSP: research_predictor.run failed", exc_info=True)
            return []

        intents = []
        for pred in result.get("predictions", []):
            topic = pred.get("topic", "")
            if not topic:
                continue
            intents.append(ExplorationIntent(
                topic=topic,
                query=pred.get("trigger", topic),
                confidence=pred.get("confidence", 0.5),
                reason=pred.get("reason", ""),
                source="fsp_predictor",
                created_at_turn=current_turn,
            ))
        return intents

    def _evaluate_and_filter(
        self,
        candidates: List[ExplorationIntent],
        user_profile_text: str,
    ) -> List[ExplorationIntent]:
        profile = self.memory.get_user_profile()
        research_history = getattr(profile, "research_history", None) or []
        filtered = []

        for intent in candidates:
            existing = self._get_existing_knowledge(intent.topic)
            try:
                result = self.research_evaluator.evaluate(
                    candidate_topic={
                        "topic": intent.topic,
                        "source": intent.source,
                        "reason": intent.reason,
                        "confidence": intent.confidence,
                    },
                    user_profile_text=user_profile_text,
                    existing_knowledge=existing,
                    research_history=research_history,
                )
            except Exception:
                logger.warning("FSP: evaluator failed for %s", intent.topic, exc_info=True)
                continue

            if result.get("should_research") and result.get("score", 0) >= EVALUATOR_THRESHOLD:
                intent.confidence = result["score"] / 100.0
                filtered.append(intent)

        return filtered

    def _get_existing_knowledge(self, topic: str) -> str:
        try:
            items = self.memory.search_knowledge(topic, n_results=3)
            if not items:
                return ""
            parts = []
            for item in items:
                content = getattr(item, "content", None) or str(item)
                parts.append(content[:200])
            return "\n".join(parts)
        except Exception:
            return ""
