"""
Benchmark ITE (Idle-Time Exploration) module.

Executes exploration intents by searching the ExternalFactStore via
SearchService, storing discovered facts into system MemorySystem, and
evaluating push worthiness via PushScoreAgent.

Also provides BlindExplorationFallback for the Blind condition: generates
random domain-agnostic intents without any conversation or fact sheet context.

Anti-leakage: this module NEVER imports or accesses user_needs, key_fact_ids,
predictable_after, or turn_order.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, TYPE_CHECKING

from experiments.ProactiveBench.eval.ablation_config import (
    DedupPriorityQueue,
    ExplorationIntent,
)

if TYPE_CHECKING:
    from agents.push_score import PushScoreAgent
    from memory.memory_system import MemorySystem
    from services.search_service import SearchService

logger = logging.getLogger(__name__)

PUSH_THRESHOLD = 70
INLINE_THRESHOLD = 40
RECENT_PUSH_HISTORY_LIMIT = 5
REACTIVE_OVERLAP_THRESHOLD = 0.6
MIN_DIRECT_NOVELTY_RATIO = 0.35
SOURCE_SCORE_MULTIPLIERS = {
    "fsp_predictor": 1.0,
    "fsp_critic": 0.9,
    "blind_random": 0.65,
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "about", "after", "and", "are", "because", "before", "being", "for",
    "from", "have", "how", "into", "just", "likely", "need", "needs",
    "not", "the", "their", "there", "this", "through", "user", "want",
    "wants", "with", "will", "would", "you", "your", "can",
}


class BenchmarkITE:

    def __init__(
        self,
        search_service: "SearchService",
        push_score_agent: "PushScoreAgent",
        memory: "MemorySystem",
        cost_tracker: Any = None,
    ):
        self.search_service = search_service
        self.push_score_agent = push_score_agent
        self.memory = memory
        self.cost_tracker = cost_tracker
        self._delivered_fact_ids: set[str] = set()
        self._delivered_source_urls: set[str] = set()
        self._recent_direct_push_fact_sets: List[set[str]] = []

    def explore(
        self,
        intent_queue: DedupPriorityQueue,
        max_intents: int = 3,
        seconds_since_last_interaction: int = 0,
        recent_interaction_count: int = 0,
    ) -> Dict[str, Any]:
        intents = intent_queue.pull(max_items=max_intents)
        if not intents:
            return {
                "has_direct_push": False,
                "push_content": None,
                "push_topic": None,
                "push_fact_ids": [],
                "facts_stored": 0,
                "intents_processed": 0,
                "inline_topics": [],
                "stored_fact_ids": [],
                "intent_results": [],
                "push_novelty_ratio": None,
                "push_undelivered_fact_ids": [],
                "push_delivered_fact_ids": [],
                "push_events": [],
            }

        total_facts_stored = 0
        best_push_score = -1
        best_push_content = None
        best_push_topic = None
        best_push_fact_ids: List[str] = []
        best_push_sources: List[Any] = []
        best_push_novelty_ratio: float | None = None
        best_push_undelivered_fact_ids: List[str] = []
        best_push_delivered_fact_ids: List[str] = []
        best_push_index: int | None = None
        inline_topics: List[str] = []
        stored_fact_ids: List[str] = []
        intent_results: List[Dict[str, Any]] = []

        for intent in intents:
            facts_stored, search_summary, fact_ids, sources = self._explore_intent(intent)
            total_facts_stored += facts_stored
            stored_fact_ids.extend(fact_ids)
            novelty = self._novelty_details(fact_ids, sources)
            scoring_summary = (
                self.search_service.format_sources_for_context(novelty["marginal_sources"])
                if novelty["marginal_sources"]
                else search_summary
            )

            intent_result = {
                "topic": intent.topic,
                "query": intent.query,
                "source": intent.source,
                "confidence": intent.confidence,
                "fact_ids": fact_ids,
                "facts_stored": facts_stored,
                "undelivered_fact_ids": novelty["undelivered_fact_ids"],
                "delivered_fact_ids": novelty["delivered_fact_ids"],
                "novelty_ratio": novelty["novelty_ratio"],
                "marginal_sources": [
                    self._source_to_dict(source)
                    for source in novelty["marginal_sources"]
                ],
                "push_score": None,
                "decision": "no_results",
            }

            if fact_ids and search_summary:
                push_eval = self._evaluate_push(
                    intent,
                    scoring_summary,
                    facts_stored=facts_stored,
                    seconds_since_last_interaction=seconds_since_last_interaction,
                    recent_interaction_count=recent_interaction_count,
                )
                score = push_eval["score"]
                intent_result["push_score"] = score
                intent_result["push_value"] = push_eval["value"]
                intent_result["push_cost"] = push_eval["cost"]
                intent_result["push_raw_score"] = push_eval["raw_score"]
                intent_result["push_reason"] = push_eval["reason"]
                direct_gate_reasons = self._direct_gate_reasons(
                    fact_ids=fact_ids,
                    facts_stored=facts_stored,
                    sources=sources,
                    undelivered_fact_ids=novelty["undelivered_fact_ids"],
                    novelty_ratio=novelty["novelty_ratio"],
                )
                intent_result["direct_gate_reasons"] = direct_gate_reasons
                if score >= PUSH_THRESHOLD:
                    if direct_gate_reasons:
                        inline_topics.append(intent.topic)
                        intent_result["decision"] = "inline"
                    else:
                        intent_result["decision"] = "direct_candidate"
                        if score > best_push_score:
                            best_push_score = score
                            best_push_content = self._format_push_content(
                                intent.topic, novelty["marginal_sources"] or sources,
                            )
                            best_push_topic = intent.topic
                            best_push_fact_ids = fact_ids
                            best_push_sources = novelty["marginal_sources"] or sources
                            best_push_novelty_ratio = novelty["novelty_ratio"]
                            best_push_undelivered_fact_ids = novelty["undelivered_fact_ids"]
                            best_push_delivered_fact_ids = novelty["delivered_fact_ids"]
                            best_push_index = len(intent_results)
                elif score >= INLINE_THRESHOLD:
                    inline_topics.append(intent.topic)
                    intent_result["decision"] = "inline"
                else:
                    intent_result["decision"] = "drop"

            intent_results.append(intent_result)

        if best_push_index is not None:
            intent_results[best_push_index]["decision"] = "direct"
            self._record_direct_push(best_push_fact_ids, best_push_sources)

        push_events = []
        if best_push_content is not None:
            push_events.append({
                "push_content": best_push_content,
                "push_topic": best_push_topic,
                "push_fact_ids": sorted(set(best_push_fact_ids)),
                "push_novelty_ratio": best_push_novelty_ratio,
                "push_undelivered_fact_ids": best_push_undelivered_fact_ids,
                "push_delivered_fact_ids": best_push_delivered_fact_ids,
                "facts_stored": total_facts_stored,
                "stored_fact_ids": sorted(set(stored_fact_ids)),
                "intent_results": intent_results,
            })

        return {
            "has_direct_push": best_push_content is not None,
            "push_content": best_push_content,
            "push_topic": best_push_topic,
            "push_fact_ids": sorted(set(best_push_fact_ids)),
            "push_novelty_ratio": best_push_novelty_ratio,
            "push_undelivered_fact_ids": best_push_undelivered_fact_ids,
            "push_delivered_fact_ids": best_push_delivered_fact_ids,
            "push_events": push_events,
            "facts_stored": total_facts_stored,
            "intents_processed": len(intents),
            "inline_topics": inline_topics,
            "stored_fact_ids": sorted(set(stored_fact_ids)),
            "intent_results": intent_results,
        }

    def _explore_intent(self, intent: ExplorationIntent) -> tuple[int, str, List[str], list]:
        try:
            if self.cost_tracker is not None:
                with self.cost_tracker.track("ite_search"):
                    result = self.search_service.search(
                        topic=intent.topic,
                        purpose=intent.reason or f"Research about {intent.topic}",
                        generate_report=False,
                    )
            else:
                result = self.search_service.search(
                    topic=intent.topic,
                    purpose=intent.reason or f"Research about {intent.topic}",
                    generate_report=False,
                )
        except Exception:
            logger.warning("ITE: search failed for %s", intent.topic, exc_info=True)
            return 0, "", [], []

        if not result.success:
            return 0, "", [], []

        facts_added = result.knowledge_stats.get("added", 0)
        summary = self.search_service.format_sources_for_context(result.sources)
        fact_ids = self._fact_ids_from_sources(result.sources)
        return facts_added, summary, fact_ids, result.sources

    @staticmethod
    def _fact_ids_from_sources(sources) -> List[str]:
        fact_ids: List[str] = []
        for source in sources:
            url = getattr(source, "url", "")
            if url.startswith("extfact://"):
                fact_id = url[len("extfact://"):].strip()
                if fact_id:
                    fact_ids.append(fact_id)
        return sorted(set(fact_ids))

    def _evaluate_push(
        self,
        intent: ExplorationIntent,
        summary: str,
        facts_stored: int = 0,
        seconds_since_last_interaction: int = 0,
        recent_interaction_count: int = 0,
    ) -> Dict[str, Any]:
        profile = self.memory.get_user_profile()
        interests = getattr(profile, "interests", None) or []
        current_context = self._current_context()
        scoring_summary = self._build_scoring_summary(intent, summary)
        try:
            if self.cost_tracker is not None:
                with self.cost_tracker.track("ite_push_score"):
                    result = self.push_score_agent.run(
                        report_topic=intent.topic,
                        report_summary=scoring_summary,
                        user_interests=interests,
                        current_context=current_context,
                        seconds_since_last_interaction=max(0, int(seconds_since_last_interaction)),
                        recent_interaction_count=max(0, int(recent_interaction_count)),
                    )
            else:
                result = self.push_score_agent.run(
                    report_topic=intent.topic,
                    report_summary=scoring_summary,
                    user_interests=interests,
                    current_context=current_context,
                    seconds_since_last_interaction=max(0, int(seconds_since_last_interaction)),
                    recent_interaction_count=max(0, int(recent_interaction_count)),
                )
        except Exception:
            logger.warning("ITE: push_score_agent failed", exc_info=True)
            return {
                "score": 0,
                "value": 0,
                "cost": 100,
                "raw_score": 0,
                "reason": "push_score_agent_failed",
            }

        value = self._coerce_score(result.get("value", 0), default=0)
        cost = self._coerce_score(result.get("cost", 100), default=100)
        raw_score = max(0, min(100, value - cost + 50))
        score = self._calibrate_push_score(raw_score, intent, facts_stored)
        return {
            "score": score,
            "value": value,
            "cost": cost,
            "raw_score": raw_score,
            "reason": result.get("reason", ""),
        }

    @staticmethod
    def _coerce_score(value: Any, default: int) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float(default)
        return int(max(0, min(100, round(numeric))))

    @staticmethod
    def _build_scoring_summary(intent: ExplorationIntent, summary: str) -> str:
        return (
            f"Exploration source: {intent.source}\n"
            f"Intent confidence: {intent.confidence:.2f}\n"
            f"Intent reason: {intent.reason}\n"
            "Benchmark note: blind_random intents are not predicted from the "
            "conversation and should be treated as weak unless the retrieved "
            "facts clearly match the current user context.\n\n"
            f"Retrieved facts:\n{summary[:700]}"
        )

    def _current_context(self) -> str:
        try:
            return self.memory.get_current_messages_formatted()[:700]
        except Exception:
            return ""

    @staticmethod
    def _calibrate_push_score(
        raw_score: int,
        intent: ExplorationIntent,
        facts_stored: int = 0,
    ) -> int:
        try:
            confidence = float(intent.confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        confidence_factor = 0.4 + (0.6 * confidence)
        source_factor = SOURCE_SCORE_MULTIPLIERS.get(intent.source, 0.8)
        score = int(round(raw_score * confidence_factor * source_factor))

        if facts_stored <= 0:
            score = int(round(score * 0.65))

        if intent.source == "blind_random" or confidence < 0.6:
            score = min(score, PUSH_THRESHOLD - 1)
        return max(0, min(100, score))

    def _direct_gate_reasons(
        self,
        fact_ids: List[str],
        facts_stored: int,
        sources: list,
        undelivered_fact_ids: List[str] | None = None,
        novelty_ratio: float | None = None,
    ) -> List[str]:
        reasons: List[str] = []
        fact_set = {
            str(fact_id).strip().upper()
            for fact_id in fact_ids
            if str(fact_id).strip()
        }

        if facts_stored <= 0:
            reasons.append("no_new_facts")
        if fact_set and not (undelivered_fact_ids or list(fact_set - self._delivered_fact_ids)):
            reasons.append("no_undelivered_facts")
        if (
            novelty_ratio is not None
            and fact_set
            and novelty_ratio < MIN_DIRECT_NOVELTY_RATIO
        ):
            reasons.append("low_novelty_ratio")
        if self._overlaps_recent_direct_push(fact_set):
            reasons.append("recent_push_overlap")
        if self._sources_recently_conveyed(sources):
            reasons.append("recent_reactive_overlap")
        return reasons

    def _overlaps_recent_direct_push(self, fact_set: set[str]) -> bool:
        if not fact_set:
            return False
        for recent in self._recent_direct_push_fact_sets:
            union = fact_set | recent
            if not union:
                continue
            if len(fact_set & recent) / len(union) >= 0.6:
                return True
        return False

    def _sources_recently_conveyed(self, sources: list) -> bool:
        assistant_context = "\n".join(
            line for line in self._current_context().splitlines()
            if line.lower().startswith("assistant:")
        )
        context_tokens = self._tokens(assistant_context)
        if not context_tokens:
            return False
        for source in sources:
            text = getattr(source, "snippet", "") or getattr(source, "full_content", "")
            source_tokens = self._tokens(text)
            if not source_tokens:
                continue
            if len(source_tokens & context_tokens) / len(source_tokens) >= REACTIVE_OVERLAP_THRESHOLD:
                return True
        return False

    def _record_direct_push(self, fact_ids: List[str], sources: list | None = None) -> None:
        fact_set = {
            str(fact_id).strip().upper()
            for fact_id in fact_ids
            if str(fact_id).strip()
        }
        source_urls = {
            str(getattr(source, "url", "")).strip()
            for source in (sources or [])
            if str(getattr(source, "url", "")).strip()
        }
        if not fact_set and not source_urls:
            return
        self._delivered_fact_ids.update(fact_set)
        self._delivered_source_urls.update(source_urls)
        self._recent_direct_push_fact_sets.append(fact_set)
        if len(self._recent_direct_push_fact_sets) > RECENT_PUSH_HISTORY_LIMIT:
            self._recent_direct_push_fact_sets = self._recent_direct_push_fact_sets[
                -RECENT_PUSH_HISTORY_LIMIT:
            ]

    def _novelty_details(self, fact_ids: List[str], sources: list) -> Dict[str, Any]:
        fact_to_source: Dict[str, list] = {}
        delivered_fact_ids: List[str] = []
        undelivered_fact_ids: List[str] = []
        marginal_sources: List[Any] = []

        for source in sources:
            fact_id = self._fact_id_from_source(source)
            if fact_id:
                fact_to_source.setdefault(fact_id, []).append(source)

        canonical_ids = [
            str(fact_id).strip().upper()
            for fact_id in fact_ids
            if str(fact_id).strip()
        ]
        for fact_id in sorted(set(canonical_ids)):
            if fact_id in self._delivered_fact_ids:
                delivered_fact_ids.append(fact_id)
            else:
                undelivered_fact_ids.append(fact_id)
                marginal_sources.extend(fact_to_source.get(fact_id, []))

        for source in sources:
            if self._fact_id_from_source(source):
                continue
            url = str(getattr(source, "url", "")).strip()
            if url and url not in self._delivered_source_urls:
                marginal_sources.append(source)

        deduped_sources: List[Any] = []
        seen_urls: set[str] = set()
        for source in marginal_sources:
            url = str(getattr(source, "url", "")).strip()
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped_sources.append(source)

        novelty_ratio = (
            len(undelivered_fact_ids) / len(set(canonical_ids))
            if canonical_ids
            else 0.0
        )
        return {
            "undelivered_fact_ids": undelivered_fact_ids,
            "delivered_fact_ids": delivered_fact_ids,
            "novelty_ratio": novelty_ratio,
            "marginal_sources": deduped_sources,
        }

    @staticmethod
    def _fact_id_from_source(source: Any) -> str:
        url = str(getattr(source, "url", "")).strip()
        if not url.lower().startswith("extfact://"):
            return ""
        return url[len("extfact://"):].strip().upper()

    @staticmethod
    def _source_to_dict(source: Any) -> Dict[str, Any]:
        return {
            "url": getattr(source, "url", ""),
            "title": getattr(source, "title", ""),
            "snippet": getattr(source, "snippet", ""),
        }

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in TOKEN_RE.findall((text or "").lower())
            if len(token) > 1 and token not in STOPWORDS
        }

    @staticmethod
    def _format_push_content(topic: str, sources: list) -> str:
        facts = []
        for src in sources:
            text = getattr(src, "snippet", "") or getattr(src, "full_content", "")
            if text and text not in facts:
                facts.append(text)
        if not facts:
            return f"I found some information regarding {topic}, but details are limited."
        bullet_list = "\n".join(f"- {fact}" for fact in facts)
        return (
            f"I found some information that might be helpful regarding {topic}:\n\n"
            f"{bullet_list}"
        )


class BlindExplorationFallback:
    """Generates random domain-agnostic intents for the Blind condition."""

    GENERIC_TOPICS: List[str] = [
        "local regulations", "timeline requirements", "cost breakdown",
        "documentation needed", "eligibility criteria", "comparison options",
        "deadline information", "safety guidelines", "registration process",
        "contact information", "operating hours", "service availability",
        "pricing structure", "warranty coverage", "maintenance schedule",
        "insurance requirements", "legal considerations", "tax implications",
        "environmental impact", "accessibility features", "quality standards",
        "certification requirements", "training programs", "support resources",
        "emergency procedures", "complaint process", "refund policy",
        "upgrade options", "compatibility requirements", "performance benchmarks",
        "industry standards", "best practices", "common issues",
        "troubleshooting steps", "user reviews", "expert recommendations",
        "alternative solutions", "integration options", "customization features",
        "data requirements", "privacy considerations", "security measures",
        "backup procedures", "recovery options", "monitoring tools",
        "reporting capabilities", "audit requirements", "compliance checklist",
        "stakeholder roles", "communication protocols", "feedback mechanisms",
    ]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._used_indices: set[int] = set()

    def generate_random_intents(
        self,
        count: int = 3,
        current_turn: int = 0,
    ) -> List[ExplorationIntent]:
        available = [
            i for i in range(len(self.GENERIC_TOPICS))
            if i not in self._used_indices
        ]
        if len(available) < count:
            self._used_indices.clear()
            available = list(range(len(self.GENERIC_TOPICS)))

        selected = self._rng.sample(available, min(count, len(available)))
        self._used_indices.update(selected)

        intents = []
        for idx in selected:
            topic = self.GENERIC_TOPICS[idx]
            intents.append(ExplorationIntent(
                topic=topic,
                query=topic,
                confidence=0.5,
                reason="blind random exploration",
                source="blind_random",
                created_at_turn=current_turn,
            ))
        return intents
