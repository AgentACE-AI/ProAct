"""
Benchmark-only fact relevance scoring for ProactiveBench Route C.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from experiments.ProactiveBench.generation.models import Fact

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "with",
}


@dataclass(frozen=True)
class ScoredFact:
    fact_id: str
    fact_text: str
    category: str
    score: float
    reason: str
    current_turn_relevance: float = 0.0
    next_step_utility: float = 0.0
    actionability: float = 0.0
    novelty: float = 1.0
    alignment_score: float = 0.0
    excluded_as_current_turn_answer: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_text": self.fact_text,
            "category": self.category,
            "score": self.score,
            "reason": self.reason,
            "current_turn_relevance": self.current_turn_relevance,
            "next_step_utility": self.next_step_utility,
            "actionability": self.actionability,
            "novelty": self.novelty,
            "alignment_score": self.alignment_score,
            "excluded_as_current_turn_answer": self.excluded_as_current_turn_answer,
        }


class BenchmarkFactScorer:
    """
    Closed-world fact scorer for Route C.

    The scorer only consumes:
    - fact_sheet
    - current user turn
    - conversation history
    - user profile text
    - memory context

    It never receives benchmark gold labels.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model

    def score_facts(
        self,
        fact_sheet: Sequence[Fact],
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
        predicted_topic: str = "",
        predicted_need: str = "",
        recently_selected_ids: Optional[set[str]] = None,
        conveyed_fact_ids: Optional[set[str]] = None,
    ) -> List[ScoredFact]:
        fact_list = list(fact_sheet)
        if not fact_list:
            return []

        fallback = self._fallback_scores(
            fact_list=fact_list,
            current_message=current_message,
            conversation_history=conversation_history,
            user_profile=user_profile,
            memory_context=memory_context,
            predicted_topic=predicted_topic,
            predicted_need=predicted_need,
            recently_selected_ids=recently_selected_ids or set(),
            conveyed_fact_ids=conveyed_fact_ids or set(),
        )
        llm_scores = self._llm_scores(
            fact_list=fact_list,
            current_message=current_message,
            conversation_history=conversation_history,
            user_profile=user_profile,
            memory_context=memory_context,
            predicted_topic=predicted_topic,
            predicted_need=predicted_need,
        )

        combined: List[ScoredFact] = []
        for fact in fact_list:
            fact_id = fact.id
            llm_override = llm_scores.get(fact_id, {})
            scored = dict(fallback[fact_id])
            scored.update(llm_override)
            current_turn_relevance = self._clamp_score(
                scored.get("current_turn_relevance", fallback[fact_id]["current_turn_relevance"])
            )
            next_step_utility = self._clamp_score(
                scored.get("next_step_utility", fallback[fact_id]["next_step_utility"])
            )
            actionability = self._clamp_score(
                scored.get("actionability", fallback[fact_id]["actionability"])
            )
            novelty = self._clamp_score(
                scored.get("novelty", fallback[fact_id]["novelty"])
            )
            alignment_score = self._clamp_score(
                scored.get("alignment_score", fallback[fact_id]["alignment_score"])
            )
            excluded = bool(
                llm_override.get(
                    "excluded_as_current_turn_answer",
                    self._should_exclude_as_current_turn_answer(
                        current_turn_relevance=current_turn_relevance,
                        next_step_utility=next_step_utility,
                    ),
                )
            )
            selection_score = self._selection_score(
                current_turn_relevance=current_turn_relevance,
                next_step_utility=next_step_utility,
                novelty=novelty,
                alignment_score=alignment_score,
            )
            combined.append(
                ScoredFact(
                    fact_id=fact_id,
                    fact_text=fact.fact,
                    category=fact.category,
                    score=selection_score,
                    reason=scored["reason"].strip() or fallback[fact_id]["reason"],
                    current_turn_relevance=current_turn_relevance,
                    next_step_utility=next_step_utility,
                    actionability=actionability,
                    novelty=novelty,
                    alignment_score=alignment_score,
                    excluded_as_current_turn_answer=excluded,
                )
            )

        return sorted(combined, key=lambda item: (-item.score, item.fact_id))

    def _llm_scores(
        self,
        fact_list: Sequence[Fact],
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
        predicted_topic: str,
        predicted_need: str,
    ) -> Dict[str, Dict[str, Any]]:
        if self.llm_client is None:
            return {}

        try:
            payload = self.llm_client.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You score fact relevance for a closed-world proactive benchmark. "
                            "Use only the provided fact sheet, current turn, conversation history, "
                            "user profile, memory context, and predicted next-step topic/need. "
                            "Do not infer from hidden labels or "
                            "future user needs. Assume the assistant will already answer the user's "
                            "current question competently in the main reply. Return JSON with a "
                            "'scored_facts' list containing fact_id, alignment_score, "
                            "current_turn_relevance, next_step_utility, and a short reason."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_prompt(
                            fact_list=fact_list,
                            current_message=current_message,
                            conversation_history=conversation_history,
                            user_profile=user_profile,
                            memory_context=memory_context,
                            predicted_topic=predicted_topic,
                            predicted_need=predicted_need,
                        ),
                    },
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=2000,
            )
        except Exception as exc:
            logger.warning("BenchmarkFactScorer fell back to deterministic scoring: %s", exc)
            return {}

        normalized: Dict[str, Dict[str, Any]] = {}
        valid_ids = {fact.id for fact in fact_list}
        for item in payload.get("scored_facts", []):
            fact_id = str(item.get("fact_id", "")).strip()
            if fact_id not in valid_ids or fact_id in normalized:
                continue
            try:
                next_step_utility = float(
                    item.get("next_step_utility", item.get("score", 0.0))
                )
            except (TypeError, ValueError):
                next_step_utility = 0.0
            try:
                alignment_score = float(
                    item.get("alignment_score", next_step_utility)
                )
            except (TypeError, ValueError):
                alignment_score = next_step_utility
            try:
                current_turn_relevance = float(
                    item.get("current_turn_relevance", 0.0)
                )
            except (TypeError, ValueError):
                current_turn_relevance = 0.0
            normalized[fact_id] = {
                "alignment_score": self._clamp_score(alignment_score),
                "next_step_utility": self._clamp_score(next_step_utility),
                "current_turn_relevance": self._clamp_score(current_turn_relevance),
                "reason": str(item.get("reason", "")).strip() or "LLM relevance score.",
            }
        return normalized

    def _fallback_scores(
        self,
        fact_list: Sequence[Fact],
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
        predicted_topic: str,
        predicted_need: str,
        recently_selected_ids: set[str],
        conveyed_fact_ids: Optional[set[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        current_tokens = self._tokenize(current_message)
        history_tokens = self._tokenize(conversation_history)
        profile_tokens = self._tokenize(user_profile)
        memory_tokens = self._tokenize(memory_context)
        seen_text = f"{conversation_history}\n{memory_context}".lower()
        seen_fact_ids = self._extract_seen_fact_ids(conversation_history, memory_context)

        # Identify current-turn dominant categories and already-discussed
        # categories to enable cross-group fact selection.
        current_dominant_categories: set[str] = set()
        discussed_category_counts: Dict[str, int] = {}
        for fact in fact_list:
            cat_lower = fact.category.lower()
            cat_tokens = self._tokenize(fact.category)
            if cat_tokens and len(cat_tokens & current_tokens) / len(cat_tokens) > 0.3:
                current_dominant_categories.add(cat_lower)
            cat_fact_tokens = self._tokenize(f"{fact.category} {fact.fact}")
            if cat_fact_tokens:
                h_overlap = len(cat_fact_tokens & history_tokens) / len(cat_fact_tokens)
                if h_overlap > 0.4:
                    discussed_category_counts[cat_lower] = (
                        discussed_category_counts.get(cat_lower, 0) + 1
                    )
        # A category is considered "discussed" when multiple facts from it
        # have high overlap with the conversation history.
        discussed_categories = {
            cat for cat, cnt in discussed_category_counts.items() if cnt >= 2
        }

        fallback: Dict[str, Dict[str, Any]] = {}
        for fact in fact_list:
            fact_tokens = self._tokenize(f"{fact.category} {fact.fact}")
            if fact_tokens:
                current_overlap = len(fact_tokens & current_tokens) / len(fact_tokens)
                history_overlap = len(fact_tokens & history_tokens) / len(fact_tokens)
                profile_overlap = len(fact_tokens & profile_tokens) / len(fact_tokens)
                memory_overlap = len(fact_tokens & memory_tokens) / len(fact_tokens)
            else:
                current_overlap = history_overlap = profile_overlap = memory_overlap = 0.0

            category_in_current = 0.15 if fact.category.lower() in current_tokens else 0.0
            current_turn_relevance = self._clamp_score(
                0.75 * current_overlap + category_in_current
            )
            novelty = self._novelty_score(
                fact=fact,
                seen_text=seen_text,
                seen_fact_ids=seen_fact_ids,
                recently_selected_ids=recently_selected_ids,
                conveyed_fact_ids=conveyed_fact_ids or set(),
            )
            actionability = self._actionability_score(fact)

            # Dampen history_overlap for current-turn categories so the
            # scorer favours cross-group facts over same-topic details.
            fact_cat = fact.category.lower()
            effective_history_overlap = history_overlap
            if fact_cat in current_dominant_categories:
                effective_history_overlap *= 0.3

            # Bonus for facts from categories not yet discussed — encourages
            # forward-looking anticipation across topic groups.
            category_novelty_bonus = (
                0.15 if fact_cat not in discussed_categories else 0.0
            )

            next_step_utility = self._clamp_score(
                0.35 * effective_history_overlap
                + 0.20 * profile_overlap
                + 0.10 * memory_overlap
                + 0.25 * actionability
                + 0.20 * novelty
                - 0.20 * current_turn_relevance
                + category_novelty_bonus
            )
            alignment_score = (
                self._alignment_score(
                    fact=fact,
                    predicted_topic=predicted_topic,
                    predicted_need=predicted_need,
                )
                if (predicted_topic or predicted_need)
                else next_step_utility
            )
            selection_score = self._selection_score(
                current_turn_relevance=current_turn_relevance,
                next_step_utility=next_step_utility,
                novelty=novelty,
                alignment_score=alignment_score,
            )

            if current_turn_relevance >= 0.75 and next_step_utility < 0.85:
                reason = (
                    f"This mainly answers the current {fact.category.replace('_', ' ')} "
                    "question rather than the next step."
                )
            elif next_step_utility >= 0.6:
                reason = "This fact looks useful for the user's next likely planning step."
            elif history_overlap > 0 or memory_overlap > 0:
                reason = "Related terms appear in the recent dialogue context."
            elif profile_overlap > 0:
                reason = "User profile context suggests this fact may be useful next."
            else:
                reason = "Fallback benchmark relevance based on available context."

            fallback[fact.id] = {
                "current_turn_relevance": current_turn_relevance,
                "next_step_utility": next_step_utility,
                "actionability": actionability,
                "novelty": novelty,
                "alignment_score": alignment_score,
                "score": selection_score,
                "excluded_as_current_turn_answer": self._should_exclude_as_current_turn_answer(
                    current_turn_relevance=current_turn_relevance,
                    next_step_utility=next_step_utility,
                ),
                "reason": reason,
            }

        return fallback

    @staticmethod
    def _build_prompt(
        fact_list: Sequence[Fact],
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
        predicted_topic: str,
        predicted_need: str,
    ) -> str:
        facts_block = "\n".join(
            f"- {fact.id} [{fact.category}]: {fact.fact}" for fact in fact_list
        )
        predicted_block = ""
        if predicted_topic or predicted_need:
            predicted_block = (
                "*** PREDICTED NEXT-STEP NEED (primary alignment target) ***\n"
                f"Topic: {predicted_topic or '[empty]'}\n"
                f"Need: {predicted_need or '[empty]'}\n"
                "Facts that directly address this predicted need should receive "
                "alignment_score >= 0.7 and high next_step_utility. "
                "Prefer facts supporting this prediction over generic context facts.\n\n"
            )
        else:
            predicted_block = (
                "Predicted next-step topic: [empty]\n"
                "Predicted next-step need: [empty]\n\n"
            )

        return (
            "Score how useful each fact is for the user's likely next information need.\n"
            "Assume the assistant will already answer the current user question well in the main reply.\n"
            "Prioritize facts that become useful for the next planning step, and down-rank facts that mainly answer the current turn.\n\n"
            f"Current user message:\n{current_message or '[empty]'}\n\n"
            f"Conversation history:\n{conversation_history or '[empty]'}\n\n"
            f"User profile:\n{user_profile or '[empty]'}\n\n"
            f"Memory context:\n{memory_context or '[empty]'}\n\n"
            f"{predicted_block}"
            "Fact sheet:\n"
            f"{facts_block}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "scored_facts": [\n'
            '    {"fact_id": "F01", "alignment_score": 0.86, "current_turn_relevance": 0.20, "next_step_utility": 0.82, "reason": "short rationale"}\n'
            "  ]\n"
            "}\n"
            "Include every fact at most once."
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in _TOKEN_RE.findall((text or "").lower())
            if token not in _STOPWORDS and len(token) > 1
        }

    @staticmethod
    def _clamp_score(score: float) -> float:
        return max(0.0, min(1.0, round(float(score), 4)))

    @staticmethod
    def _should_exclude_as_current_turn_answer(
        *,
        current_turn_relevance: float,
        next_step_utility: float,
    ) -> bool:
        # Exclude when current_turn_relevance is high AND meaningfully dominates
        # next_step_utility.  A 0.10 gap prevents excluding facts that are
        # almost equally useful for the next step.
        if current_turn_relevance >= 0.75 and current_turn_relevance >= next_step_utility + 0.10:
            return True
        # Also exclude obvious current-turn answers with low future utility.
        if current_turn_relevance >= 0.75 and next_step_utility < 0.45:
            return True
        return False

    def _selection_score(
        self,
        *,
        current_turn_relevance: float,
        next_step_utility: float,
        novelty: float,
        alignment_score: float,
    ) -> float:
        return self._clamp_score(
            0.50 * alignment_score
            + 0.30 * next_step_utility
            + 0.20 * novelty
            - 0.25 * current_turn_relevance
        )

    def _alignment_score(
        self,
        *,
        fact: Fact,
        predicted_topic: str,
        predicted_need: str,
    ) -> float:
        target_tokens = self._tokenize(f"{predicted_topic} {predicted_need}")
        if not target_tokens:
            return 0.0

        topic_tokens = self._tokenize(predicted_topic)
        need_tokens = self._tokenize(predicted_need)
        fact_tokens = self._tokenize(f"{fact.category} {fact.fact}")
        if not fact_tokens:
            return 0.0

        combined_overlap = len(fact_tokens & target_tokens) / len(target_tokens)
        topic_overlap = (
            len(fact_tokens & topic_tokens) / len(topic_tokens) if topic_tokens else 0.0
        )
        need_overlap = (
            len(fact_tokens & need_tokens) / len(need_tokens) if need_tokens else 0.0
        )
        anchor_tokens = {
            token for token in target_tokens if len(token) >= 4 or any(ch.isdigit() for ch in token)
        }
        anchor_overlap = (
            len(fact_tokens & anchor_tokens) / len(anchor_tokens) if anchor_tokens else 0.0
        )
        category_bonus = 0.12 if self._tokenize(fact.category) & target_tokens else 0.0

        return self._clamp_score(
            0.35 * combined_overlap
            + 0.20 * topic_overlap
            + 0.20 * need_overlap
            + 0.25 * anchor_overlap
            + category_bonus
        )

    @staticmethod
    def _actionability_score(fact: Fact) -> float:
        text = f"{fact.category} {fact.fact}".lower()
        markers = [
            "required",
            "must",
            "deadline",
            "fee",
            "fees",
            "cost",
            "costs",
            "pay",
            "paid",
            "deposit",
            "opens",
            "closes",
            "open",
            "close",
            "register",
            "registration",
            "apply",
            "application",
            "documents",
            "document",
            "passport",
            "statement",
            "visa",
            "address",
            "located",
            "schedule",
        ]
        hits = sum(1 for marker in markers if marker in text)
        return max(0.0, min(1.0, round(min(hits, 5) / 5, 4)))

    @staticmethod
    def _novelty_score(
        *,
        fact: Fact,
        seen_text: str,
        seen_fact_ids: set[str],
        recently_selected_ids: set[str],
        conveyed_fact_ids: Optional[set[str]] = None,
    ) -> float:
        if fact.id.lower() in seen_fact_ids:
            return 0.0
        if conveyed_fact_ids and fact.id in conveyed_fact_ids:
            return 0.0
        if fact.id in recently_selected_ids:
            return 0.3
        if fact.fact.lower() in seen_text:
            return 0.0
        # Content-overlap check: if a substantial proportion of the fact's
        # meaningful tokens already appear in the conversation/memory context,
        # the fact's information has likely been conveyed (possibly paraphrased).
        fact_tokens = {
            token
            for token in _TOKEN_RE.findall(fact.fact.lower())
            if token not in _STOPWORDS and len(token) > 1
        }
        if fact_tokens:
            seen_tokens = {
                token
                for token in _TOKEN_RE.findall(seen_text)
                if token not in _STOPWORDS and len(token) > 1
            }
            overlap_ratio = len(fact_tokens & seen_tokens) / len(fact_tokens)
            if overlap_ratio >= 0.35:
                return 0.0
        return 1.0

    @staticmethod
    def _extract_seen_fact_ids(*texts: str) -> set[str]:
        combined = "\n".join(texts).lower()
        return {
            match.group(1).lower()
            for match in re.finditer(r"fact:([a-z0-9_-]+)", combined)
        }
