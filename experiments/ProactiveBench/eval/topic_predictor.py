"""
Benchmark-only topic-first proactive predictor for Route C main runtime.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Optional, Sequence

from experiments.ProactiveBench.eval.fact_scorer import BenchmarkFactScorer, ScoredFact
from experiments.ProactiveBench.generation.models import Fact


class BenchmarkTopicFirstPredictor:
    """
    Need-predictor-compatible wrapper that predicts future topics first,
    retrieves supporting memory, reranks only the retrieved subset, and uses
    a bounded repair step when all initial hypotheses are unsupported.
    """

    _RECENT_SELECTED_LIMIT = 8

    def __init__(
        self,
        memory_system: Any,
        llm_client: Any,
        scorer: Optional[BenchmarkFactScorer] = None,
        model: str = "gpt-4o-mini",
        top_k: int = 2,
        retrieval_k: int = 5,
        min_fact_score: float = 0.2,
        support_threshold: float = 0.35,
        repair_limit: int = 2,
    ) -> None:
        self.memory = memory_system
        self.llm_client = llm_client
        self.scorer = scorer or BenchmarkFactScorer(llm_client=llm_client, model=model)
        self.model = model
        self.top_k = max(1, int(top_k))
        self.retrieval_k = max(1, int(retrieval_k))
        self.min_fact_score = max(0.0, float(min_fact_score))
        self.support_threshold = max(0.0, float(support_threshold))
        self.repair_limit = max(0, int(repair_limit))
        self._recent_selected_fact_ids: List[str] = []
        self._conveyed_fact_ids: set[str] = set()
        self.last_trace: Dict[str, Any] = {
            "fact_scores": [],
            "selected_facts": [],
            "fact_scoring_rationales": {},
            "topic_candidates": [],
            "repair_used": False,
            "repair_raw_output": None,
            "repair_candidate_count": 0,
        }

    def mark_conveyed(self, fact_ids: Sequence[str]) -> None:
        self._conveyed_fact_ids.update(fact_ids)

    def run(
        self,
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
    ) -> Dict[str, Any]:
        repair_used = False
        repair_raw_output: Optional[Dict[str, Any]] = None
        repair_candidate_count = 0
        candidates = self._predict_topics(
            current_message=current_message,
            conversation_history=conversation_history,
            user_profile=user_profile,
            memory_context=memory_context,
        )
        evaluated = self._evaluate_candidates(
            candidates,
            current_message=current_message,
            conversation_history=conversation_history,
            user_profile=user_profile,
            memory_context=memory_context,
        )
        supported = [item for item in evaluated if item["supported"]]

        if not supported and candidates and self.repair_limit > 0:
            repaired, repair_raw_output = self._repair_topics(
                current_message=current_message,
                conversation_history=conversation_history,
                user_profile=user_profile,
                memory_context=memory_context,
                evaluated_candidates=evaluated,
            )
            repair_used = bool(repaired)
            repair_candidate_count = len(repaired)
            if repaired:
                evaluated = self._evaluate_candidates(
                    repaired,
                    current_message=current_message,
                    conversation_history=conversation_history,
                    user_profile=user_profile,
                    memory_context=memory_context,
                )
                supported = [item for item in evaluated if item["supported"]]

        selected_candidates = sorted(
            supported,
            key=lambda item: (
                -float(item["support_score"]),
                -(item["candidate"].get("confidence", 0.0) or 0.0),
            ),
        )[: self.top_k]
        selected_facts: List[ScoredFact] = []
        for item in selected_candidates:
            selected_facts.extend(item["selected_facts"])

        self._record_selected_fact_ids(selected_facts)
        self.last_trace = {
            "fact_scores": self._dedupe_fact_dicts(
                [fact.to_dict() for item in evaluated for fact in item["scored_facts"]]
            ),
            "selected_facts": self._dedupe_fact_dicts(
                [fact.to_dict() for fact in selected_facts]
            ),
            "fact_scoring_rationales": {
                fact["fact_id"]: fact.get("reason", "")
                for fact in self._dedupe_fact_dicts(
                    [fact.to_dict() for item in evaluated for fact in item["scored_facts"]]
                )
            },
            "topic_candidates": [
                {
                    "topic": item["candidate"].get("topic", ""),
                    "need": item["candidate"].get("need", ""),
                    "reason": item["candidate"].get("reason", ""),
                    "confidence": item["candidate"].get("confidence", 0.0),
                    "retrieval_query": item["candidate"].get("retrieval_query", ""),
                    "support_score": item["support_score"],
                    "supported": item["supported"],
                }
                for item in evaluated
            ],
            "repair_used": repair_used,
            "repair_raw_output": repair_raw_output,
            "repair_candidate_count": repair_candidate_count,
        }

        return {
            "predicted_needs": [
                self._candidate_to_prediction(item) for item in selected_candidates
            ]
        }

    def _predict_topics(
        self,
        *,
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
    ) -> List[Dict[str, Any]]:
        payload = self.llm_client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You predict what the user will ask about NEXT (after their "
                        "current question is answered) in a closed-world benchmark. "
                        "Focus on anticipating future topics, not deepening the "
                        "current one. Use only the conversation context. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prediction_prompt(
                        current_message=current_message,
                        conversation_history=conversation_history,
                        user_profile=user_profile,
                        memory_context=memory_context,
                    ),
                },
            ],
            model=self.model,
            temperature=0.0,
            max_tokens=600,
        )
        return self._normalize_candidates(payload)[: self.top_k]

    def _repair_topics(
        self,
        *,
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
        evaluated_candidates: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        payload = self.llm_client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You revise a failed proactive topic prediction after "
                        "retrieval found insufficient support. Suggest one topic "
                        "the user will move to NEXT (a different area from the "
                        "current discussion) that is better grounded in the "
                        "benchmark memory. Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_repair_prompt(
                        current_message=current_message,
                        conversation_history=conversation_history,
                        user_profile=user_profile,
                        memory_context=memory_context,
                        evaluated_candidates=evaluated_candidates,
                    ),
                },
            ],
            model=self.model,
            temperature=0.0,
            max_tokens=400,
        )
        return self._normalize_candidates(payload)[:1], payload

    def _evaluate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
    ) -> List[Dict[str, Any]]:
        evaluated: List[Dict[str, Any]] = []
        for candidate in candidates:
            query = (candidate.get("retrieval_query") or candidate.get("topic") or "").strip()
            retrieved_items = self.memory.search_knowledge(query, n_results=self.retrieval_k)
            retrieved_facts = self._facts_from_retrieved_items(retrieved_items)
            if not retrieved_facts:
                evaluated.append(
                    {
                        "candidate": candidate,
                        "scored_facts": [],
                        "selected_facts": [],
                        "support_score": 0.0,
                        "supported": False,
                    }
                )
                continue

            score_kwargs = {
                "fact_sheet": retrieved_facts,
                "current_message": current_message,
                "conversation_history": conversation_history,
                "user_profile": user_profile,
                "memory_context": memory_context,
                "predicted_topic": str(candidate.get("topic", "")).strip(),
                "predicted_need": str(candidate.get("need", "")).strip(),
                "recently_selected_ids": set(self._recent_selected_fact_ids),
            }
            if self._scorer_supports_conveyed_fact_ids():
                score_kwargs["conveyed_fact_ids"] = self._conveyed_fact_ids

            scored_facts = self.scorer.score_facts(**score_kwargs)
            selected_facts = self._select_facts(scored_facts)
            support_score = self._support_score(
                selected_facts=selected_facts,
                candidate_confidence=float(candidate.get("confidence", 0.0) or 0.0),
            )
            evaluated.append(
                {
                    "candidate": candidate,
                    "scored_facts": scored_facts,
                    "selected_facts": selected_facts,
                    "support_score": support_score,
                    "supported": bool(selected_facts) and support_score >= self.support_threshold,
                }
            )
        return evaluated

    def _select_facts(self, scored_facts: Sequence[ScoredFact]) -> List[ScoredFact]:
        selected = [
            fact
            for fact in scored_facts
            if fact.score >= self.min_fact_score and not fact.excluded_as_current_turn_answer
        ][: self.top_k]
        if selected:
            return selected

        fallback = [
            fact for fact in scored_facts if not fact.excluded_as_current_turn_answer
        ][:1]
        return fallback

    @staticmethod
    def _support_score(
        *,
        selected_facts: Sequence[ScoredFact],
        candidate_confidence: float,
    ) -> float:
        if not selected_facts:
            return 0.0
        best_fact = max(fact.score for fact in selected_facts)
        score = 0.75 * float(best_fact) + 0.25 * float(candidate_confidence)
        return max(0.0, min(1.0, round(score, 4)))

    @staticmethod
    def _candidate_to_prediction(item: Dict[str, Any]) -> Dict[str, Any]:
        fact_ids = [fact.fact_id for fact in item["selected_facts"]]
        candidate = item["candidate"]
        need_text = (candidate.get("need") or candidate.get("topic") or "").strip()
        if not need_text:
            need_text = "Likely next planning step."
        return {
            "need": need_text,
            "reason": (candidate.get("reason") or need_text).strip(),
            "confidence": item["support_score"],
            "lookup_query": f"factset:{','.join(fact_ids)}",
        }

    @staticmethod
    def _build_prediction_prompt(
        *,
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
    ) -> str:
        anchors = BenchmarkTopicFirstPredictor._extract_anchor_hints(
            current_message,
            conversation_history,
            memory_context,
        )
        anchor_block = "\n".join(f"- {anchor}" for anchor in anchors) if anchors else "- [none]"
        return (
            "Predict exactly 2 information needs the user is likely to ask about NEXT in this closed-world benchmark.\n\n"
            "*** CRITICAL DISTINCTION ***\n"
            "You MUST predict what the user will need AFTER their current question is answered, "
            "NOT what helps answer the current question.\n"
            "Think: once the assistant answers the current message, what will the user logically ask about next?\n\n"
            "The 2 predictions should be:\n"
            "1. A NEXT-STEP need: the immediate follow-up within the current topic area "
            "(e.g., after learning a deadline, the user will ask about required documents).\n"
            "2. A NEXT-TOPIC need: a NEW topic area the user hasn't explored yet but will "
            "naturally move to given the overall task context "
            "(e.g., after discussing requirements, the user will move to costs/fees or scheduling).\n\n"
            "For each prediction, think about the user's overall goal (from the profile) and what "
            "logical gaps remain in the conversation so far. The most valuable predictions address "
            "topics NOT YET discussed in the conversation history.\n\n"
            "Rules:\n"
            "- Do not use any benchmark gold labels.\n"
            "- Use only information grounded in the conversation context and already retrieved memory.\n"
            "- Prefer concrete anchors (entities, names, dates, IDs) from the current context.\n"
            "- Do not generate generic advice or open-world suggestions not anchored in context.\n"
            "- retrieval_query should reuse concrete anchors verbatim when possible.\n\n"
            "Return JSON with a `predicted_topics` list. Each item must contain:\n"
            "- topic\n"
            "- need\n"
            "- reason (explain WHY the user will ask this NEXT — what logical step leads here)\n"
            "- confidence (0-1)\n"
            "- retrieval_query\n\n"
            "Concrete anchors from current context:\n"
            f"{anchor_block}\n\n"
            f"Current user message:\n{current_message or '[empty]'}\n\n"
            f"Conversation history:\n{conversation_history or '[empty]'}\n\n"
            f"User profile:\n{user_profile or '[empty]'}\n\n"
            f"Already retrieved memory context:\n{memory_context or '[empty]'}\n"
        )

    @staticmethod
    def _build_repair_prompt(
        *,
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
        evaluated_candidates: List[Dict[str, Any]],
    ) -> str:
        anchors = BenchmarkTopicFirstPredictor._extract_anchor_hints(
            current_message,
            conversation_history,
            memory_context,
        )
        anchor_block = "\n".join(f"- {anchor}" for anchor in anchors) if anchors else "- [none]"
        candidate_lines = []
        for item in evaluated_candidates:
            candidate = item["candidate"]
            candidate_lines.append(
                "- topic: {topic}\n"
                "  need: {need}\n"
                "  reason: {reason}\n"
                "  retrieval_query: {query}\n"
                "  support_score: {support:.2f}\n"
                "  supported: {supported}".format(
                    topic=candidate.get("topic", ""),
                    need=candidate.get("need", ""),
                    reason=candidate.get("reason", ""),
                    query=candidate.get("retrieval_query", ""),
                    support=float(item.get("support_score", 0.0)),
                    supported=item.get("supported", False),
                )
            )
        return (
            "The earlier next-step predictions were not well supported by retrieval.\n"
            "Suggest one revised next-step topic that is:\n"
            "1. About a DIFFERENT topic area than what was already discussed in the conversation.\n"
            "2. Something the user will logically need AFTER their current question is answered.\n"
            "3. Grounded in concrete entities already mentioned in the context.\n\n"
            "Use a more concrete, entity-anchored retrieval query.\n"
            "Do not generate generic advice, broad best-practice topics, or unsupported open-world suggestions.\n"
            "Return JSON in the same schema as before.\n\n"
            "Concrete anchors from current context:\n"
            f"{anchor_block}\n\n"
            f"Current user message:\n{current_message or '[empty]'}\n\n"
            f"Conversation history:\n{conversation_history or '[empty]'}\n\n"
            f"User profile:\n{user_profile or '[empty]'}\n\n"
            f"Already retrieved memory context:\n{memory_context or '[empty]'}\n\n"
            "Unsupported candidates:\n"
            + ("\n".join(candidate_lines) if candidate_lines else "[none]")
        )

    @staticmethod
    def _normalize_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries = payload.get("predicted_topics")
        if entries is None and payload.get("topic"):
            entries = [payload]
        if entries is None and payload.get("next_step_topic") is not None:
            next_step_topic = payload.get("next_step_topic")
            if isinstance(next_step_topic, dict):
                merged = dict(next_step_topic)
                for key in ("need", "reason", "retrieval_query", "confidence"):
                    if key not in merged and key in payload:
                        merged[key] = payload.get(key)
                entries = [merged]
            else:
                merged = {
                    "topic": next_step_topic,
                    "need": payload.get("need", ""),
                    "reason": payload.get("reason", ""),
                    "retrieval_query": payload.get("retrieval_query", ""),
                    "confidence": payload.get("confidence", 0.0),
                }
                if not merged["retrieval_query"] and isinstance(payload.get("details"), dict):
                    details = payload.get("details", {})
                    merged["need"] = merged["need"] or details.get("need", "")
                    merged["reason"] = merged["reason"] or details.get("reason", "")
                    merged["retrieval_query"] = details.get("retrieval_query", "")
                entries = [merged]
        if not isinstance(entries, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for entry in entries:
            topic = str(entry.get("topic", "")).strip()
            need = str(entry.get("need", "")).strip()
            reason = str(entry.get("reason", "")).strip()
            query = str(entry.get("retrieval_query", "")).strip() or topic or need
            if not query:
                continue
            try:
                confidence = float(entry.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            normalized.append(
                {
                    "topic": topic or need or query,
                    "need": need or topic or query,
                    "reason": reason or need or topic or query,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "retrieval_query": query,
                }
            )
        return normalized

    def _scorer_supports_conveyed_fact_ids(self) -> bool:
        try:
            signature = inspect.signature(self.scorer.score_facts)
        except (TypeError, ValueError):
            return False
        parameters = signature.parameters
        if "conveyed_fact_ids" in parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _record_selected_fact_ids(self, selected_facts: Sequence[ScoredFact]) -> None:
        if not selected_facts:
            return
        combined = list(self._recent_selected_fact_ids)
        for fact in selected_facts:
            combined = [fact_id for fact_id in combined if fact_id != fact.fact_id]
            combined.append(fact.fact_id)
        self._recent_selected_fact_ids = combined[-self._RECENT_SELECTED_LIMIT:]

    @staticmethod
    def _dedupe_fact_dicts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for item in items:
            fact_id = str(item.get("fact_id", "")).strip()
            if not fact_id or fact_id in seen:
                continue
            seen.add(fact_id)
            deduped.append(item)
        return deduped

    @staticmethod
    def _extract_anchor_hints(*texts: str) -> List[str]:
        combined = "\n".join(text for text in texts if text)
        patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-zA-Z'&.-]+){0,4}\b",
            r"\b[A-Z]{2,}[A-Z0-9-]*\b",
            r"\b\d{1,4}[A-Za-z0-9-]*\b",
            r"\(\d{3}\)\s*\d{3}-\d{4}",
            r"\b[A-Za-z0-9-]{4,}\b",
        ]
        anchors: List[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, combined):
                value = match.group(0).strip(" ,.;:")
                if len(value) < 4:
                    continue
                lowered = value.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                anchors.append(value)
                if len(anchors) >= 12:
                    return anchors
        return anchors

    @classmethod
    def _facts_from_retrieved_items(cls, items: Sequence[Any]) -> List[Fact]:
        facts: List[Fact] = []
        seen: set[str] = set()
        for item in items:
            fact = cls._knowledge_item_to_fact(item)
            if fact is None or fact.id in seen:
                continue
            seen.add(fact.id)
            facts.append(fact)
        return facts

    @staticmethod
    def _knowledge_item_to_fact(item: Any) -> Optional[Fact]:
        if isinstance(item, dict):
            title = str(item.get("title", ""))
            content = str(item.get("content", "") or "")
            summary = str(item.get("summary", "") or "")
            source_url = str(item.get("source_url", ""))
        else:
            title = str(getattr(item, "title", ""))
            content = str(getattr(item, "content", "") or "")
            summary = str(getattr(item, "summary", "") or "")
            source_url = str(getattr(item, "source_url", ""))

        text = content or summary
        fact_id_match = re.search(r"fact:([A-Za-z0-9_-]+)", source_url, flags=re.IGNORECASE)
        if fact_id_match:
            fact_id = fact_id_match.group(1).upper()
        else:
            title_match = re.match(r"^\s*\[[^\]]+\]\s*([A-Za-z0-9_-]+)\s*$", title)
            if title_match:
                fact_id = title_match.group(1).upper()
            else:
                content_match = re.match(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.+)$", text)
                if not content_match:
                    return None
                fact_id = content_match.group(1).upper()

        category_match = re.match(r"^\s*\[([^\]]+)\]", title)
        category = category_match.group(1).strip() if category_match else "benchmark"

        fact_text = text
        content_match = re.match(r"^\s*[A-Za-z0-9_-]+\s*:\s*(.+)$", text)
        if content_match:
            fact_text = content_match.group(1).strip()
        fact_text = fact_text.strip()
        if not fact_text:
            return None

        return Fact(id=fact_id, category=category, fact=fact_text)
