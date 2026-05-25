"""
Benchmark-only fact-grounded proactive predictor for Route C.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

from experiments.ProactiveBench.eval.fact_scorer import BenchmarkFactScorer, ScoredFact
from experiments.ProactiveBench.generation.models import Fact


class BenchmarkFactGroundedPredictor:
    """
    Need-predictor-compatible wrapper that scores benchmark facts first and then
    synthesizes closed-world predictions for the existing proactive pipeline.
    """

    _RECENT_SELECTED_LIMIT = 8

    def __init__(
        self,
        fact_sheet: Sequence[Fact],
        scorer: Optional[BenchmarkFactScorer] = None,
        top_k: int = 2,
        min_score: float = 0.2,
    ) -> None:
        self.fact_sheet = list(fact_sheet)
        self.scorer = scorer or BenchmarkFactScorer()
        self.top_k = max(1, int(top_k))
        self.min_score = max(0.0, float(min_score))
        self.last_scored_facts: List[ScoredFact] = []
        self.last_selected_facts: List[ScoredFact] = []
        self._recent_selected_fact_ids: List[str] = []
        self._conveyed_fact_ids: set[str] = set()
        self.last_trace: Dict[str, Any] = {
            "fact_scores": [],
            "selected_facts": [],
            "fact_scoring_rationales": {},
        }

    def mark_conveyed(self, fact_ids: Sequence[str]) -> None:
        """Mark fact IDs as having been conveyed in assistant responses."""
        self._conveyed_fact_ids.update(fact_ids)

    def run(
        self,
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
    ) -> Dict[str, Any]:
        score_kwargs = {
            "fact_sheet": self.fact_sheet,
            "current_message": current_message,
            "conversation_history": conversation_history,
            "user_profile": user_profile,
            "memory_context": memory_context,
            "recently_selected_ids": set(self._recent_selected_fact_ids),
        }
        if self._scorer_supports_conveyed_fact_ids():
            score_kwargs["conveyed_fact_ids"] = self._conveyed_fact_ids

        scored_facts = self.scorer.score_facts(**score_kwargs)
        selected_facts = [
            fact
            for fact in scored_facts
            if fact.score >= self.min_score and not fact.excluded_as_current_turn_answer
        ][: self.top_k]
        if not selected_facts:
            selected_facts = [
                fact for fact in scored_facts if fact.score >= self.min_score
            ][: self.top_k]
        if not selected_facts and scored_facts:
            non_excluded = [
                fact for fact in scored_facts if not fact.excluded_as_current_turn_answer
            ]
            selected_facts = (non_excluded or scored_facts)[:1]

        self.last_scored_facts = list(scored_facts)
        self.last_selected_facts = list(selected_facts)
        self._record_selected_fact_ids(selected_facts)
        self.last_trace = {
            "fact_scores": [fact.to_dict() for fact in scored_facts],
            "selected_facts": [fact.to_dict() for fact in selected_facts],
            "fact_scoring_rationales": {
                fact.fact_id: fact.reason for fact in scored_facts
            },
        }

        return {
            "predicted_needs": [
                self._fact_to_prediction(fact) for fact in selected_facts
            ]
        }

    def _record_selected_fact_ids(self, selected_facts: Sequence[ScoredFact]) -> None:
        if not selected_facts:
            return
        combined = list(self._recent_selected_fact_ids)
        for fact in selected_facts:
            combined = [fact_id for fact_id in combined if fact_id != fact.fact_id]
            combined.append(fact.fact_id)
        self._recent_selected_fact_ids = combined[-self._RECENT_SELECTED_LIMIT:]

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

    @staticmethod
    def _fact_to_prediction(fact: ScoredFact) -> Dict[str, Any]:
        return {
            "need": f"Likely next planning step: {fact.fact_text}",
            "reason": fact.reason,
            "confidence": fact.score,
            "lookup_query": f"fact:{fact.fact_id}",
        }
