"""
ProactiveBench evaluation data models.

Defines all intermediate and final result structures for the evaluation pipeline.
Every dataclass provides to_dict() / from_dict() for JSON serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Turn-level structures
# ---------------------------------------------------------------------------


@dataclass
class NeedCoverage:
    """Records how a particular user need was covered."""

    need_id: str           # e.g., "N1"
    mode: str              # "reactive" | "proactive"

    def to_dict(self) -> Dict[str, Any]:
        return {"need_id": self.need_id, "mode": self.mode}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeedCoverage":
        return cls(need_id=data["need_id"], mode=data["mode"])


@dataclass
class JudgmentResult:
    """CoverageJudge output for a single agent response turn."""

    turn_number: int
    facts_conveyed: List[str]           # Fact IDs accurately mentioned
    facts_distorted: List[str]          # Fact IDs mentioned with errors
    hallucinated_claims: List[str]      # Claims not grounded in fact_sheet
    needs_addressed: List[NeedCoverage] # Needs covered this turn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "facts_conveyed": self.facts_conveyed,
            "facts_distorted": self.facts_distorted,
            "hallucinated_claims": self.hallucinated_claims,
            "needs_addressed": [nc.to_dict() for nc in self.needs_addressed],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgmentResult":
        return cls(
            turn_number=data["turn_number"],
            facts_conveyed=data.get("facts_conveyed", []),
            facts_distorted=data.get("facts_distorted", []),
            hallucinated_claims=data.get("hallucinated_claims", []),
            needs_addressed=[
                NeedCoverage.from_dict(nc) for nc in data.get("needs_addressed", [])
            ],
        )


@dataclass
class TurnTrace:
    """Full trace for a single conversation turn."""

    turn_number: int
    user_message: str
    active_need_id: Optional[str]
    needs_skipped: List[str]
    agent_response: str
    proactive_predictions: List[Dict[str, Any]]
    proactive_approved: List[Dict[str, Any]]
    memory_context: str  # Retrieved memory context for this turn (for audit)
    judgment: JudgmentResult
    cumulative_covered_needs: Set[str]
    satellite_need_ids: List[str] = field(default_factory=list)
    fact_scores: List[Dict[str, Any]] = field(default_factory=list)
    selected_facts: List[Dict[str, Any]] = field(default_factory=list)
    fact_scoring_rationales: Dict[str, str] = field(default_factory=dict)
    delivered_inline_facts: List[Dict[str, Any]] = field(default_factory=list)
    admission_trace: Optional[Dict[str, Any]] = None
    decision_trace: Optional[Dict[str, Any]] = None
    is_push_turn: bool = False
    push_content: Optional[str] = None
    push_type: Optional[str] = None
    push_fact_ids: List[str] = field(default_factory=list)
    exploration_intents: List[Dict[str, Any]] = field(default_factory=list)
    fsp_filter_reasons: List[Dict[str, Any]] = field(default_factory=list)
    simulated_idle_seconds: float = 0.0
    idle_trigger_seconds: float = 5.0
    max_intents_per_idle: int = 3
    max_queries_per_search: int = 1
    search_budget: int = 999
    total_searches_used: int = 0
    remaining_search_budget: int = 999
    ite_ran: bool = False
    intent_queue_size: int = 0
    idle_window_mode: bool = False
    idle_tick_count: int = 0
    idle_tick_results: List[Dict[str, Any]] = field(default_factory=list)
    queue_size_before: int = 0
    queue_size_after_generation: int = 0
    queue_size_after_exploration: int = 0
    generated_intents: int = 0
    accepted_intents: int = 0
    filtered_intents: int = 0
    duplicate_intents: int = 0
    facts_stored: int = 0
    stored_fact_ids: List[str] = field(default_factory=list)
    intent_results: List[Dict[str, Any]] = field(default_factory=list)
    inline_topics: List[str] = field(default_factory=list)
    push_novelty_ratio: Optional[float] = None
    push_undelivered_fact_ids: List[str] = field(default_factory=list)
    push_delivered_fact_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "active_need_id": self.active_need_id,
            "primary_need_id": self.active_need_id,
            "satellite_need_ids": self.satellite_need_ids,
            "needs_skipped": self.needs_skipped,
            "agent_response": self.agent_response,
            "proactive_predictions": self.proactive_predictions,
            "proactive_approved": self.proactive_approved,
            "memory_context": self.memory_context,
            "fact_scores": self.fact_scores,
            "selected_facts": self.selected_facts,
            "fact_scoring_rationales": self.fact_scoring_rationales,
            "delivered_inline_facts": self.delivered_inline_facts,
            "judgment": self.judgment.to_dict(),
            "cumulative_covered_needs": sorted(self.cumulative_covered_needs),
        }
        if self.admission_trace is not None:
            payload["admission_trace"] = self.admission_trace
        if self.decision_trace is not None:
            payload["decision_trace"] = self.decision_trace
        payload["is_push_turn"] = self.is_push_turn
        if self.push_content is not None:
            payload["push_content"] = self.push_content
        if self.push_type is not None:
            payload["push_type"] = self.push_type
        if self.push_fact_ids:
            payload["push_fact_ids"] = self.push_fact_ids
        if self.exploration_intents:
            payload["exploration_intents"] = self.exploration_intents
        if self.fsp_filter_reasons:
            payload["fsp_filter_reasons"] = self.fsp_filter_reasons
        payload["simulated_idle_seconds"] = self.simulated_idle_seconds
        payload["idle_trigger_seconds"] = self.idle_trigger_seconds
        payload["max_intents_per_idle"] = self.max_intents_per_idle
        payload["max_queries_per_search"] = self.max_queries_per_search
        payload["search_budget"] = self.search_budget
        payload["total_searches_used"] = self.total_searches_used
        payload["remaining_search_budget"] = self.remaining_search_budget
        payload["ite_ran"] = self.ite_ran
        payload["intent_queue_size"] = self.intent_queue_size
        payload["idle_window_mode"] = self.idle_window_mode
        payload["idle_tick_count"] = self.idle_tick_count
        if self.idle_tick_results:
            payload["idle_tick_results"] = self.idle_tick_results
        payload["queue_size_before"] = self.queue_size_before
        payload["queue_size_after_generation"] = self.queue_size_after_generation
        payload["queue_size_after_exploration"] = self.queue_size_after_exploration
        payload["generated_intents"] = self.generated_intents
        payload["accepted_intents"] = self.accepted_intents
        payload["filtered_intents"] = self.filtered_intents
        payload["duplicate_intents"] = self.duplicate_intents
        payload["facts_stored"] = self.facts_stored
        if self.stored_fact_ids:
            payload["stored_fact_ids"] = self.stored_fact_ids
        if self.intent_results:
            payload["intent_results"] = self.intent_results
        if self.inline_topics:
            payload["inline_topics"] = self.inline_topics
        if self.push_novelty_ratio is not None:
            payload["push_novelty_ratio"] = self.push_novelty_ratio
        if self.push_undelivered_fact_ids:
            payload["push_undelivered_fact_ids"] = self.push_undelivered_fact_ids
        if self.push_delivered_fact_ids:
            payload["push_delivered_fact_ids"] = self.push_delivered_fact_ids
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnTrace":
        return cls(
            turn_number=data["turn_number"],
            user_message=data["user_message"],
            active_need_id=data.get("active_need_id", data.get("primary_need_id")),
            needs_skipped=data.get("needs_skipped", []),
            agent_response=data["agent_response"],
            proactive_predictions=data.get("proactive_predictions", []),
            proactive_approved=data.get("proactive_approved", []),
            memory_context=data.get("memory_context", ""),
            judgment=JudgmentResult.from_dict(data["judgment"]),
            cumulative_covered_needs=set(data.get("cumulative_covered_needs", [])),
            satellite_need_ids=data.get("satellite_need_ids", []),
            fact_scores=data.get("fact_scores", []),
            selected_facts=data.get("selected_facts", []),
            fact_scoring_rationales=data.get("fact_scoring_rationales", {}),
            delivered_inline_facts=data.get("delivered_inline_facts", []),
            admission_trace=data.get("admission_trace"),
            decision_trace=data.get("decision_trace"),
            is_push_turn=data.get("is_push_turn", False),
            push_content=data.get("push_content"),
            push_type=data.get("push_type"),
            push_fact_ids=data.get("push_fact_ids", []),
            exploration_intents=data.get("exploration_intents", []),
            fsp_filter_reasons=data.get("fsp_filter_reasons", []),
            simulated_idle_seconds=data.get("simulated_idle_seconds", 0.0),
            idle_trigger_seconds=data.get("idle_trigger_seconds", 5.0),
            max_intents_per_idle=data.get("max_intents_per_idle", 3),
            max_queries_per_search=data.get("max_queries_per_search", 1),
            search_budget=data.get("search_budget", 999),
            total_searches_used=data.get("total_searches_used", 0),
            remaining_search_budget=data.get("remaining_search_budget", 999),
            ite_ran=data.get("ite_ran", False),
            intent_queue_size=data.get("intent_queue_size", 0),
            idle_window_mode=data.get("idle_window_mode", False),
            idle_tick_count=data.get("idle_tick_count", 0),
            idle_tick_results=data.get("idle_tick_results", []),
            queue_size_before=data.get("queue_size_before", 0),
            queue_size_after_generation=data.get("queue_size_after_generation", 0),
            queue_size_after_exploration=data.get("queue_size_after_exploration", 0),
            generated_intents=data.get("generated_intents", 0),
            accepted_intents=data.get("accepted_intents", 0),
            filtered_intents=data.get("filtered_intents", 0),
            duplicate_intents=data.get("duplicate_intents", 0),
            facts_stored=data.get("facts_stored", 0),
            stored_fact_ids=data.get("stored_fact_ids", []),
            intent_results=data.get("intent_results", []),
            inline_topics=data.get("inline_topics", []),
            push_novelty_ratio=data.get("push_novelty_ratio"),
            push_undelivered_fact_ids=data.get("push_undelivered_fact_ids", []),
            push_delivered_fact_ids=data.get("push_delivered_fact_ids", []),
        )

    @property
    def primary_need_id(self) -> Optional[str]:
        return self.active_need_id


# ---------------------------------------------------------------------------
# Conversation-level structures
# ---------------------------------------------------------------------------


@dataclass
class ConversationResult:
    """Evaluation result for one complete conversation (scenario + condition)."""

    scenario_id: str
    condition: str          # "Full" | "Blind" | "Paralyzed" | "Baseline"
    domain: str
    seed: int
    total_turns: int
    turn_traces: List[TurnTrace]
    final_covered_needs: Set[str]
    all_need_ids: Set[str]
    must_have_ids: Set[str]
    predictable_need_ids: Set[str]
    wall_clock_seconds: float = 0.0
    llm_usage_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    llm_usage_by_module: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "domain": self.domain,
            "seed": self.seed,
            "total_turns": self.total_turns,
            "turn_traces": [t.to_dict() for t in self.turn_traces],
            "final_covered_needs": sorted(self.final_covered_needs),
            "all_need_ids": sorted(self.all_need_ids),
            "must_have_ids": sorted(self.must_have_ids),
            "predictable_need_ids": sorted(self.predictable_need_ids),
            "wall_clock_seconds": self.wall_clock_seconds,
            "llm_usage_stats": self.llm_usage_stats,
            "llm_usage_by_module": self.llm_usage_by_module,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationResult":
        return cls(
            scenario_id=data["scenario_id"],
            condition=data["condition"],
            domain=data["domain"],
            seed=data["seed"],
            total_turns=data["total_turns"],
            turn_traces=[TurnTrace.from_dict(t) for t in data["turn_traces"]],
            final_covered_needs=set(data.get("final_covered_needs", [])),
            all_need_ids=set(data.get("all_need_ids", [])),
            must_have_ids=set(data.get("must_have_ids", [])),
            predictable_need_ids=set(data.get("predictable_need_ids", [])),
            wall_clock_seconds=data.get("wall_clock_seconds", 0.0),
            llm_usage_stats=data.get("llm_usage_stats", {}),
            llm_usage_by_module=data.get("llm_usage_by_module", {}),
        )


# ---------------------------------------------------------------------------
# Metrics & aggregation
# ---------------------------------------------------------------------------


@dataclass
class ScenarioMetrics:
    """Per-scenario evaluation metrics under a specific experimental condition."""

    scenario_id: str
    condition: str
    domain: str
    t80: int                                        # Turns to reach 80% must-have coverage
    t100: int                                       # Turns to reach 100% must-have coverage
    anticipation_precision: Optional[float]         # Proactive anticipation precision
    anticipation_recall: Optional[float]            # Proactive anticipation recall
    fact_accuracy: float                            # Fraction of conveyed facts that are correct
    hallucination_rate: float                       # Hallucinated claims per turn
    user_effort: int                                # Number of explicit user questions
    total_coverage: float                           # Coverage over all needs
    must_have_coverage: float                       # Coverage over must-have needs only
    push_precision: Optional[float] = None
    push_coverage: Optional[float] = None
    intent_hit_rate: Optional[float] = None
    memory_utilization: Optional[float] = None
    wall_clock_seconds: float = 0.0
    llm_calls: int = 0
    llm_total_tokens: int = 0
    stored_facts: int = 0
    processed_intents: int = 0
    peak_queue_size: int = 0
    mean_push_novelty_ratio: Optional[float] = None
    low_novelty_push_rate: Optional[float] = None
    mean_queue_size: float = 0.0
    peak_queue_pressure: float = 0.0
    queue_saturation_turns: int = 0
    duplicate_topic_count: int = 0
    duplicate_topic_rate: Optional[float] = None
    fsp_semantic_duplicate_count: int = 0
    fsp_cross_turn_duplicate_count: int = 0
    queue_semantic_duplicate_count: int = 0
    active_llm_calls: int = 0
    active_llm_total_tokens: int = 0
    module_llm_calls: Dict[str, int] = field(default_factory=dict)
    module_llm_total_tokens: Dict[str, int] = field(default_factory=dict)
    cost_per_new_need: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "domain": self.domain,
            "t80": self.t80,
            "t100": self.t100,
            "anticipation_precision": self.anticipation_precision,
            "anticipation_recall": self.anticipation_recall,
            "fact_accuracy": self.fact_accuracy,
            "hallucination_rate": self.hallucination_rate,
            "user_effort": self.user_effort,
            "total_coverage": self.total_coverage,
            "must_have_coverage": self.must_have_coverage,
            "push_precision": self.push_precision,
            "push_coverage": self.push_coverage,
            "intent_hit_rate": self.intent_hit_rate,
            "memory_utilization": self.memory_utilization,
            "wall_clock_seconds": self.wall_clock_seconds,
            "llm_calls": self.llm_calls,
            "llm_total_tokens": self.llm_total_tokens,
            "stored_facts": self.stored_facts,
            "processed_intents": self.processed_intents,
            "peak_queue_size": self.peak_queue_size,
            "mean_push_novelty_ratio": self.mean_push_novelty_ratio,
            "low_novelty_push_rate": self.low_novelty_push_rate,
            "mean_queue_size": self.mean_queue_size,
            "peak_queue_pressure": self.peak_queue_pressure,
            "queue_saturation_turns": self.queue_saturation_turns,
            "duplicate_topic_count": self.duplicate_topic_count,
            "duplicate_topic_rate": self.duplicate_topic_rate,
            "fsp_semantic_duplicate_count": self.fsp_semantic_duplicate_count,
            "fsp_cross_turn_duplicate_count": self.fsp_cross_turn_duplicate_count,
            "queue_semantic_duplicate_count": self.queue_semantic_duplicate_count,
            "active_llm_calls": self.active_llm_calls,
            "active_llm_total_tokens": self.active_llm_total_tokens,
            "module_llm_calls": self.module_llm_calls,
            "module_llm_total_tokens": self.module_llm_total_tokens,
            "cost_per_new_need": self.cost_per_new_need,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioMetrics":
        return cls(
            scenario_id=data["scenario_id"],
            condition=data["condition"],
            domain=data["domain"],
            t80=data["t80"],
            t100=data["t100"],
            anticipation_precision=data.get("anticipation_precision"),
            anticipation_recall=data.get("anticipation_recall"),
            fact_accuracy=data["fact_accuracy"],
            hallucination_rate=data["hallucination_rate"],
            user_effort=data["user_effort"],
            total_coverage=data["total_coverage"],
            must_have_coverage=data["must_have_coverage"],
            push_precision=data.get("push_precision"),
            push_coverage=data.get("push_coverage"),
            intent_hit_rate=data.get("intent_hit_rate"),
            memory_utilization=data.get("memory_utilization"),
            wall_clock_seconds=data.get("wall_clock_seconds", 0.0),
            llm_calls=data.get("llm_calls", 0),
            llm_total_tokens=data.get("llm_total_tokens", 0),
            stored_facts=data.get("stored_facts", 0),
            processed_intents=data.get("processed_intents", 0),
            peak_queue_size=data.get("peak_queue_size", 0),
            mean_push_novelty_ratio=data.get("mean_push_novelty_ratio"),
            low_novelty_push_rate=data.get("low_novelty_push_rate"),
            mean_queue_size=data.get("mean_queue_size", 0.0),
            peak_queue_pressure=data.get("peak_queue_pressure", 0.0),
            queue_saturation_turns=data.get("queue_saturation_turns", 0),
            duplicate_topic_count=data.get("duplicate_topic_count", 0),
            duplicate_topic_rate=data.get("duplicate_topic_rate"),
            fsp_semantic_duplicate_count=data.get("fsp_semantic_duplicate_count", 0),
            fsp_cross_turn_duplicate_count=data.get("fsp_cross_turn_duplicate_count", 0),
            queue_semantic_duplicate_count=data.get("queue_semantic_duplicate_count", 0),
            active_llm_calls=data.get("active_llm_calls", 0),
            active_llm_total_tokens=data.get("active_llm_total_tokens", 0),
            module_llm_calls=data.get("module_llm_calls", {}),
            module_llm_total_tokens=data.get("module_llm_total_tokens", {}),
            cost_per_new_need=data.get("cost_per_new_need"),
        )


@dataclass
class ConditionAggregate:
    """Aggregated metrics across all scenarios for one experimental condition."""

    condition: str
    n_scenarios: int
    mean_t80: float
    std_t80: float
    mean_t100: float
    std_t100: float
    mean_anticipation_precision: Optional[float]
    mean_anticipation_recall: Optional[float]
    mean_fact_accuracy: float
    mean_hallucination_rate: float
    mean_user_effort: float
    mean_total_coverage: float
    mean_must_have_coverage: float
    mean_push_precision: Optional[float] = None
    mean_push_coverage: Optional[float] = None
    mean_intent_hit_rate: Optional[float] = None
    mean_memory_utilization: Optional[float] = None
    mean_wall_clock_seconds: float = 0.0
    mean_llm_calls: float = 0.0
    mean_llm_total_tokens: float = 0.0
    mean_stored_facts: float = 0.0
    mean_processed_intents: float = 0.0
    mean_peak_queue_size: float = 0.0
    mean_push_novelty_ratio: Optional[float] = None
    low_novelty_push_rate: Optional[float] = None
    mean_queue_size: float = 0.0
    mean_peak_queue_pressure: float = 0.0
    mean_queue_saturation_turns: float = 0.0
    mean_duplicate_topic_count: float = 0.0
    mean_duplicate_topic_rate: Optional[float] = None
    mean_fsp_semantic_duplicate_count: float = 0.0
    mean_fsp_cross_turn_duplicate_count: float = 0.0
    mean_queue_semantic_duplicate_count: float = 0.0
    mean_active_llm_calls: float = 0.0
    mean_active_llm_total_tokens: float = 0.0
    mean_module_llm_calls: Dict[str, float] = field(default_factory=dict)
    mean_module_llm_total_tokens: Dict[str, float] = field(default_factory=dict)
    mean_cost_per_new_need: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "n_scenarios": self.n_scenarios,
            "mean_t80": self.mean_t80,
            "std_t80": self.std_t80,
            "mean_t100": self.mean_t100,
            "std_t100": self.std_t100,
            "mean_anticipation_precision": self.mean_anticipation_precision,
            "mean_anticipation_recall": self.mean_anticipation_recall,
            "mean_fact_accuracy": self.mean_fact_accuracy,
            "mean_hallucination_rate": self.mean_hallucination_rate,
            "mean_user_effort": self.mean_user_effort,
            "mean_total_coverage": self.mean_total_coverage,
            "mean_must_have_coverage": self.mean_must_have_coverage,
            "mean_push_precision": self.mean_push_precision,
            "mean_push_coverage": self.mean_push_coverage,
            "mean_intent_hit_rate": self.mean_intent_hit_rate,
            "mean_memory_utilization": self.mean_memory_utilization,
            "mean_wall_clock_seconds": self.mean_wall_clock_seconds,
            "mean_llm_calls": self.mean_llm_calls,
            "mean_llm_total_tokens": self.mean_llm_total_tokens,
            "mean_stored_facts": self.mean_stored_facts,
            "mean_processed_intents": self.mean_processed_intents,
            "mean_peak_queue_size": self.mean_peak_queue_size,
            "mean_push_novelty_ratio": self.mean_push_novelty_ratio,
            "low_novelty_push_rate": self.low_novelty_push_rate,
            "mean_queue_size": self.mean_queue_size,
            "mean_peak_queue_pressure": self.mean_peak_queue_pressure,
            "mean_queue_saturation_turns": self.mean_queue_saturation_turns,
            "mean_duplicate_topic_count": self.mean_duplicate_topic_count,
            "mean_duplicate_topic_rate": self.mean_duplicate_topic_rate,
            "mean_fsp_semantic_duplicate_count": self.mean_fsp_semantic_duplicate_count,
            "mean_fsp_cross_turn_duplicate_count": self.mean_fsp_cross_turn_duplicate_count,
            "mean_queue_semantic_duplicate_count": self.mean_queue_semantic_duplicate_count,
            "mean_active_llm_calls": self.mean_active_llm_calls,
            "mean_active_llm_total_tokens": self.mean_active_llm_total_tokens,
            "mean_module_llm_calls": self.mean_module_llm_calls,
            "mean_module_llm_total_tokens": self.mean_module_llm_total_tokens,
            "mean_cost_per_new_need": self.mean_cost_per_new_need,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionAggregate":
        return cls(
            condition=data["condition"],
            n_scenarios=data["n_scenarios"],
            mean_t80=data["mean_t80"],
            std_t80=data["std_t80"],
            mean_t100=data["mean_t100"],
            std_t100=data["std_t100"],
            mean_anticipation_precision=data.get("mean_anticipation_precision"),
            mean_anticipation_recall=data.get("mean_anticipation_recall"),
            mean_fact_accuracy=data["mean_fact_accuracy"],
            mean_hallucination_rate=data["mean_hallucination_rate"],
            mean_user_effort=data["mean_user_effort"],
            mean_total_coverage=data["mean_total_coverage"],
            mean_must_have_coverage=data["mean_must_have_coverage"],
            mean_push_precision=data.get("mean_push_precision"),
            mean_push_coverage=data.get("mean_push_coverage"),
            mean_intent_hit_rate=data.get("mean_intent_hit_rate"),
            mean_memory_utilization=data.get("mean_memory_utilization"),
            mean_wall_clock_seconds=data.get("mean_wall_clock_seconds", 0.0),
            mean_llm_calls=data.get("mean_llm_calls", 0.0),
            mean_llm_total_tokens=data.get("mean_llm_total_tokens", 0.0),
            mean_stored_facts=data.get("mean_stored_facts", 0.0),
            mean_processed_intents=data.get("mean_processed_intents", 0.0),
            mean_peak_queue_size=data.get("mean_peak_queue_size", 0.0),
            mean_push_novelty_ratio=data.get("mean_push_novelty_ratio"),
            low_novelty_push_rate=data.get("low_novelty_push_rate"),
            mean_queue_size=data.get("mean_queue_size", 0.0),
            mean_peak_queue_pressure=data.get("mean_peak_queue_pressure", 0.0),
            mean_queue_saturation_turns=data.get("mean_queue_saturation_turns", 0.0),
            mean_duplicate_topic_count=data.get("mean_duplicate_topic_count", 0.0),
            mean_duplicate_topic_rate=data.get("mean_duplicate_topic_rate"),
            mean_fsp_semantic_duplicate_count=data.get("mean_fsp_semantic_duplicate_count", 0.0),
            mean_fsp_cross_turn_duplicate_count=data.get("mean_fsp_cross_turn_duplicate_count", 0.0),
            mean_queue_semantic_duplicate_count=data.get("mean_queue_semantic_duplicate_count", 0.0),
            mean_active_llm_calls=data.get("mean_active_llm_calls", 0.0),
            mean_active_llm_total_tokens=data.get("mean_active_llm_total_tokens", 0.0),
            mean_module_llm_calls=data.get("mean_module_llm_calls", {}),
            mean_module_llm_total_tokens=data.get("mean_module_llm_total_tokens", {}),
            mean_cost_per_new_need=data.get("mean_cost_per_new_need"),
        )


# ---------------------------------------------------------------------------
# Top-level evaluation report
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Final evaluation report with multi-granularity metrics and statistical tests."""

    per_scenario: List[ScenarioMetrics]
    per_condition: Dict[str, ConditionAggregate]
    per_domain: Dict[str, Dict[str, ConditionAggregate]]
    statistical_tests: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_scenario": [s.to_dict() for s in self.per_scenario],
            "per_condition": {
                k: v.to_dict() for k, v in self.per_condition.items()
            },
            "per_domain": {
                domain: {cond: agg.to_dict() for cond, agg in conds.items()}
                for domain, conds in self.per_domain.items()
            },
            "statistical_tests": self.statistical_tests,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalReport":
        return cls(
            per_scenario=[
                ScenarioMetrics.from_dict(s) for s in data["per_scenario"]
            ],
            per_condition={
                k: ConditionAggregate.from_dict(v)
                for k, v in data["per_condition"].items()
            },
            per_domain={
                domain: {
                    cond: ConditionAggregate.from_dict(agg)
                    for cond, agg in conds.items()
                }
                for domain, conds in data["per_domain"].items()
            },
            statistical_tests=data.get("statistical_tests", {}),
        )
