"""
ProactiveBench system adapter.

Wraps ChatHandler for benchmark evaluation under four ablation conditions:
  Full      -- FSP + ITE enabled (directed exploration + execution)
  Blind     -- ITE with random intents (no FSP direction)
  Paralyzed -- FSP generates intents but ITE disabled (no execution)
  Baseline  -- No FSP, no ITE (pure reactive search)

All conditions use ExternalFactStore for facts (system memory starts empty)
and BenchmarkSearchTool/BenchmarkWebFetcher for SearchService integration.
"""

import copy
import logging
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.assistant_agent import AssistantAgent
from agents.need_predictor import ConversationNeedPredictor
from agents.planning_agent import PlanningAgent
from agents.push_score import PushScoreAgent
from agents.report_generator import ReportGeneratorAgent
from agents.search_planner import SearchPlannerAgent
from agents.topic_monitor import TopicMonitorAgent
from core.config import Config
from core.models import KnowledgeItem, SearchSource
from experiments.ProactiveBench.eval.ablation_config import (
    AblationConfig,
    DedupPriorityQueue,
    ExplorationIntent,
    VALID_CONDITIONS,
)
from experiments.ProactiveBench.eval.benchmark_search import (
    BenchmarkSearchTool,
    BenchmarkWebFetcher,
)
from experiments.ProactiveBench.eval.external_fact_store import ExternalFactStore
from experiments.ProactiveBench.eval.fsp_module import BenchmarkFSP
from experiments.ProactiveBench.eval.ite_module import BenchmarkITE, BlindExplorationFallback
from memory.memory_system import MemorySystem
from server.handlers.chat_handler import ChatHandler
from services.search_service import SearchService
from services.proactive import (
    ProactiveBriefService,
    ProactiveDecisionService,
    ProactiveEligibilityGate,
    ProactiveItemService,
    ProactivePolicy,
    build_turn_signal_extractor,
)
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BenchmarkDeps -- lightweight stand-in for server.dependencies.UserDependencies
# ---------------------------------------------------------------------------

class BenchmarkDeps:
    """
    Minimal dependency container that satisfies ChatHandler's contract.

    Accepts injected search_tool and web_fetcher so that SearchService routes
    queries through the benchmark's ExternalFactStore.
    """

    def __init__(
        self,
        user_id: str,
        config: Config,
        llm_client: LLMClient,
        memory: MemorySystem,
        search_tool,
        web_fetcher,
    ):
        self.user_id = user_id
        self._config = config
        self._llm = llm_client
        self.memory = memory

        self.last_interaction: datetime = datetime.now()

        self.topic_monitor = TopicMonitorAgent(
            config=config.agents.topic_monitor,
            llm_client=llm_client,
        )
        self.planning_agent = PlanningAgent(
            config=config.agents.planning,
            llm_client=llm_client,
            memory_system=memory,
        )
        self.assistant_agent = AssistantAgent(
            config=config.agents.assistant,
            llm_client=llm_client,
            memory_system=memory,
        )
        self.need_predictor = ConversationNeedPredictor(
            config=config.agents.need_predictor,
            llm_client=llm_client,
            memory_system=memory,
        )

        search_planner = SearchPlannerAgent(
            config=config.agents.search_planner,
            llm_client=llm_client,
        )
        report_generator = ReportGeneratorAgent(
            config=config.agents.report_generator,
            llm_client=llm_client,
        )
        self.search_service = SearchService(
            memory_system=memory,
            search_planner=search_planner,
            report_generator=report_generator,
            search_tool=search_tool,
            web_fetcher=web_fetcher,
            llm_client=llm_client,
            config=config,
        )

        self.deep_research_orchestrator = None

        push_score_agent = PushScoreAgent(
            config=config.agents.push_score,
            llm_client=llm_client,
        )
        self.proactive_policy = ProactivePolicy(config.proactive.mode)
        self.proactive_turn_signal_extractor = build_turn_signal_extractor(
            config=config.proactive,
            llm_client=llm_client,
        )
        self.proactive_eligibility_gate = ProactiveEligibilityGate(config.proactive)
        self.proactive_brief_service = ProactiveBriefService()
        self.proactive_decision_service = ProactiveDecisionService(
            config=config.proactive,
            push_score_agent=push_score_agent,
        )
        self.proactive_item_service = ProactiveItemService(
            user_id=user_id,
            storage_config=config.storage,
            include_delivered_in_active=False,
        )

    def record_interaction(self) -> None:
        pass

    def get_interaction_stats(self) -> Dict[str, int]:
        return {
            "seconds_since_last_interaction": 0,
            "recent_interaction_count": 0,
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sanitize_user_id_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value.lower())


def _make_user_id(
    scenario_id: str,
    condition: str,
    seed: int,
    run_id_suffix: str = "",
) -> str:
    raw = f"pb_{scenario_id}_{condition}_{seed}"
    sanitized = _sanitize_user_id_part(raw)
    suffix = _sanitize_user_id_part(run_id_suffix.strip()) if run_id_suffix else ""
    if not suffix:
        return sanitized[:63]

    suffix_part = f"_{suffix}"[:63]
    if len(suffix_part) >= 63:
        return suffix_part[-63:]
    base_len = 63 - len(suffix_part)
    return f"{sanitized[:base_len]}{suffix_part}"


# ---------------------------------------------------------------------------
# ProactiveBenchAdapter
# ---------------------------------------------------------------------------

class ProactiveBenchAdapter:
    """
    Adapter that translates ProactiveBench evaluation protocol into
    ChatHandler calls with FSP/ITE integration.

    Usage::

        adapter = ProactiveBenchAdapter("Full", llm_client)
        adapter.reset(scenario)
        abl_result = adapter.run_ablation_turn(1)
        chat_result = await adapter.send_message("Hi, I need help with ...")
    """

    def __init__(
        self,
        condition: str,
        llm_client: LLMClient,
        seed: int = 42,
        max_queries_per_search: int = 1,
        max_intents_per_idle: int = 3,
        idle_trigger_seconds: float = 5.0,
        max_idle_ticks_per_window: int = 4,
        max_direct_pushes_per_idle_window: int = 2,
        min_idle_tick_interval_seconds: float = 3.0,
        max_intents_per_idle_window: int = 6,
        stop_on_consecutive_no_new_facts: int = 2,
        cold_start_turns: int = 2,
        cold_start_max_intents: int = 1,
        min_anchor_tokens_for_fsp: int = 6,
        cross_turn_duplicate_jaccard: float = 0.6,
        max_total_searches: int = 999,
        run_id_suffix: str = "",
        cost_tracker: Any = None,
        proactive_variant: str = "topic_first",
    ):
        if condition not in VALID_CONDITIONS:
            raise ValueError(
                f"Invalid condition {condition!r}. "
                f"Must be one of {sorted(VALID_CONDITIONS)}."
            )
        if max_queries_per_search < 1:
            raise ValueError("max_queries_per_search must be >= 1")
        if max_intents_per_idle < 0:
            raise ValueError("max_intents_per_idle must be >= 0")
        if idle_trigger_seconds < 0:
            raise ValueError("idle_trigger_seconds must be >= 0")
        if max_idle_ticks_per_window < 1:
            raise ValueError("max_idle_ticks_per_window must be >= 1")
        if max_direct_pushes_per_idle_window < 0:
            raise ValueError("max_direct_pushes_per_idle_window must be >= 0")
        if min_idle_tick_interval_seconds < 0:
            raise ValueError("min_idle_tick_interval_seconds must be >= 0")
        if max_intents_per_idle_window < 0:
            raise ValueError("max_intents_per_idle_window must be >= 0")
        if stop_on_consecutive_no_new_facts < 1:
            raise ValueError("stop_on_consecutive_no_new_facts must be >= 1")
        if max_total_searches < 0:
            raise ValueError("max_total_searches must be >= 0")
        self.condition = condition
        self.llm = llm_client
        self.seed = seed
        self.max_queries_per_search = int(max_queries_per_search)
        self.max_intents_per_idle = int(max_intents_per_idle)
        self.idle_trigger_seconds = float(idle_trigger_seconds)
        self.max_idle_ticks_per_window = int(max_idle_ticks_per_window)
        self.max_direct_pushes_per_idle_window = int(max_direct_pushes_per_idle_window)
        self.min_idle_tick_interval_seconds = float(min_idle_tick_interval_seconds)
        self.max_intents_per_idle_window = int(max_intents_per_idle_window)
        self.stop_on_consecutive_no_new_facts = int(stop_on_consecutive_no_new_facts)
        self.cold_start_turns = int(cold_start_turns)
        self.cold_start_max_intents = int(cold_start_max_intents)
        self.min_anchor_tokens_for_fsp = int(min_anchor_tokens_for_fsp)
        self.cross_turn_duplicate_jaccard = float(cross_turn_duplicate_jaccard)
        self.max_total_searches = int(max_total_searches)
        self.run_id_suffix = run_id_suffix
        self.cost_tracker = cost_tracker
        self.proactive_variant = proactive_variant

        self.user_id: Optional[str] = None
        self.config: Optional[Config] = None
        self.memory: Optional[MemorySystem] = None
        self.deps: Optional[BenchmarkDeps] = None
        self.chat_handler: Optional[ChatHandler] = None

        self.ablation_config: Optional[AblationConfig] = None
        self.external_fact_store: Optional[ExternalFactStore] = None
        self.fsp: Optional[BenchmarkFSP] = None
        self.ite: Optional[BenchmarkITE] = None
        self.intent_queue: Optional[DedupPriorityQueue] = None
        self.blind_fallback: Optional[BlindExplorationFallback] = None
        self._idle_rng: Optional[random.Random] = None
        self._total_searches_used = 0

    # ------------------------------------------------------------------
    # reset -- called once per scenario
    # ------------------------------------------------------------------

    def reset(self, scenario: "Scenario") -> None:  # noqa: F821
        """
        Prepare a fresh environment for *scenario*.

        1. Derive deterministic user_id.
        2. Create isolated Config with proactive prediction disabled.
        3. Create empty MemorySystem.
        4. Load facts into ExternalFactStore (NOT system memory).
        5. Wire BenchmarkDeps with BenchmarkSearchTool/WebFetcher.
        6. Create FSP/ITE modules based on ablation config.
        """
        from experiments.ProactiveBench.generation.models import Scenario

        if not isinstance(scenario, Scenario):
            raise TypeError(f"Expected Scenario, got {type(scenario).__name__}")

        self.ablation_config = AblationConfig.from_condition(self.condition)
        self.ablation_config.max_total_searches = self.max_total_searches
        self._total_searches_used = 0

        self.user_id = _make_user_id(
            scenario.scenario_id,
            self.condition,
            self.seed,
            self.run_id_suffix,
        )
        self.config = self._build_config()

        user_dir = self.config.storage.get_user_dir(self.user_id)
        if user_dir.exists():
            shutil.rmtree(user_dir)

        # Empty memory system — facts go to ExternalFactStore, not here
        self.memory = MemorySystem(self.user_id, self.config, self.llm)

        # External fact store
        self.external_fact_store = ExternalFactStore(scenario.scenario_id)
        self.external_fact_store.load_facts(scenario.fact_sheet, scenario.domain)

        # Set topic and load profile
        self.memory.set_current_topic(scenario.domain)
        self._load_user_profile(scenario)
        self._load_legacy_route_facts(scenario)

        # Wire deps with benchmark search tools
        search_tool = BenchmarkSearchTool(self.external_fact_store)
        web_fetcher = BenchmarkWebFetcher(self.external_fact_store)
        self.deps = BenchmarkDeps(
            user_id=self.user_id,
            config=self.config,
            llm_client=self.llm,
            memory=self.memory,
            search_tool=search_tool,
            web_fetcher=web_fetcher,
        )
        self.chat_handler = ChatHandler(self.deps)

        # FSP module
        if self.ablation_config.fsp_enabled:
            self.fsp = BenchmarkFSP(self.memory, self.llm, self.config)
            self.fsp.cold_start_turns = self.cold_start_turns
            self.fsp.cold_start_max_intents = self.cold_start_max_intents
            self.fsp.min_anchor_tokens_for_fsp = self.min_anchor_tokens_for_fsp
            self.fsp.cross_turn_duplicate_jaccard = self.cross_turn_duplicate_jaccard
        else:
            self.fsp = None

        # ITE module
        if self.ablation_config.ite_enabled:
            push_agent = PushScoreAgent(
                config=self.config.agents.push_score,
                llm_client=self.llm,
            )
            self.ite = BenchmarkITE(
                self.deps.search_service,
                push_agent,
                self.memory,
                cost_tracker=self.cost_tracker,
            )
        else:
            self.ite = None

        self._configure_legacy_route_c_predictor(scenario)

        # Blind fallback (Blind condition: ITE enabled but FSP disabled)
        if self.ablation_config.ite_enabled and not self.ablation_config.fsp_enabled:
            self.blind_fallback = BlindExplorationFallback(seed=self.seed)
        else:
            self.blind_fallback = None

        self.intent_queue = DedupPriorityQueue()
        self._idle_rng = random.Random(self.seed)

        logger.info(
            "Reset adapter: scenario=%s condition=%s fsp=%s ite=%s facts=%d",
            scenario.scenario_id, self.condition,
            self.ablation_config.fsp_enabled, self.ablation_config.ite_enabled,
            self.external_fact_store.get_fact_count(),
        )

    # ------------------------------------------------------------------
    # run_ablation_turn -- FSP + ITE per turn, called BEFORE send_message
    # ------------------------------------------------------------------

    def run_ablation_turn(self, turn: int) -> Dict[str, Any]:
        """
        Execute FSP and ITE phases for this turn.

        Returns a dict with exploration metadata and optional push content.
        """
        if self.memory is None or self.intent_queue is None or self._idle_rng is None:
            raise RuntimeError("Adapter not initialised. Call reset() first.")

        # Simulated idle time: first 2 turns short (user typing), then 5-10s
        if turn <= 2:
            idle_seconds = self._idle_rng.uniform(1.0, 3.0)
        else:
            idle_seconds = self._idle_rng.uniform(5.0, 10.0)

        if getattr(self, "ablation_config", None) is None:
            self.ablation_config = AblationConfig.from_condition(
                getattr(self, "condition", "Full")
            )
        search_budget = int(getattr(
            self,
            "max_total_searches",
            getattr(self.ablation_config, "max_total_searches", 999),
        ))
        self.ablation_config.max_total_searches = search_budget
        if not hasattr(self, "_total_searches_used"):
            self._total_searches_used = 0

        queue_size_before = len(self.intent_queue)

        # Phase 0: FSP or Blind intent generation
        intents: List[ExplorationIntent] = []
        if self.fsp is not None:
            try:
                profile = self.memory.get_user_profile()
                profile_text = getattr(profile, "to_prompt_text", lambda: str(profile))()
                history_text = self.memory.get_current_messages_formatted()
                cost_tracker = getattr(self, "cost_tracker", None)
                if cost_tracker is not None:
                    with cost_tracker.track("fsp"):
                        intents = self.fsp.generate_intents(
                            history_text,
                            profile_text,
                            turn,
                            queue_size=len(self.intent_queue),
                            max_queue_size=getattr(self.intent_queue, "MAX_SIZE", 20),
                        )
                else:
                    intents = self.fsp.generate_intents(
                        history_text,
                        profile_text,
                        turn,
                        queue_size=len(self.intent_queue),
                        max_queue_size=getattr(self.intent_queue, "MAX_SIZE", 20),
                    )
            except Exception:
                logger.warning("FSP generate_intents failed", exc_info=True)
            self.intent_queue.clear_drop_reasons()
            self.intent_queue.push_all(intents)
        elif self.blind_fallback is not None:
            intents = self.blind_fallback.generate_random_intents(
                count=3, current_turn=turn,
            )
            self.intent_queue.clear_drop_reasons()
            self.intent_queue.push_all(intents)

        queue_size_after_generation = len(self.intent_queue)

        # Phase 1: ITE exploration (gated by simulated idle time >= threshold)
        tick_results: List[Dict[str, Any]] = []
        push_events: List[Dict[str, Any]] = []
        ite_ran = False
        if (
            self.ite is not None
            and self.max_intents_per_idle > 0
            and idle_seconds >= self.idle_trigger_seconds
        ):
            allowed_ticks = self._allowed_idle_ticks(idle_seconds)
            if not self.ablation_config.idle_window_mode:
                allowed_ticks = min(allowed_ticks, 1)

            processed_intents = 0
            direct_pushes = 0
            consecutive_no_new_facts = 0

            for tick_index in range(1, allowed_ticks + 1):
                if self.intent_queue.is_empty():
                    break
                max_window_intents = getattr(
                    self,
                    "max_intents_per_idle_window",
                    self.max_intents_per_idle,
                )
                if processed_intents >= max_window_intents:
                    break
                if direct_pushes >= getattr(self, "max_direct_pushes_per_idle_window", 2):
                    break
                if consecutive_no_new_facts >= getattr(self, "stop_on_consecutive_no_new_facts", 2):
                    break

                max_intents = min(
                    self.max_intents_per_idle,
                    max_window_intents - processed_intents,
                )
                if max_intents <= 0:
                    break
                remaining_search_budget = search_budget - self._total_searches_used
                if remaining_search_budget <= 0:
                    break
                max_intents = min(max_intents, remaining_search_budget)

                try:
                    exploration = self.ite.explore(
                        self.intent_queue,
                        max_intents=max_intents,
                        seconds_since_last_interaction=int(idle_seconds),
                    )
                    ite_ran = True
                except Exception:
                    logger.warning("ITE explore failed", exc_info=True)
                    break

                searches_used = int(exploration.get("intents_processed", 0) or 0)
                processed_intents += searches_used
                self._total_searches_used += searches_used
                tick_event_count = len(exploration.get("push_events", []) or [])
                direct_pushes += tick_event_count
                if int(exploration.get("facts_stored", 0) or 0) <= 0:
                    consecutive_no_new_facts += 1
                else:
                    consecutive_no_new_facts = 0

                tick_result = {
                    **exploration,
                    "tick_index": tick_index,
                    "queue_size_after_tick": len(self.intent_queue),
                    "search_budget": search_budget,
                    "total_searches_used": self._total_searches_used,
                    "remaining_search_budget": max(
                        0,
                        search_budget - self._total_searches_used,
                    ),
                }
                tick_results.append(tick_result)
                for event in exploration.get("push_events", []) or []:
                    push_events.append({
                        **event,
                        "tick_index": tick_index,
                    })

        first_tick = tick_results[0] if tick_results else {}
        first_push = push_events[0] if push_events else {}
        fsp_filter_reasons = (
            list(getattr(self.fsp, "last_filter_reasons", []))
            if self.fsp is not None
            else []
        )
        fsp_filter_reasons.extend(getattr(self.intent_queue, "drop_reasons", []))

        return {
            "exploration_intents": [i.to_dict() for i in intents],
            "fsp_filter_reasons": fsp_filter_reasons,
            "simulated_idle_seconds": idle_seconds,
            "idle_trigger_seconds": self.idle_trigger_seconds,
            "max_intents_per_idle": self.max_intents_per_idle,
            "max_queries_per_search": self.max_queries_per_search,
            "search_budget": search_budget,
            "total_searches_used": self._total_searches_used,
            "remaining_search_budget": max(0, search_budget - self._total_searches_used),
            "ite_ran": ite_ran,
            "intent_queue_size": len(self.intent_queue),
            "idle_window_mode": self.ablation_config.idle_window_mode,
            "idle_tick_count": len(tick_results),
            "idle_tick_results": tick_results,
            "push_events": push_events,
            "queue_size_before": queue_size_before,
            "queue_size_after_generation": queue_size_after_generation,
            "queue_size_after_exploration": len(self.intent_queue),
            "generated_intents": int(getattr(self.fsp, "last_generated_count", len(intents))),
            "accepted_intents": int(getattr(self.fsp, "last_accepted_count", len(intents))),
            "filtered_intents": int(getattr(self.fsp, "last_filtered_count", 0)),
            "duplicate_intents": int(getattr(self.fsp, "last_duplicate_count", 0)),
            "has_direct_push": bool(first_push),
            "push_content": first_push.get("push_content"),
            "push_topic": first_push.get("push_topic"),
            "push_fact_ids": first_push.get("push_fact_ids", []),
            "push_novelty_ratio": first_push.get("push_novelty_ratio"),
            "push_undelivered_fact_ids": first_push.get("push_undelivered_fact_ids", []),
            "push_delivered_fact_ids": first_push.get("push_delivered_fact_ids", []),
            "facts_stored": sum(int(t.get("facts_stored", 0) or 0) for t in tick_results),
            "stored_fact_ids": sorted({
                str(fact_id)
                for tick in tick_results
                for fact_id in tick.get("stored_fact_ids", [])
            }),
            "intent_results": first_tick.get("intent_results", []),
            "inline_topics": sorted({
                str(topic)
                for tick in tick_results
                for topic in tick.get("inline_topics", [])
            }),
        }

    def _allowed_idle_ticks(self, idle_seconds: float) -> int:
        if idle_seconds < self.idle_trigger_seconds:
            return 0
        max_ticks = getattr(self, "max_idle_ticks_per_window", 1)
        interval = getattr(self, "min_idle_tick_interval_seconds", 3.0)
        if interval <= 0:
            return max_ticks
        extra = idle_seconds - self.idle_trigger_seconds
        ticks = 1 + int(extra // interval)
        return max(0, min(max_ticks, ticks))

    # ------------------------------------------------------------------
    # send_message -- called once per conversational turn
    # ------------------------------------------------------------------

    def record_direct_push(self, push_content: str) -> None:
        """Record a direct push in the active memory session."""
        if self.memory is None:
            raise RuntimeError("Adapter not initialised. Call reset() first.")
        if push_content:
            self.memory.add_message("assistant", push_content)

    async def send_message(self, message: str) -> Dict[str, Any]:
        """
        Send a user message through ChatHandler.

        In the new ablation conditions, in-conversation proactive prediction
        is disabled. The system uses whatever knowledge ITE has stored in
        memory, plus reactive search via BenchmarkSearchTool.
        """
        if self.chat_handler is None or self.memory is None:
            raise RuntimeError("Adapter not initialised. Call reset() first.")

        chat_result = await self.chat_handler.handle(message)

        memory_context = ""
        try:
            memory_context = self.memory.format_context_for_query("")
        except Exception:
            pass

        proactive_trace = self._last_proactive_trace()
        predictor_trace = self._predictor_trace()
        delivered_inline = proactive_trace.get("delivered_inline", [])
        self._mark_delivered_inline_facts(delivered_inline)

        return {
            "agent_response": chat_result.reply,
            "proactive_predictions": proactive_trace.get("predictions", []),
            "proactive_approved": proactive_trace.get("approved", []),
            "topic": chat_result.topic,
            "topic_changed": chat_result.topic_changed,
            "search_performed": chat_result.search_performed,
            "memory_context": memory_context,
            "fact_scores": predictor_trace.get("fact_scores", []),
            "selected_facts": predictor_trace.get("selected_facts", []),
            "fact_scoring_rationales": predictor_trace.get(
                "fact_scoring_rationales", {}
            ),
            "delivered_inline_facts": delivered_inline,
            "admission_trace": self._admission_trace(),
            "decision_trace": self._decision_trace(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_config(self) -> Config:
        cfg = copy.deepcopy(Config())

        cfg.storage.data_dir = (
            Path(__file__).resolve().parents[1] / "runtime"
        )

        # Proactive prediction disabled for all new conditions —
        # FSP/ITE handle proactive behavior externally.
        cfg.proactive.in_conversation_enabled = False
        cfg.proactive.mode = "need_only"
        cfg.proactive.turn_signal_backend = "rule"
        cfg.proactive.enable_llm_turn_signals_in_need_only = False
        cfg.proactive.inline_same_topic_cooldown_turns = 0
        cfg.proactive.max_active_items_per_topic = 5
        cfg.proactive.max_active_items_per_user = 10

        cfg.deep_research.enabled = False
        cfg.search.max_queries_per_search = self.max_queries_per_search

        cfg.agents.set_seed_all(self.seed)

        return cfg

    def _configure_legacy_route_c_predictor(self, scenario: "Scenario") -> None:  # noqa: F821
        if self.deps is None or self.memory is None:
            return
        if self.condition != "C2":
            return
        try:
            if self.proactive_variant == "oracle_fact":
                from experiments.ProactiveBench.eval.fact_predictor import (
                    BenchmarkFactGroundedPredictor,
                )

                self.deps.need_predictor = BenchmarkFactGroundedPredictor(
                    scenario.fact_sheet
                )
            else:
                from experiments.ProactiveBench.eval.topic_predictor import (
                    BenchmarkTopicFirstPredictor,
                )

                self.deps.need_predictor = BenchmarkTopicFirstPredictor(
                    memory_system=self.memory,
                    llm_client=self.llm,
                )
        except Exception:
            logger.warning("Failed to configure legacy C2 predictor", exc_info=True)

    def _last_proactive_trace(self) -> Dict[str, Any]:
        if self.chat_handler is None:
            return {}
        trace = getattr(self.chat_handler, "_last_proactive_trace", None)
        return trace if isinstance(trace, dict) else {}

    def _predictor_trace(self) -> Dict[str, Any]:
        if self.deps is None:
            return {}
        trace = getattr(getattr(self.deps, "need_predictor", None), "last_trace", None)
        return trace if isinstance(trace, dict) else {}

    def _admission_trace(self) -> Optional[Dict[str, Any]]:
        trace = self._last_proactive_trace()
        predictor_trace = self._predictor_trace()
        if not trace and not predictor_trace:
            return None
        if trace.get("delivered_inline"):
            stage = "delivered_inline"
        elif trace.get("approved"):
            stage = "approved"
        elif trace.get("predictions"):
            stage = "predicted"
        else:
            stage = "none"
        return {
            "stage": stage,
            "predictor_trace": predictor_trace,
        }

    def _decision_trace(self) -> Optional[Dict[str, Any]]:
        trace = self._last_proactive_trace()
        if not trace:
            return None
        return {
            "summary": {
                "candidate_count": len(trace.get("predictions", []) or []),
                "approved_count": len(trace.get("approved", []) or []),
                "delivered_count": len(trace.get("delivered_inline", []) or []),
            }
        }

    def _mark_delivered_inline_facts(self, delivered_inline: List[Dict[str, Any]]) -> None:
        if self.deps is None:
            return
        predictor = getattr(self.deps, "need_predictor", None)
        marker = getattr(predictor, "mark_conveyed", None)
        if not callable(marker):
            return
        fact_ids = sorted({
            fact_id
            for item in delivered_inline or []
            if isinstance(item, dict)
            for fact_id in self._delivered_fact_ids_from_item(item)
        })
        if fact_ids:
            marker(fact_ids)

    @staticmethod
    def _delivered_fact_ids_from_item(item: Dict[str, Any]) -> set[str]:
        explicit = {
            str(fact_id).strip().upper()
            for fact_id in item.get("delivered_fact_ids", [])
            if str(fact_id).strip()
        }
        if explicit:
            return explicit
        lookup = str(item.get("lookup_query", "")).strip()
        factset_match = re.match(r"^factset:([A-Za-z0-9_, -]+)$", lookup, flags=re.IGNORECASE)
        if factset_match:
            return {
                fact_id.strip().upper()
                for fact_id in factset_match.group(1).split(",")
                if fact_id.strip()
            }
        fact_match = re.match(r"^fact:([A-Za-z0-9_-]+)$", lookup, flags=re.IGNORECASE)
        if fact_match:
            return {fact_match.group(1).strip().upper()}
        return set()

    def _load_user_profile(self, scenario: "Scenario") -> None:  # noqa: F821
        if self.memory is None:
            raise RuntimeError("MemorySystem must be initialised before loading profile.")

        profile = self.memory.get_user_profile()
        sp = scenario.user_profile

        profile.role = sp.persona
        profile.traits = [sp.context]
        profile.communication_style = [sp.communication_style]

        self.memory._profile_store.save(profile)

        logger.info(
            "Loaded user profile for user=%s condition=%s: role=%s",
            self.user_id, self.condition, sp.persona[:60],
        )

    def _load_legacy_route_facts(self, scenario: "Scenario") -> None:  # noqa: F821
        if self.memory is None or self.condition not in {"C1", "C2"}:
            return
        fact_lookup = {
            fact.id.upper(): KnowledgeItem(
                id=fact.id,
                content=f"{fact.id}: {fact.fact}",
                summary=fact.fact,
                topic=scenario.domain,
                source_url=f"extfact://{fact.id}",
                title=f"[{fact.category}] {fact.id}",
            )
            for fact in scenario.fact_sheet
        }
        base_search = self.memory.search_knowledge

        def _legacy_search(query: str, n_results: int = 5):
            requested = self._legacy_fact_ids_from_query(query)
            if requested:
                return [
                    fact_lookup[fact_id]
                    for fact_id in requested
                    if fact_id in fact_lookup
                ][:n_results]
            return base_search(query, n_results)

        self.memory.search_knowledge = _legacy_search  # type: ignore[method-assign]

    @staticmethod
    def _legacy_fact_ids_from_query(query: str) -> List[str]:
        query = (query or "").strip()
        factset_match = re.match(r"^factset:([A-Za-z0-9_, -]+)$", query, flags=re.IGNORECASE)
        if factset_match:
            return [
                fact_id.strip().upper()
                for fact_id in factset_match.group(1).split(",")
                if fact_id.strip()
            ]
        fact_match = re.match(r"^fact:([A-Za-z0-9_-]+)$", query, flags=re.IGNORECASE)
        if fact_match:
            return [fact_match.group(1).strip().upper()]
        return []
