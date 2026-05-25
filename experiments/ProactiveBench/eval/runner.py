"""
ProactiveBench experiment runner.

Drives the scenario x condition evaluation loop, coordinating the adapter,
user simulator, and coverage judge to produce ConversationResult traces.

Usage:
    python -m experiments.ProactiveBench.eval.runner \
        --scenario-dir ../data/scenarios/ \
        --output-dir ../results/ \
        --conditions Full Blind Paralyzed Baseline \
        --seed 42
"""

import argparse
import asyncio
import inspect
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Project root on sys.path so that `from tools.llm_client import ...` works
# when invoked as `python -m experiments.ProactiveBench.eval.runner`.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.eval.adapter import ProactiveBenchAdapter
from experiments.ProactiveBench.eval.cost_tracking import LLMCostTracker, snapshot_usage, usage_delta
from experiments.ProactiveBench.eval.coverage_judge import CoverageJudge
from experiments.ProactiveBench.eval.models import ConversationResult, TurnTrace
from experiments.ProactiveBench.eval.user_simulator import BenchmarkUserSimulator
from experiments.ProactiveBench.generation.models import Scenario
from tools.llm_client import LLMClient
from core.config import config as app_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Configuration for a ProactiveBench evaluation run."""

    scenario_dir: str = "../data/scenarios"       # relative to eval/
    output_dir: str = "../results"                 # relative to eval/
    conditions: List[str] = field(
        default_factory=lambda: ["Full", "Blind", "Paralyzed", "Baseline"],
    )
    seed: int = 42
    judge_model: str = "gpt-4o-mini"
    simulator_model: str = "gpt-4o"
    max_turns_override: Optional[int] = None
    max_queries_per_search: int = 1
    max_intents_per_idle: int = 3
    idle_trigger_seconds: float = 5.0
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
    run_id_suffix: str = ""
    checkpoint_file: str = "checkpoint.json"
    scenarios: Optional[List[str]] = None          # filter by scenario_id


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ProactiveBenchRunner:
    """
    Main experiment runner.

    Iterates over every (scenario, condition) pair, orchestrates the
    adapter / simulator / judge pipeline, and persists results with a
    resumable checkpoint system.
    """

    def __init__(
        self,
        config: RunnerConfig,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.config = config

        # Resolve paths relative to this file when they look relative.
        eval_dir = Path(__file__).resolve().parent
        self.scenario_dir = self._resolve_path(config.scenario_dir, eval_dir)
        self.output_dir = self._resolve_path(config.output_dir, eval_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm = llm_client or LLMClient(app_config.llm)

        # Checkpoint paths
        self.checkpoint_path = self.output_dir / config.checkpoint_file
        self.detailed_results_path = self.output_dir / "detailed_results.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> List[ConversationResult]:
        """
        Execute the full evaluation matrix (scenario x condition).

        Returns:
            All ConversationResult objects (including previously checkpointed).
        """
        scenarios = self._load_scenarios()
        if not scenarios:
            logger.warning("No scenarios found in %s", self.scenario_dir)
            return []

        logger.info(
            "Loaded %d scenario(s), conditions=%s",
            len(scenarios),
            self.config.conditions,
        )

        # Load checkpoint state
        completed_keys, results = self._load_checkpoint()
        if completed_keys:
            logger.info("Resuming: %d completed keys in checkpoint", len(completed_keys))

        total_pairs = len(scenarios) * len(self.config.conditions)
        done_count = len(completed_keys)

        for scenario in scenarios:
            for condition in self.config.conditions:
                key = f"{scenario.scenario_id}__{condition}"
                if key in completed_keys:
                    continue

                done_count += 1
                logger.info(
                    "[%d/%d] Running scenario=%s condition=%s",
                    done_count,
                    total_pairs,
                    scenario.scenario_id,
                    condition,
                )

                result = await self.run_single(scenario, condition)
                results.append(result)

                # Persist checkpoint immediately after each completion
                completed_keys.add(key)
                self._save_checkpoint(completed_keys)
                self._save_detailed_results(results)

                logger.info(
                    "  Completed: %d turns, covered %d/%d needs",
                    result.total_turns,
                    len(result.final_covered_needs),
                    len(result.all_need_ids),
                )

        # Final save
        self._save_detailed_results(results)
        logger.info(
            "Evaluation finished. %d result(s) saved to %s",
            len(results),
            self.detailed_results_path,
        )
        return results

    async def run_single(
        self,
        scenario: Scenario,
        condition: str,
    ) -> ConversationResult:
        """
        Run a single scenario under a given experimental condition.

        Loop structure per turn:
          Phase 0+1: FSP/Blind intent generation + ITE exploration
          Phase 2:   Direct push delivery (if ITE found high-value content)
          Phase 3:   User message generation + system reply + judge evaluation
        """
        start_time = time.perf_counter()
        usage_before = snapshot_usage(self.llm)
        cost_tracker = LLMCostTracker(self.llm, initial_snapshot=usage_before)

        adapter = self._build_adapter(
            condition,
            self.llm,
            seed=self.config.seed,
            max_queries_per_search=self.config.max_queries_per_search,
            max_intents_per_idle=self.config.max_intents_per_idle,
            idle_trigger_seconds=self.config.idle_trigger_seconds,
            max_idle_ticks_per_window=self.config.max_idle_ticks_per_window,
            max_direct_pushes_per_idle_window=self.config.max_direct_pushes_per_idle_window,
            min_idle_tick_interval_seconds=self.config.min_idle_tick_interval_seconds,
            max_intents_per_idle_window=self.config.max_intents_per_idle_window,
            stop_on_consecutive_no_new_facts=self.config.stop_on_consecutive_no_new_facts,
            cold_start_turns=self.config.cold_start_turns,
            cold_start_max_intents=self.config.cold_start_max_intents,
            min_anchor_tokens_for_fsp=self.config.min_anchor_tokens_for_fsp,
            cross_turn_duplicate_jaccard=self.config.cross_turn_duplicate_jaccard,
            max_total_searches=self.config.max_total_searches,
            run_id_suffix=self.config.run_id_suffix,
            cost_tracker=cost_tracker,
        )
        adapter.reset(scenario)

        simulator = BenchmarkUserSimulator(
            scenario, self.llm, model=self.config.simulator_model,
        )
        judge = CoverageJudge(
            scenario, self.llm, model=self.config.judge_model,
        )

        history: List[Dict[str, str]] = []
        turn_traces: List[TurnTrace] = []
        covered_needs: Set[str] = set()
        max_turns = self.config.max_turns_override or scenario.simulator_config.max_turns

        for turn in range(1, max_turns + 1):
            # Phase 0+1: FSP + ITE (sync agents, run in thread)
            if hasattr(adapter, "run_ablation_turn"):
                abl = await asyncio.to_thread(adapter.run_ablation_turn, turn)
            else:
                abl = {}

            # Phase 2: Direct push
            push_events = self._push_events_from_ablation(abl)
            has_push = bool(push_events)
            for push_event in push_events:
                push_content = push_event.get("push_content") or push_event.get("content")
                if not push_content:
                    continue
                try:
                    with cost_tracker.track("push_judge"):
                        push_judgment = judge.judge_push_turn(
                            turn_number=turn,
                            push_content=push_content,
                            conversation_history=history,
                            allowed_fact_ids=push_event.get("push_fact_ids"),
                        )
                except TypeError as exc:
                    if "allowed_fact_ids" not in str(exc):
                        raise
                    with cost_tracker.track("push_judge"):
                        push_judgment = judge.judge_push_turn(
                            turn_number=turn,
                            push_content=push_content,
                            conversation_history=history,
                        )
                for nc in push_judgment.needs_addressed:
                    covered_needs.add(nc.need_id)
                history.append({"role": "assistant", "content": push_content})
                if hasattr(adapter, "record_direct_push"):
                    adapter.record_direct_push(push_content)
                turn_traces.append(TurnTrace(
                    turn_number=turn,
                    user_message="",
                    active_need_id=None,
                    needs_skipped=[],
                    agent_response=push_content,
                    proactive_predictions=[],
                    proactive_approved=[],
                    memory_context="",
                    judgment=push_judgment,
                    cumulative_covered_needs=set(covered_needs),
                    is_push_turn=True,
                    push_content=push_content,
                    push_type="direct",
                    push_fact_ids=push_event.get("push_fact_ids", []),
                    push_novelty_ratio=push_event.get("push_novelty_ratio"),
                    push_undelivered_fact_ids=push_event.get("push_undelivered_fact_ids", []),
                    push_delivered_fact_ids=push_event.get("push_delivered_fact_ids", []),
                    exploration_intents=abl.get("exploration_intents", []),
                    fsp_filter_reasons=abl.get("fsp_filter_reasons", []),
                    simulated_idle_seconds=abl.get("simulated_idle_seconds", 0.0),
                    idle_trigger_seconds=abl.get("idle_trigger_seconds", 5.0),
                    max_intents_per_idle=abl.get("max_intents_per_idle", 3),
                    max_queries_per_search=abl.get("max_queries_per_search", 1),
                    search_budget=abl.get("search_budget", 999),
                    total_searches_used=abl.get("total_searches_used", 0),
                    remaining_search_budget=abl.get("remaining_search_budget", 999),
                    ite_ran=abl.get("ite_ran", False),
                    intent_queue_size=abl.get("intent_queue_size", 0),
                    idle_window_mode=abl.get("idle_window_mode", False),
                    idle_tick_count=abl.get("idle_tick_count", 0),
                    idle_tick_results=abl.get("idle_tick_results", []),
                    queue_size_before=abl.get("queue_size_before", 0),
                    queue_size_after_generation=abl.get("queue_size_after_generation", 0),
                    queue_size_after_exploration=abl.get("queue_size_after_exploration", 0),
                    generated_intents=abl.get("generated_intents", 0),
                    accepted_intents=abl.get("accepted_intents", 0),
                    filtered_intents=abl.get("filtered_intents", 0),
                    duplicate_intents=abl.get("duplicate_intents", 0),
                    facts_stored=push_event.get("facts_stored", 0),
                    stored_fact_ids=push_event.get("stored_fact_ids", []),
                    intent_results=push_event.get("intent_results", abl.get("intent_results", [])),
                    inline_topics=abl.get("inline_topics", []),
                ))

            # Phase 3: User message + reply
            with cost_tracker.track("simulator"):
                sim_out = simulator.next_message(history, covered_needs)
            if sim_out is None:
                logger.info("  Simulator signalled end-of-conversation at turn %d", turn)
                break

            with cost_tracker.track("reactive_reply"):
                result = await adapter.send_message(sim_out.message)

            satellite_need_ids = getattr(sim_out, "satellite_need_ids", [])
            try:
                with cost_tracker.track("turn_judge"):
                    judgment = judge.judge_turn(
                        turn_number=turn,
                        user_message=sim_out.message,
                        agent_response=result["agent_response"],
                        active_need_id=sim_out.active_need_id,
                        conversation_history=history,
                        satellite_need_ids=satellite_need_ids,
                    )
            except TypeError as exc:
                if "satellite_need_ids" not in str(exc):
                    raise
                with cost_tracker.track("turn_judge"):
                    judgment = judge.judge_turn(
                        turn_number=turn,
                        user_message=sim_out.message,
                        agent_response=result["agent_response"],
                        active_need_id=sim_out.active_need_id,
                        conversation_history=history,
                    )

            for nc in judgment.needs_addressed:
                covered_needs.add(nc.need_id)

            history.append({"role": "user", "content": sim_out.message})
            history.append({"role": "assistant", "content": result["agent_response"]})

            turn_traces.append(TurnTrace(
                turn_number=turn,
                user_message=sim_out.message,
                active_need_id=sim_out.active_need_id,
                needs_skipped=sim_out.needs_skipped,
                agent_response=result["agent_response"],
                proactive_predictions=result.get("proactive_predictions", []),
                proactive_approved=result.get("proactive_approved", []),
                memory_context=result.get("memory_context", ""),
                judgment=judgment,
                cumulative_covered_needs=set(covered_needs),
                satellite_need_ids=satellite_need_ids,
                fact_scores=result.get("fact_scores", []),
                selected_facts=result.get("selected_facts", []),
                fact_scoring_rationales=result.get("fact_scoring_rationales", {}),
                delivered_inline_facts=result.get("delivered_inline_facts", []),
                admission_trace=result.get("admission_trace"),
                decision_trace=result.get("decision_trace"),
                push_type="inline" if abl.get("inline_topics") else None,
                exploration_intents=abl.get("exploration_intents", []),
                fsp_filter_reasons=abl.get("fsp_filter_reasons", []),
                simulated_idle_seconds=abl.get("simulated_idle_seconds", 0.0),
                idle_trigger_seconds=abl.get("idle_trigger_seconds", 5.0),
                max_intents_per_idle=abl.get("max_intents_per_idle", 3),
                max_queries_per_search=abl.get("max_queries_per_search", 1),
                search_budget=abl.get("search_budget", 999),
                total_searches_used=abl.get("total_searches_used", 0),
                remaining_search_budget=abl.get("remaining_search_budget", 999),
                ite_ran=abl.get("ite_ran", False),
                intent_queue_size=abl.get("intent_queue_size", 0),
                idle_window_mode=abl.get("idle_window_mode", False),
                idle_tick_count=abl.get("idle_tick_count", 0),
                idle_tick_results=abl.get("idle_tick_results", []),
                queue_size_before=abl.get("queue_size_before", 0),
                queue_size_after_generation=abl.get("queue_size_after_generation", 0),
                queue_size_after_exploration=abl.get("queue_size_after_exploration", 0),
                generated_intents=abl.get("generated_intents", 0),
                accepted_intents=abl.get("accepted_intents", 0),
                filtered_intents=abl.get("filtered_intents", 0),
                duplicate_intents=abl.get("duplicate_intents", 0),
                facts_stored=0 if has_push else abl.get("facts_stored", 0),
                stored_fact_ids=[] if has_push else abl.get("stored_fact_ids", []),
                intent_results=[] if has_push else abl.get("intent_results", []),
                inline_topics=[] if has_push else abl.get("inline_topics", []),
            ))

            logger.info(
                "  Turn %d: need=%s, covered=%d/%d, skipped=%s, ite=%s",
                turn,
                sim_out.active_need_id,
                len(covered_needs),
                len(scenario.user_needs),
                sim_out.needs_skipped,
                abl.get("ite_ran", False),
            )

        return ConversationResult(
            scenario_id=scenario.scenario_id,
            condition=condition,
            domain=scenario.domain,
            seed=self.config.seed,
            total_turns=max((t.turn_number for t in turn_traces), default=0),
            turn_traces=turn_traces,
            final_covered_needs=covered_needs,
            all_need_ids={n.id for n in scenario.user_needs},
            must_have_ids={n.id for n in scenario.user_needs if n.level == "must-have"},
            predictable_need_ids={
                n.id for n in scenario.user_needs if n.predictable_after is not None
            },
            wall_clock_seconds=time.perf_counter() - start_time,
            llm_usage_stats=usage_delta(
                usage_before,
                snapshot_usage(self.llm) or cost_tracker.latest_snapshot,
            ),
            llm_usage_by_module=cost_tracker.module_usage,
        )

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------

    def _load_scenarios(self) -> List[Scenario]:
        """Load and optionally filter scenarios from the scenario directory."""
        scenario_files = sorted(self.scenario_dir.glob("*.json"))
        scenarios: List[Scenario] = []

        for path in scenario_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                scenario = Scenario.from_dict(data)
                scenarios.append(scenario)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

        # Apply optional filter
        if self.config.scenarios:
            allowed = set(self.config.scenarios)
            scenarios = [s for s in scenarios if s.scenario_id in allowed]

        return scenarios

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> tuple:
        """
        Load checkpoint state.

        Returns:
            (completed_keys: Set[str], results: List[ConversationResult])
        """
        completed: Set[str] = set()
        results: List[ConversationResult] = []

        if not self.checkpoint_path.exists():
            return completed, results

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            completed = set(data.get("completed_keys", []))

            # Load detailed results alongside checkpoint
            if self.detailed_results_path.exists():
                with open(self.detailed_results_path, "r", encoding="utf-8") as f:
                    results_data = json.load(f)
                results = [ConversationResult.from_dict(r) for r in results_data]

        except Exception as exc:
            logger.warning("Failed to load checkpoint, starting fresh: %s", exc)
            completed = set()
            results = []

        return completed, results

    def _save_checkpoint(self, completed_keys: Set[str]) -> None:
        """Persist checkpoint with completed composite keys."""
        data: Dict[str, Any] = {
            "schema_version": 1,
            "completed_keys": sorted(completed_keys),
            "total_completed": len(completed_keys),
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_detailed_results(self, results: List[ConversationResult]) -> None:
        """Persist all conversation results to disk."""
        with open(self.detailed_results_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
                ensure_ascii=False,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_adapter(condition: str, llm: LLMClient, **kwargs) -> Any:
        try:
            signature = inspect.signature(ProactiveBenchAdapter)
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
            return ProactiveBenchAdapter(condition, llm, **accepted)
        except (TypeError, ValueError):
            minimal = {
                key: value
                for key, value in kwargs.items()
                if key == "seed"
            }
            return ProactiveBenchAdapter(condition, llm, **minimal)

    @staticmethod
    def _push_events_from_ablation(abl: Dict[str, Any]) -> List[Dict[str, Any]]:
        events = [
            event for event in (abl.get("push_events", []) or [])
            if isinstance(event, dict)
        ]
        if events:
            return events
        if abl.get("has_direct_push") and abl.get("push_content"):
            return [{
                "push_content": abl.get("push_content"),
                "push_topic": abl.get("push_topic"),
                "push_fact_ids": abl.get("push_fact_ids", []),
                "push_novelty_ratio": abl.get("push_novelty_ratio"),
                "push_undelivered_fact_ids": abl.get("push_undelivered_fact_ids", []),
                "push_delivered_fact_ids": abl.get("push_delivered_fact_ids", []),
                "facts_stored": abl.get("facts_stored", 0),
                "stored_fact_ids": abl.get("stored_fact_ids", []),
                "intent_results": abl.get("intent_results", []),
            }]
        return []

    @staticmethod
    def _resolve_path(raw: str, base: Path) -> Path:
        """Resolve a possibly-relative path against a base directory."""
        p = Path(raw)
        if p.is_absolute():
            return p
        return (base / p).resolve()

    def _llm_usage_snapshot(self) -> Dict[str, Dict[str, int]]:
        getter = getattr(self.llm, "get_usage_stats", None)
        if not callable(getter):
            return {}
        try:
            return getter()
        except Exception:
            logger.warning("Failed to read LLM usage stats", exc_info=True)
            return {}

    @staticmethod
    def _llm_usage_delta(
        before: Dict[str, Dict[str, int]],
        after: Dict[str, Dict[str, int]],
    ) -> Dict[str, Dict[str, int]]:
        keys = {"calls", "prompt_tokens", "completion_tokens", "total_tokens"}
        delta: Dict[str, Dict[str, int]] = {}
        for model in sorted(set(before) | set(after)):
            model_delta = {}
            for key in keys:
                model_delta[key] = int(after.get(model, {}).get(key, 0)) - int(
                    before.get(model, {}).get(key, 0)
                )
            if any(value != 0 for value in model_delta.values()):
                delta[model] = model_delta
        return delta


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProactiveBench evaluation runner",
    )
    parser.add_argument(
        "--scenario-dir",
        default="../data/scenarios",
        help="Path to scenario JSON files (default: ../data/scenarios relative to eval/)",
    )
    parser.add_argument(
        "--output-dir",
        default="../results",
        help="Path for output results (default: ../results relative to eval/)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Full", "Blind", "Paralyzed", "Baseline"],
        help=(
            "Ablation conditions to evaluate. Full and Full-single-idle are "
            "single-idle; Full-idle-window enables multi-tick idle."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Model for the coverage judge (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--simulator-model",
        default="gpt-4o",
        help="Model for the user simulator (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max_turns from scenario config",
    )
    parser.add_argument(
        "--max-queries-per-search",
        type=int,
        default=1,
        help="Maximum planned search queries to execute per SearchService search (default: 1)",
    )
    parser.add_argument(
        "--max-intents-per-idle",
        type=int,
        default=3,
        help="Maximum exploration intents ITE may process per idle window (default: 3)",
    )
    parser.add_argument(
        "--idle-trigger-seconds",
        type=float,
        default=5.0,
        help="Minimum simulated idle seconds required to run ITE (default: 5.0)",
    )
    parser.add_argument("--max-idle-ticks-per-window", type=int, default=4)
    parser.add_argument("--max-direct-pushes-per-idle-window", type=int, default=2)
    parser.add_argument("--min-idle-tick-interval-seconds", type=float, default=3.0)
    parser.add_argument("--max-intents-per-idle-window", type=int, default=6)
    parser.add_argument("--stop-on-consecutive-no-new-facts", type=int, default=2)
    parser.add_argument("--cold-start-turns", type=int, default=2)
    parser.add_argument("--cold-start-max-intents", type=int, default=1)
    parser.add_argument("--min-anchor-tokens-for-fsp", type=int, default=6)
    parser.add_argument("--cross-turn-duplicate-jaccard", type=float, default=0.6)
    parser.add_argument(
        "--max-total-searches",
        type=int,
        default=999,
        help="Maximum ITE intent searches allowed across one conversation (default: 999)",
    )
    parser.add_argument(
        "--run-id-suffix",
        default="",
        help="Optional suffix for runtime user IDs; use this for parallel runs of the same condition/seed",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        dest="scenarios",
        help="Only run these scenario IDs (space-separated)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    runner_config = RunnerConfig(
        scenario_dir=args.scenario_dir,
        output_dir=args.output_dir,
        conditions=args.conditions,
        seed=args.seed,
        judge_model=args.judge_model,
        simulator_model=args.simulator_model,
        max_turns_override=args.max_turns,
        max_queries_per_search=args.max_queries_per_search,
        max_intents_per_idle=args.max_intents_per_idle,
        idle_trigger_seconds=args.idle_trigger_seconds,
        max_idle_ticks_per_window=args.max_idle_ticks_per_window,
        max_direct_pushes_per_idle_window=args.max_direct_pushes_per_idle_window,
        min_idle_tick_interval_seconds=args.min_idle_tick_interval_seconds,
        max_intents_per_idle_window=args.max_intents_per_idle_window,
        stop_on_consecutive_no_new_facts=args.stop_on_consecutive_no_new_facts,
        cold_start_turns=args.cold_start_turns,
        cold_start_max_intents=args.cold_start_max_intents,
        min_anchor_tokens_for_fsp=args.min_anchor_tokens_for_fsp,
        cross_turn_duplicate_jaccard=args.cross_turn_duplicate_jaccard,
        max_total_searches=args.max_total_searches,
        run_id_suffix=args.run_id_suffix,
        scenarios=args.scenarios,
    )

    runner = ProactiveBenchRunner(runner_config)
    results = asyncio.run(runner.run())

    logger.info("Done. %d conversation result(s) produced.", len(results))


if __name__ == "__main__":
    main()
