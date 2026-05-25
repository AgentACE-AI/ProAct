"""
Run the ProactiveAgent-style GPT-4o baseline on ProactiveBench.

This runner reuses the existing user simulator, coverage judge, trace models,
checkpointing, cost tracking, and persistence from runner.py, but swaps in the
ProactiveAgentAdapter as the system under test.

Usage:
    python -m experiments.ProactiveBench.eval.run_proactive_agent \
        --scenario-dir ../data/scenarios/ \
        --output-dir ../results/proactive_agent/ \
        --seed 42
"""

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.eval.cost_tracking import (
    LLMCostTracker,
    snapshot_usage,
    usage_delta,
)
from experiments.ProactiveBench.eval.coverage_judge import CoverageJudge
from experiments.ProactiveBench.eval.models import ConversationResult, TurnTrace
from experiments.ProactiveBench.eval.proactive_agent_adapter import (
    DEFAULT_PROACTIVE_AGENT_MODEL,
    PROACTIVE_AGENT_CONDITION,
    ProactiveAgentAdapter,
)
from experiments.ProactiveBench.eval.runner import ProactiveBenchRunner, RunnerConfig
from experiments.ProactiveBench.eval.user_simulator import BenchmarkUserSimulator
from experiments.ProactiveBench.generation.models import Scenario
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ProactiveAgentRunnerConfig(RunnerConfig):
    """Configuration for the ProactiveAgent-style baseline run."""

    output_dir: str = "../results/proactive_agent"
    conditions: List[str] = field(
        default_factory=lambda: [PROACTIVE_AGENT_CONDITION],
    )
    proactive_agent_model: str = DEFAULT_PROACTIVE_AGENT_MODEL


class ProactiveAgentRunner(ProactiveBenchRunner):
    """Runner that evaluates only the ProactiveAgent-style baseline."""

    config: ProactiveAgentRunnerConfig

    def __init__(
        self,
        config: ProactiveAgentRunnerConfig,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        super().__init__(config, llm_client=llm_client)

    async def run_single(
        self,
        scenario: Scenario,
        condition: str,
    ) -> ConversationResult:
        if condition != PROACTIVE_AGENT_CONDITION:
            raise ValueError(
                "ProactiveAgentRunner only supports condition "
                f"{PROACTIVE_AGENT_CONDITION!r}, got {condition!r}."
            )

        start_time = time.perf_counter()
        usage_before = snapshot_usage(self.llm)
        cost_tracker = LLMCostTracker(self.llm, initial_snapshot=usage_before)

        adapter = ProactiveAgentAdapter(
            self.llm,
            model=self.config.proactive_agent_model,
            seed=self.config.seed,
        )
        adapter.reset(scenario)

        simulator = BenchmarkUserSimulator(
            scenario,
            self.llm,
            model=self.config.simulator_model,
        )
        judge = CoverageJudge(
            scenario,
            self.llm,
            model=self.config.judge_model,
        )

        history: List[Dict[str, str]] = []
        turn_traces: List[TurnTrace] = []
        covered_needs: Set[str] = set()
        max_turns = self.config.max_turns_override or scenario.simulator_config.max_turns

        for turn in range(1, max_turns + 1):
            with cost_tracker.track("simulator"):
                sim_out = simulator.next_message(history, covered_needs)
            if sim_out is None:
                logger.info("  Simulator signalled end-of-conversation at turn %d", turn)
                break

            with cost_tracker.track("proactive_agent_reply"):
                result = await adapter.send_message(sim_out.message)

            satellite_need_ids = getattr(sim_out, "satellite_need_ids", [])
            with cost_tracker.track("turn_judge"):
                judgment = judge.judge_turn(
                    turn_number=turn,
                    user_message=sim_out.message,
                    agent_response=result["agent_response"],
                    active_need_id=sim_out.active_need_id,
                    conversation_history=history,
                    satellite_need_ids=satellite_need_ids,
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
                decision_trace=result.get("decision_trace"),
            ))

            logger.info(
                "  Turn %d: need=%s, covered=%d/%d, skipped=%s",
                turn,
                sim_out.active_need_id,
                len(covered_needs),
                len(scenario.user_needs),
                sim_out.needs_skipped,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProactiveAgent-style GPT-4o baseline runner for ProactiveBench",
    )
    parser.add_argument(
        "--scenario-dir",
        default="../data/scenarios",
        help="Path to scenario JSON files (default: ../data/scenarios relative to eval/)",
    )
    parser.add_argument(
        "--output-dir",
        default="../results/proactive_agent",
        help="Path for output results (default: ../results/proactive_agent relative to eval/)",
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
        "--adapter-model",
        default=DEFAULT_PROACTIVE_AGENT_MODEL,
        help="Model for the ProactiveAgent-style adapter (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max_turns from scenario config",
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

    runner_config = ProactiveAgentRunnerConfig(
        scenario_dir=args.scenario_dir,
        output_dir=args.output_dir,
        conditions=[PROACTIVE_AGENT_CONDITION],
        seed=args.seed,
        judge_model=args.judge_model,
        simulator_model=args.simulator_model,
        max_turns_override=args.max_turns,
        scenarios=args.scenarios,
        proactive_agent_model=args.adapter_model,
    )

    runner = ProactiveAgentRunner(runner_config)
    results = asyncio.run(runner.run())

    logger.info(
        "Done. %d %s conversation result(s) produced.",
        len(results),
        PROACTIVE_AGENT_CONDITION,
    )


if __name__ == "__main__":
    main()
