"""
ProactiveBench post-hoc metrics evaluator.

Loads ConversationResult traces produced by the runner and computes all
benchmark metrics, aggregations, and statistical tests.

Usage:
    python -m experiments.ProactiveBench.eval.evaluator \
        --results-dir ../results/ \
        --scenario-dir ../data/scenarios/ \
        --output ../results/report.json
"""

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.eval.ablation_config import DedupPriorityQueue
from experiments.ProactiveBench.eval.cost_tracking import flatten_usage
from experiments.ProactiveBench.eval.models import (
    ConditionAggregate,
    ConversationResult,
    EvalReport,
    ScenarioMetrics,
)
from experiments.ProactiveBench.generation.models import Scenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ProactiveBenchEvaluator:
    """
    Post-hoc metrics computation from runner output.

    Reads detailed_results.json produced by ProactiveBenchRunner and
    computes per-scenario, per-condition, and per-domain aggregated
    metrics plus optional statistical significance tests.
    """

    def __init__(
        self,
        results_dir: str,
        scenario_dir: Optional[str] = None,
    ) -> None:
        eval_dir = Path(__file__).resolve().parent
        self.results_dir = self._resolve_path(results_dir, eval_dir)
        self.scenario_dir = (
            self._resolve_path(scenario_dir, eval_dir) if scenario_dir else None
        )

        # Scenario lookup: scenario_id -> Scenario
        self._scenarios: Dict[str, Scenario] = {}
        if self.scenario_dir:
            self._scenarios = self._load_scenario_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_results(self) -> List[ConversationResult]:
        """Load ConversationResult objects from detailed_results.json."""
        path = self.results_dir / "detailed_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [ConversationResult.from_dict(item) for item in data]

    def compute_all(self) -> EvalReport:
        """
        Compute the full evaluation report.

        Returns:
            An EvalReport containing per-scenario metrics, per-condition
            aggregates, per-domain breakdowns, and statistical tests.
        """
        results = self.load_results()
        if not results:
            raise ValueError("No results to evaluate")

        # --- Per-scenario metrics ---
        per_scenario: List[ScenarioMetrics] = []
        for result in results:
            scenario = self._scenarios.get(result.scenario_id)
            metrics = self.compute_scenario_metrics(result, scenario)
            per_scenario.append(metrics)

        # --- Per-condition aggregation ---
        by_condition: Dict[str, List[ScenarioMetrics]] = defaultdict(list)
        for m in per_scenario:
            by_condition[m.condition].append(m)

        per_condition: Dict[str, ConditionAggregate] = {}
        for condition, metrics_list in sorted(by_condition.items()):
            per_condition[condition] = self.aggregate_condition(metrics_list, condition)

        # --- Per-domain x condition aggregation ---
        by_domain_condition: Dict[str, Dict[str, List[ScenarioMetrics]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for m in per_scenario:
            by_domain_condition[m.domain][m.condition].append(m)

        per_domain: Dict[str, Dict[str, ConditionAggregate]] = {}
        for domain in sorted(by_domain_condition):
            per_domain[domain] = {}
            for condition, metrics_list in sorted(by_domain_condition[domain].items()):
                per_domain[domain][condition] = self.aggregate_condition(
                    metrics_list, condition,
                )

        # --- Statistical tests (Full vs each control condition) ---
        statistical_tests = self._run_statistical_tests(per_scenario)

        return EvalReport(
            per_scenario=per_scenario,
            per_condition=per_condition,
            per_domain=per_domain,
            statistical_tests=statistical_tests,
        )

    def compute_scenario_metrics(
        self,
        result: ConversationResult,
        scenario: Optional[Scenario] = None,
    ) -> ScenarioMetrics:
        """
        Compute all metrics for a single scenario-condition pair.

        Args:
            result: The conversation result to evaluate.
            scenario: The original Scenario (needed for anticipation precision).

        Returns:
            ScenarioMetrics with all computed fields.
        """
        must_have_ids = result.must_have_ids
        predictable_ids = result.predictable_need_ids
        traces = result.turn_traces

        # -- T80 --
        t80 = self._compute_tN(traces, must_have_ids, fraction=0.8, total_turns=result.total_turns)

        # -- T100 --
        t100 = self._compute_tN(traces, must_have_ids, fraction=1.0, total_turns=result.total_turns)

        # -- Anticipation Precision --
        anticipation_precision = self._compute_anticipation_precision(
            result, scenario,
        )

        # -- Anticipation Recall --
        anticipation_recall = self._compute_anticipation_recall(result, scenario)

        # -- Fact Accuracy --
        fact_accuracy = self._compute_fact_accuracy(traces)

        # -- Hallucination Rate --
        hallucination_rate = self._compute_hallucination_rate(traces)

        # -- User Effort (exclude push-only turns) --
        user_effort = sum(
            1 for trace in traces
            if trace.active_need_id is not None and not trace.is_push_turn
        )

        # -- Coverage metrics --
        all_ids = result.all_need_ids
        covered = result.final_covered_needs
        total_coverage = len(covered) / len(all_ids) if all_ids else 0.0
        must_have_coverage = (
            len(covered & must_have_ids) / len(must_have_ids)
            if must_have_ids
            else 0.0
        )

        # -- Push metrics --
        push_precision, push_coverage = self._compute_push_metrics(traces, all_ids)
        intent_hit_rate, memory_utilization = self._compute_intent_memory_metrics(
            traces
        )
        budget_cost_metrics = self._compute_budget_cost_metrics(result)
        novelty_metrics = self._compute_push_novelty_metrics(traces)
        queue_duplicate_metrics = self._compute_queue_duplicate_metrics(traces)
        active_cost_metrics = self._compute_active_cost_metrics(result)

        return ScenarioMetrics(
            scenario_id=result.scenario_id,
            condition=result.condition,
            domain=result.domain,
            t80=t80,
            t100=t100,
            anticipation_precision=anticipation_precision,
            anticipation_recall=anticipation_recall,
            fact_accuracy=fact_accuracy,
            hallucination_rate=hallucination_rate,
            user_effort=user_effort,
            total_coverage=total_coverage,
            must_have_coverage=must_have_coverage,
            push_precision=push_precision,
            push_coverage=push_coverage,
            intent_hit_rate=intent_hit_rate,
            memory_utilization=memory_utilization,
            **budget_cost_metrics,
            **novelty_metrics,
            **queue_duplicate_metrics,
            **active_cost_metrics,
        )

    def aggregate_condition(
        self,
        metrics: List[ScenarioMetrics],
        condition: str,
    ) -> ConditionAggregate:
        """
        Aggregate a list of ScenarioMetrics into a ConditionAggregate.

        Computes mean and standard deviation for each numeric metric.
        """
        n = len(metrics)
        if n == 0:
            return ConditionAggregate(
                condition=condition,
                n_scenarios=0,
                mean_t80=0.0, std_t80=0.0,
                mean_t100=0.0, std_t100=0.0,
                mean_anticipation_precision=None,
                mean_anticipation_recall=None,
                mean_fact_accuracy=0.0,
                mean_hallucination_rate=0.0,
                mean_user_effort=0.0,
                mean_total_coverage=0.0,
                mean_must_have_coverage=0.0,
                mean_push_precision=None,
                mean_push_coverage=None,
                mean_intent_hit_rate=None,
                mean_memory_utilization=None,
            )

        def _mean(values: List[float]) -> float:
            return sum(values) / len(values)

        def _std(values: List[float]) -> float:
            if len(values) < 2:
                return 0.0
            mu = _mean(values)
            variance = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
            return math.sqrt(variance)

        def _mean_optional(values: List[Optional[float]]) -> Optional[float]:
            filtered = [v for v in values if v is not None]
            return _mean(filtered) if filtered else None

        def _mean_dict(dicts: List[Dict[str, int]]) -> Dict[str, float]:
            keys = sorted({key for item in dicts for key in item})
            return {
                key: _mean([
                    float(item.get(key, 0) or 0)
                    for item in dicts
                ])
                for key in keys
            }

        t80_vals = [float(m.t80) for m in metrics]
        t100_vals = [float(m.t100) for m in metrics]

        return ConditionAggregate(
            condition=condition,
            n_scenarios=n,
            mean_t80=_mean(t80_vals),
            std_t80=_std(t80_vals),
            mean_t100=_mean(t100_vals),
            std_t100=_std(t100_vals),
            mean_anticipation_precision=_mean_optional(
                [m.anticipation_precision for m in metrics]
            ),
            mean_anticipation_recall=_mean_optional(
                [m.anticipation_recall for m in metrics]
            ),
            mean_fact_accuracy=_mean([m.fact_accuracy for m in metrics]),
            mean_hallucination_rate=_mean([m.hallucination_rate for m in metrics]),
            mean_user_effort=_mean([float(m.user_effort) for m in metrics]),
            mean_total_coverage=_mean([m.total_coverage for m in metrics]),
            mean_must_have_coverage=_mean([m.must_have_coverage for m in metrics]),
            mean_push_precision=_mean_optional(
                [m.push_precision for m in metrics]
            ),
            mean_push_coverage=_mean_optional(
                [m.push_coverage for m in metrics]
            ),
            mean_intent_hit_rate=_mean_optional(
                [m.intent_hit_rate for m in metrics]
            ),
            mean_memory_utilization=_mean_optional(
                [m.memory_utilization for m in metrics]
            ),
            mean_wall_clock_seconds=_mean([m.wall_clock_seconds for m in metrics]),
            mean_llm_calls=_mean([float(m.llm_calls) for m in metrics]),
            mean_llm_total_tokens=_mean([float(m.llm_total_tokens) for m in metrics]),
            mean_stored_facts=_mean([float(m.stored_facts) for m in metrics]),
            mean_processed_intents=_mean([float(m.processed_intents) for m in metrics]),
            mean_peak_queue_size=_mean([float(m.peak_queue_size) for m in metrics]),
            mean_push_novelty_ratio=_mean_optional(
                [m.mean_push_novelty_ratio for m in metrics]
            ),
            low_novelty_push_rate=_mean_optional(
                [m.low_novelty_push_rate for m in metrics]
            ),
            mean_queue_size=_mean([float(m.mean_queue_size) for m in metrics]),
            mean_peak_queue_pressure=_mean([
                float(m.peak_queue_pressure) for m in metrics
            ]),
            mean_queue_saturation_turns=_mean([
                float(m.queue_saturation_turns) for m in metrics
            ]),
            mean_duplicate_topic_count=_mean([
                float(m.duplicate_topic_count) for m in metrics
            ]),
            mean_duplicate_topic_rate=_mean_optional(
                [m.duplicate_topic_rate for m in metrics]
            ),
            mean_fsp_semantic_duplicate_count=_mean([
                float(m.fsp_semantic_duplicate_count) for m in metrics
            ]),
            mean_fsp_cross_turn_duplicate_count=_mean([
                float(m.fsp_cross_turn_duplicate_count) for m in metrics
            ]),
            mean_queue_semantic_duplicate_count=_mean([
                float(m.queue_semantic_duplicate_count) for m in metrics
            ]),
            mean_active_llm_calls=_mean([
                float(m.active_llm_calls) for m in metrics
            ]),
            mean_active_llm_total_tokens=_mean([
                float(m.active_llm_total_tokens) for m in metrics
            ]),
            mean_module_llm_calls=_mean_dict([m.module_llm_calls for m in metrics]),
            mean_module_llm_total_tokens=_mean_dict([
                m.module_llm_total_tokens for m in metrics
            ]),
            mean_cost_per_new_need=_mean_optional(
                [m.cost_per_new_need for m in metrics]
            ),
        )

    def print_summary(self, report: EvalReport) -> None:
        """Print a formatted comparison table across conditions."""
        conditions = sorted(report.per_condition.keys())
        if not conditions:
            print("No results to display.")
            return

        # Define metric rows: (label, key, direction, format_fn)
        metric_rows = [
            ("T80", "mean_t80", "down", lambda v: f"{v:.1f}"),
            ("T100", "mean_t100", "down", lambda v: f"{v:.1f}"),
            ("User Effort", "mean_user_effort", "down", lambda v: f"{v:.1f}"),
            ("Fact Accuracy", "mean_fact_accuracy", "up", lambda v: f"{v:.3f}"),
            ("Hallucination Rate", "mean_hallucination_rate", "down", lambda v: f"{v:.3f}"),
            ("Total Coverage", "mean_total_coverage", "up", lambda v: f"{v:.3f}"),
            ("Must-Have Coverage", "mean_must_have_coverage", "up", lambda v: f"{v:.3f}"),
            ("Anticipation Prec.", "mean_anticipation_precision", "up", lambda v: f"{v:.3f}" if v is not None else "  -  "),
            ("Anticipation Rec.", "mean_anticipation_recall", "up", lambda v: f"{v:.3f}" if v is not None else "  -  "),
            ("Push Precision", "mean_push_precision", "up", lambda v: f"{v:.3f}" if v is not None else "  -  "),
            ("Push Coverage", "mean_push_coverage", "up", lambda v: f"{v:.3f}" if v is not None else "  -  "),
            ("Push Novelty", "mean_push_novelty_ratio", "up", lambda v: f"{v:.3f}" if v is not None else "  -  "),
            ("Low Novelty Pushes", "low_novelty_push_rate", "down", lambda v: f"{v:.3f}" if v is not None else "  -  "),
            ("Wall Time", "mean_wall_clock_seconds", "down", lambda v: f"{v:.1f}"),
            ("LLM Calls", "mean_llm_calls", "down", lambda v: f"{v:.1f}"),
            ("LLM Tokens", "mean_llm_total_tokens", "down", lambda v: f"{v:.0f}"),
            ("Active LLM Tokens", "mean_active_llm_total_tokens", "down", lambda v: f"{v:.0f}"),
            ("Cost/New Need", "mean_cost_per_new_need", "down", lambda v: f"{v:.1f}" if v is not None else "  -  "),
            ("Stored Facts", "mean_stored_facts", "down", lambda v: f"{v:.1f}"),
            ("Processed Intents", "mean_processed_intents", "down", lambda v: f"{v:.1f}"),
            ("Peak Queue", "mean_peak_queue_size", "down", lambda v: f"{v:.1f}"),
            ("Queue Pressure", "mean_peak_queue_pressure", "down", lambda v: f"{v:.3f}"),
            ("Duplicate Topics", "mean_duplicate_topic_rate", "down", lambda v: f"{v:.3f}" if v is not None else "  -  "),
        ]

        # Column widths
        label_width = 22
        col_width = 10

        # Build header
        header_line = "Metric".ljust(label_width)
        for cond in conditions:
            header_line += cond.center(col_width)
        sep_line = "-" * (label_width + col_width * len(conditions))

        # Top border
        print()
        print("=" * len(sep_line))
        print("ProactiveBench Evaluation Summary")
        print("=" * len(sep_line))
        print()

        # N scenarios
        n_line = "N scenarios".ljust(label_width)
        for cond in conditions:
            agg = report.per_condition[cond]
            n_line += str(agg.n_scenarios).center(col_width)
        print(n_line)
        print(sep_line)
        print(header_line)
        print(sep_line)

        for label, attr, direction, fmt_fn in metric_rows:
            arrow = " (down)" if direction == "down" else " (up)"
            row_label = f"{label}{arrow}".ljust(label_width)
            for cond in conditions:
                agg = report.per_condition[cond]
                value = getattr(agg, attr)
                row_label += fmt_fn(value).center(col_width)
            print(row_label)

        print(sep_line)

        # Statistical tests
        if report.statistical_tests:
            print()
            print("Statistical Tests (Wilcoxon signed-rank, paired by scenario):")
            for key, p_value in sorted(report.statistical_tests.items()):
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {key}: p={p_value:.4f} {sig}")

        # Per-domain breakdown
        if report.per_domain:
            print()
            print("Per-Domain Breakdown:")
            print(sep_line)
            for domain in sorted(report.per_domain):
                print(f"\n  Domain: {domain}")
                domain_aggs = report.per_domain[domain]
                row = "    ".ljust(label_width)
                for cond in conditions:
                    row += cond.center(col_width)
                print(row)

                for label, attr, _direction, fmt_fn in metric_rows[:3]:
                    row = f"    {label}".ljust(label_width)
                    for cond in conditions:
                        if cond in domain_aggs:
                            value = getattr(domain_aggs[cond], attr)
                            row += fmt_fn(value).center(col_width)
                        else:
                            row += "-".center(col_width)
                    print(row)

        print()

    # ------------------------------------------------------------------
    # Metric computation (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tN(
        traces: list,
        must_have_ids: Set[str],
        fraction: float,
        total_turns: int,
    ) -> int:
        """
        Find the first turn where cumulative coverage of must-have needs
        reaches the target fraction.

        Returns total_turns + 1 as a penalty if the threshold is never met.
        """
        if not must_have_ids:
            return 0

        threshold = math.ceil(fraction * len(must_have_ids))

        for trace in traces:
            covered_must_have = trace.cumulative_covered_needs & must_have_ids
            if len(covered_must_have) >= threshold:
                return trace.turn_number

        # Never reached
        return total_turns + 1

    @staticmethod
    def _compute_anticipation_precision(
        result: ConversationResult,
        scenario: Optional[Scenario],
    ) -> Optional[float]:
        """
        Compute anticipation precision.

        For each turn with proactive need coverage, count facts conveyed that
        belong to those proactive needs. Check which are relevant (i.e., in
        any uncovered need's key_fact_ids at that point).
        """
        if scenario is None:
            return None

        # Build a lookup: need_id -> set of key_fact_ids
        need_fact_map: Dict[str, Set[str]] = {
            n.id: set(n.key_fact_ids) for n in scenario.user_needs
        }

        total_proactive_facts = 0
        relevant_proactive_facts = 0

        for trace in result.turn_traces:
            proactive_need_ids, proactive_facts = (
                ProactiveBenchEvaluator._runtime_proactive_needs_and_facts(
                    trace, scenario
                )
            )
            if not proactive_need_ids or not proactive_facts:
                continue

            total_proactive_facts += len(proactive_facts)

            # Determine which needs were uncovered *before* this turn's coverage.
            # "Uncovered before" = all needs minus those already covered before
            # this turn's addressed needs took effect.
            previously_covered = trace.cumulative_covered_needs - {
                nc.need_id for nc in trace.judgment.needs_addressed
            }
            uncovered_before = result.all_need_ids - previously_covered

            # Collect all fact IDs that are relevant to uncovered needs
            relevant_fact_ids: Set[str] = set()
            for need_id in uncovered_before:
                if need_id in need_fact_map:
                    relevant_fact_ids.update(need_fact_map[need_id])

            relevant_proactive_facts += len(proactive_facts & relevant_fact_ids)

        if total_proactive_facts == 0:
            return None

        return relevant_proactive_facts / total_proactive_facts

    @staticmethod
    def _compute_anticipation_recall(
        result: ConversationResult,
        scenario: Optional[Scenario] = None,
    ) -> Optional[float]:
        """
        Compute anticipation recall.

        Fraction of predictable needs that were proactively addressed.
        """
        predictable_needs = result.predictable_need_ids
        if not predictable_needs:
            return None

        if scenario is not None:
            anticipated: Set[str] = set()
            for trace in result.turn_traces:
                proactive_need_ids, _ = (
                    ProactiveBenchEvaluator._runtime_proactive_needs_and_facts(
                        trace, scenario
                    )
                )
                anticipated.update(proactive_need_ids & predictable_needs)
            return len(anticipated) / len(predictable_needs)

        anticipated: Set[str] = set()
        for trace in result.turn_traces:
            for nc in trace.judgment.needs_addressed:
                if nc.mode == "proactive" and nc.need_id in predictable_needs:
                    anticipated.add(nc.need_id)

        return len(anticipated) / len(predictable_needs)

    @staticmethod
    def _lookup_query_fact_ids(lookup_query: str) -> Set[str]:
        query = (lookup_query or "").strip()
        if not query:
            return set()

        factset_match = re.match(r"^factset:([A-Za-z0-9_, -]+)$", query, flags=re.IGNORECASE)
        if factset_match:
            return {
                fact_id.strip().upper()
                for fact_id in factset_match.group(1).split(",")
                if fact_id.strip()
            }

        fact_match = re.match(r"^fact:([A-Za-z0-9_-]+)$", query, flags=re.IGNORECASE)
        if fact_match:
            return {fact_match.group(1).strip().upper()}

        return set()

    @classmethod
    def _delivered_fact_ids(cls, trace) -> Set[str]:
        delivered_fact_ids: Set[str] = set()
        for item in trace.delivered_inline_facts or []:
            if not isinstance(item, dict):
                continue
            explicit_ids = {
                str(fact_id).strip().upper()
                for fact_id in item.get("delivered_fact_ids", [])
                if str(fact_id).strip()
            }
            if explicit_ids:
                delivered_fact_ids.update(explicit_ids)
                continue
            delivered_fact_ids.update(
                cls._lookup_query_fact_ids(str(item.get("lookup_query", "")))
            )
        return delivered_fact_ids

    @classmethod
    def _runtime_proactive_needs_and_facts(
        cls,
        trace,
        scenario: Scenario,
    ) -> tuple[Set[str], Set[str]]:
        if trace.is_push_turn and trace.push_type == "direct":
            return cls._direct_push_needs_and_facts(trace, scenario)

        delivered_fact_ids = cls._delivered_fact_ids(trace)
        if not delivered_fact_ids:
            return set(), set()

        conveyed_or_distorted = {
            fact_id.upper()
            for fact_id in (
                list(trace.judgment.facts_conveyed) + list(trace.judgment.facts_distorted)
            )
        }
        expressed_delivered_fact_ids = delivered_fact_ids & conveyed_or_distorted
        if not expressed_delivered_fact_ids:
            return set(), set()

        need_fact_map: Dict[str, Set[str]] = {
            need.id: {fact_id.upper() for fact_id in need.key_fact_ids}
            for need in scenario.user_needs
        }
        addressed_need_ids = {
            coverage.need_id for coverage in trace.judgment.needs_addressed
        }
        user_mentioned_need_ids = set(trace.satellite_need_ids or [])
        if trace.active_need_id:
            user_mentioned_need_ids.add(trace.active_need_id)

        proactive_need_ids: Set[str] = set()
        proactive_fact_ids: Set[str] = set()
        for need_id in addressed_need_ids:
            if need_id in user_mentioned_need_ids:
                continue
            key_facts = need_fact_map.get(need_id, set())
            matching_fact_ids = key_facts & expressed_delivered_fact_ids
            if not matching_fact_ids:
                continue
            proactive_need_ids.add(need_id)
            proactive_fact_ids.update(matching_fact_ids)

        return proactive_need_ids, proactive_fact_ids

    @staticmethod
    def _direct_push_needs_and_facts(
        trace,
        scenario: Scenario,
    ) -> tuple[Set[str], Set[str]]:
        conveyed_or_distorted = {
            fact_id.upper()
            for fact_id in (
                list(trace.judgment.facts_conveyed) + list(trace.judgment.facts_distorted)
            )
        }
        if not conveyed_or_distorted:
            return set(), set()

        need_fact_map: Dict[str, Set[str]] = {
            need.id: {fact_id.upper() for fact_id in need.key_fact_ids}
            for need in scenario.user_needs
        }
        proactive_need_ids: Set[str] = set()
        proactive_fact_ids: Set[str] = set()
        for coverage in trace.judgment.needs_addressed:
            if coverage.mode != "proactive":
                continue
            matching_fact_ids = (
                need_fact_map.get(coverage.need_id, set()) & conveyed_or_distorted
            )
            if not matching_fact_ids:
                continue
            proactive_need_ids.add(coverage.need_id)
            proactive_fact_ids.update(matching_fact_ids)

        return proactive_need_ids, proactive_fact_ids

    @staticmethod
    def _compute_push_metrics(
        traces: list,
        all_need_ids: Set[str],
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute push_precision and push_coverage from trace data.

        push_precision: fraction of push turns that covered new needs.
        push_coverage: fraction of all needs first covered during a push turn.
        """
        push_turns = [
            t for t in traces
            if t.is_push_turn and t.push_type == "direct"
        ]
        if not push_turns:
            return None, None

        prev_covered: Set[str] = set()
        push_hits = 0
        needs_from_push: Set[str] = set()

        for trace in traces:
            newly_covered = trace.cumulative_covered_needs - prev_covered
            if trace.is_push_turn and trace.push_type == "direct" and newly_covered:
                push_hits += 1
                needs_from_push.update(newly_covered)
            prev_covered = set(trace.cumulative_covered_needs)

        push_precision = push_hits / len(push_turns) if push_turns else None
        push_coverage = (
            len(needs_from_push) / len(all_need_ids)
            if all_need_ids
            else None
        )
        return push_precision, push_coverage

    @staticmethod
    def _compute_intent_memory_metrics(
        traces: list,
    ) -> tuple[Optional[float], Optional[float]]:
        used_fact_ids: Set[str] = set()
        stored_fact_ids: Set[str] = set()
        intent_results: List[Dict[str, Any]] = []

        for trace in traces:
            used_fact_ids.update(
                fact_id.upper()
                for fact_id in (
                    list(trace.judgment.facts_conveyed)
                    + list(trace.judgment.facts_distorted)
                )
            )
            stored_fact_ids.update(
                str(fact_id).strip().upper()
                for fact_id in getattr(trace, "stored_fact_ids", [])
                if str(fact_id).strip()
            )
            for item in getattr(trace, "intent_results", []) or []:
                if isinstance(item, dict):
                    intent_results.append(item)

        intent_hit_rate = None
        if intent_results:
            hit_count = 0
            for item in intent_results:
                fact_ids = {
                    str(fact_id).strip().upper()
                    for fact_id in item.get("fact_ids", [])
                    if str(fact_id).strip()
                }
                if fact_ids & used_fact_ids:
                    hit_count += 1
            intent_hit_rate = hit_count / len(intent_results)

        memory_utilization = None
        if stored_fact_ids:
            memory_utilization = len(stored_fact_ids & used_fact_ids) / len(stored_fact_ids)

        return intent_hit_rate, memory_utilization

    @staticmethod
    def _compute_budget_cost_metrics(result: ConversationResult) -> Dict[str, Any]:
        traces = result.turn_traces
        usage_stats = result.llm_usage_stats or {}

        llm_calls = sum(
            int(stats.get("calls", 0) or 0)
            for stats in usage_stats.values()
            if isinstance(stats, dict)
        )
        llm_total_tokens = sum(
            int(stats.get("total_tokens", 0) or 0)
            for stats in usage_stats.values()
            if isinstance(stats, dict)
        )

        return {
            "wall_clock_seconds": result.wall_clock_seconds,
            "llm_calls": llm_calls,
            "llm_total_tokens": llm_total_tokens,
            "stored_facts": sum(int(getattr(trace, "facts_stored", 0) or 0) for trace in traces),
            "processed_intents": sum(
                len(getattr(trace, "intent_results", []) or [])
                for trace in traces
            ),
            "peak_queue_size": max(
                (int(getattr(trace, "intent_queue_size", 0) or 0) for trace in traces),
                default=0,
            ),
        }

    @staticmethod
    def _compute_push_novelty_metrics(traces: list) -> Dict[str, Any]:
        novelty_values = [
            float(trace.push_novelty_ratio)
            for trace in traces
            if (
                trace.is_push_turn
                and trace.push_type == "direct"
                and trace.push_novelty_ratio is not None
            )
        ]
        if not novelty_values:
            return {
                "mean_push_novelty_ratio": None,
                "low_novelty_push_rate": None,
            }
        return {
            "mean_push_novelty_ratio": sum(novelty_values) / len(novelty_values),
            "low_novelty_push_rate": (
                sum(1 for value in novelty_values if value < 0.35)
                / len(novelty_values)
            ),
        }

    @staticmethod
    def _compute_queue_duplicate_metrics(traces: list) -> Dict[str, Any]:
        max_queue_size = getattr(DedupPriorityQueue, "MAX_SIZE", 20) or 20
        queue_sizes = []
        generated_total = 0
        duplicate_reasons: Dict[str, int] = defaultdict(int)

        for trace in traces:
            queue_size = int(
                getattr(trace, "queue_size_after_exploration", 0)
                or getattr(trace, "intent_queue_size", 0)
                or 0
            )
            queue_sizes.append(queue_size)
            generated_total += int(
                getattr(trace, "generated_intents", 0)
                or len(getattr(trace, "exploration_intents", []) or [])
            )
            for item in getattr(trace, "fsp_filter_reasons", []) or []:
                if not isinstance(item, dict):
                    continue
                reason = str(item.get("reason", ""))
                if reason in {
                    "fsp_semantic_duplicate",
                    "semantic_duplicate_topic",
                }:
                    duplicate_reasons["fsp_semantic_duplicate"] += 1
                elif reason == "fsp_cross_turn_duplicate":
                    duplicate_reasons["fsp_cross_turn_duplicate"] += 1
                elif reason == "queue_semantic_duplicate":
                    duplicate_reasons["queue_semantic_duplicate"] += 1

        duplicate_count = sum(duplicate_reasons.values())
        peak_queue_size = max(queue_sizes, default=0)
        return {
            "mean_queue_size": (
                sum(queue_sizes) / len(queue_sizes) if queue_sizes else 0.0
            ),
            "peak_queue_pressure": peak_queue_size / max_queue_size,
            "queue_saturation_turns": sum(
                1 for size in queue_sizes if size >= max_queue_size
            ),
            "duplicate_topic_count": duplicate_count,
            "duplicate_topic_rate": (
                duplicate_count / generated_total
                if generated_total
                else None
            ),
            "fsp_semantic_duplicate_count": duplicate_reasons["fsp_semantic_duplicate"],
            "fsp_cross_turn_duplicate_count": duplicate_reasons["fsp_cross_turn_duplicate"],
            "queue_semantic_duplicate_count": duplicate_reasons["queue_semantic_duplicate"],
        }

    @staticmethod
    def _compute_active_cost_metrics(result: ConversationResult) -> Dict[str, Any]:
        active_modules = {"fsp", "ite_search", "ite_push_score"}
        module_usage = result.llm_usage_by_module or {}
        module_calls: Dict[str, int] = {}
        module_tokens: Dict[str, int] = {}

        for module, usage_by_model in module_usage.items():
            totals = flatten_usage(usage_by_model)
            module_calls[module] = totals["calls"]
            module_tokens[module] = totals["total_tokens"]

        active_llm_calls = sum(
            module_calls.get(module, 0) for module in active_modules
        )
        active_llm_total_tokens = sum(
            module_tokens.get(module, 0) for module in active_modules
        )

        first_push_covered_need_count = 0
        prev_covered: Set[str] = set()
        for trace in result.turn_traces:
            newly_covered = trace.cumulative_covered_needs - prev_covered
            if trace.is_push_turn and trace.push_type == "direct":
                first_push_covered_need_count += len(newly_covered)
            prev_covered = set(trace.cumulative_covered_needs)

        return {
            "active_llm_calls": active_llm_calls,
            "active_llm_total_tokens": active_llm_total_tokens,
            "module_llm_calls": module_calls,
            "module_llm_total_tokens": module_tokens,
            "cost_per_new_need": (
                active_llm_total_tokens / first_push_covered_need_count
                if first_push_covered_need_count
                else None
            ),
        }

    @staticmethod
    def _compute_fact_accuracy(traces: list) -> float:
        """
        Compute fact accuracy across all turns.

        accuracy = conveyed / (conveyed + distorted)
        """
        total_conveyed = sum(len(t.judgment.facts_conveyed) for t in traces)
        total_distorted = sum(len(t.judgment.facts_distorted) for t in traces)
        denominator = total_conveyed + total_distorted
        if denominator == 0:
            return 1.0
        return total_conveyed / denominator

    @staticmethod
    def _compute_hallucination_rate(traces: list) -> float:
        """
        Compute hallucination rate across all turns.

        rate = hallucinated / (conveyed + distorted + hallucinated)
        """
        total_conveyed = sum(len(t.judgment.facts_conveyed) for t in traces)
        total_distorted = sum(len(t.judgment.facts_distorted) for t in traces)
        total_hallucinated = sum(len(t.judgment.hallucinated_claims) for t in traces)
        total_claims = total_conveyed + total_distorted + total_hallucinated
        if total_claims == 0:
            return 0.0
        return total_hallucinated / total_claims

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    @staticmethod
    def _run_statistical_tests(
        per_scenario: List[ScenarioMetrics],
    ) -> Dict[str, float]:
        """
        Run paired Wilcoxon signed-rank tests: Full vs each control condition.

        Tests efficiency metrics (expecting Full < control) and coverage/quality
        metrics (expecting Full > control).
        Gracefully degrades if scipy is unavailable.
        """
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            logger.info("scipy not available; skipping statistical tests")
            return {}

        by_condition: Dict[str, Dict[str, ScenarioMetrics]] = defaultdict(dict)
        for m in per_scenario:
            by_condition[m.condition][m.scenario_id] = m

        full_map = by_condition.get("Full", {})
        if not full_map:
            full_map = by_condition.get("Full-single-idle", {})
        if not full_map:
            return {}

        control_conditions = ["Baseline", "Blind", "Paralyzed"]
        test_metrics = [
            ("t80", "greater"),
            ("t100", "greater"),
            ("user_effort", "greater"),
            ("fact_accuracy", "less"),
            ("hallucination_rate", "greater"),
            ("total_coverage", "less"),
            ("must_have_coverage", "less"),
            ("anticipation_recall", "less"),
            ("push_precision", "less"),
            ("push_coverage", "less"),
            ("active_llm_total_tokens", "greater"),
            ("peak_queue_pressure", "greater"),
            ("duplicate_topic_rate", "greater"),
            ("mean_push_novelty_ratio", "less"),
            ("cost_per_new_need", "greater"),
        ]

        results: Dict[str, float] = {}

        def _run_pair(
            left_label: str,
            left_map: Dict[str, ScenarioMetrics],
            right_label: str,
            right_map: Dict[str, ScenarioMetrics],
        ) -> None:
            paired_ids = sorted(set(left_map) & set(right_map))
            if len(paired_ids) < 5:
                return

            for metric_label, alternative in test_metrics:
                pairs = [
                    (
                        getattr(left_map[sid], metric_label),
                        getattr(right_map[sid], metric_label),
                    )
                    for sid in paired_ids
                ]
                pairs = [
                    (left, right)
                    for left, right in pairs
                    if left is not None and right is not None
                ]
                if len(pairs) < 5:
                    continue
                ctrl_vals = [float(left) for left, _right in pairs]
                full_vals = [float(right) for _left, right in pairs]

                differences = [a - b for a, b in zip(ctrl_vals, full_vals)]
                if all(d == 0 for d in differences):
                    results[f"{metric_label}__{right_label}_vs_{left_label}"] = 1.0
                    continue

                try:
                    _stat, p_value = wilcoxon(
                        ctrl_vals, full_vals, alternative=alternative,
                    )
                    results[f"{metric_label}__{right_label}_vs_{left_label}"] = p_value
                except Exception as exc:
                    logger.warning(
                        "Wilcoxon test failed for %s (%s vs %s): %s",
                        metric_label, right_label, left_label, exc,
                    )

        full_label = "Full" if "Full" in by_condition else "Full-single-idle"
        for control in control_conditions:
            ctrl_map = by_condition.get(control, {})
            _run_pair(control, ctrl_map, full_label, full_map)

        single_map = by_condition.get("Full-single-idle", {})
        window_map = by_condition.get("Full-idle-window", {})
        if single_map and window_map:
            _run_pair("Full-single-idle", single_map, "Full-idle-window", window_map)

        return results

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------

    def _load_scenario_map(self) -> Dict[str, Scenario]:
        """Load scenarios into a lookup map by scenario_id."""
        if not self.scenario_dir or not self.scenario_dir.exists():
            return {}

        result: Dict[str, Scenario] = {}
        for path in sorted(self.scenario_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                scenario = Scenario.from_dict(data)
                result[scenario.scenario_id] = scenario
            except Exception as exc:
                logger.warning("Skipping scenario %s: %s", path.name, exc)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(raw: str, base: Path) -> Path:
        """Resolve a possibly-relative path against a base directory."""
        p = Path(raw)
        if p.is_absolute():
            return p
        return (base / p).resolve()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProactiveBench post-hoc evaluator",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing detailed_results.json",
    )
    parser.add_argument(
        "--scenario-dir",
        default=None,
        help="Directory containing scenario JSON files (needed for anticipation precision)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the evaluation report JSON (default: <results-dir>/report.json)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    evaluator = ProactiveBenchEvaluator(
        results_dir=args.results_dir,
        scenario_dir=args.scenario_dir,
    )

    report = evaluator.compute_all()

    # Print summary to stdout
    evaluator.print_summary(report)

    # Save report JSON
    output_path = args.output
    if output_path is None:
        eval_dir = Path(__file__).resolve().parent
        results_dir = ProactiveBenchEvaluator._resolve_path(args.results_dir, eval_dir)
        output_path = str(results_dir / "report.json")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Report saved to %s", output_file)


if __name__ == "__main__":
    main()
