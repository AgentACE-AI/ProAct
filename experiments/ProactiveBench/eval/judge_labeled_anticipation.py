"""
Post-hoc judge-labeled anticipation analysis.

The main evaluator's anticipation recall is tied to runtime delivery artifacts
such as delivered_inline_facts and push_fact_ids.  This script computes a
separate, model-agnostic diagnostic metric from CoverageJudge labels:

    judge_labeled_anticipation_recall =
        predictable needs covered with mode="proactive" / predictable needs

Use this for baselines like ProactiveAgent-style that can proactively add text
but do not expose structured delivered fact IDs.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.eval.models import ConversationResult

logger = logging.getLogger(__name__)


@dataclass
class JudgeLabeledAnticipationMetrics:
    scenario_id: str
    condition: str
    domain: str
    n_predictable_needs: int
    n_judge_proactive_needs: int
    n_anticipated_predictable_needs: int
    judge_labeled_anticipation_recall: Optional[float]
    anticipated_predictable_need_ids: List[str]
    non_predictable_proactive_need_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_judge_labeled_anticipation(
    result: ConversationResult,
) -> JudgeLabeledAnticipationMetrics:
    predictable_need_ids = set(result.predictable_need_ids)
    judge_proactive_need_ids = set()

    for trace in result.turn_traces:
        for coverage in trace.judgment.needs_addressed:
            if coverage.mode == "proactive":
                judge_proactive_need_ids.add(coverage.need_id)

    anticipated = sorted(judge_proactive_need_ids & predictable_need_ids)
    non_predictable = sorted(judge_proactive_need_ids - predictable_need_ids)
    recall = (
        len(anticipated) / len(predictable_need_ids)
        if predictable_need_ids
        else None
    )

    return JudgeLabeledAnticipationMetrics(
        scenario_id=result.scenario_id,
        condition=result.condition,
        domain=result.domain,
        n_predictable_needs=len(predictable_need_ids),
        n_judge_proactive_needs=len(judge_proactive_need_ids),
        n_anticipated_predictable_needs=len(anticipated),
        judge_labeled_anticipation_recall=recall,
        anticipated_predictable_need_ids=anticipated,
        non_predictable_proactive_need_ids=non_predictable,
    )


def load_results(results_dir: str) -> List[ConversationResult]:
    path = Path(results_dir) / "detailed_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [ConversationResult.from_dict(item) for item in payload]


def aggregate_by_condition(
    metrics: List[JudgeLabeledAnticipationMetrics],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[JudgeLabeledAnticipationMetrics]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.condition].append(metric)

    aggregates: Dict[str, Dict[str, Any]] = {}
    for condition, items in sorted(grouped.items()):
        recalls = [
            item.judge_labeled_anticipation_recall
            for item in items
            if item.judge_labeled_anticipation_recall is not None
        ]
        total_predictable = sum(item.n_predictable_needs for item in items)
        total_anticipated = sum(item.n_anticipated_predictable_needs for item in items)
        total_judge_proactive = sum(item.n_judge_proactive_needs for item in items)
        scenarios_with_any_proactive = sum(
            1 for item in items if item.n_judge_proactive_needs > 0
        )
        scenarios_with_predictable_proactive = sum(
            1 for item in items if item.n_anticipated_predictable_needs > 0
        )

        aggregates[condition] = {
            "condition": condition,
            "n_scenarios": len(items),
            "mean_judge_labeled_anticipation_recall": (
                sum(recalls) / len(recalls) if recalls else None
            ),
            "micro_judge_labeled_anticipation_recall": (
                total_anticipated / total_predictable
                if total_predictable
                else None
            ),
            "total_predictable_needs": total_predictable,
            "total_anticipated_predictable_needs": total_anticipated,
            "total_judge_proactive_needs": total_judge_proactive,
            "scenarios_with_any_judge_proactive": scenarios_with_any_proactive,
            "scenarios_with_predictable_judge_proactive": (
                scenarios_with_predictable_proactive
            ),
        }

    return aggregates


def build_report(results_dir: str) -> Dict[str, Any]:
    results = load_results(results_dir)
    per_scenario = [
        compute_judge_labeled_anticipation(result) for result in results
    ]
    return {
        "per_scenario": [metric.to_dict() for metric in per_scenario],
        "per_condition": aggregate_by_condition(per_scenario),
    }


def print_summary(report: Dict[str, Any]) -> None:
    print()
    print("Judge-Labeled Anticipation Summary")
    print("=" * 42)
    for condition, aggregate in report["per_condition"].items():
        macro = aggregate["mean_judge_labeled_anticipation_recall"]
        micro = aggregate["micro_judge_labeled_anticipation_recall"]
        macro_text = f"{macro:.4f}" if macro is not None else "-"
        micro_text = f"{micro:.4f}" if micro is not None else "-"
        print(f"\nCondition: {condition}")
        print(f"  N scenarios: {aggregate['n_scenarios']}")
        print(f"  Mean judge-labeled anticipation recall: {macro_text}")
        print(f"  Micro judge-labeled anticipation recall: {micro_text}")
        print(
            "  Anticipated predictable needs: "
            f"{aggregate['total_anticipated_predictable_needs']}/"
            f"{aggregate['total_predictable_needs']}"
        )
        print(
            "  Judge-proactive needs: "
            f"{aggregate['total_judge_proactive_needs']}"
        )
        print(
            "  Scenarios with any judge-proactive need: "
            f"{aggregate['scenarios_with_any_judge_proactive']}/"
            f"{aggregate['n_scenarios']}"
        )
        print(
            "  Scenarios with predictable judge-proactive need: "
            f"{aggregate['scenarios_with_predictable_judge_proactive']}/"
            f"{aggregate['n_scenarios']}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute judge-labeled anticipation metrics from detailed_results.json",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing detailed_results.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON report (default: <results-dir>/judge_labeled_anticipation.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    report = build_report(args.results_dir)
    output = (
        Path(args.output)
        if args.output
        else Path(args.results_dir) / "judge_labeled_anticipation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Judge-labeled anticipation report saved to %s", output)
    print_summary(report)


if __name__ == "__main__":
    main()
