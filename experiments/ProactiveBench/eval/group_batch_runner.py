"""Run ProactiveBench by scenario groups and aggregate group-level reports."""

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.eval.scenario_groups import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SCENARIO_DIR,
    load_manifest,
)


logger = logging.getLogger(__name__)

SUMMARY_SCHEMA_VERSION = 1
DEFAULT_CONDITIONS = [
    "Baseline",
    "Blind",
    "Paralyzed",
    "Full-single-idle",
    "Full-idle-window",
]
MEAN_METRICS = [
    "mean_t80",
    "mean_t100",
    "mean_anticipation_precision",
    "mean_anticipation_recall",
    "mean_fact_accuracy",
    "mean_hallucination_rate",
    "mean_user_effort",
    "mean_total_coverage",
    "mean_must_have_coverage",
    "mean_wall_clock_seconds",
    "mean_llm_calls",
    "mean_llm_total_tokens",
    "mean_stored_facts",
    "mean_processed_intents",
    "mean_peak_queue_size",
    "mean_push_novelty_ratio",
    "low_novelty_push_rate",
    "mean_queue_size",
    "mean_peak_queue_pressure",
    "mean_queue_saturation_turns",
    "mean_duplicate_topic_count",
    "mean_duplicate_topic_rate",
    "mean_fsp_semantic_duplicate_count",
    "mean_fsp_cross_turn_duplicate_count",
    "mean_queue_semantic_duplicate_count",
    "mean_active_llm_calls",
    "mean_active_llm_total_tokens",
    "mean_cost_per_new_need",
]
SCENARIO_TO_MEAN_METRIC = {
    "mean_t80": "t80",
    "mean_t100": "t100",
    "mean_anticipation_precision": "anticipation_precision",
    "mean_anticipation_recall": "anticipation_recall",
    "mean_fact_accuracy": "fact_accuracy",
    "mean_hallucination_rate": "hallucination_rate",
    "mean_user_effort": "user_effort",
    "mean_total_coverage": "total_coverage",
    "mean_must_have_coverage": "must_have_coverage",
    "mean_wall_clock_seconds": "wall_clock_seconds",
    "mean_llm_calls": "llm_calls",
    "mean_llm_total_tokens": "llm_total_tokens",
    "mean_stored_facts": "stored_facts",
    "mean_processed_intents": "processed_intents",
    "mean_peak_queue_size": "peak_queue_size",
    "mean_push_novelty_ratio": "mean_push_novelty_ratio",
    "low_novelty_push_rate": "low_novelty_push_rate",
    "mean_queue_size": "mean_queue_size",
    "mean_peak_queue_pressure": "peak_queue_pressure",
    "mean_queue_saturation_turns": "queue_saturation_turns",
    "mean_duplicate_topic_count": "duplicate_topic_count",
    "mean_duplicate_topic_rate": "duplicate_topic_rate",
    "mean_fsp_semantic_duplicate_count": "fsp_semantic_duplicate_count",
    "mean_fsp_cross_turn_duplicate_count": "fsp_cross_turn_duplicate_count",
    "mean_queue_semantic_duplicate_count": "queue_semantic_duplicate_count",
    "mean_active_llm_calls": "active_llm_calls",
    "mean_active_llm_total_tokens": "active_llm_total_tokens",
    "mean_cost_per_new_need": "cost_per_new_need",
}


@dataclass(frozen=True)
class GroupRunSpec:
    group_type: str
    group_name: str
    selector: str
    scenario_ids: List[str]
    output_dir: Path


def build_group_run_specs(
    *,
    manifest: Dict[str, Any],
    output_root: Path,
    group_types: Optional[Iterable[str]] = None,
) -> List[GroupRunSpec]:
    selected_group_types = list(group_types or manifest.get("group_types", {}).keys())
    specs: List[GroupRunSpec] = []

    for group_type in selected_group_types:
        groups = manifest["group_types"][group_type]["groups"]
        for group_name, scenario_ids in sorted(groups.items()):
            specs.append(
                GroupRunSpec(
                    group_type=group_type,
                    group_name=group_name,
                    selector=f"{group_type}:{group_name}",
                    scenario_ids=list(scenario_ids),
                    output_dir=Path(output_root) / _safe_path_part(group_type) / _safe_path_part(group_name),
                )
            )
    return specs


def aggregate_group_reports(
    *,
    manifest: Dict[str, Any],
    output_root: Path,
    group_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    specs = build_group_run_specs(
        manifest=manifest,
        output_root=output_root,
        group_types=group_types,
    )
    summary: Dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "output_root": str(Path(output_root)),
        "group_types": {},
    }

    specs_by_type: Dict[str, List[GroupRunSpec]] = {}
    for spec in specs:
        specs_by_type.setdefault(spec.group_type, []).append(spec)

    for group_type, group_specs in sorted(specs_by_type.items()):
        group_entries: List[Dict[str, Any]] = []
        for spec in group_specs:
            report_path = spec.output_dir / "report.json"
            if not report_path.exists():
                raise FileNotFoundError(f"Missing report for {spec.selector}: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            group_entries.append(_group_entry_from_report(spec, report, report_path))

        summary["group_types"][group_type] = {
            "n_groups": len(group_entries),
            "n_scenario_condition_rows": sum(
                len(group_entry["per_scenario"])
                for group_entry in group_entries
            ),
            "groups": [
                {
                    key: value
                    for key, value in group_entry.items()
                    if key != "per_scenario"
                }
                for group_entry in group_entries
            ],
            "macro_average": _macro_average(group_entries),
            "micro_average": _micro_average(group_entries),
        }

    return summary


def write_summary_outputs(summary: Dict[str, Any], output_root: Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "group_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(summary, output_root / "group_summary.csv")


def run_all_groups(
    *,
    manifest: Dict[str, Any],
    scenario_dir: Path,
    output_root: Path,
    group_types: Optional[Iterable[str]],
    conditions: List[str],
    seed: int,
    judge_model: str,
    simulator_model: str,
    max_turns: Optional[int],
    max_queries_per_search: int,
    max_intents_per_idle: int,
    idle_trigger_seconds: float,
    max_idle_ticks_per_window: int,
    max_direct_pushes_per_idle_window: int,
    min_idle_tick_interval_seconds: float,
    max_intents_per_idle_window: int,
    stop_on_consecutive_no_new_facts: int,
    cold_start_turns: int,
    cold_start_max_intents: int,
    min_anchor_tokens_for_fsp: int,
    cross_turn_duplicate_jaccard: float,
    max_total_searches: int,
    aggregate_only: bool,
    evaluate_only: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    specs = build_group_run_specs(
        manifest=manifest,
        output_root=output_root,
        group_types=group_types,
    )

    for index, spec in enumerate(specs, start=1):
        logger.info(
            "[%d/%d] %s (%d scenario IDs)",
            index,
            len(specs),
            spec.selector,
            len(spec.scenario_ids),
        )
        if not dry_run:
            spec.output_dir.mkdir(parents=True, exist_ok=True)
        if not aggregate_only and not evaluate_only:
            _run_command(
                _runner_command(
                    scenario_dir=scenario_dir,
                    output_dir=spec.output_dir,
                    conditions=conditions,
                    seed=seed,
                    judge_model=judge_model,
                    simulator_model=simulator_model,
                    max_turns=max_turns,
                    max_queries_per_search=max_queries_per_search,
                    max_intents_per_idle=max_intents_per_idle,
                    idle_trigger_seconds=idle_trigger_seconds,
                    max_idle_ticks_per_window=max_idle_ticks_per_window,
                    max_direct_pushes_per_idle_window=max_direct_pushes_per_idle_window,
                    min_idle_tick_interval_seconds=min_idle_tick_interval_seconds,
                    max_intents_per_idle_window=max_intents_per_idle_window,
                    stop_on_consecutive_no_new_facts=stop_on_consecutive_no_new_facts,
                    cold_start_turns=cold_start_turns,
                    cold_start_max_intents=cold_start_max_intents,
                    min_anchor_tokens_for_fsp=min_anchor_tokens_for_fsp,
                    cross_turn_duplicate_jaccard=cross_turn_duplicate_jaccard,
                    max_total_searches=max_total_searches,
                    scenario_ids=spec.scenario_ids,
                ),
                dry_run=dry_run,
            )
        if not aggregate_only:
            _run_command(
                _evaluator_command(
                    scenario_dir=scenario_dir,
                    output_dir=spec.output_dir,
                ),
                dry_run=dry_run,
            )

    if dry_run:
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "output_root": str(Path(output_root)),
            "group_types": {},
            "dry_run": True,
        }

    summary = aggregate_group_reports(
        manifest=manifest,
        output_root=output_root,
        group_types=group_types,
    )
    write_summary_outputs(summary, output_root)
    return summary


def _group_entry_from_report(
    spec: GroupRunSpec,
    report: Dict[str, Any],
    report_path: Path,
) -> Dict[str, Any]:
    return {
        "group_type": spec.group_type,
        "group_name": spec.group_name,
        "selector": spec.selector,
        "scenario_count": len(spec.scenario_ids),
        "scenario_ids": spec.scenario_ids,
        "output_dir": str(spec.output_dir),
        "report_path": str(report_path),
        "conditions": report.get("per_condition", {}),
        "per_scenario": report.get("per_scenario", []),
    }


def _macro_average(group_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for group_entry in group_entries:
        for condition, aggregate in group_entry["conditions"].items():
            by_condition.setdefault(condition, []).append(aggregate)

    result: Dict[str, Dict[str, Any]] = {}
    for condition, aggregates in sorted(by_condition.items()):
        row: Dict[str, Any] = {"n_groups": len(aggregates)}
        for metric in MEAN_METRICS:
            row[metric] = _mean_optional(aggregate.get(metric) for aggregate in aggregates)
        result[condition] = row
    return result


def _micro_average(group_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for group_entry in group_entries:
        for scenario_row in group_entry["per_scenario"]:
            by_condition.setdefault(scenario_row["condition"], []).append(scenario_row)

    result: Dict[str, Dict[str, Any]] = {}
    for condition, scenario_rows in sorted(by_condition.items()):
        row: Dict[str, Any] = {"n_scenarios": len(scenario_rows)}
        for mean_metric, scenario_metric in SCENARIO_TO_MEAN_METRIC.items():
            row[mean_metric] = _mean_optional(
                scenario_row.get(scenario_metric)
                for scenario_row in scenario_rows
            )
        result[condition] = row
    return result


def _mean_optional(values: Iterable[Any]) -> Optional[float]:
    numeric_values = [value for value in values if isinstance(value, (int, float))]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _write_summary_csv(summary: Dict[str, Any], output_path: Path) -> None:
    fieldnames = [
        "row_type",
        "group_type",
        "group_name",
        "condition",
        "n_groups",
        "n_scenarios",
        "selector",
        *MEAN_METRICS,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_type, group_type_summary in sorted(summary.get("group_types", {}).items()):
            for group_entry in group_type_summary.get("groups", []):
                for condition, aggregate in sorted(group_entry.get("conditions", {}).items()):
                    writer.writerow(
                        _csv_row(
                            row_type="group",
                            group_type=group_type,
                            group_name=group_entry["group_name"],
                            condition=condition,
                            selector=group_entry["selector"],
                            n_scenarios=aggregate.get("n_scenarios"),
                            metrics=aggregate,
                        )
                    )
            for row_type in ["macro_average", "micro_average"]:
                for condition, aggregate in sorted(group_type_summary.get(row_type, {}).items()):
                    writer.writerow(
                        _csv_row(
                            row_type=row_type,
                            group_type=group_type,
                            group_name="",
                            condition=condition,
                            selector="",
                            n_groups=aggregate.get("n_groups"),
                            n_scenarios=aggregate.get("n_scenarios"),
                            metrics=aggregate,
                        )
                    )


def _csv_row(
    *,
    row_type: str,
    group_type: str,
    group_name: str,
    condition: str,
    selector: str,
    metrics: Dict[str, Any],
    n_groups: Any = None,
    n_scenarios: Any = None,
) -> Dict[str, Any]:
    row = {
        "row_type": row_type,
        "group_type": group_type,
        "group_name": group_name,
        "condition": condition,
        "n_groups": n_groups,
        "n_scenarios": n_scenarios,
        "selector": selector,
    }
    for metric in MEAN_METRICS:
        row[metric] = metrics.get(metric)
    return row


def _runner_command(
    *,
    scenario_dir: Path,
    output_dir: Path,
    conditions: List[str],
    seed: int,
    judge_model: str,
    simulator_model: str,
    max_turns: Optional[int],
    max_queries_per_search: int,
    max_intents_per_idle: int,
    idle_trigger_seconds: float,
    max_idle_ticks_per_window: int,
    max_direct_pushes_per_idle_window: int,
    min_idle_tick_interval_seconds: float,
    max_intents_per_idle_window: int,
    stop_on_consecutive_no_new_facts: int,
    cold_start_turns: int,
    cold_start_max_intents: int,
    min_anchor_tokens_for_fsp: int,
    cross_turn_duplicate_jaccard: float,
    max_total_searches: int,
    scenario_ids: List[str],
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.ProactiveBench.eval.runner",
        "--scenario-dir",
        str(scenario_dir),
        "--output-dir",
        str(output_dir),
        "--conditions",
        *conditions,
        "--seed",
        str(seed),
        "--judge-model",
        judge_model,
        "--simulator-model",
        simulator_model,
        "--max-queries-per-search",
        str(max_queries_per_search),
        "--max-intents-per-idle",
        str(max_intents_per_idle),
        "--idle-trigger-seconds",
        str(idle_trigger_seconds),
        "--max-idle-ticks-per-window",
        str(max_idle_ticks_per_window),
        "--max-direct-pushes-per-idle-window",
        str(max_direct_pushes_per_idle_window),
        "--min-idle-tick-interval-seconds",
        str(min_idle_tick_interval_seconds),
        "--max-intents-per-idle-window",
        str(max_intents_per_idle_window),
        "--stop-on-consecutive-no-new-facts",
        str(stop_on_consecutive_no_new_facts),
        "--cold-start-turns",
        str(cold_start_turns),
        "--cold-start-max-intents",
        str(cold_start_max_intents),
        "--min-anchor-tokens-for-fsp",
        str(min_anchor_tokens_for_fsp),
        "--cross-turn-duplicate-jaccard",
        str(cross_turn_duplicate_jaccard),
        "--max-total-searches",
        str(max_total_searches),
    ]
    if max_turns is not None:
        command.extend(["--max-turns", str(max_turns)])
    command.extend(["--only", *scenario_ids])
    return command


def _evaluator_command(*, scenario_dir: Path, output_dir: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "experiments.ProactiveBench.eval.evaluator",
        "--results-dir",
        str(output_dir),
        "--scenario-dir",
        str(scenario_dir),
        "--output",
        str(output_dir / "report.json"),
    ]


def _run_command(command: List[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return
    logger.info("Running: %s", printable)
    subprocess.run(command, check=True)


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ProactiveBench for every scenario group and aggregate macro/micro averages",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--group-types",
        nargs="+",
        default=None,
        help="Group types to run. Default: all group types in manifest.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--simulator-model", default="gpt-4o")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-queries-per-search", type=int, default=1)
    parser.add_argument("--max-intents-per-idle", type=int, default=3)
    parser.add_argument("--idle-trigger-seconds", type=float, default=5.0)
    parser.add_argument("--max-idle-ticks-per-window", type=int, default=4)
    parser.add_argument("--max-direct-pushes-per-idle-window", type=int, default=2)
    parser.add_argument("--min-idle-tick-interval-seconds", type=float, default=3.0)
    parser.add_argument("--max-intents-per-idle-window", type=int, default=6)
    parser.add_argument("--stop-on-consecutive-no-new-facts", type=int, default=2)
    parser.add_argument("--cold-start-turns", type=int, default=2)
    parser.add_argument("--cold-start-max-intents", type=int, default=1)
    parser.add_argument("--min-anchor-tokens-for-fsp", type=int, default=6)
    parser.add_argument("--cross-turn-duplicate-jaccard", type=float, default=0.6)
    parser.add_argument("--max-total-searches", type=int, default=999)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip runner/evaluator and only aggregate existing group report.json files.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip runner and run evaluator + aggregation for existing detailed_results.json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print runner/evaluator commands without executing them.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )
    if args.aggregate_only and args.evaluate_only:
        raise ValueError("Use at most one of --aggregate-only and --evaluate-only")

    manifest = load_manifest(args.manifest)
    summary = run_all_groups(
        manifest=manifest,
        scenario_dir=args.scenario_dir,
        output_root=args.output_root,
        group_types=args.group_types,
        conditions=args.conditions,
        seed=args.seed,
        judge_model=args.judge_model,
        simulator_model=args.simulator_model,
        max_turns=args.max_turns,
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
        aggregate_only=args.aggregate_only,
        evaluate_only=args.evaluate_only,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        logger.info("Group summary written to %s", args.output_root / "group_summary.json")
        logger.info("Group CSV written to %s", args.output_root / "group_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
