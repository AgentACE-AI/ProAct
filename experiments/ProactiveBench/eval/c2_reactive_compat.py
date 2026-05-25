"""
Utilities for the 2026-04-13 C2 reactive compatibility dev12 gate.

This module compares a frozen baseline C2 report against a candidate
dev12 C2 report and computes:

- per-scenario metric deltas
- aggregate deltas over the fixed 12-scenario set
- the task-specific acceptance gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEV12_REGRESSION_TARGETS = [
    "legal_form_completion_04",
    "negotiate_legal_settlement_04",
    "negotiate_vendor_02",
]

DEV12_POSITIVE_CONTROLS = [
    "insure_claim_timeline_01",
    "insure_denial_language_05",
    "negotiate_salary_01",
]

DEV12_SCENARIOS = [
    *DEV12_REGRESSION_TARGETS,
    *DEV12_POSITIVE_CONTROLS,
    "finance_basic_01",
    "finance_buy_vs_rent_03",
    "finance_estate_05",
    "finance_tax_loss_04",
    "legal_contract_review_01",
    "negotiate_equity_conflict_03",
]

# "materially worsen" should be explicit and conservative. We allow a very small
# absolute increase in the dev12 mean hallucination rate to avoid treating tiny
# judge noise as an automatic gate failure.
MATERIAL_HALLUCINATION_RATE_DELTA_TOLERANCE = 0.01


def _load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_c2_per_scenario(path: Path) -> Dict[str, Dict[str, Any]]:
    report = _load_report(path)
    per_scenario = report.get("per_scenario", [])
    c2_entries = [entry for entry in per_scenario if entry.get("condition") == "C2"]
    by_scenario: Dict[str, Dict[str, Any]] = {}
    for entry in c2_entries:
        scenario_id = entry["scenario_id"]
        if scenario_id in by_scenario:
            raise ValueError(f"Duplicate C2 scenario entry in {path}: {scenario_id}")
        by_scenario[scenario_id] = entry
    return by_scenario


def _require_scenarios(
    by_scenario: Dict[str, Dict[str, Any]],
    scenarios: Iterable[str],
    *,
    label: str,
) -> None:
    missing = [scenario_id for scenario_id in scenarios if scenario_id not in by_scenario]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing {label} scenarios: {missing_text}")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _scenario_summary(entry: Dict[str, Any]) -> Dict[str, float]:
    return {
        "must_have_coverage": float(entry["must_have_coverage"]),
        "t100": float(entry["t100"]),
        "user_effort": float(entry["user_effort"]),
        "hallucination_rate": float(entry["hallucination_rate"]),
        "total_coverage": float(entry["total_coverage"]),
        "fact_accuracy": float(entry["fact_accuracy"]),
    }


def _aggregate(entries: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    entries = list(entries)
    return {
        "mean_must_have_coverage": _mean(
            float(entry["must_have_coverage"]) for entry in entries
        ),
        "mean_t100": _mean(float(entry["t100"]) for entry in entries),
        "mean_user_effort": _mean(float(entry["user_effort"]) for entry in entries),
        "mean_hallucination_rate": _mean(
            float(entry["hallucination_rate"]) for entry in entries
        ),
        "mean_total_coverage": _mean(float(entry["total_coverage"]) for entry in entries),
        "mean_fact_accuracy": _mean(float(entry["fact_accuracy"]) for entry in entries),
    }


def compare_dev12_reports(
    baseline_report_path: str | Path,
    candidate_report_path: str | Path,
) -> Dict[str, Any]:
    baseline_path = Path(baseline_report_path)
    candidate_path = Path(candidate_report_path)

    baseline = _load_c2_per_scenario(baseline_path)
    candidate = _load_c2_per_scenario(candidate_path)

    _require_scenarios(baseline, DEV12_SCENARIOS, label="baseline")
    _require_scenarios(candidate, DEV12_SCENARIOS, label="candidate")

    baseline_entries = [baseline[scenario_id] for scenario_id in DEV12_SCENARIOS]
    candidate_entries = [candidate[scenario_id] for scenario_id in DEV12_SCENARIOS]
    baseline_aggregate = _aggregate(baseline_entries)
    candidate_aggregate = _aggregate(candidate_entries)

    scenario_deltas: List[Dict[str, Any]] = []
    regression_targets_improved = 0
    positive_controls_ok = True

    for scenario_id in DEV12_SCENARIOS:
        baseline_entry = baseline[scenario_id]
        candidate_entry = candidate[scenario_id]
        baseline_metrics = _scenario_summary(baseline_entry)
        candidate_metrics = _scenario_summary(candidate_entry)

        deltas = {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
        }
        deltas["t100"] = candidate_metrics["t100"] - baseline_metrics["t100"]

        regression_improved = (
            deltas["must_have_coverage"] > 0 or deltas["t100"] < 0
        )
        if scenario_id in DEV12_REGRESSION_TARGETS and regression_improved:
            regression_targets_improved += 1

        positive_control_guard_ok = True
        if scenario_id in DEV12_POSITIVE_CONTROLS:
            positive_control_guard_ok = (
                deltas["must_have_coverage"] >= 0 and deltas["t100"] <= 1
            )
            positive_controls_ok = positive_controls_ok and positive_control_guard_ok

        scenario_deltas.append(
            {
                "scenario_id": scenario_id,
                "is_regression_target": scenario_id in DEV12_REGRESSION_TARGETS,
                "is_positive_control": scenario_id in DEV12_POSITIVE_CONTROLS,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "deltas": deltas,
                "regression_target_improved": regression_improved,
                "positive_control_guard_ok": positive_control_guard_ok,
            }
        )

    aggregate_deltas = {
        key: candidate_aggregate[key] - baseline_aggregate[key]
        for key in baseline_aggregate
    }

    mean_user_effort_ok = (
        candidate_aggregate["mean_user_effort"]
        <= baseline_aggregate["mean_user_effort"]
    )
    mean_hallucination_ok = (
        candidate_aggregate["mean_hallucination_rate"]
        <= baseline_aggregate["mean_hallucination_rate"]
        + MATERIAL_HALLUCINATION_RATE_DELTA_TOLERANCE
    )

    gate = {
        "regression_targets_improved": regression_targets_improved,
        "regression_targets_required": 2,
        "regression_targets_pass": regression_targets_improved >= 2,
        "positive_controls_checked": len(DEV12_POSITIVE_CONTROLS),
        "positive_controls_pass": positive_controls_ok,
        "mean_user_effort_pass": mean_user_effort_ok,
        "mean_hallucination_rate_pass": mean_hallucination_ok,
        "hallucination_rule": (
            "Candidate mean hallucination_rate may increase by at most "
            f"{MATERIAL_HALLUCINATION_RATE_DELTA_TOLERANCE:.2f} absolute."
        ),
        "hallucination_tolerance": MATERIAL_HALLUCINATION_RATE_DELTA_TOLERANCE,
    }
    gate["overall_pass"] = all(
        [
            gate["regression_targets_pass"],
            gate["positive_controls_pass"],
            gate["mean_user_effort_pass"],
            gate["mean_hallucination_rate_pass"],
        ]
    )

    return {
        "baseline_report_path": str(baseline_path),
        "candidate_report_path": str(candidate_path),
        "scenario_order": DEV12_SCENARIOS,
        "scenario_deltas": scenario_deltas,
        "aggregate_baseline": baseline_aggregate,
        "aggregate_candidate": candidate_aggregate,
        "aggregate_deltas": aggregate_deltas,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen C2 baseline vs dev12 reactive-compat run."
    )
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    comparison = compare_dev12_reports(
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
    )

    rendered = json.dumps(comparison, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
