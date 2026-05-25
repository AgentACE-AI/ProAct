"""Merge multiple ProactiveBench report.json files into one unified report.

Reads per_scenario rows from each input report, deduplicates by
(scenario_id, condition), and recomputes per_condition / per_domain
aggregates and statistical tests from scratch.

Usage:
    python -m experiments.ProactiveBench.eval.merge_reports \
        --inputs report_batch12.json report_batch34.json \
        --output merged_report.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def load_per_scenario(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("per_scenario", [])


def merge_per_scenario(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for row in all_rows:
        key = f"{row['scenario_id']}__{row['condition']}"
        seen[key] = row
    return sorted(seen.values(), key=lambda r: (r["scenario_id"], r["condition"]))


def aggregate_condition(
    rows: List[Dict[str, Any]], condition: str
) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"condition": condition, "n_scenarios": 0}

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals)

    def _std(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        mu = _mean(vals)
        return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))

    def _mean_opt(vals: List[Optional[float]]) -> Optional[float]:
        filtered = [v for v in vals if v is not None]
        return _mean(filtered) if filtered else None

    t80 = [float(r["t80"]) for r in rows]
    t100 = [float(r["t100"]) for r in rows]

    return {
        "condition": condition,
        "n_scenarios": n,
        "mean_t80": _mean(t80),
        "std_t80": _std(t80),
        "mean_t100": _mean(t100),
        "std_t100": _std(t100),
        "mean_anticipation_precision": _mean_opt(
            [r.get("anticipation_precision") for r in rows]
        ),
        "mean_anticipation_recall": _mean_opt(
            [r.get("anticipation_recall") for r in rows]
        ),
        "mean_fact_accuracy": _mean([r["fact_accuracy"] for r in rows]),
        "mean_hallucination_rate": _mean([r["hallucination_rate"] for r in rows]),
        "mean_user_effort": _mean([float(r["user_effort"]) for r in rows]),
        "mean_total_coverage": _mean([r["total_coverage"] for r in rows]),
        "mean_must_have_coverage": _mean([r["must_have_coverage"] for r in rows]),
    }


def run_statistical_tests(
    per_scenario: List[Dict[str, Any]],
) -> Dict[str, float]:
    by_scenario: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in per_scenario:
        by_scenario[row["scenario_id"]][row["condition"]] = row

    paired = [
        (v["C1"], v["C2"])
        for v in by_scenario.values()
        if "C1" in v and "C2" in v
    ]
    if len(paired) < 5:
        return {}

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {}

    tests = {}
    metrics_greater = [
        ("t80", True),
        ("t100", True),
        ("user_effort", True),
        ("hallucination_rate", True),
    ]
    metrics_less = [
        ("fact_accuracy", False),
        ("total_coverage", False),
        ("must_have_coverage", False),
    ]

    for metric, greater in metrics_greater + metrics_less:
        diffs = [c1[metric] - c2[metric] for c1, c2 in paired]
        if all(d == 0.0 for d in diffs):
            continue
        try:
            alt = "greater" if greater else "less"
            _, p = wilcoxon(diffs, alternative=alt)
            tests[metric] = round(p, 6)
        except Exception:
            pass

    return tests


def build_merged_report(
    per_scenario: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_scenario:
        by_condition[row["condition"]].append(row)

    per_condition = {
        cond: aggregate_condition(rows, cond)
        for cond, rows in sorted(by_condition.items())
    }

    by_domain_cond: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in per_scenario:
        by_domain_cond[row["domain"]][row["condition"]].append(row)

    per_domain: Dict[str, Dict[str, Any]] = {}
    for domain in sorted(by_domain_cond):
        per_domain[domain] = {
            cond: aggregate_condition(rows, cond)
            for cond, rows in sorted(by_domain_cond[domain].items())
        }

    statistical_tests = run_statistical_tests(per_scenario)

    return {
        "per_scenario": per_scenario,
        "per_condition": per_condition,
        "per_domain": per_domain,
        "statistical_tests": statistical_tests,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple ProactiveBench report.json files"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input report.json files to merge",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged report path",
    )
    args = parser.parse_args(argv)

    all_rows: List[Dict[str, Any]] = []
    for path in args.inputs:
        rows = load_per_scenario(path)
        print(f"Loaded {len(rows)} per_scenario rows from {path}")
        all_rows.extend(rows)

    merged = merge_per_scenario(all_rows)
    scenarios = set(r["scenario_id"] for r in merged)
    conditions = set(r["condition"] for r in merged)
    print(f"Merged: {len(merged)} rows, {len(scenarios)} scenarios, conditions={sorted(conditions)}")

    report = build_merged_report(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Written to {args.output}")

    for cond, agg in sorted(report["per_condition"].items()):
        print(f"\n{cond} (n={agg['n_scenarios']}):")
        for k in ["mean_t80", "mean_t100", "mean_user_effort", "mean_hallucination_rate", "mean_must_have_coverage", "mean_anticipation_recall"]:
            v = agg.get(k)
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if report["statistical_tests"]:
        print("\nStatistical tests (Wilcoxon):")
        for metric, p in sorted(report["statistical_tests"].items()):
            sig = "*" if p < 0.05 else ""
            print(f"  {metric}: p={p:.4f} {sig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
