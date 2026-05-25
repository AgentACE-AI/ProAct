"""Summarize ProactiveBench ablation and search-budget experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CORE_METRICS = [
    "n_scenarios",
    "mean_t100",
    "mean_t80",
    "mean_total_coverage",
    "mean_must_have_coverage",
    "mean_anticipation_recall",
    "mean_anticipation_precision",
    "mean_user_effort",
    "mean_processed_intents",
    "mean_stored_facts",
    "mean_active_llm_calls",
    "mean_active_llm_total_tokens",
    "mean_cost_per_new_need",
]

DELTA_METRICS = [
    "mean_t100",
    "mean_t80",
    "mean_total_coverage",
    "mean_must_have_coverage",
    "mean_anticipation_recall",
    "mean_user_effort",
]


def analyze_ablation(root: Path) -> List[Dict[str, Any]]:
    """Return one row per condition report under *root*."""
    rows: List[Dict[str, Any]] = []
    for report_path in _iter_report_paths(root):
        report = _load_report(report_path)
        for raw_condition, aggregate in sorted(report.get("per_condition", {}).items()):
            row = _row_from_aggregate(aggregate)
            row.update({
                "condition": _display_condition(raw_condition),
                "raw_condition": raw_condition,
                "report_path": str(report_path),
            })
            rows.append(row)
    return sorted(rows, key=lambda row: _condition_sort_key(row["condition"]))


def analyze_scaling(root: Path) -> List[Dict[str, Any]]:
    """Return one row per budget_N report under *root*, with deltas vs budget 0."""
    rows: List[Dict[str, Any]] = []
    for report_path in _iter_report_paths(root):
        budget = _budget_from_path(report_path)
        if budget is None:
            continue
        report = _load_report(report_path)
        aggregate = _single_condition_aggregate(report, report_path)
        row = _row_from_aggregate(aggregate)
        row.update({
            "max_total_searches": budget,
            "condition": _display_condition(str(aggregate.get("condition", ""))),
            "raw_condition": aggregate.get("condition"),
            "report_path": str(report_path),
        })
        rows.append(row)

    rows.sort(key=lambda row: row["max_total_searches"])
    baseline = next(
        (row for row in rows if row["max_total_searches"] == 0),
        rows[0] if rows else None,
    )
    if baseline is not None:
        for row in rows:
            for metric in DELTA_METRICS:
                value = row.get(metric)
                base_value = baseline.get(metric)
                delta_key = f"delta_{metric}_vs_budget_0"
                row[delta_key] = (
                    value - base_value
                    if isinstance(value, (int, float))
                    and isinstance(base_value, (int, float))
                    else None
                )
    return rows


def write_summary_outputs(rows: List[Dict[str, Any]], output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(rows, output_dir / f"{stem}.csv")
    (output_dir / f"{stem}.md").write_text(_markdown_table(rows), encoding="utf-8")


def _iter_report_paths(root: Path) -> Iterable[Path]:
    root = Path(root)
    if (root / "report.json").exists():
        yield root / "report.json"
        return
    yield from sorted(root.glob("*/report.json"))


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _single_condition_aggregate(report: Dict[str, Any], report_path: Path) -> Dict[str, Any]:
    per_condition = report.get("per_condition", {})
    if len(per_condition) != 1:
        raise ValueError(f"Expected exactly one condition in {report_path}")
    return next(iter(per_condition.values()))


def _row_from_aggregate(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    return {metric: aggregate.get(metric) for metric in CORE_METRICS}


def _display_condition(condition: str) -> str:
    return "Full" if condition == "Full-single-idle" else condition


def _condition_sort_key(condition: str) -> tuple[int, str]:
    order = {
        "Baseline": 0,
        "Paralyzed": 1,
        "Blind": 2,
        "Full": 3,
    }
    return (order.get(condition, 99), condition)


def _budget_from_path(report_path: Path) -> Optional[int]:
    for part in reversed(report_path.parts):
        match = re.fullmatch(r"budget_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = _fieldnames(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    fieldnames = [
        field
        for field in _fieldnames(rows)
        if field not in {"report_path", "raw_condition"}
    ]
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_markdown_value(row.get(field)) for field in fieldnames)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = [
        "condition",
        "raw_condition",
        "max_total_searches",
        *CORE_METRICS,
        *[f"delta_{metric}_vs_budget_0" for metric in DELTA_METRICS],
        "report_path",
    ]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ProactiveBench experiment report.json files",
    )
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--scaling-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.ablation_root and not args.scaling_root:
        raise ValueError("Provide --ablation-root, --scaling-root, or both")

    if args.ablation_root:
        rows = analyze_ablation(args.ablation_root)
        write_summary_outputs(rows, args.output_dir, "ablation_summary")
        print(f"Wrote ablation summary for {len(rows)} condition(s)")

    if args.scaling_root:
        rows = analyze_scaling(args.scaling_root)
        write_summary_outputs(rows, args.output_dir, "scaling_summary")
        print(f"Wrote scaling summary for {len(rows)} budget point(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
