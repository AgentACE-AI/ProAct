"""Scenario grouping helpers for ProactiveBench.

Builds a frozen grouping manifest from the current scenario set and exposes a
small CLI for printing scenario IDs in formats convenient for batched eval
runs.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.generation.models import Scenario


SCHEMA_VERSION = 1
DEFAULT_SCENARIO_DIR = Path("experiments/ProactiveBench/data/scenarios")
DEFAULT_MANIFEST_PATH = Path("experiments/ProactiveBench/data/scenario_groups.json")

ARCHETYPE_LABELS = {
    "01": "01_foundational_memory",
    "02": "02_translation_and_gap_resolution",
    "03": "03_trace_and_dependency_reasoning",
    "04": "04_handoff_and_consistency_control",
    "05": "05_readiness_and_followthrough",
}

OPPORTUNITY_THRESHOLDS = {
    "high_predictable_ratio": 0.72,
    "high_cross_group_ratio": 0.55,
    "low_predictable_ratio": 0.55,
}

FRAGMENTATION_THRESHOLDS = {
    "high_reveal_groups": 10,
    "high_root_groups": 6,
    "high_singleton_groups": 7,
    "low_reveal_groups": 7,
    "low_root_groups": 3,
}


def load_scenarios(scenario_dir: Path) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    for path in sorted(scenario_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scenarios.append(
            {
                "scenario": Scenario.from_dict(data),
                "source_filename": path.name,
            }
        )
    return sorted(
        scenarios,
        key=lambda entry: entry["scenario"].scenario_id,
    )


def build_scenario_group_manifest(scenario_dir: Path) -> Dict[str, Any]:
    scenario_dir = Path(scenario_dir)
    scenarios = load_scenarios(scenario_dir)

    domain_groups: dict[str, list[str]] = defaultdict(list)
    archetype_groups: dict[str, list[str]] = {label: [] for label in ARCHETYPE_LABELS.values()}
    opportunity_groups: dict[str, list[str]] = {
        "high_opportunity": [],
        "medium_opportunity": [],
        "low_opportunity": [],
    }
    fragmentation_groups: dict[str, list[str]] = {
        "high_fragmentation": [],
        "medium_fragmentation": [],
        "low_fragmentation": [],
    }
    scenario_metrics: dict[str, dict[str, Any]] = {}

    for entry in scenarios:
        scenario = entry["scenario"]
        scenario_id = scenario.scenario_id
        domain_groups[scenario.domain].append(scenario_id)

        archetype_key = ARCHETYPE_LABELS[_extract_archetype_suffix(scenario_id)]
        archetype_groups[archetype_key].append(scenario_id)

        metrics = _scenario_metrics(
            scenario,
            source_filename=entry["source_filename"],
        )
        scenario_metrics[scenario_id] = metrics

        opportunity_groups[_opportunity_bucket(metrics)].append(scenario_id)
        fragmentation_groups[_fragmentation_bucket(metrics)].append(scenario_id)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scenario_dir": str(scenario_dir).replace("\\", "/"),
        "scenario_count": len(scenarios),
        "group_types": {
            "domain": {
                "description": "Group by the scenario `domain` field.",
                "groups": _sorted_group_mapping(domain_groups),
            },
            "archetype": {
                "description": (
                    "Cross-domain task archetypes derived from the stable 01-05 "
                    "scenario suffix family."
                ),
                "groups": _sorted_group_mapping(archetype_groups),
            },
            "opportunity": {
                "description": (
                    "Structural proactive opportunity buckets derived from "
                    "predictable_after density and cross-group predictability."
                ),
                "thresholds": OPPORTUNITY_THRESHOLDS,
                "groups": _sorted_group_mapping(opportunity_groups),
            },
            "fragmentation": {
                "description": (
                    "Structural topic-fragmentation buckets derived from reveal "
                    "group count, root-group count, and singleton-group count."
                ),
                "thresholds": FRAGMENTATION_THRESHOLDS,
                "groups": _sorted_group_mapping(fragmentation_groups),
            },
        },
        "scenario_metrics": dict(sorted(scenario_metrics.items())),
    }
    return manifest


def write_scenario_group_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def select_scenarios(manifest: Dict[str, Any], selector: str) -> List[str]:
    group_type, group_name = _parse_selector(selector)
    group_types = manifest.get("group_types", {})
    if group_type not in group_types:
        raise ValueError(f"Unknown group type '{group_type}'")

    groups = group_types[group_type].get("groups", {})
    if group_name not in groups:
        raise ValueError(f"Unknown group '{group_name}' for group type '{group_type}'")
    return list(groups[group_name])


def format_scenarios(scenarios: Iterable[str], output_format: str = "lines") -> str:
    scenario_list = list(scenarios)
    if output_format == "lines":
        return "\n".join(scenario_list)
    if output_format == "space":
        return " ".join(scenario_list)
    if output_format == "json":
        return json.dumps(scenario_list, ensure_ascii=False, indent=2)
    if output_format == "runner":
        if not scenario_list:
            return "--only"
        rendered = ["--only \\"]
        for index, scenario_id in enumerate(scenario_list):
            suffix = " \\" if index < len(scenario_list) - 1 else ""
            rendered.append(f"  {scenario_id}{suffix}")
        return "\n".join(rendered)
    raise ValueError(f"Unsupported output format '{output_format}'")


def _parse_selector(selector: str) -> tuple[str, str]:
    if ":" not in selector:
        raise ValueError("Selector must use '<group_type>:<group_name>' format")
    group_type, group_name = selector.split(":", 1)
    group_type = group_type.strip()
    group_name = group_name.strip()
    if not group_type or not group_name:
        raise ValueError("Selector must include both group type and group name")
    return group_type, group_name


def _extract_archetype_suffix(scenario_id: str) -> str:
    match = re.search(r"_(\d{2})(?:[^/]*)$", scenario_id)
    if match is None or match.group(1) not in ARCHETYPE_LABELS:
        raise ValueError(f"Could not derive archetype suffix from scenario_id '{scenario_id}'")
    return match.group(1)


def _scenario_metrics(
    scenario: Scenario,
    *,
    source_filename: str,
) -> Dict[str, Any]:
    needs = list(scenario.user_needs)
    predictable_needs = [need for need in needs if need.predictable_after is not None]
    needs_by_id = {need.id: need for need in needs}

    cross_group_predictable_count = 0
    for need in predictable_needs:
        predecessor = needs_by_id.get(need.predictable_after)
        if predecessor is None:
            continue
        if not predecessor.reveal_group or not need.reveal_group:
            continue
        if predecessor.reveal_group != need.reveal_group:
            cross_group_predictable_count += 1

    reveal_groups = list(scenario.reveal_groups)
    group_sizes = [len(group.member_need_ids) for group in reveal_groups]

    predictable_count = len(predictable_needs)
    needs_count = len(needs)
    return {
        "domain": scenario.domain,
        "source_filename": source_filename,
        "needs_count": needs_count,
        "must_have_count": sum(1 for need in needs if need.level == "must-have"),
        "nice_to_have_count": sum(1 for need in needs if need.level != "must-have"),
        "predictable_count": predictable_count,
        "predictable_ratio": _safe_ratio(predictable_count, needs_count),
        "cross_group_predictable_count": cross_group_predictable_count,
        "cross_group_predictable_ratio": _safe_ratio(
            cross_group_predictable_count,
            predictable_count,
        ),
        "reveal_groups_count": len(reveal_groups),
        "root_groups_count": sum(1 for group in reveal_groups if group.trigger_after is None),
        "singleton_groups_count": sum(
            1 for group in reveal_groups if len(group.member_need_ids) == 1
        ),
        "max_group_size": max(group_sizes) if group_sizes else 0,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _opportunity_bucket(metrics: Dict[str, Any]) -> str:
    predictable_ratio = metrics["predictable_ratio"]
    cross_group_ratio = metrics["cross_group_predictable_ratio"]
    if (
        predictable_ratio >= OPPORTUNITY_THRESHOLDS["high_predictable_ratio"]
        and cross_group_ratio >= OPPORTUNITY_THRESHOLDS["high_cross_group_ratio"]
    ):
        return "high_opportunity"
    if predictable_ratio < OPPORTUNITY_THRESHOLDS["low_predictable_ratio"]:
        return "low_opportunity"
    return "medium_opportunity"


def _fragmentation_bucket(metrics: Dict[str, Any]) -> str:
    reveal_groups_count = metrics["reveal_groups_count"]
    root_groups_count = metrics["root_groups_count"]
    singleton_groups_count = metrics["singleton_groups_count"]

    if (
        reveal_groups_count >= FRAGMENTATION_THRESHOLDS["high_reveal_groups"]
        or root_groups_count >= FRAGMENTATION_THRESHOLDS["high_root_groups"]
        or singleton_groups_count >= FRAGMENTATION_THRESHOLDS["high_singleton_groups"]
    ):
        return "high_fragmentation"
    if (
        reveal_groups_count <= FRAGMENTATION_THRESHOLDS["low_reveal_groups"]
        and root_groups_count <= FRAGMENTATION_THRESHOLDS["low_root_groups"]
    ):
        return "low_fragmentation"
    return "medium_fragmentation"


def _sorted_group_mapping(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        group_name: sorted(scenario_ids)
        for group_name, scenario_ids in sorted(groups.items())
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query ProactiveBench scenario groups")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the frozen scenario group manifest")
    build_parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=DEFAULT_SCENARIO_DIR,
        help="Directory containing scenario JSON files",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output path for the generated manifest",
    )

    list_parser = subparsers.add_parser("list-groups", help="List group names available in a manifest")
    list_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to a frozen scenario group manifest",
    )
    list_parser.add_argument(
        "--group-type",
        choices=["domain", "archetype", "opportunity", "fragmentation"],
        required=True,
        help="Which group type to list",
    )

    print_parser = subparsers.add_parser("print", help="Print scenario IDs for a selector")
    print_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to a frozen scenario group manifest",
    )
    print_parser.add_argument(
        "--selector",
        required=True,
        help="Scenario selector in '<group_type>:<group_name>' form",
    )
    print_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["lines", "space", "json", "runner"],
        default="lines",
        help="Output format for selected scenarios",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        manifest = build_scenario_group_manifest(args.scenario_dir)
        write_scenario_group_manifest(manifest, args.output)
        print(args.output)
        return 0

    manifest = load_manifest(args.manifest)
    if args.command == "list-groups":
        groups = manifest["group_types"][args.group_type]["groups"]
        print("\n".join(groups.keys()))
        return 0

    if args.command == "print":
        selected = select_scenarios(manifest, args.selector)
        print(format_scenarios(selected, output_format=args.output_format))
        return 0

    parser.error(f"Unsupported command '{args.command}'")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
