#!/usr/bin/env python3
"""
场景校验器。

对生成的 Fact-Grounded Scenario 执行结构校验（programmatic）和预判合理性校验（LLM）。

用法:
    # 仅结构校验
    python validate_scenario.py --scenario-dir ../data/scenarios/

    # 结构 + LLM 预判校验，并保存报告
    python validate_scenario.py --scenario-dir ../data/scenarios/ --llm-check --report validation_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到 Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.llm_client import LLMClient
from core.config import config

try:
    from .bench_config import bench_config
    from .models import Scenario, ScenarioValidator
    from .prompts import GenerationPrompts
except ImportError:  # pragma: no cover - keeps direct script execution working
    from bench_config import bench_config
    from models import Scenario, ScenarioValidator
    from prompts import GenerationPrompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_scenarios(scenario_dir: Path) -> List[Scenario]:
    """加载目录下所有场景 JSON 文件。"""
    scenarios: List[Scenario] = []
    if not scenario_dir.exists():
        logger.error(f"场景目录不存在: {scenario_dir}")
        return scenarios

    for path in sorted(scenario_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scenarios.append(Scenario.from_dict(data))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"跳过无效文件 {path.name}: {e}")

    return scenarios


def run_structural_validation(scenario: Scenario) -> List[str]:
    """执行结构校验，返回问题列表。"""
    validator = ScenarioValidator()
    return validator.validate(scenario)


def run_predictability_check(
    scenario: Scenario, llm: LLMClient, model: str,
) -> List[Dict[str, Any]]:
    """
    对场景中每个有 predictable_after 的 UserNeed 执行 LLM 预判合理性校验。

    Returns:
        每个校验项的结果列表，包含 need_id、predecessor_id 和 LLM 评估结果。
    """
    results: List[Dict[str, Any]] = []

    need_map = {n.id: n for n in scenario.user_needs}

    user_profile_text = (
        f"Persona: {scenario.user_profile.persona}; "
        f"Context: {scenario.user_profile.context}; "
        f"Style: {scenario.user_profile.communication_style}"
    )

    for need in scenario.user_needs:
        if need.predictable_after is None:
            continue

        predecessor = need_map.get(need.predictable_after)
        if predecessor is None:
            # 引用了不存在的前序需求 -- 结构校验会捕获这个问题
            results.append({
                "need_id": need.id,
                "predecessor_id": need.predictable_after,
                "error": f"前序需求 '{need.predictable_after}' 不存在",
            })
            continue

        prompt = GenerationPrompts.VALIDATE_PREDICTABILITY.format(
            scenario_description=scenario.description,
            user_profile=user_profile_text,
            predecessor_need=predecessor.description,
            target_need=need.description,
            prediction_reason=need.prediction_reason or "(未提供)",
        )

        try:
            llm_result = llm.chat_json(
                messages=[
                    {"role": "system", "content": "You are a logical reasoning expert. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
            )
            results.append({
                "need_id": need.id,
                "predecessor_id": need.predictable_after,
                "is_reasonable": llm_result.get("is_reasonable"),
                "confidence": llm_result.get("confidence"),
                "reason": llm_result.get("reason"),
                "suggestion": llm_result.get("suggestion"),
            })
        except (RuntimeError, ValueError) as e:
            logger.warning(f"[{scenario.scenario_id}] Need {need.id} LLM 校验失败: {e}")
            results.append({
                "need_id": need.id,
                "predecessor_id": need.predictable_after,
                "error": str(e),
            })

    return results


def run_answerability_check(
    scenario: Scenario, llm: LLMClient, model: str,
) -> List[Dict[str, Any]]:
    """
    对场景中每个 UserNeed 执行 LLM 可答性校验。

    Returns:
        每个校验项的结果列表，包含 need_id 和 LLM 评估结果。
    """
    results: List[Dict[str, Any]] = []

    fact_map = {fact.id: fact for fact in scenario.fact_sheet}
    user_profile_text = (
        f"Persona: {scenario.user_profile.persona}; "
        f"Context: {scenario.user_profile.context}; "
        f"Style: {scenario.user_profile.communication_style}"
    )

    for need in scenario.user_needs:
        key_fact_lines = []
        missing_fact_ids = []
        for fact_id in need.key_fact_ids:
            fact = fact_map.get(fact_id)
            if fact is None:
                missing_fact_ids.append(fact_id)
                continue
            key_fact_lines.append(f"- {fact.id} [{fact.category}]: {fact.fact}")

        if missing_fact_ids:
            results.append({
                "need_id": need.id,
                "error": f"缺少 key facts: {missing_fact_ids}",
            })
            continue

        prompt = GenerationPrompts.VALIDATE_ANSWERABILITY.format(
            scenario_description=scenario.description,
            user_profile=user_profile_text,
            target_need=need.description,
            key_facts_text="\n".join(key_fact_lines),
        )

        try:
            llm_result = llm.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict benchmark data auditor. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
            )
            results.append({
                "need_id": need.id,
                "is_answerable": llm_result.get("is_answerable"),
                "confidence": llm_result.get("confidence"),
                "reason": llm_result.get("reason"),
                "suggestion": llm_result.get("suggestion"),
            })
        except (RuntimeError, ValueError) as e:
            logger.warning(f"[{scenario.scenario_id}] Need {need.id} answerability 校验失败: {e}")
            results.append({
                "need_id": need.id,
                "error": str(e),
            })

    return results


def build_report_entry(
    scenario: Scenario,
    structural_errors: List[str],
    structural_warnings: List[str],
    predictability_checks: Optional[List[Dict[str, Any]]] = None,
    answerability_checks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """构建单个场景的校验报告条目。"""
    entry: Dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "structural_errors": structural_errors,
        "structural_warnings": structural_warnings,
    }
    if predictability_checks is not None:
        entry["predictability_checks"] = predictability_checks
    if answerability_checks is not None:
        entry["answerability_checks"] = answerability_checks
    return entry


def print_summary(report: List[Dict[str, Any]], llm_checked: bool) -> None:
    """打印校验结果摘要表格到控制台。"""
    header_parts = ["Scenario ID", "Structural", "Warnings", "Status"]
    if llm_checked:
        header_parts = [
            "Scenario ID",
            "Structural",
            "Warnings",
            "Predictability",
            "Answerability",
            "Status",
        ]

    col_widths = (
        [24, 14, 10, 10]
        if not llm_checked
        else [24, 14, 10, 16, 16, 10]
    )
    separator = "+" + "+".join("-" * w for w in col_widths) + "+"

    def row(values: List[str]) -> str:
        return "|" + "|".join(v.center(w) for v, w in zip(values, col_widths)) + "|"

    print()
    print(separator)
    print(row(header_parts))
    print(separator)

    for entry in report:
        sid = entry["scenario_id"]
        n_errors = len(entry.get("structural_errors", []))
        n_warnings = len(entry.get("structural_warnings", []))
        struct_cell = "PASS" if n_errors == 0 else f"{n_errors} error(s)"
        warn_cell = str(n_warnings) if n_warnings > 0 else "-"

        if llm_checked and "predictability_checks" in entry:
            pred_checks = entry["predictability_checks"]
            if not pred_checks:
                pred_cell = "N/A"
            else:
                n_ok = sum(
                    1 for c in pred_checks
                    if c.get("is_reasonable") is True and "error" not in c
                )
                pred_cell = f"{n_ok}/{len(pred_checks)}"

            answer_checks = entry.get("answerability_checks", [])
            if not answer_checks:
                answer_cell = "N/A"
            else:
                n_answerable = sum(
                    1 for c in answer_checks
                    if c.get("is_answerable") is True and "error" not in c
                )
                answer_cell = f"{n_answerable}/{len(answer_checks)}"
        else:
            pred_cell = ""
            answer_cell = ""

        if n_errors > 0:
            status = "FAIL"
        elif llm_checked and "predictability_checks" in entry:
            has_unreasonable = any(
                c.get("is_reasonable") is False
                for c in entry.get("predictability_checks", [])
                if "error" not in c
            )
            has_unanswerable = any(
                c.get("is_answerable") is False
                for c in entry.get("answerability_checks", [])
                if "error" not in c
            )
            status = "WARN" if has_unreasonable or has_unanswerable else "PASS"
        else:
            status = "PASS"

        cells = [sid, struct_cell, warn_cell, status]
        if llm_checked:
            cells = [sid, struct_cell, warn_cell, pred_cell, answer_cell, status]

        print(row(cells))

    print(separator)

    total = len(report)
    passed = sum(1 for e in report if not e.get("structural_errors", []))
    failed = total - passed
    total_warnings = sum(len(e.get("structural_warnings", [])) for e in report)
    print(f"\nTotal: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Warnings: {total_warnings}")

    if llm_checked:
        all_checks = [
            c
            for e in report
            for c in e.get("predictability_checks", [])
            if "error" not in c
        ]
        if all_checks:
            n_reasonable = sum(1 for c in all_checks if c.get("is_reasonable"))
            print(f"Predictability checks: {n_reasonable}/{len(all_checks)} reasonable")

        all_answerability_checks = [
            c
            for e in report
            for c in e.get("answerability_checks", [])
            if "error" not in c
        ]
        if all_answerability_checks:
            n_answerable = sum(
                1 for c in all_answerability_checks if c.get("is_answerable")
            )
            print(
                f"Answerability checks: {n_answerable}/{len(all_answerability_checks)} answerable"
            )

    print()


def main():
    bc = bench_config
    parser = argparse.ArgumentParser(description="校验 Fact-Grounded Scenarios")
    parser.add_argument(
        "--scenario-dir", type=Path, default=Path(bc.paths.scenarios),
        help="场景 JSON 文件目录",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="输出详细 JSON 报告的路径（可选）",
    )
    parser.add_argument(
        "--llm-check", action="store_true",
        help="启用 LLM 预判合理性校验（需要 API key）",
    )
    parser.add_argument(
        "--model", type=str, default=bc.llm.validation_model,
        help="LLM 校验使用的模型",
    )
    args = parser.parse_args()

    # 加载场景
    scenarios = load_scenarios(args.scenario_dir)
    if not scenarios:
        logger.error("未找到任何有效场景文件")
        sys.exit(1)

    logger.info(f"加载了 {len(scenarios)} 个场景")

    # 初始化 LLM（仅在 llm-check 模式下）
    llm: Optional[LLMClient] = None
    if args.llm_check:
        llm = LLMClient(config.llm)
        logger.info(f"LLM 预判/可答性校验已启用，模型: {args.model}")

    # 校验每个场景
    report: List[Dict[str, Any]] = []
    for scenario in scenarios:
        logger.info(f"校验 [{scenario.scenario_id}]...")

        # 结构校验 — separate hard errors from warnings
        all_issues = run_structural_validation(scenario)
        structural_errors = [i for i in all_issues if not i.startswith("[WARNING]")]
        structural_warnings = [i for i in all_issues if i.startswith("[WARNING]")]
        if structural_errors:
            for issue in structural_errors:
                logger.warning(f"  [{scenario.scenario_id}] {issue}")
        elif structural_warnings:
            for w in structural_warnings:
                logger.warning(f"  [{scenario.scenario_id}] {w}")
            logger.info(f"  [{scenario.scenario_id}] 结构校验通过 (with {len(structural_warnings)} warning(s))")
        else:
            logger.info(f"  [{scenario.scenario_id}] 结构校验通过")

        # LLM 预判校验
        predictability_checks = None
        answerability_checks = None
        if llm is not None:
            predictability_checks = run_predictability_check(
                scenario, llm, model=args.model,
            )
            n_reasonable = sum(
                1 for c in predictability_checks
                if c.get("is_reasonable") is True and "error" not in c
            )
            logger.info(
                f"  [{scenario.scenario_id}] 预判校验: "
                f"{n_reasonable}/{len(predictability_checks)} reasonable"
            )
            answerability_checks = run_answerability_check(
                scenario, llm, model=args.model,
            )
            n_answerable = sum(
                1 for c in answerability_checks
                if c.get("is_answerable") is True and "error" not in c
            )
            logger.info(
                f"  [{scenario.scenario_id}] 可答性校验: "
                f"{n_answerable}/{len(answerability_checks)} answerable"
            )

        report.append(
            build_report_entry(
                scenario,
                structural_errors,
                structural_warnings,
                predictability_checks,
                answerability_checks,
            )
        )

    # 打印摘要
    print_summary(report, llm_checked=args.llm_check)

    # 保存报告
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"详细报告已保存到 {report_path}")

    # LLM 使用报告
    if llm is not None:
        print(llm.format_usage_report())


if __name__ == "__main__":
    main()
