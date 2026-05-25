#!/usr/bin/env python3
"""
场景生成器。

从种子描述生成完整的 Fact-Grounded Scenario（fact_sheet + user_needs）。

用法:
    python generate_scenarios.py --seeds ../data/seeds/seed_descriptions.json --output ../data/scenarios/
    python generate_scenarios.py --seeds ../data/seeds/seed_descriptions.json --output ../data/tmp_scenarios/ --model gemini-2.5-pro --only relocate_city_01
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

# 添加项目根目录到 Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.llm_client import LLMClient
from core.config import config

try:
    from .bench_config import bench_config
    from .models import (
        Fact,
        RevealGroupMeta,
        Scenario,
        ScenarioMetadata,
        ScenarioValidator,
        SimulatorConfig,
        UserNeed,
        UserProfile,
    )
    from .prompts import GenerationPrompts
except ImportError:  # pragma: no cover - keeps direct script execution working
    from bench_config import bench_config
    from models import (
        Fact,
        RevealGroupMeta,
        Scenario,
        ScenarioMetadata,
        ScenarioValidator,
        SimulatorConfig,
        UserNeed,
        UserProfile,
    )
    from prompts import GenerationPrompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """场景生成器：seed → complete scenario。"""

    def __init__(
        self, llm: LLMClient, model: str = "gpt-4o", temperature: float = 0.7,
        max_retries: int = 2,
        num_facts_range: tuple[int, int] = (12, 20),
        num_needs_range: tuple[int, int] = (5, 8),
    ):
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.num_facts_range = num_facts_range
        self.num_needs_range = num_needs_range
        self.validator = ScenarioValidator()

    def generate_fact_sheet(
        self, domain: str, description: str, user_profile: str,
        num_facts: int = 16,
    ) -> list[Fact]:
        """生成事实表。"""
        category_hints = GenerationPrompts.CATEGORY_HINTS.get(domain, "根据场景自行确定合适的类别")
        prompt = GenerationPrompts.GENERATE_FACT_SHEET.format(
            domain=domain,
            description=description,
            user_profile=user_profile,
            num_facts=num_facts,
            category_hints=category_hints,
        )

        result = self.llm.chat_json(
            messages=[
                {"role": "system", "content": "You are a dataset design expert. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=self.temperature,
        )

        facts = []
        for item in result.get("fact_sheet", []):
            try:
                facts.append(Fact.from_dict(item))
            except (KeyError, TypeError) as e:
                logger.warning(f"跳过无效 fact: {item} ({e})")
        return facts

    def generate_user_needs(
        self, domain: str, description: str, user_profile: str,
        fact_sheet: list[Fact], num_needs: int = 6,
    ) -> tuple[list[UserNeed], list[RevealGroupMeta]]:
        """生成用户信息需求与 grouped reveal 元数据。"""
        fact_sheet_text = "\n".join(
            f"- {f.id} [{f.category}]: {f.fact}" for f in fact_sheet
        )
        prompt = GenerationPrompts.GENERATE_USER_NEEDS.format(
            domain=domain,
            description=description,
            user_profile=user_profile,
            fact_sheet_text=fact_sheet_text,
            num_needs=num_needs,
            min_must_have=max(3, math.ceil(num_needs * 0.6)),
            min_predictable=max(2, math.ceil(num_needs * 0.4)),
        )

        result = self.llm.chat_json(
            messages=[
                {"role": "system", "content": "You are a user behavior analysis expert. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=max(0.1, self.temperature - 0.2),  # needs 生成用略低温度保证结构
        )

        needs = []
        for item in result.get("user_needs", []):
            try:
                needs.append(UserNeed.from_dict(item))
            except (KeyError, TypeError) as e:
                logger.warning(f"跳过无效 need: {item} ({e})")

        reveal_groups = []
        for item in result.get("reveal_groups", []):
            try:
                reveal_groups.append(RevealGroupMeta.from_dict(item))
            except (KeyError, TypeError) as e:
                logger.warning(f"跳过无效 reveal_group: {item} ({e})")

        if not reveal_groups:
            reveal_groups = self._derive_reveal_groups(needs)

        return needs, reveal_groups

    def generate_scenario(self, seed: dict) -> Scenario | None:
        """
        从种子描述生成完整场景。

        Args:
            seed: 种子描述 dict，包含 scenario_id, domain, description, user_profile

        Returns:
            完整 Scenario 或 None（如果生成失败）
        """
        sid = seed["scenario_id"]
        domain = seed["domain"]
        description = seed["description"]
        profile_data = seed["user_profile"]
        user_profile_text = (
            f"Persona: {profile_data['persona']}; "
            f"Context: {profile_data['context']}; "
            f"Style: {profile_data.get('communication_style', 'Direct and concise')}"
        )

        # 种子可精确指定数量，未指定时从配置范围内随机
        num_facts = seed.get("num_facts") or random.randint(*self.num_facts_range)
        num_needs = seed.get("num_needs") or random.randint(*self.num_needs_range)

        for attempt in range(1, self.max_retries + 2):
            logger.info(f"[{sid}] 尝试 {attempt}/{self.max_retries + 1}")

            # Step 1: 生成 fact_sheet
            logger.info(f"[{sid}] 生成 fact_sheet ({num_facts} facts)...")
            facts = self.generate_fact_sheet(domain, description, user_profile_text, num_facts)
            if len(facts) < 8:
                logger.warning(f"[{sid}] fact_sheet 生成不足 ({len(facts)})，重试")
                continue

            # Step 2: 生成 user_needs
            logger.info(f"[{sid}] 生成 user_needs ({num_needs} needs)...")
            needs, reveal_groups = self.generate_user_needs(
                domain,
                description,
                user_profile_text,
                facts,
                num_needs,
            )
            if len(needs) < 3:
                logger.warning(f"[{sid}] user_needs 生成不足 ({len(needs)})，重试")
                continue

            # Step 3: 组装 Scenario
            simulator_config_data = dict(seed.get("simulator_config", {}))
            if reveal_groups and "need_reveal_strategy" not in simulator_config_data:
                simulator_config_data["need_reveal_strategy"] = "grouped"

            scenario = Scenario(
                scenario_id=sid,
                domain=domain,
                description=description,
                user_profile=UserProfile.from_dict(profile_data),
                fact_sheet=facts,
                user_needs=needs,
                reveal_groups=reveal_groups,
                simulator_config=SimulatorConfig.from_dict(simulator_config_data),
                metadata=ScenarioMetadata(
                    seed_id=sid,
                    generated_by=self.model,
                ),
            )

            # Step 4: 校验 — separate warnings from hard errors
            issues = self.validator.validate(scenario)
            errors = [i for i in issues if not i.startswith("[WARNING]")]
            warnings = [i for i in issues if i.startswith("[WARNING]")]
            for w in warnings:
                logger.warning(f"[{sid}] {w}")
            if not errors:
                logger.info(f"[{sid}] 生成成功，{len(facts)} facts, {len(needs)} needs")
                return scenario

            logger.warning(f"[{sid}] 校验发现 {len(errors)} 个错误: {errors}")
            if attempt <= self.max_retries:
                logger.info(f"[{sid}] 重试中...")

        logger.error(f"[{sid}] 生成失败（超过最大重试次数）")
        return None

    def _derive_reveal_groups(self, needs: list[UserNeed]) -> list[RevealGroupMeta]:
        """Derive reveal_groups from need-level metadata when LLM omits the top-level list.

        Instead of a conservative linear chain, inspects cross-group
        predictable_after links to infer trigger_after relationships.
        Groups with no cross-group dependency become root groups (trigger_after=null).
        """
        grouped: dict[str, list[UserNeed]] = {}
        for need in needs:
            if not need.reveal_group:
                continue
            grouped.setdefault(need.reveal_group, []).append(need)

        if not grouped:
            return []

        logger.warning(
            "LLM output missing top-level reveal_groups; deriving trigger_after from cross-group predictable_after links."
        )

        needs_by_id = {need.id: need for need in needs}

        # Sort groups by earliest turn_order of their members
        ordered_groups: list[tuple[str, list[UserNeed]]] = sorted(
            grouped.items(),
            key=lambda item: min(member.turn_order for member in item[1]),
        )

        # For each group, find the trigger_after by scanning cross-group
        # predictable_after links among its members
        group_trigger: dict[str, str | None] = {}
        for group_id, members in ordered_groups:
            trigger: str | None = None
            for member in members:
                if member.predictable_after is None:
                    continue
                pred = needs_by_id.get(member.predictable_after)
                if pred is None or not pred.reveal_group:
                    continue
                if pred.reveal_group != group_id:
                    # Cross-group link found — use predecessor's group as trigger
                    trigger = pred.reveal_group
                    break
            group_trigger[group_id] = trigger

        reveal_groups: list[RevealGroupMeta] = []
        for group_id, members in ordered_groups:
            ordered_members = sorted(
                members,
                key=lambda need: (
                    need.reveal_priority if need.reveal_priority is not None else 999,
                    need.turn_order,
                ),
            )
            reveal_groups.append(
                RevealGroupMeta(
                    group_id=group_id,
                    label=ordered_members[0].reveal_group_label or group_id,
                    member_need_ids=[need.id for need in ordered_members],
                    trigger_after=group_trigger.get(group_id),
                )
            )
        return reveal_groups


def load_seeds(path: Path) -> list[dict]:
    """加载种子描述文件。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("seeds", data) if isinstance(data, dict) else data


def save_scenario(scenario: Scenario, output_dir: Path) -> Path:
    """保存场景到 JSON 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scenario.scenario_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenario.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def main():
    bc = bench_config
    parser = argparse.ArgumentParser(description="从种子描述生成 Fact-Grounded Scenarios")
    parser.add_argument("--seeds", type=Path, default=Path(bc.paths.seeds), help="种子描述 JSON 文件路径")
    parser.add_argument("--output", type=Path, default=Path(bc.paths.scenarios), help="场景输出目录")
    parser.add_argument("--model", type=str, default=bc.llm.generation_model, help="生成使用的 LLM 模型")
    parser.add_argument("--temperature", type=float, default=bc.llm.generation_temperature, help="生成温度")
    parser.add_argument("--only", type=str, default=None, help="只生成指定 scenario_id")
    parser.add_argument("--retries", type=int, default=bc.generation.max_retries, help="每个场景的最大重试次数")
    args = parser.parse_args()

    # 初始化
    llm = LLMClient(config.llm)
    generator = ScenarioGenerator(
        llm, model=args.model, temperature=args.temperature,
        max_retries=args.retries,
        num_facts_range=(bc.generation.num_facts_min, bc.generation.num_facts_max),
        num_needs_range=(bc.generation.num_needs_min, bc.generation.num_needs_max),
    )

    # 加载种子
    seeds = load_seeds(args.seeds)
    logger.info(f"加载了 {len(seeds)} 个种子描述")

    # 过滤
    if args.only:
        seeds = [s for s in seeds if s["scenario_id"] == args.only]
        if not seeds:
            logger.error(f"未找到 scenario_id='{args.only}'")
            return

    # 检查已有场景（支持 resume）
    existing = set()
    if args.output.exists():
        for f in args.output.glob("*.json"):
            existing.add(f.stem)

    # 生成
    success, failed = 0, 0
    for seed in seeds:
        sid = seed["scenario_id"]
        if sid in existing:
            logger.info(f"[{sid}] 已存在，跳过")
            continue

        scenario = generator.generate_scenario(seed)
        if scenario:
            path = save_scenario(scenario, args.output)
            logger.info(f"[{sid}] 已保存到 {path}")
            success += 1
        else:
            failed += 1

    logger.info(f"完成: 成功 {success}, 失败 {failed}, 跳过 {len(existing)}")

    # 打印 LLM 使用报告
    print("\n" + llm.format_usage_report())


if __name__ == "__main__":
    main()
