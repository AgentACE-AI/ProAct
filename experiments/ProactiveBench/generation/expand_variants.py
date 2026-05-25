#!/usr/bin/env python3
"""
变体扩展器。

基于种子场景生成参数变体：替换具体细节（名称、数字、地点），保持结构不变。

用法:
    python expand_variants.py --seed-dir ../data/scenarios/ --variants 5 --output ../data/scenarios/
    python expand_variants.py --seed-dir ../data/scenarios/ --variants 3 --output ../data/scenarios/ --model gpt-4o --only onboard_01
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.llm_client import LLMClient
from core.config import config

try:
    from .bench_config import bench_config
    from .models import Scenario, ScenarioMetadata, ScenarioValidator
    from .prompts import GenerationPrompts
except ImportError:  # pragma: no cover - keeps direct script execution working
    from bench_config import bench_config
    from models import Scenario, ScenarioMetadata, ScenarioValidator
    from prompts import GenerationPrompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VariantExpander:
    """变体扩展器：seed scenario -> N parameter variants。"""

    def __init__(self, llm: LLMClient, model: str = "gpt-4o", temperature: float = 0.8, max_retries: int = 2):
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.validator = ScenarioValidator()

    def expand_variant(self, seed: Scenario, variant_num: int) -> Scenario | None:
        """
        基于种子场景生成一个参数变体。

        Args:
            seed: 种子场景
            variant_num: 变体编号（从 1 开始）

        Returns:
            变体 Scenario 或 None（如果生成失败）
        """
        seed_id = seed.scenario_id
        variant_id = f"{seed_id}_v{variant_num}"

        seed_json = json.dumps(seed.to_dict(), ensure_ascii=False, indent=2)
        prompt = GenerationPrompts.EXPAND_VARIANT.format(
            seed_scenario_json=seed_json,
            domain=seed.domain,
        )

        for attempt in range(1, self.max_retries + 2):
            logger.info(f"[{variant_id}] attempt {attempt}/{self.max_retries + 1}")

            try:
                result = self.llm.chat_json(
                    messages=[
                        {"role": "system", "content": "You are a data augmentation expert. Return JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    temperature=self.temperature,
                )
            except (RuntimeError, ValueError) as e:
                logger.warning(f"[{variant_id}] LLM call failed: {e}")
                if attempt <= self.max_retries:
                    continue
                break

            # Parse the variant scenario from the LLM response
            try:
                variant = Scenario.from_dict(result)
            except (KeyError, TypeError) as e:
                logger.warning(f"[{variant_id}] failed to parse scenario: {e}")
                if attempt <= self.max_retries:
                    continue
                break

            # Override IDs and metadata to ensure correctness
            variant.scenario_id = variant_id
            variant.metadata = ScenarioMetadata(
                seed_id=seed_id,
                variant=f"v{variant_num}",
                generated_by=self.model,
            )

            # Validate
            issues = self.validator.validate(variant)
            if not issues:
                logger.info(
                    f"[{variant_id}] generated successfully, "
                    f"{len(variant.fact_sheet)} facts, {len(variant.user_needs)} needs"
                )
                return variant

            logger.warning(f"[{variant_id}] validation found {len(issues)} issues: {issues}")
            if attempt <= self.max_retries:
                logger.info(f"[{variant_id}] retrying...")

        logger.error(f"[{variant_id}] generation failed (max retries exceeded)")
        return None


def load_seed_scenarios(seed_dir: Path) -> list[Scenario]:
    """
    Load seed scenarios from a directory.

    A seed scenario is one where metadata.variant is None (i.e., it is not
    itself a variant of another seed).
    """
    seeds = []
    if not seed_dir.exists():
        logger.error(f"Seed directory does not exist: {seed_dir}")
        return seeds

    for path in sorted(seed_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scenario = Scenario.from_dict(data)
            if scenario.metadata.variant is None:
                seeds.append(scenario)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Skipping invalid file {path.name}: {e}")

    return seeds


def save_scenario(scenario: Scenario, output_dir: Path) -> Path:
    """Save scenario to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scenario.scenario_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenario.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def get_existing_ids(output_dir: Path) -> set[str]:
    """Collect scenario IDs already present in the output directory."""
    existing = set()
    if output_dir.exists():
        for path in output_dir.glob("*.json"):
            existing.add(path.stem)
    return existing


def main():
    bc = bench_config
    parser = argparse.ArgumentParser(description="Expand seed scenarios into parameter variants")
    parser.add_argument("--seed-dir", type=Path, default=Path(bc.paths.scenarios), help="Directory containing seed scenario JSONs")
    parser.add_argument("--variants", type=int, default=5, help="Number of variants per seed")
    parser.add_argument("--output", type=Path, default=Path(bc.paths.scenarios), help="Output directory for variants")
    parser.add_argument("--model", type=str, default=bc.llm.expansion_model, help="LLM model for generation")
    parser.add_argument("--temperature", type=float, default=bc.llm.expansion_temperature, help="Generation temperature")
    parser.add_argument("--only", type=str, default=None, help="Only expand this seed scenario_id")
    parser.add_argument("--retries", type=int, default=bc.generation.max_retries, help="Max retries per variant")
    args = parser.parse_args()

    # Initialize
    llm = LLMClient(config.llm)
    expander = VariantExpander(llm, model=args.model, temperature=args.temperature, max_retries=args.retries)

    # Load seeds
    seeds = load_seed_scenarios(args.seed_dir)
    logger.info(f"Loaded {len(seeds)} seed scenarios from {args.seed_dir}")

    if not seeds:
        logger.error("No seed scenarios found. Nothing to do.")
        return

    # Filter
    if args.only:
        seeds = [s for s in seeds if s.scenario_id == args.only]
        if not seeds:
            logger.error(f"Seed scenario_id '{args.only}' not found")
            return

    # Check existing variants for resume support
    existing = get_existing_ids(args.output)

    # Generate variants
    success, failed, skipped = 0, 0, 0
    for seed in seeds:
        logger.info(f"[{seed.scenario_id}] expanding {args.variants} variants...")

        for n in range(1, args.variants + 1):
            variant_id = f"{seed.scenario_id}_v{n}"

            if variant_id in existing:
                logger.info(f"[{variant_id}] already exists, skipping")
                skipped += 1
                continue

            variant = expander.expand_variant(seed, variant_num=n)
            if variant:
                path = save_scenario(variant, args.output)
                logger.info(f"[{variant_id}] saved to {path}")
                success += 1
            else:
                failed += 1

    logger.info(f"Done: success {success}, failed {failed}, skipped {skipped}")

    # Print LLM usage report
    print("\n" + llm.format_usage_report())


if __name__ == "__main__":
    main()
