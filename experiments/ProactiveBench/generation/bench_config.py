"""
ProactiveBench 生成配置加载器。

从 config.yaml 读取默认配置，支持 CLI 参数覆盖。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class LLMSettings:
    """LLM 模型配置。"""
    generation_model: str = "gpt-4o"
    generation_temperature: float = 0.7
    expansion_model: str = "gpt-4o"
    expansion_temperature: float = 0.8
    validation_model: str = "gpt-4o-mini"
    validation_temperature: float = 0.2


@dataclass
class GenerationSettings:
    """生成参数。"""
    max_retries: int = 2
    num_facts_min: int = 25
    num_facts_max: int = 35
    num_needs_min: int = 10
    num_needs_max: int = 15


@dataclass
class PathSettings:
    """路径配置（相对于 generation/ 目录）。"""
    seeds: str = "../data/seeds/seed_descriptions.json"
    scenarios: str = "../data/scenarios/"


@dataclass
class BenchConfig:
    """ProactiveBench 数据生成完整配置。"""
    llm: LLMSettings = field(default_factory=LLMSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    paths: PathSettings = field(default_factory=PathSettings)


def load_bench_config(config_path: Optional[Path] = None) -> BenchConfig:
    """
    从 YAML 文件加载配置。文件不存在时返回默认值。

    Args:
        config_path: 配置文件路径，默认为 generation/config.yaml
    """
    path = config_path or _CONFIG_PATH
    cfg = BenchConfig()

    if not path.exists():
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # LLM
    llm_raw = raw.get("llm", {})
    for k, v in llm_raw.items():
        if hasattr(cfg.llm, k):
            setattr(cfg.llm, k, v)

    # Generation
    gen_raw = raw.get("generation", {})
    for k, v in gen_raw.items():
        if hasattr(cfg.generation, k):
            setattr(cfg.generation, k, v)

    # Paths
    paths_raw = raw.get("paths", {})
    for k, v in paths_raw.items():
        if hasattr(cfg.paths, k):
            setattr(cfg.paths, k, v)

    return cfg


# 全局单例
bench_config = load_bench_config()
