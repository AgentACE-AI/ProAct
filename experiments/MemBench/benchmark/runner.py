"""
MemBench Benchmark 运行器。

协调评估流程的主入口。
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters.memory_v2_adapter import MemorySystemV2Adapter
from .data_loader import MemBenchDataLoader
from .evaluator import MemBenchEvaluator
from .metrics import MetricsCalculator
from .models import AggregateMetrics, BenchmarkConfig

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark 主运行器"""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        初始化运行器。

        Args:
            config: Benchmark 配置
        """
        self.config = config or BenchmarkConfig()

        # 输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        agent_type: str = "FirstAgent",
        level: str = "LowLevel",
        memory_adapter: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        运行评估。

        Args:
            agent_type: "FirstAgent" 或 "ThirdAgent"
            level: "LowLevel" 或 "HighLevel"
            memory_adapter: 记忆适配器，默认使用 MemorySystemV2Adapter
            llm_client: LLM 客户端
            verbose: 是否打印详细日志

        Returns:
            评估报告
        """
        import time

        start_time = time.time()

        # 初始化 LLM 客户端
        if llm_client is None:
            from core.config import Config
            from tools.llm_client import LLMClient

            app_config = Config()
            llm_client = LLMClient(app_config.llm)

        # 初始化记忆适配器
        if memory_adapter is None:
            memory_adapter = MemorySystemV2Adapter(
                {
                    "user_id": f"membench_{agent_type}_{level}",
                    "args": {"max_words": self.config.max_words},
                    "llm_client": llm_client,
                    "llm_model": self.config.llm_model,
                }
            )

        # 创建评估器
        evaluator = MemBenchEvaluator(
            memory_adapter=memory_adapter,
            llm_client=llm_client,
            config=self.config,
        )

        # 运行评估
        if verbose:
            logger.info(f"Starting evaluation: {agent_type} / {level}")

        metrics = await evaluator.run_evaluation(
            level=level,
            verbose=verbose,
        )

        elapsed_time = time.time() - start_time

        # 生成报告
        report = self._generate_report(metrics, elapsed_time, agent_type, level)

        return report

    def _generate_report(
        self,
        metrics: AggregateMetrics,
        elapsed_time: float,
        agent_type: str,
        level: str,
    ) -> Dict[str, Any]:
        """
        生成评估报告。

        Args:
            metrics: 聚合指标
            elapsed_time: 运行时间（秒）
            agent_type: 代理类型
            level: 数据级别

        Returns:
            报告字典
        """
        report = {
            "metadata": {
                "benchmark": "MemBench",
                "agent_type": agent_type,
                "level": level,
                "memory_system": "Memory System v2",
                "elapsed_time_seconds": elapsed_time,
                "created_at": datetime.now().isoformat(),
            },
            "summary": {
                "accuracy": metrics.accuracy,
                "total_trajectories": metrics.total_trajectories,
                "correct_count": metrics.correct_count,
                "avg_recall_at_10": metrics.avg_recall_at_10,
            },
            "by_question_type": metrics.accuracy_by_type,
            "by_scenario": metrics.accuracy_by_scenario,
            "recall_by_type": metrics.recall_by_type,
        }

        # 保存报告
        report_path = self.output_dir / "final_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def print_dataset_summary(
        self,
        agent_type: str = "FirstAgent",
        level: str = "LowLevel",
    ) -> None:
        """
        打印数据集摘要。

        Args:
            agent_type: 代理类型
            level: 数据级别
        """
        data_loader = MemBenchDataLoader(self.config)
        summary = data_loader.get_dataset_summary(level=level)

        print(f"\n{'='*60}")
        print(f"Dataset Summary: {agent_type} / {level}")
        print("=" * 60)
        print(f"Total Trajectories: {summary['total_trajectories']}")
        print("\nBy Question Type:")
        for qtype, info in sorted(summary["by_question_type"].items()):
            print(f"  {qtype}: {info['count']} (scenarios: {', '.join(info['scenarios'])})")
        print("=" * 60)


async def run_benchmark(
    agent_type: str = "FirstAgent",
    level: str = "LowLevel",
    question_types: Optional[List[str]] = None,
    max_trajectories: Optional[int] = None,
    output_dir: str = "./experiments/MemBench/results",
    verbose: bool = True,
    llm_client: Optional[Any] = None,
    local_model: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行 benchmark 的快捷函数。

    Args:
        agent_type: "FirstAgent" 或 "ThirdAgent"
        level: "LowLevel" 或 "HighLevel"
        question_types: 要评估的问题类型
        max_trajectories: 每种类型最大轨迹数
        output_dir: 输出目录
        verbose: 是否打印详细日志
        llm_client: LLM 客户端
        local_model: 本地模型路径或 HuggingFace ID (使用 transformers 推理)
        model: API 模型名 (使用 OpenAI 兼容 API)

    Returns:
        评估报告
    """
    model_name = local_model or model or "gpt-4o-mini"

    config = BenchmarkConfig(
        question_types=question_types,
        max_trajectories_per_type=max_trajectories,
        output_dir=output_dir,
        llm_model=model_name,
    )

    # 创建本地 LLM 客户端（如果指定且未预置）
    if local_model and llm_client is None:
        from tools.local_llm_client import LocalLLMClient

        llm_client = LocalLLMClient(local_model)

    runner = BenchmarkRunner(config=config)
    return await runner.run(
        agent_type=agent_type,
        level=level,
        verbose=verbose,
        llm_client=llm_client,
    )


# 同步包装器
def run_benchmark_sync(**kwargs) -> Dict[str, Any]:
    """同步版本的 run_benchmark"""
    return asyncio.run(run_benchmark(**kwargs))


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    async def main():
        report = await run_benchmark(
            agent_type="FirstAgent",
            level="HighLevel",  # 先用小数据集测试
            question_types=["Emotion"],
            max_trajectories=5,
            verbose=True,
        )
        print(json.dumps(report, indent=2))

    asyncio.run(main())
