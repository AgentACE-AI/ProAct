"""
MemBench 对比实验脚本。

复现论文 Table 3, 4, 5 的实验结果，并与 Memory System v2 进行对比。
注意：只测试 Participation (FirstAgent) 场景，因为 Memory System V2 是对话记忆系统。

使用方式:
    # 运行 Table 3 实验 (Factual Memory)
    python -m experiments.MemBench.benchmark.comparison_experiment --table 3

    # 运行 Table 4 实验 (Reflective Memory)
    python -m experiments.MemBench.benchmark.comparison_experiment --table 4

    # 运行完整对比
    python -m experiments.MemBench.benchmark.comparison_experiment --all
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.MemBench.benchmark.adapters.memory_v2_adapter import MemorySystemV2Adapter
from experiments.MemBench.benchmark.evaluator import MemBenchEvaluator
from experiments.MemBench.benchmark.models import BenchmarkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============== 实验配置 ==============

# Table 3: Factual Memory (LowLevel)
TABLE3_CONFIG = {
    "name": "Table 3 - Factual Memory",
    "experiments": [
        {
            "name": "10k",
            "level": "LowLevel",
            "max_trajectories": None,  # 全量测评
            "question_types": None,  # All factual types
            "capacity_test": False,
        },
        {
            "name": "100k",
            "level": "LowLevel",
            "max_trajectories": None,  # 全量测评
            "question_types": None,
            "capacity_test": True,
        },
    ]
}

# Table 4: Reflective Memory (HighLevel)
TABLE4_CONFIG = {
    "name": "Table 4 - Reflective Memory",
    "experiments": [
        {
            "name": "10k",
            "level": "HighLevel",
            "max_trajectories": None,  # 全量测评
            "question_types": ["Emotion", "Preference"],
            "capacity_test": False,
        },
        {
            "name": "100k",
            "level": "HighLevel",
            "max_trajectories": None,  # 全量测评
            "question_types": ["Emotion", "Preference"],
            "capacity_test": True,
        },
    ]
}

# Table 5: Different LLMs
TABLE5_CONFIG = {
    "name": "Table 5 - LLM Comparison",
    "experiments": [
        {
            "name": "Factual",
            "level": "LowLevel",
            "max_trajectories": None,  # 全量测评
            "question_types": None,
        },
        {
            "name": "Reflective",
            "level": "HighLevel",
            "max_trajectories": None,  # 全量测评
            "question_types": ["Emotion", "Preference"],
        },
    ]
}


# ============== 论文基线结果 (用于对比) ==============

# Table 3: Factual Memory (from paper, using Qwen2.5-7B-Instruct)
PAPER_RESULTS_TABLE3 = {
    "10k": {
        "FullMemory": 0.647,
        "RecentMemory": 0.639,
        "RetrievalMemory": 0.692,
        "GenerativeAgent": 0.478,
        "MemoryBank": 0.442,
        "MemGPT": 0.455,
        "SCMemory": 0.355,
    },
    "100k": {
        "FullMemory": 0.489,
        "RecentMemory": 0.422,
        "RetrievalMemory": 0.833,
        "GenerativeAgent": 0.455,
        "MemoryBank": 0.456,
        "MemGPT": 0.411,
        "SCMemory": 0.444,
    },
}

# Table 3: Efficiency data (RT=Read Time, WT=Write Time, in seconds)
PAPER_EFFICIENCY_TABLE3 = {
    "FullMemory": {"RT": 0.001, "WT": 0.001},
    "RecentMemory": {"RT": 0.001, "WT": 0.001},
    "RetrievalMemory": {"RT": 0.041, "WT": 0.058},
    "GenerativeAgent": {"RT": 0.045, "WT": 6.116},
    "MemoryBank": {"RT": 0.035, "WT": 8.047},
    "MemGPT": {"RT": 4.549, "WT": 0.106},
    "SCMemory": {"RT": 1.531, "WT": 2.276},
}

# Table 3: Recall@10
PAPER_RECALL_TABLE3 = {
    "10k": 0.776,
    "100k": 0.749,
}

# Table 4: Reflective Memory (from paper, using Qwen2.5-7B-Instruct)
PAPER_RESULTS_TABLE4 = {
    "10k": {
        "FullMemory": 0.733,
        "RecentMemory": 0.700,
        "RetrievalMemory": 0.692,
        "GenerativeAgent": 0.742,
        "MemoryBank": 0.692,
        "MemGPT": 0.733,
        "SCMemory": 0.542,
    },
    "100k": {
        "FullMemory": 0.533,
        "RecentMemory": 0.333,
        "RetrievalMemory": 0.833,
        "GenerativeAgent": 0.333,
        "MemoryBank": 0.400,
        "MemGPT": 0.367,
        "SCMemory": 0.267,
    },
}

# Table 4: Efficiency data
PAPER_EFFICIENCY_TABLE4 = {
    "FullMemory": {"RT": 0.001, "WT": 0.001},
    "RecentMemory": {"RT": 0.001, "WT": 0.001},
    "RetrievalMemory": {"RT": 0.036, "WT": 0.057},
    "GenerativeAgent": {"RT": 0.028, "WT": 6.064},
    "MemoryBank": {"RT": 0.033, "WT": 15.705},
    "MemGPT": {"RT": 1.042, "WT": 0.001},
    "SCMemory": {"RT": 0.036, "WT": 0.057},
}


class ComparisonExperiment:
    """对比实验运行器"""

    def __init__(
        self,
        output_dir: str = "./experiments/MemBench/results/comparison",
        local_model: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化对比实验运行器。

        Args:
            output_dir: 输出目录
            local_model: 本地模型路径或 HuggingFace ID (使用 transformers 推理)
            model: API 模型名 (使用 OpenAI 兼容 API)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict[str, Any]] = {}
        self.local_model = local_model
        self.model = model

        # 预加载本地模型（只加载一次，跨实验复用）
        self._local_llm_client = None
        if self.local_model:
            from tools.local_llm_client import LocalLLMClient
            logger.info(f"Loading local model: {self.local_model} ...")
            self._local_llm_client = LocalLLMClient(self.local_model)

    async def run_table3_experiments(self) -> Dict[str, Any]:
        """运行 Table 3 实验 (Factual Memory)"""
        logger.info("=" * 60)
        logger.info("Running Table 3 Experiments (Factual Memory)")
        logger.info("=" * 60)

        results = {}
        for exp_config in TABLE3_CONFIG["experiments"]:
            exp_name = exp_config["name"]
            logger.info(f"\n--- {exp_name} ---")

            result = await self._run_single_experiment(exp_config, table_name="table3")
            results[exp_name] = result

            # 与论文结果对比
            if exp_name in PAPER_RESULTS_TABLE3:
                self._print_comparison(
                    exp_name, result,
                    PAPER_RESULTS_TABLE3[exp_name],
                    PAPER_RECALL_TABLE3.get(exp_name),
                )

        # 保存结果
        self._save_results("table3", results)
        return results

    async def run_table4_experiments(self) -> Dict[str, Any]:
        """运行 Table 4 实验 (Reflective Memory)"""
        logger.info("=" * 60)
        logger.info("Running Table 4 Experiments (Reflective Memory)")
        logger.info("=" * 60)

        results = {}
        for exp_config in TABLE4_CONFIG["experiments"]:
            exp_name = exp_config["name"]
            logger.info(f"\n--- {exp_name} ---")

            result = await self._run_single_experiment(exp_config, table_name="table4")
            results[exp_name] = result

            # 与论文结果对比
            if exp_name in PAPER_RESULTS_TABLE4:
                self._print_comparison(
                    exp_name, result,
                    PAPER_RESULTS_TABLE4[exp_name],
                    None,
                )

        self._save_results("table4", results)
        return results

    async def run_table5_experiments(self) -> Dict[str, Any]:
        """运行 Table 5 实验 (LLM Comparison)"""
        logger.info("=" * 60)
        logger.info("Running Table 5 Experiments (LLM Comparison)")
        logger.info("=" * 60)

        results = {}
        for exp_config in TABLE5_CONFIG["experiments"]:
            exp_name = exp_config["name"]
            logger.info(f"\n--- {exp_name} ---")

            result = await self._run_single_experiment(exp_config, table_name="table5")
            results[exp_name] = result

        self._save_results("table5", results)
        return results

    async def _run_single_experiment(
        self,
        exp_config: Dict[str, Any],
        table_name: str = "",
    ) -> Dict[str, Any]:
        """运行单个实验"""
        # 确定 LLM 客户端和模型名
        if self._local_llm_client:
            llm_client = self._local_llm_client
            model_name = self.local_model
        else:
            from core.config import Config
            from tools.llm_client import LLMClient
            app_config = Config()
            llm_client = LLMClient(app_config.llm)
            model_name = self.model or "gpt-4o-mini"

        # 创建 Memory System v2 适配器
        adapter = MemorySystemV2Adapter({
            "user_id": f"membench_exp_{table_name}_{exp_config['name'].lower().replace('-', '_')}",
            "args": {
                "max_words": 2000,
                "enable_profile_extraction": True,
                "profile_update_interval": 5,
                "enable_sentiment_extraction": True,
                "enable_fact_extraction": True,
            },
            "llm_client": llm_client,
            "llm_model": model_name,
        })

        # 判断是否为容量测试 (100k)
        is_capacity_test = exp_config.get("capacity_test", False)

        # 根据实验名称选择正确的数据集
        exp_name = exp_config["name"].lower()
        if "100k" in exp_name:
            dataset_variant = "100k"
        elif "10k" in exp_name:
            dataset_variant = "10k"
        else:
            dataset_variant = "10k"

        # 输出目录: table_name 前缀避免 Table 间同名实验冲突
        # 模型名纳入路径，防止不同模型的 checkpoint/结果互相覆盖
        model_label = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        exp_output_dir = str(self.output_dir / f"{table_name}_{exp_config['name']}_{model_label}")

        # 创建评估配置
        benchmark_config = BenchmarkConfig(
            dataset_variant=dataset_variant,
            question_types=exp_config.get("question_types"),
            max_trajectories_per_type=exp_config.get("max_trajectories"),
            output_dir=exp_output_dir,
            capacity_test=is_capacity_test,
            min_tokens_for_capacity=8000 if is_capacity_test else 0,
            llm_model=model_name,
        )

        # 生成 run manifest
        manifest = {
            "table_name": table_name,
            "experiment_name": exp_config["name"],
            "level": exp_config["level"],
            "dataset_variant": dataset_variant,
            "question_types": exp_config.get("question_types"),
            "max_trajectories": exp_config.get("max_trajectories"),
            "llm_model": model_name,
            "llm_backend": "local" if self._local_llm_client else "api",
            "started_at": datetime.now().isoformat(),
        }
        manifest_dir = Path(exp_output_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # 创建评估器并运行
        evaluator = MemBenchEvaluator(
            memory_adapter=adapter,
            llm_client=llm_client,
            config=benchmark_config,
        )

        metrics = await evaluator.run_evaluation(
            level=exp_config["level"],
            verbose=True,
        )

        # 打印并保存 token 用量报告
        usage_report = llm_client.format_usage_report()
        print(f"\n{usage_report}")
        with open(manifest_dir / "token_usage.json", "w", encoding="utf-8") as f:
            json.dump(llm_client.get_usage_stats(), f, indent=2, ensure_ascii=False)
        logger.info(f"Token usage saved to {manifest_dir / 'token_usage.json'}")
        llm_client.reset_usage_stats()

        return {
            "MemorySystemV2": {
                "accuracy": metrics.accuracy,
                "recall_at_10": metrics.avg_recall_at_10,
                "total_trajectories": metrics.total_trajectories,
                "correct_count": metrics.correct_count,
                "accuracy_by_type": metrics.accuracy_by_type,
            }
        }

    def _print_comparison(
        self,
        exp_name: str,
        our_result: Dict[str, Any],
        paper_results: Dict[str, float],
        paper_recall: Optional[float],
    ) -> None:
        """打印与论文结果的对比"""
        our_data = our_result["MemorySystemV2"]
        our_accuracy = our_data["accuracy"]
        our_recall = our_data.get("recall_at_10", 0)

        print(f"\n{'='*70}")
        print(f"Comparison for {exp_name}")
        print(f"{'='*70}")

        # 准确率对比
        print(f"\n--- Accuracy ---")
        print(f"{'Method':<20} {'Accuracy':>10} {'vs Ours':>12}")
        print("-" * 44)

        for method, acc in sorted(paper_results.items(), key=lambda x: -x[1]):
            diff = our_accuracy - acc
            diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
            print(f"{method:<20} {acc:>10.3f} {diff_str:>12}")

        print("-" * 44)
        print(f"{'MemorySystemV2 (Ours)':<20} {our_accuracy:>10.3f}")

        # Recall@10
        if our_recall > 0 and paper_recall:
            print(f"\n--- Recall@10 ---")
            print(f"RetrievalMemory (Paper): {paper_recall:.3f}")
            print(f"MemorySystemV2 (Ours):   {our_recall:.3f}")

        print(f"{'='*70}")

    def _save_results(self, table_name: str, results: Dict[str, Any]) -> None:
        """保存结果到文件"""
        # 模型名纳入结果文件名，防止不同模型结果互相覆盖
        model_name = self.local_model or self.model or "gpt-4o-mini"
        model_label = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        output_file = self.output_dir / f"{table_name}_{model_label}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "table": table_name,
                "llm_model": model_name,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")

    def generate_comparison_table(self) -> str:
        """生成对比表格 (Markdown 格式)"""
        model_name = self.local_model or self.model or "gpt-4o-mini"
        model_label = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)

        table3_file = self.output_dir / f"table3_{model_label}_results.json"
        table4_file = self.output_dir / f"table4_{model_label}_results.json"

        tables = []

        if table3_file.exists():
            with open(table3_file) as f:
                data = json.load(f)
            tables.append(self._format_comparison_table(
                "Table 3 - Factual Memory",
                data["results"],
                PAPER_RESULTS_TABLE3,
                PAPER_RECALL_TABLE3,
            ))

        if table4_file.exists():
            with open(table4_file) as f:
                data = json.load(f)
            tables.append(self._format_comparison_table(
                "Table 4 - Reflective Memory",
                data["results"],
                PAPER_RESULTS_TABLE4,
                {},
            ))

        return "\n\n".join(tables)

    def _format_comparison_table(
        self,
        title: str,
        our_results: Dict[str, Any],
        paper_results: Dict[str, Dict[str, float]],
        paper_recall: Dict[str, float],
    ) -> str:
        """格式化对比表格"""
        lines = [f"## {title}\n"]

        # 准确率对比表
        lines.append("### Accuracy Comparison\n")
        lines.append("| Experiment | Method | Accuracy | vs Ours |")
        lines.append("|------------|--------|----------|---------|")

        for exp_name in our_results:
            if exp_name not in paper_results:
                our_acc = our_results[exp_name]["MemorySystemV2"]["accuracy"]
                lines.append(f"| {exp_name} | **MemorySystemV2** | **{our_acc:.3f}** | - |")
                continue

            our_acc = our_results[exp_name]["MemorySystemV2"]["accuracy"]

            for method, acc in sorted(paper_results[exp_name].items(), key=lambda x: -x[1]):
                diff = our_acc - acc
                diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
                lines.append(f"| {exp_name} | {method} | {acc:.3f} | {diff_str} |")

            lines.append(f"| {exp_name} | **MemorySystemV2 (Ours)** | **{our_acc:.3f}** | - |")

        # Recall@10 对比表
        if paper_recall:
            lines.append("\n### Recall@10 Comparison\n")
            lines.append("| Experiment | RetrievalMemory (Paper) | MemorySystemV2 (Ours) |")
            lines.append("|------------|-------------------------|----------------------|")

            for exp_name in our_results:
                if exp_name in paper_recall:
                    paper_r = paper_recall[exp_name]
                    our_r = our_results[exp_name]["MemorySystemV2"].get("recall_at_10", 0)
                    lines.append(f"| {exp_name} | {paper_r:.3f} | {our_r:.3f} |")

        return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="MemBench Comparison Experiments")
    parser.add_argument(
        "--table",
        type=int,
        choices=[3, 4, 5],
        help="Run specific table experiment (3, 4, or 5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer trajectories (5 per type)"
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=None,
        help="Override max trajectories per type (e.g., --num-trajectories 2)"
    )
    parser.add_argument(
        "--output-dir",
        default="./experiments/MemBench/results/comparison",
        help="Output directory"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comparison report from existing results"
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Local model path or HuggingFace ID for transformers inference "
             "(e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="API model name (e.g., gpt-4o-mini). Default: gpt-4o-mini"
    )
    args = parser.parse_args()

    runner = ComparisonExperiment(
        output_dir=args.output_dir,
        local_model=args.local_model,
        model=args.model,
    )

    if args.report:
        report = runner.generate_comparison_table()
        print(report)
        report_file = Path(args.output_dir) / "comparison_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"\nReport saved to {report_file}")
        return

    # 自定义轨迹数量
    if args.num_trajectories:
        for config in [TABLE3_CONFIG, TABLE4_CONFIG, TABLE5_CONFIG]:
            for exp in config["experiments"]:
                exp["max_trajectories"] = args.num_trajectories
    # 快速测试模式
    elif args.quick:
        for config in [TABLE3_CONFIG, TABLE4_CONFIG, TABLE5_CONFIG]:
            for exp in config["experiments"]:
                exp["max_trajectories"] = 5

    if args.all:
        await runner.run_table3_experiments()
        await runner.run_table4_experiments()
        await runner.run_table5_experiments()
    elif args.table == 3:
        await runner.run_table3_experiments()
    elif args.table == 4:
        await runner.run_table4_experiments()
    elif args.table == 5:
        await runner.run_table5_experiments()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
