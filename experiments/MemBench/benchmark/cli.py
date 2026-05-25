"""
MemBench Benchmark 命令行接口。

使用方式:
    python -m experiments.MemBench.benchmark.cli --help
    python -m experiments.MemBench.benchmark.cli run --agent-type FirstAgent --level LowLevel
    python -m experiments.MemBench.benchmark.cli summary --agent-type FirstAgent --level LowLevel
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.MemBench.benchmark.models import ALL_QUESTION_TYPES, BenchmarkConfig
from experiments.MemBench.benchmark.runner import BenchmarkRunner, run_benchmark


def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> None:
    """运行评估命令"""
    setup_logging(args.verbose)

    # 解析问题类型
    question_types: Optional[List[str]] = None
    if args.question_types:
        question_types = args.question_types

    async def _run():
        report = await run_benchmark(
            agent_type=args.agent_type,
            level=args.level,
            question_types=question_types,
            max_trajectories=args.max_trajectories,
            output_dir=args.output_dir,
            verbose=args.verbose,
            local_model=args.local_model,
            model=args.model,
        )

        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)
        print(f"Accuracy: {report['summary']['accuracy']:.2%}")
        print(f"Total Trajectories: {report['summary']['total_trajectories']}")
        print(f"Correct: {report['summary']['correct_count']}")
        print(f"Avg Recall@10: {report['summary']['avg_recall_at_10']:.2%}")

        print("\nAccuracy by Question Type:")
        for qtype, acc in sorted(report["by_question_type"].items()):
            print(f"  {qtype}: {acc:.2%}")

        print(f"\nResults saved to: {args.output_dir}")

    asyncio.run(_run())


def cmd_summary(args: argparse.Namespace) -> None:
    """打印数据集摘要"""
    setup_logging(False)

    config = BenchmarkConfig()
    runner = BenchmarkRunner(config=config)
    runner.print_dataset_summary(args.agent_type, args.level)


def cmd_reset(args: argparse.Namespace) -> None:
    """重置 checkpoint"""
    setup_logging(False)

    config = BenchmarkConfig(output_dir=args.output_dir)
    checkpoint_path = Path(config.output_dir) / config.checkpoint_file

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint reset: {checkpoint_path}")
    else:
        print("No checkpoint found")

    # 也删除详细结果
    results_path = Path(config.output_dir) / "detailed_results.json"
    if results_path.exists():
        results_path.unlink()
        print(f"Results cleared: {results_path}")


def cmd_list_types(args: argparse.Namespace) -> None:
    """列出所有问题类型"""
    print("\nAvailable Question Types:")
    print("-" * 40)
    for qtype in ALL_QUESTION_TYPES:
        print(f"  - {qtype}")
    print()


def main() -> None:
    """主入口"""
    parser = argparse.ArgumentParser(
        description="MemBench Benchmark CLI for Memory System v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation on FirstAgent LowLevel data
  python -m experiments.MemBench.benchmark.cli run --agent-type FirstAgent --level LowLevel

  # Run with specific question types and limited trajectories
  python -m experiments.MemBench.benchmark.cli run --agent-type FirstAgent --level LowLevel \\
      --question-types Single-hop Multi-hop --max-trajectories 10

  # Print dataset summary
  python -m experiments.MemBench.benchmark.cli summary --agent-type FirstAgent --level LowLevel

  # Reset checkpoint to start fresh
  python -m experiments.MemBench.benchmark.cli reset
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run 命令
    run_parser = subparsers.add_parser("run", help="Run the evaluation")
    run_parser.add_argument(
        "--agent-type",
        choices=["FirstAgent", "ThirdAgent"],
        default="FirstAgent",
        help="Agent type (default: FirstAgent)",
    )
    run_parser.add_argument(
        "--level",
        choices=["LowLevel", "HighLevel"],
        default="LowLevel",
        help="Data level (default: LowLevel)",
    )
    run_parser.add_argument(
        "--question-types",
        nargs="+",
        choices=ALL_QUESTION_TYPES,
        help="Question types to evaluate (default: all)",
    )
    run_parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Max trajectories per question type (default: all)",
    )
    run_parser.add_argument(
        "--output-dir",
        default="./experiments/MemBench/results",
        help="Output directory (default: ./experiments/MemBench/results)",
    )
    run_parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Local model path or HuggingFace ID for transformers inference "
             "(e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="API model name (e.g., gpt-4o-mini). Default: gpt-4o-mini",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    run_parser.set_defaults(func=cmd_run)

    # summary 命令
    summary_parser = subparsers.add_parser("summary", help="Print dataset summary")
    summary_parser.add_argument(
        "--agent-type",
        choices=["FirstAgent", "ThirdAgent"],
        default="FirstAgent",
        help="Agent type",
    )
    summary_parser.add_argument(
        "--level",
        choices=["LowLevel", "HighLevel"],
        default="LowLevel",
        help="Data level",
    )
    summary_parser.set_defaults(func=cmd_summary)

    # reset 命令
    reset_parser = subparsers.add_parser("reset", help="Reset checkpoint")
    reset_parser.add_argument(
        "--output-dir",
        default="./experiments/MemBench/results",
        help="Output directory",
    )
    reset_parser.set_defaults(func=cmd_reset)

    # list-types 命令
    types_parser = subparsers.add_parser("list-types", help="List all question types")
    types_parser.set_defaults(func=cmd_list_types)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
