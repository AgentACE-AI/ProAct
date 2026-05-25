"""
MemBench Benchmark 评估器。

核心评估逻辑实现。
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .data_loader import MemBenchDataLoader
from .metrics import MetricsCalculator, calculate_recall_at_k
from .models import (
    AggregateMetrics,
    BenchmarkConfig,
    EvaluationResult,
    MemBenchMessage,
    Trajectory,
)

logger = logging.getLogger(__name__)


def _trajectory_key(trajectory) -> str:
    """计算轨迹的唯一复合 key，用于 checkpoint 去重。"""
    return f"{trajectory.question_type}__{trajectory.scenario}__{trajectory.tid}"


class MemBenchEvaluator:
    """MemBench 评估器"""

    def __init__(
        self,
        memory_adapter: Any,  # BaseMemory
        llm_client: Any,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        初始化评估器。

        Args:
            memory_adapter: 实现 BaseMemory 接口的记忆适配器
            llm_client: LLM 客户端
            config: Benchmark 配置
        """
        self.memory = memory_adapter
        self.llm = llm_client
        self.config = config or BenchmarkConfig()

        # 数据加载器
        self.data_loader = MemBenchDataLoader(self.config)

        # 结果存储
        self.results: List[EvaluationResult] = []

        # 输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint
        self.checkpoint_path = self.output_dir / self.config.checkpoint_file

    async def run_evaluation(
        self,
        level: str = "LowLevel",
        verbose: bool = True,
    ) -> AggregateMetrics:
        """
        运行完整评估。

        注意：只支持 FirstAgent (Participation) 场景。

        Args:
            level: "LowLevel" 或 "HighLevel"
            verbose: 是否打印详细日志

        Returns:
            聚合评估指标
        """
        # 加载轨迹 (只使用 FirstAgent)
        trajectories = self.data_loader.load_trajectories(
            level=level,
            question_types=self.config.question_types,
            max_per_type=self.config.max_trajectories_per_type,
            capacity_test=self.config.capacity_test,
            min_tokens=self.config.min_tokens_for_capacity,
        )

        # print(f"Trajectories loaded: {trajectories}")
        # input("Press enter to continue...")

        if verbose:
            logger.info(f"Loaded {len(trajectories)} trajectories")

        # 生成本次运行的 run_id
        run_id = f"run_{int(time.time())}"

        # 加载 checkpoint
        completed_keys, self.results = self._load_checkpoint()

        if verbose and completed_keys:
            logger.info(f"Resuming from checkpoint: {len(completed_keys)} completed")

        # 评估每个轨迹
        for idx, trajectory in enumerate(trajectories):
            if _trajectory_key(trajectory) in completed_keys:
                continue

            if verbose:
                logger.info(
                    f"[{idx + 1}/{len(trajectories)}] "
                    f"Evaluating tid={trajectory.tid} ({trajectory.question_type})"
                )

            try:
                result = await self._evaluate_trajectory(
                    trajectory,
                    run_id=run_id,
                    verbose=verbose,
                )
                self.results.append(result)

                # 保存 checkpoint
                completed_keys.add(_trajectory_key(trajectory))
                self._save_checkpoint(completed_keys)

                if verbose:
                    status = "✓" if result.is_correct else "✗"
                    logger.info(
                        f"  {status} Predicted: {result.predicted_answer}, "
                        f"Ground Truth: {result.ground_truth}"
                    )

            except Exception as e:
                logger.error(f"Error evaluating tid={trajectory.tid}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # 计算并返回聚合指标
        metrics_calc = MetricsCalculator(self.results)
        metrics = metrics_calc.compute_all()

        # 保存最终结果
        self._save_results(metrics)

        if verbose:
            metrics_calc.print_summary()

        return metrics

    async def _evaluate_trajectory(
        self,
        trajectory: Trajectory,
        run_id: str = "",
        verbose: bool = False,
    ) -> EvaluationResult:
        """
        评估单个轨迹。

        Args:
            trajectory: 轨迹数据
            run_id: 本次运行的唯一标识
            verbose: 是否打印详细日志

        Returns:
            评估结果
        """
        # Step 1: 设置轨迹上下文并重置记忆
        if hasattr(self.memory, "set_trajectory_context"):
            self.memory.set_trajectory_context(
                question_type=trajectory.question_type,
                scenario=trajectory.scenario,
                tid=trajectory.tid,
                run_id=run_id,
            )
        self.memory.reset()

        # Step 2: 转换消息并注入历史
        messages = self.data_loader.convert_to_messages(trajectory)

        for msg in messages:
            observation = msg.to_observation()
            self.memory.store(observation)

        # 刷新剩余消息缓冲区的事实提取
        self.memory.manage()

        # Step 3: 构建查询并检索
        qa = trajectory.qa
        query = qa.question
        if qa.time:
            query = f"(current time is {qa.time}) {query}"

        context = self.memory.recall(query)
        if verbose:
            logger.debug(f"Retrieved context ({len(context)} chars): {context[:200]}...")

        # Step 4: 使用 LLM 回答问题
        predicted = await self._answer_question(
            context=context,
            question=qa.question,
            time_context=qa.time,
            choices=qa.choices,
        )

        # Step 5: 计算 Recall
        target_ids = self.data_loader.get_target_step_ids(trajectory)
        retrieved_ids = self.memory.retri(query)
        recall_score = calculate_recall_at_k(retrieved_ids, target_ids, k=10)

        # 估算 token 数
        memory_tokens = len(context.split())

        return EvaluationResult(
            tid=trajectory.tid,
            question_type=trajectory.question_type,
            scenario=trajectory.scenario,
            question=qa.question,
            is_correct=(predicted == qa.ground_truth),
            predicted_answer=predicted,
            ground_truth=qa.ground_truth,
            user_id=getattr(self.memory, "user_id", ""),
            context_preview=context[:500] if context else "",
            run_id=run_id,
            recall_score=recall_score,
            write_time_avg_ms=0.0,
            read_time_ms=0.0,
            memory_size_tokens=memory_tokens,
            retrieved_step_ids=retrieved_ids,
            target_step_ids=target_ids,
        )

    async def _answer_question(
        self,
        context: str,
        question: str,
        time_context: str,
        choices: Dict[str, str],
    ) -> str:
        """
        使用 LLM 回答多选题。

        Args:
            context: 检索到的记忆上下文
            question: 问题
            time_context: 时间上下文
            choices: 选项 {"A": "...", "B": "...", "C": "...", "D": "..."}

        Returns:
            预测的答案 (A/B/C/D)
        """
        # 检测是否为情感类问题，使用专门的 system prompt
        is_emotion_query = "sentiment" in question.lower()

        prompt = f"""Please answer the following question based on past memories.

Past memory:
{context}

Question: {f"(current time is {time_context}) " if time_context else ""}{question}

Choices:
A. {choices.get('A', '')}
B. {choices.get('B', '')}
C. {choices.get('C', '')}
D. {choices.get('D', '')}

Please output only the correct option letter (A, B, C, or D), without any other text."""

        try:
            if is_emotion_query:
                system_content = (
                    "You are classifying user sentiment at a specific time point based on "
                    "appraisal theory. In Past memory, lines prefixed with >>> are closest "
                    "to the queried timestamp — focus primarily on those lines.\n\n"
                    "IMPORTANT: Classify based on the SITUATION the user describes, not "
                    "the surface tone of their words:\n"
                    "- Witnessing something unexpected/novel → Surprise (even if described positively)\n"
                    "- Aversion to an unpleasant/oppressive environment → Disgust (even if expressed passively)\n"
                    "- Being unfairly dismissed/ignored/wronged → Anger (even if expressed with resignation)\n"
                    "- Loss, helplessness, or grief → Sadness\n"
                    "- Threat or danger → Fear\n"
                    "- Positive outcome or fulfillment → Happiness\n\n"
                    "Output only a single letter (A, B, C, or D)."
                )
            else:
                system_content = (
                    "You are a precise assistant that answers factual questions based on "
                    "provided memory context. Read ALL facts carefully before answering.\n\n"
                    "IMPORTANT:\n"
                    "- For comparison questions: extract the exact values for both entities, "
                    "then compare them systematically\n"
                    "- For counting questions: list each qualifying item explicitly, then count\n"
                    "- For multi-hop questions: first identify the intermediate entity, "
                    "then look up its attributes\n"
                    "- The same person may appear under different names or references "
                    "(e.g., full name vs. relationship like 'subordinate')\n"
                    "- If a fact was updated, use the MOST RECENT value\n\n"
                    "Output only a single letter (A, B, C, or D)."
                )

            response = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_temperature,
                model=self.config.llm_model,
                max_tokens=32,
            )

            # 解析答案
            answer = response.strip().upper()

            # 1. 精确单字母匹配（最常见情况）
            if answer in {"A", "B", "C", "D"}:
                return answer

            # 2. 匹配独立选项字母（单词边界），取最后一个
            boundary_matches = re.findall(r"\b([ABCD])\b", answer)
            if boundary_matches:
                return boundary_matches[-1]

            # 3. 兜底：取最后出现的选项字母
            letter_matches = re.findall(r"[ABCD]", answer)
            if letter_matches:
                return letter_matches[-1]

            logger.warning(f"Could not parse LLM answer: {response.strip()!r}")
            return "ERROR_PARSE"

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "ERROR_LLM"

    def _load_checkpoint(self) -> tuple:
        """
        加载 checkpoint。

        Returns:
            (completed_key_set, results_list)
        """
        completed: Set[str] = set()
        results: List[EvaluationResult] = []

        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "completed_keys" in data:
                    # 新格式 (schema_version >= 2)
                    completed = set(str(k) for k in data["completed_keys"])
                else:
                    # 旧格式: completed_tids 存的是裸 tid，已知不可靠
                    # 尝试从 detailed_results.json 重建 composite key
                    results_path = self.output_dir / "detailed_results.json"
                    if results_path.exists():
                        with open(results_path, "r", encoding="utf-8") as f:
                            results_data = json.load(f)
                            results = [
                                EvaluationResult.from_dict(r) for r in results_data
                            ]
                        # 从已有结果重建 composite key
                        completed = {
                            f"{r.question_type}__{r.scenario}__{r.tid}"
                            for r in results
                        }
                        logger.info(
                            f"Migrated old checkpoint: rebuilt {len(completed)} keys from detailed_results"
                        )
                    return completed, results

                # 新格式: 正常加载结果
                results_path = self.output_dir / "detailed_results.json"
                if results_path.exists():
                    with open(results_path, "r", encoding="utf-8") as f:
                        results_data = json.load(f)
                        results = [
                            EvaluationResult.from_dict(r) for r in results_data
                        ]
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        return completed, results

    def _save_checkpoint(self, completed_keys: Set[str]) -> None:
        """
        保存 checkpoint。

        Args:
            completed_keys: 已完成的轨迹 composite key 集合
        """
        data = {
            "schema_version": 2,
            "completed_keys": sorted(completed_keys),
            "total_completed": len(completed_keys),
        }

        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # 同时保存详细结果
        self._save_detailed_results()

    def _save_detailed_results(self) -> None:
        """保存详细结果"""
        results_path = self.output_dir / "detailed_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, ensure_ascii=False)

    def _save_results(self, metrics: AggregateMetrics) -> None:
        """
        保存最终结果和指标。

        Args:
            metrics: 聚合指标
        """
        # 保存详细结果
        self._save_detailed_results()

        # 保存指标摘要
        metrics_path = self.output_dir / "metrics_summary.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        # 保存完整报告
        metrics_calc = MetricsCalculator(self.results)
        report = metrics_calc.get_full_report()

        report_path = self.output_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")

    def reset_checkpoint(self) -> None:
        """重置 checkpoint"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

        results_path = self.output_dir / "detailed_results.json"
        if results_path.exists():
            results_path.unlink()

        self.results = []
        logger.info("Checkpoint reset")
