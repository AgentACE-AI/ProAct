"""
MemBench Benchmark 指标计算器。

计算和聚合评估指标。
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .models import AggregateMetrics, EvaluationResult


class MetricsCalculator:
    """指标计算器"""

    def __init__(self, results: List[EvaluationResult]):
        """
        初始化指标计算器。

        Args:
            results: 评估结果列表
        """
        self.results = results

    def compute_all(self) -> AggregateMetrics:
        """
        计算所有聚合指标。

        Returns:
            聚合指标对象
        """
        if not self.results:
            return AggregateMetrics()

        # 基础统计
        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)

        # 按类型统计
        by_type_correct: Dict[str, int] = defaultdict(int)
        by_type_total: Dict[str, int] = defaultdict(int)
        by_type_recall: Dict[str, List[float]] = defaultdict(list)
        by_type_write_time: Dict[str, List[float]] = defaultdict(list)
        by_type_read_time: Dict[str, List[float]] = defaultdict(list)

        # 按场景统计
        by_scenario_correct: Dict[str, int] = defaultdict(int)
        by_scenario_total: Dict[str, int] = defaultdict(int)

        # 召回率
        all_recalls: List[float] = []

        # 效率
        all_write_times: List[float] = []
        all_read_times: List[float] = []

        # 容量曲线数据
        capacity_data: List[Tuple[int, bool]] = []

        for r in self.results:
            qtype = r.question_type
            scenario = r.scenario

            # 按类型统计
            by_type_total[qtype] += 1
            if r.is_correct:
                by_type_correct[qtype] += 1

            # 按场景统计
            by_scenario_total[scenario] += 1
            if r.is_correct:
                by_scenario_correct[scenario] += 1

            # 召回率
            if r.recall_score is not None:
                all_recalls.append(r.recall_score)
                by_type_recall[qtype].append(r.recall_score)

            # 效率
            if r.write_time_avg_ms > 0:
                all_write_times.append(r.write_time_avg_ms)
                by_type_write_time[qtype].append(r.write_time_avg_ms)

            if r.read_time_ms > 0:
                all_read_times.append(r.read_time_ms)
                by_type_read_time[qtype].append(r.read_time_ms)

            # 容量数据
            if r.memory_size_tokens > 0:
                capacity_data.append((r.memory_size_tokens, r.is_correct))

        # 计算聚合指标
        accuracy = correct / total
        accuracy_by_type = {
            qtype: by_type_correct[qtype] / by_type_total[qtype]
            for qtype in by_type_total
        }
        accuracy_by_scenario = {
            scenario: by_scenario_correct[scenario] / by_scenario_total[scenario]
            for scenario in by_scenario_total
        }

        avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
        recall_by_type = {
            qtype: sum(recalls) / len(recalls)
            for qtype, recalls in by_type_recall.items()
            if recalls
        }

        avg_write_time = (
            sum(all_write_times) / len(all_write_times) if all_write_times else 0.0
        )
        avg_read_time = (
            sum(all_read_times) / len(all_read_times) if all_read_times else 0.0
        )

        write_time_by_type = {
            qtype: sum(times) / len(times)
            for qtype, times in by_type_write_time.items()
            if times
        }
        read_time_by_type = {
            qtype: sum(times) / len(times)
            for qtype, times in by_type_read_time.items()
            if times
        }

        # 计算容量曲线
        capacity_curve = self._compute_capacity_curve(capacity_data)

        # 按类型统计
        by_type_stats = {
            qtype: {"correct": by_type_correct[qtype], "total": by_type_total[qtype]}
            for qtype in by_type_total
        }

        return AggregateMetrics(
            accuracy=accuracy,
            accuracy_by_type=accuracy_by_type,
            accuracy_by_scenario=accuracy_by_scenario,
            avg_recall_at_10=avg_recall,
            recall_by_type=recall_by_type,
            avg_write_time_ms=avg_write_time,
            avg_read_time_ms=avg_read_time,
            write_time_by_type=write_time_by_type,
            read_time_by_type=read_time_by_type,
            capacity_curve=capacity_curve,
            total_trajectories=total,
            correct_count=correct,
            by_type_stats=by_type_stats,
        )

    def _compute_capacity_curve(
        self,
        capacity_data: List[Tuple[int, bool]],
    ) -> List[Tuple[int, float]]:
        """
        计算容量曲线。

        按 token 数量排序，计算累积准确率。

        Args:
            capacity_data: [(token_count, is_correct), ...]

        Returns:
            [(token_count, cumulative_accuracy), ...]
        """
        if not capacity_data:
            return []

        # 按 token 数量排序
        sorted_data = sorted(capacity_data, key=lambda x: x[0])

        # 计算累积准确率
        curve = []
        cumulative_correct = 0

        for i, (tokens, is_correct) in enumerate(sorted_data):
            if is_correct:
                cumulative_correct += 1
            cumulative_accuracy = cumulative_correct / (i + 1)
            curve.append((tokens, cumulative_accuracy))

        # 采样（减少数据点）
        if len(curve) > 100:
            step = len(curve) // 100
            curve = curve[::step]

        return curve

    def print_summary(self) -> None:
        """打印指标摘要"""
        metrics = self.compute_all()

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\nOverall Accuracy: {metrics.accuracy:.2%}")
        print(f"Total Trajectories: {metrics.total_trajectories}")
        print(f"Correct: {metrics.correct_count}")

        print("\n--- Accuracy by Question Type ---")
        for qtype, acc in sorted(metrics.accuracy_by_type.items()):
            stats = metrics.by_type_stats.get(qtype, {})
            correct = stats.get("correct", 0)
            total = stats.get("total", 0)
            print(f"  {qtype}: {acc:.2%} ({correct}/{total})")

        if metrics.avg_recall_at_10 > 0:
            print(f"\n--- Recall@10 ---")
            print(f"Average: {metrics.avg_recall_at_10:.2%}")
            for qtype, recall in sorted(metrics.recall_by_type.items()):
                print(f"  {qtype}: {recall:.2%}")

        print("\n" + "=" * 60)

    def get_full_report(self) -> Dict[str, Any]:
        """
        获取完整报告。

        Returns:
            报告字典
        """
        metrics = self.compute_all()

        return {
            "summary": {
                "accuracy": metrics.accuracy,
                "total_trajectories": metrics.total_trajectories,
                "correct_count": metrics.correct_count,
                "avg_recall_at_10": metrics.avg_recall_at_10,
                "avg_write_time_ms": metrics.avg_write_time_ms,
                "avg_read_time_ms": metrics.avg_read_time_ms,
            },
            "by_question_type": {
                qtype: {
                    "accuracy": metrics.accuracy_by_type.get(qtype, 0),
                    "recall": metrics.recall_by_type.get(qtype, 0),
                    "write_time_ms": metrics.write_time_by_type.get(qtype, 0),
                    "read_time_ms": metrics.read_time_by_type.get(qtype, 0),
                    **metrics.by_type_stats.get(qtype, {}),
                }
                for qtype in metrics.by_type_stats
            },
            "by_scenario": {
                scenario: {"accuracy": acc}
                for scenario, acc in metrics.accuracy_by_scenario.items()
            },
            "capacity_curve": metrics.capacity_curve,
        }


def calculate_recall_at_k(
    retrieved: List[int],
    targets: List[int],
    k: int = 10,
) -> float:
    """
    计算 Recall@K。

    Args:
        retrieved: 检索到的索引列表
        targets: 目标索引列表
        k: top-k

    Returns:
        Recall@K 分数
    """
    if not targets:
        return 0.0

    # 只考虑 top-k
    retrieved_set = set(retrieved[:k])
    targets_set = set(targets)

    hits = len(retrieved_set & targets_set)
    return hits / len(targets_set)
