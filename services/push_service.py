"""
推送决策服务。

评估是否应该向用户推送信息，以及以何种方式推送。
"""

from typing import TYPE_CHECKING, Any, Dict, List

from core.config import TasksConfig
from core.models import PushAction, PushDecision, Report

if TYPE_CHECKING:
    from agents.push_score import PushScoreAgent


class PushService:
    """
    推送决策服务。

    职责:
    1. 调用 PushScoreAgent 评估信息价值和打断成本
    2. 根据阈值配置决定推送动作
    3. 提供统一的推送决策接口

    推送决策逻辑:
    - Score = Value - Cost + 50 (归一化到 0-100)
    - Score > push_threshold_notify: 提醒推送 (NOTIFY)
    - 否则: 静默存储 (SILENT)
    """

    def __init__(
        self,
        push_score_agent: "PushScoreAgent",
        config: TasksConfig,
    ):
        """
        初始化推送服务。

        Args:
            push_score_agent: 推送评估 Agent
            config: 任务配置（包含推送阈值）
        """
        self.scorer = push_score_agent
        self.config = config

    def evaluate(
        self,
        report: Report,
        user_interests: List[str],
        current_context: str,
        seconds_since_last_interaction: int,
        recent_interaction_count: int = 0,
    ) -> PushDecision:
        """
        评估是否应该推送报告。

        Args:
            report: 待推送的报告
            user_interests: 用户兴趣列表
            current_context: 当前对话上下文
            seconds_since_last_interaction: 距离上次交互的秒数
            recent_interaction_count: 最近 5 分钟的交互次数

        Returns:
            推送决策
        """
        # 调用 PushScoreAgent 评估
        eval_result = self.scorer.run(
            report_topic=report.topic,
            report_summary=report.summary[:300] if report.summary else "",
            user_interests=user_interests,
            current_context=current_context[:500] if current_context else "",
            seconds_since_last_interaction=seconds_since_last_interaction,
            recent_interaction_count=recent_interaction_count,
        )

        value = eval_result.get("value", 50)
        cost = eval_result.get("cost", 50)
        reason = eval_result.get("reason", "")

        # 计算综合分数
        score = self._calculate_score(value, cost)

        # 根据阈值决定动作
        action = self._decide_action(score)

        return PushDecision(
            action=action,
            score=score,
            value=value,
            cost=cost,
            reason=reason,
        )

    def evaluate_multiple(
        self,
        reports: List[Report],
        user_interests: List[str],
        current_context: str,
        seconds_since_last_interaction: int,
        recent_interaction_count: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        批量评估多个报告。

        按推送分数降序排列返回结果。

        Args:
            reports: 报告列表
            user_interests: 用户兴趣列表
            current_context: 当前对话上下文
            seconds_since_last_interaction: 距离上次交互的秒数
            recent_interaction_count: 最近 5 分钟的交互次数

        Returns:
            评估结果列表，包含 report 和 decision
        """
        results = []

        for report in reports:
            decision = self.evaluate(
                report=report,
                user_interests=user_interests,
                current_context=current_context,
                seconds_since_last_interaction=seconds_since_last_interaction,
                recent_interaction_count=recent_interaction_count,
            )
            results.append({
                "report": report,
                "decision": decision,
            })

        # 按分数排序
        results.sort(key=lambda x: x["decision"].score, reverse=True)

        return results

    def should_push(
        self,
        report: Report,
        user_interests: List[str],
        current_context: str,
        seconds_since_last_interaction: int,
        recent_interaction_count: int = 0,
    ) -> bool:
        """
        快速判断是否应该推送。

        这是一个便捷方法，用于快速判断。

        Args:
            report: 报告
            user_interests: 用户兴趣列表
            current_context: 当前对话上下文
            seconds_since_last_interaction: 距离上次交互的秒数
            recent_interaction_count: 最近 5 分钟的交互次数

        Returns:
            是否应该推送（非静默）
        """
        decision = self.evaluate(
            report=report,
            user_interests=user_interests,
            current_context=current_context,
            seconds_since_last_interaction=seconds_since_last_interaction,
            recent_interaction_count=recent_interaction_count,
        )
        return decision.action != PushAction.SILENT

    def get_best_to_push(
        self,
        reports: List[Report],
        user_interests: List[str],
        current_context: str,
        seconds_since_last_interaction: int,
        recent_interaction_count: int = 0,
    ) -> Dict[str, Any]:
        """
        从多个报告中选择最适合推送的一个。

        Args:
            reports: 报告列表
            user_interests: 用户兴趣列表
            current_context: 当前对话上下文
            seconds_since_last_interaction: 距离上次交互的秒数
            recent_interaction_count: 最近 5 分钟的交互次数

        Returns:
            最佳推送候选（包含 report 和 decision），
            如果没有适合推送的则返回 None
        """
        if not reports:
            return None

        results = self.evaluate_multiple(
            reports=reports,
            user_interests=user_interests,
            current_context=current_context,
            seconds_since_last_interaction=seconds_since_last_interaction,
            recent_interaction_count=recent_interaction_count,
        )

        # 找到分数最高且非静默的
        for result in results:
            if result["decision"].action != PushAction.SILENT:
                return result

        return None

    def _calculate_score(self, value: int, cost: int) -> int:
        """
        计算推送分数。

        公式: Score = Value - Cost + 50，归一化到 0-100

        Args:
            value: 信息价值（0-100）
            cost: 打断成本（0-100）

        Returns:
            综合分数（0-100）
        """
        raw_score = value - cost
        score = max(0, min(100, raw_score + 50))
        return score

    def _decide_action(self, score: int) -> PushAction:
        """
        根据分数决定推送动作。

        Args:
            score: 推送分数（0-100）

        Returns:
            推送动作
        """
        if score > self.config.push_threshold_notify:
            return PushAction.NOTIFY
        else:
            return PushAction.SILENT

    @staticmethod
    def get_action_description(action: PushAction) -> str:
        """
        获取推送动作的人类可读描述。

        Args:
            action: 推送动作

        Returns:
            描述文本
        """
        descriptions = {
            PushAction.NOTIFY: "提醒推送 (Notification)",
            PushAction.SILENT: "静默存储 (Silent Log)",
        }
        return descriptions.get(action, str(action))

    def format_decision_summary(self, decision: PushDecision) -> str:
        """
        格式化推送决策摘要。

        Args:
            decision: 推送决策

        Returns:
            格式化的摘要文本
        """
        action_desc = self.get_action_description(decision.action)
        return (
            f"推送决策: {action_desc}\n"
            f"综合分数: {decision.score}\n"
            f"信息价值: {decision.value}\n"
            f"打断成本: {decision.cost}\n"
            f"原因: {decision.reason}"
        )
