"""
报告处理器。

处理报告生成、查询和管理相关的操作。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.models import Report
from server.dependencies import UserDependencies


@dataclass
class GenerateReportResult:
    """
    报告生成结果。
    """
    success: bool
    report: Optional[Report] = None
    report_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "report": self.report.to_dict() if self.report else None,
            "report_path": self.report_path,
            "error": self.error,
        }


class ReportHandler:
    """
    报告处理器。

    处理报告相关的操作：
    1. 生成调研报告
    2. 获取报告列表
    3. 获取报告详情
    4. 管理待推送报告
    """

    def __init__(self, deps: UserDependencies):
        """
        初始化报告处理器。

        Args:
            deps: 用户依赖
        """
        self.deps = deps
        self.report_service = deps.report_service
        self.search_service = deps.search_service

    async def generate_report(
        self,
        topic: str,
        purpose: str = "Research requested by the user",
    ) -> GenerateReportResult:
        """
        生成调研报告。

        Args:
            topic: 调研主题
            purpose: 调研目的

        Returns:
            报告生成结果
        """
        try:
            # 执行搜索并生成报告
            result = self.search_service.search_for_report(
                topic=topic,
                purpose=purpose,
            )

            if not result.success:
                return GenerateReportResult(
                    success=False,
                    error=result.error or "搜索失败",
                )

            if not result.report:
                return GenerateReportResult(
                    success=False,
                    error="未能生成报告",
                )

            # 创建报告对象
            import uuid
            report = Report(
                id=str(uuid.uuid4()),
                topic=topic,
                title=topic,
                summary=result.report[:200] if result.report else "",
                content=result.report,
                sources=result.sources,
            )

            # 添加到待推送队列
            self.report_service.add_pending(report)

            return GenerateReportResult(
                success=True,
                report=report,
                report_path=result.report_path,
            )

        except Exception as e:
            print(f"[ReportHandler] 生成报告失败: {e}")
            import traceback
            traceback.print_exc()
            return GenerateReportResult(
                success=False,
                error=str(e),
            )

    def list_reports(self) -> List[Dict[str, Any]]:
        """
        获取所有报告列表。

        Returns:
            报告列表
        """
        return self.report_service.list_all()

    def get_report(self, report_id: str) -> Optional[Report]:
        """
        获取报告详情。

        Args:
            report_id: 报告 ID

        Returns:
            报告对象或 None
        """
        return self.report_service.get_by_id(report_id)

    def get_pending_reports(self) -> List[Report]:
        """
        获取待推送报告列表。

        Returns:
            待推送报告列表
        """
        return self.report_service.get_pending()

    def mark_report_pushed(self, report_id: str) -> bool:
        """
        标记报告为已推送。

        Args:
            report_id: 报告 ID

        Returns:
            是否成功
        """
        try:
            self.report_service.mark_pushed(report_id)
            return True
        except Exception as e:
            print(f"[ReportHandler] 标记报告已推送失败: {e}")
            return False

    async def evaluate_push(
        self,
        report: Report,
        current_context: str = "",
        seconds_since_last_interaction: int = 0,
        recent_interaction_count: int = 0,
    ) -> Dict[str, Any]:
        """
        评估报告是否应该推送。

        Args:
            report: 报告对象
            current_context: 当前对话上下文
            seconds_since_last_interaction: 距离上次交互的秒数
            recent_interaction_count: 最近 5 分钟的交互次数

        Returns:
            推送决策
        """
        # 获取用户兴趣
        profile = self.deps.memory.get_user_profile()
        user_interests = profile.interests

        # 使用推送服务评估
        decision = self.deps.push_service.evaluate(
            report=report,
            user_interests=user_interests,
            current_context=current_context,
            seconds_since_last_interaction=seconds_since_last_interaction,
            recent_interaction_count=recent_interaction_count,
        )

        return {
            "action": decision.action.value,
            "score": decision.score,
            "value": decision.value,
            "cost": decision.cost,
            "reason": decision.reason,
        }

    def get_report_stats(self) -> Dict[str, Any]:
        """
        获取报告统计信息。

        Returns:
            统计信息
        """
        all_reports = self.report_service.list_all()
        pending_reports = self.report_service.get_pending()

        return {
            "total_reports": len(all_reports),
            "pending_reports": len(pending_reports),
        }
