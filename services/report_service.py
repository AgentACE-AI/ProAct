"""
报告管理服务。

管理待推送报告队列、报告持久化和报告查询。
"""

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import StorageConfig
from core.models import Report, SearchSource


class ReportService:
    """
    报告管理服务。

    职责:
    1. 管理待推送报告队列
    2. 报告持久化
    3. 报告查询
    4. 报告状态管理（待推送、已推送）
    """

    def __init__(self, user_id: str, storage_config: StorageConfig):
        """
        初始化报告服务。

        Args:
            user_id: 用户 ID
            storage_config: 存储配置
        """
        self.user_id = user_id
        self.storage_config = storage_config

        # 待推送报告存储路径
        self._pending_path = storage_config.get_pending_reports_path(user_id)

        # 报告目录
        self._reports_dir = storage_config.get_reports_dir(user_id)

        # 内存中的待推送报告
        self._pending_reports: Dict[str, Dict[str, Any]] = {}

        # 加载待推送报告
        self._load_pending()

    # ==================== 待推送报告管理 ====================

    def add_pending(self, report: Report) -> str:
        """
        添加待推送报告。

        Args:
            report: 报告对象

        Returns:
            报告 ID
        """
        # 确保有 ID
        if not report.id:
            report.id = str(uuid.uuid4())[:8]

        report_data = report.to_dict()
        report_data["status"] = "pending"
        report_data["added_at"] = datetime.now().isoformat()

        self._pending_reports[report.id] = report_data
        self._save_pending()

        return report.id

    def get_pending(self) -> List[Report]:
        """
        获取待推送报告列表。

        按 relevance * urgency 降序排序。

        Returns:
            待推送报告列表
        """
        pending = [
            Report.from_dict(data)
            for data in self._pending_reports.values()
            if data.get("status") == "pending"
        ]

        # 按综合分数排序
        pending.sort(
            key=lambda r: r.relevance * r.urgency,
            reverse=True,
        )

        return pending

    def mark_pushed(self, report_id: str) -> None:
        """
        标记报告为已推送。

        Args:
            report_id: 报告 ID
        """
        if report_id in self._pending_reports:
            self._pending_reports[report_id]["status"] = "pushed"
            self._pending_reports[report_id]["pushed_at"] = datetime.now().isoformat()
            self._save_pending()

    def remove_pending(self, report_id: str) -> None:
        """
        从待推送队列移除报告。

        Args:
            report_id: 报告 ID
        """
        if report_id in self._pending_reports:
            del self._pending_reports[report_id]
            self._save_pending()

    def clear_pushed(self) -> int:
        """
        清除所有已推送的报告。

        Returns:
            清除的报告数量
        """
        to_remove = [
            rid for rid, data in self._pending_reports.items()
            if data.get("status") == "pushed"
        ]

        for rid in to_remove:
            del self._pending_reports[rid]

        self._save_pending()
        return len(to_remove)

    def update_scores(
        self,
        report_id: str,
        relevance: Optional[int] = None,
        urgency: Optional[int] = None,
    ) -> bool:
        """
        更新报告的相关度/紧急度分数。

        Args:
            report_id: 报告 ID
            relevance: 新的相关度分数（可选）
            urgency: 新的紧急度分数（可选）

        Returns:
            是否更新成功
        """
        if report_id not in self._pending_reports:
            return False

        if relevance is not None:
            self._pending_reports[report_id]["relevance"] = max(0, min(100, relevance))
        if urgency is not None:
            self._pending_reports[report_id]["urgency"] = max(0, min(100, urgency))

        self._save_pending()
        return True

    # ==================== 报告查询 ====================

    def get_by_id(self, report_id: str) -> Optional[Report]:
        """
        获取报告详情。

        首先检查待推送队列，然后检查已保存的报告文件。

        Args:
            report_id: 报告 ID

        Returns:
            报告对象，不存在返回 None
        """
        # 检查待推送队列
        if report_id in self._pending_reports:
            return Report.from_dict(self._pending_reports[report_id])

        # 检查报告文件
        for report_file in self._reports_dir.glob("*.json"):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("id") == report_id:
                        return Report.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    def list_all(self, limit: int = 50) -> List[Report]:
        """
        列出所有报告。

        包括待推送的和已保存的报告。

        Args:
            limit: 返回数量限制

        Returns:
            报告列表（按创建时间降序）
        """
        reports = []

        # 添加待推送报告
        for data in self._pending_reports.values():
            reports.append(Report.from_dict(data))

        # 添加已保存的报告
        for report_file in self._reports_dir.glob("*.json"):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 跳过已在待推送队列中的报告
                    if data.get("id") not in self._pending_reports:
                        reports.append(Report.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue

        # 按创建时间排序
        reports.sort(key=lambda r: r.created_at, reverse=True)

        return reports[:limit]

    def list_by_topic(self, topic: str, limit: int = 20) -> List[Report]:
        """
        按话题列出报告。

        Args:
            topic: 话题关键词
            limit: 返回数量限制

        Returns:
            匹配的报告列表
        """
        all_reports = self.list_all(limit=100)
        matched = [
            r for r in all_reports
            if topic.lower() in r.topic.lower()
        ]
        return matched[:limit]

    # ==================== 统计 ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        获取报告统计信息。

        Returns:
            统计信息
        """
        pending_count = len([
            r for r in self._pending_reports.values()
            if r.get("status") == "pending"
        ])
        pushed_count = len([
            r for r in self._pending_reports.values()
            if r.get("status") == "pushed"
        ])

        # 统计已保存的报告文件数
        saved_count = len(list(self._reports_dir.glob("*.json")))

        return {
            "pending_count": pending_count,
            "pushed_count": pushed_count,
            "saved_count": saved_count,
            "total_in_queue": len(self._pending_reports),
        }

    # ==================== 报告创建辅助 ====================

    def create_report(
        self,
        topic: str,
        title: str,
        summary: str,
        content: str,
        sources: Optional[List[SearchSource]] = None,
        relevance: int = 50,
        urgency: int = 50,
        add_to_pending: bool = True,
    ) -> Report:
        """
        创建新报告。

        这是一个便捷方法，用于创建报告对象并可选添加到待推送队列。

        Args:
            topic: 话题
            title: 标题
            summary: 摘要
            content: 内容
            sources: 来源列表（可选）
            relevance: 相关度（0-100）
            urgency: 紧急度（0-100）
            add_to_pending: 是否添加到待推送队列

        Returns:
            创建的报告对象
        """
        report = Report(
            id=str(uuid.uuid4())[:8],
            topic=topic,
            title=title,
            summary=summary,
            content=content,
            sources=sources or [],
            relevance=relevance,
            urgency=urgency,
        )

        if add_to_pending:
            self.add_pending(report)

        return report

    def save_report_to_file(self, report: Report) -> str:
        """
        保存报告到文件。

        Args:
            report: 报告对象

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in report.topic
        )[:50]
        filename = f"{timestamp}_{safe_topic}.json"

        report_path = self._reports_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        return str(report_path)

    # ==================== 私有方法 ====================

    def _load_pending(self) -> None:
        """从文件加载待推送报告"""
        if self._pending_path.exists():
            try:
                with open(self._pending_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for report_data in data.get("reports", []):
                        report_id = report_data.get("id")
                        if report_id:
                            self._pending_reports[report_id] = report_data
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[ReportService] 加载待推送报告失败: {e}")

    def _save_pending(self) -> None:
        """保存待推送报告到文件"""
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "user_id": self.user_id,
            "reports": list(self._pending_reports.values()),
            "updated_at": datetime.now().isoformat(),
        }

        with open(self._pending_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
