"""
统一搜索服务。

协调搜索规划、执行搜索、存储结果、生成报告的完整流程。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.config import Config
from core.models import (
    MemoryUpdateRequest,
    Report,
    SearchSource,
    UpdateType,
)
from core.prompts import Prompts

if TYPE_CHECKING:
    from agents.report_generator import ReportGeneratorAgent
    from agents.search_planner import SearchPlannerAgent
    from memory.memory_system import MemorySystem
    from tools.llm_client import LLMClient
    from tools.search_tool import SearchTool
    from tools.web_fetcher import WebFetcher


@dataclass
class SearchServiceResult:
    """
    搜索服务结果。

    包含搜索执行的完整结果和统计信息。
    """
    success: bool
    sources: List[SearchSource] = field(default_factory=list)
    knowledge_stats: Dict[str, int] = field(default_factory=dict)
    report: Optional[str] = None
    report_path: Optional[str] = None
    error: Optional[str] = None
    mode: str = "simple"  # "simple" | "comprehensive"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "sources": [s.to_dict() for s in self.sources],
            "knowledge_stats": self.knowledge_stats,
            "report": self.report,
            "report_path": self.report_path,
            "error": self.error,
            "mode": self.mode,
        }


class SearchService:
    """
    统一搜索服务。

    职责:
    1. 调用 SearchPlannerAgent 规划搜索
    2. 执行搜索并抓取内容
    3. 调用 MemorySystem.submit_update 存储结果
    4. 可选生成报告

    所有搜索场景统一通过此服务:
    - 记忆验证发现知识缺口
    - NeedsPrediction 主动搜索
    - 用户 Query 需要补充信息
    """

    def __init__(
        self,
        memory_system: "MemorySystem",
        search_planner: "SearchPlannerAgent",
        report_generator: "ReportGeneratorAgent",
        search_tool: "SearchTool",
        web_fetcher: "WebFetcher",
        llm_client: "LLMClient",
        config: Config,
    ):
        """
        初始化搜索服务。

        Args:
            memory_system: 记忆系统
            search_planner: 搜索规划 Agent
            report_generator: 报告生成 Agent
            search_tool: 搜索工具
            web_fetcher: 网页抓取工具
            llm_client: LLM 客户端（用于生成摘要）
            config: 全局配置
        """
        self.memory = memory_system
        self.search_planner = search_planner
        self.report_generator = report_generator
        self.search_tool = search_tool
        self.web_fetcher = web_fetcher
        self.llm_client = llm_client
        self.config = config

    def search(
        self,
        topic: str,
        purpose: str,
        generate_report: bool = False,
    ) -> SearchServiceResult:
        """
        执行搜索。

        完整流程：
        1. 规划搜索策略
        2. 执行搜索查询
        3. 抓取网页内容
        4. 生成内容摘要
        5. 存储到记忆系统
        6. 可选生成报告

        Args:
            topic: 搜索主题
            purpose: 搜索目的
            generate_report: 是否生成报告

        Returns:
            搜索结果，包含来源列表、存储统计和可选报告
        """
        try:
            # 1. 规划搜索
            user_profile = self.memory.get_user_profile()
            plan = self.search_planner.run(
                topic=topic,
                purpose=purpose,
                user_profile=user_profile,
            )

            queries = plan.get("queries", [])
            if not queries:
                return SearchServiceResult(
                    success=False,
                    error="搜索规划失败：未生成查询",
                )

            mode = plan.get("mode", "simple")

            # 2. 执行搜索
            sources = self._execute_search(queries, mode)

            if not sources:
                return SearchServiceResult(
                    success=False,
                    mode=mode,
                    error="搜索未返回结果",
                )

            # 3. 存储到记忆系统
            stats = self._store_results(sources, topic)

            # 4. 可选生成报告
            report = None
            report_path = None
            if generate_report and sources:
                report = self.report_generator.run(
                    topic=topic,
                    sources=sources,
                    user_profile=user_profile,
                )
                report_path = self._save_report(topic, report, sources)

            return SearchServiceResult(
                success=True,
                sources=sources,
                knowledge_stats=stats,
                report=report,
                report_path=report_path,
                mode=mode,
            )

        except Exception as e:
            print(f"[SearchService] 搜索错误: {e}")
            import traceback
            traceback.print_exc()
            return SearchServiceResult(
                success=False,
                error=str(e),
            )

    def search_for_query(
        self,
        query: str,
        purpose: str = "Answer the user's question",
    ) -> SearchServiceResult:
        """
        为用户查询执行搜索（不生成报告）。

        这是一个便捷方法，用于对话中需要补充信息的场景。

        Args:
            query: 用户查询
            purpose: 搜索目的

        Returns:
            搜索结果
        """
        return self.search(
            topic=query,
            purpose=purpose,
            generate_report=False,
        )

    def search_for_report(
        self,
        topic: str,
        purpose: str = "Generate a research report",
    ) -> SearchServiceResult:
        """
        执行搜索并生成报告。

        这是一个便捷方法，用于主动搜索场景。

        Args:
            topic: 搜索主题
            purpose: 搜索目的

        Returns:
            搜索结果（包含报告）
        """
        return self.search(
            topic=topic,
            purpose=purpose,
            generate_report=True,
        )

    def _execute_search(
        self,
        queries: List[Dict[str, Any]],
        mode: str,
    ) -> List[SearchSource]:
        """
        执行搜索计划。

        Args:
            queries: 搜索查询列表
            mode: 搜索模式 ("simple" | "comprehensive")

        Returns:
            搜索来源列表
        """
        # 根据模式决定查询数量。Benchmark 可通过 config 注入不同预算做 sweep。
        max_queries = getattr(self.config.search, "max_queries_per_search", 1)
        try:
            max_queries = int(max_queries)
        except (TypeError, ValueError):
            max_queries = 1
        max_queries = max(1, max_queries)
        if mode == "comprehensive":
            max_queries = min(max_queries + 3, 8)

        results_per_query = self.config.search.max_results_per_query

        # 按优先级排序查询
        sorted_queries = sorted(
            queries,
            key=lambda x: x.get("priority", 5),
            reverse=True,
        )

        sources = []
        seen_urls = set()

        for q in sorted_queries[:max_queries]:
            query_text = q.get("query", "")
            if not query_text:
                continue

            # 执行搜索
            raw_results = self.search_tool.search(query_text, results_per_query)

            for result in raw_results:
                # 去重
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)

                # 抓取完整内容
                full_content = self.web_fetcher.fetch(result.url)

                # 更新 SearchSource
                source = SearchSource(
                    url=result.url,
                    title=result.title,
                    snippet=result.snippet,
                    full_content=full_content or "",
                )
                sources.append(source)

        return sources

    def _store_results(
        self,
        sources: List[SearchSource],
        topic: str,
    ) -> Dict[str, int]:
        """
        存储搜索结果到记忆系统。

        通过 MemorySystem.submit_update 提交更新请求，
        由 MemoryUpdateAgent 处理查重、合并等逻辑。

        Args:
            sources: 搜索来源列表
            topic: 主题

        Returns:
            存储统计 {"added": n, "skipped": n, "merged": n, "replaced": n}
        """
        stats = {"added": 0, "skipped": 0, "merged": 0, "replaced": 0}

        for source in sources:
            # 跳过没有内容的来源
            content = source.full_content or source.snippet
            if not content:
                continue

            # 生成摘要
            summary = self._generate_summary(source.url, content)
            if not summary:
                summary = source.snippet

            # 提交更新请求
            request = MemoryUpdateRequest(
                update_type=UpdateType.ADD_KNOWLEDGE,
                source="search_service",
                user_id=self.memory.user_id,
                data={
                    "content": content,
                    "summary": summary,
                    "topic": topic,
                    "source_url": source.url,
                    "title": source.title,
                },
            )

            result = self.memory.submit_update(request)

            # 统计结果
            action = result.action.value
            if action in stats:
                stats[action] += 1

        return stats

    def _generate_summary(self, url: str, content: str) -> str:
        """
        生成内容摘要。

        Args:
            url: 来源 URL
            content: 内容

        Returns:
            摘要文本
        """
        max_content_length = 8000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        prompt = Prompts.CONTENT_SUMMARY.format(
            url=url,
            content=content,
        )

        try:
            summary = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "You generate concise summaries for web content."},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4o-mini",  # 使用快速模型生成摘要
                temperature=0.3,
            )
            return summary.strip()
        except Exception as e:
            print(f"[SearchService] 生成摘要失败: {e}")
            # 返回截断的原始内容作为摘要
            return content[:500] if len(content) > 500 else content

    def _save_report(
        self,
        topic: str,
        report_content: str,
        sources: List[SearchSource],
    ) -> str:
        """
        保存报告到文件。

        Args:
            topic: 主题
            report_content: 报告内容
            sources: 来源列表

        Returns:
            报告文件路径
        """
        reports_dir = self.config.storage.get_reports_dir(self.memory.user_id)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in topic
        )[:50]
        filename = f"{timestamp}_{safe_topic}.json"

        # 生成标题
        title = self._generate_title(report_content)

        # 生成摘要
        summary = self._generate_report_summary(report_content)

        report_data = {
            "topic": topic,
            "title": title,
            "summary": summary,
            "content": report_content,
            "created_at": datetime.now().isoformat(),
            "sources": [s.to_dict() for s in sources],
        }

        report_path = reports_dir / filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return str(report_path)

    def _generate_title(self, report_content: str) -> str:
        """
        生成报告标题。

        Args:
            report_content: 报告内容

        Returns:
            标题
        """
        preview = report_content[:500] if len(report_content) > 500 else report_content

        try:
            title = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "Generate a concise title (under 50 characters) for this report."},
                    {"role": "user", "content": f"Report content:\n{preview}"},
                ],
                model="gpt-4o-mini",  # 使用快速模型生成标题
                temperature=0.3,
            )
            return title.strip()[:100]
        except Exception:
            # 从报告中提取第一行作为标题
            lines = report_content.strip().split("\n")
            for line in lines:
                line = line.strip().lstrip("#").strip()
                if line:
                    return line[:100]
            return "Research Report"

    def _generate_report_summary(self, report_content: str) -> str:
        """
        生成报告摘要。

        Args:
            report_content: 报告内容

        Returns:
            摘要
        """
        try:
            summary = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "Generate a brief summary (under 200 characters) for this report."},
                    {"role": "user", "content": f"Report:\n{report_content[:2000]}"},
                ],
                model="gpt-4o-mini",  # 使用快速模型生成摘要
                temperature=0.3,
            )
            return summary.strip()[:300]
        except Exception:
            # 返回报告开头作为摘要
            return report_content[:200] + "..." if len(report_content) > 200 else report_content

    def format_sources_for_context(self, sources: List[SearchSource]) -> str:
        """
        将搜索来源格式化为对话上下文。

        用于将搜索结果传递给 AssistantAgent。

        Args:
            sources: 搜索来源列表

        Returns:
            格式化的上下文字符串
        """
        if not sources:
            return ""

        context_parts = []
        for i, source in enumerate(sources, 1):
            content = source.full_content or source.snippet
            if len(content) > 1000:
                content = content[:1000] + "..."

            context_parts.append(
                f"[Source {i}] {source.title}\n"
                f"URL: {source.url}\n"
                f"Content: {content}\n"
            )

        return "\n---\n".join(context_parts)
