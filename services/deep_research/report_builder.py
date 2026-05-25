"""
报告生成器。

负责生成个性化调研报告。
"""

import asyncio
import uuid
from typing import Any, Dict, List, TYPE_CHECKING

from core.config import AgentModelConfig
from core.models import UserProfile
from services.deep_research.models import (
    DeepResearchReport,
    KnowledgeGraph,
    ResearchPlan,
    SearchIteration,
    SourceResult,
)
from services.deep_research.prompts import DeepResearchPrompts

if TYPE_CHECKING:
    from tools.llm_client import LLMClient


class ReportBuilder:
    """
    报告生成器。

    职责:
    1. 生成执行摘要
    2. 分章节撰写内容
    3. 添加引用
    4. 生成方法论说明
    5. 标注局限性
    """

    def __init__(
        self,
        config: AgentModelConfig,
        llm_client: "LLMClient",
    ):
        """
        初始化报告生成器。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        self.config = config
        self.llm = llm_client

    async def build(
        self,
        plan: ResearchPlan,
        knowledge_graph: KnowledgeGraph,
        iterations: List[SearchIteration],
        user_profile: UserProfile,
    ) -> DeepResearchReport:
        """
        生成调研报告。

        直接把 user_profile 传给 LLM，让它理解用户偏好并生成个性化报告。

        Args:
            plan: 研究计划
            knowledge_graph: 知识图谱
            iterations: 搜索迭代列表
            user_profile: 用户画像

        Returns:
            深度调研报告
        """
        # 准备上下文
        user_context = user_profile.to_prompt_text()
        facts_summary = self._summarize_facts(knowledge_graph)
        sources_list = self._format_sources(iterations)
        conflicts_text = self._format_conflicts(knowledge_graph)

        # 生成报告内容
        prompt = DeepResearchPrompts.REPORT_GENERATION.format(
            topic=plan.refined_topic,
            research_goals="\n".join(f"- {g}" for g in plan.research_goals),
            outline=self._format_outline(plan),
            user_profile=user_context,
            facts=facts_summary,
            sources=sources_list,
            conflicts=conflicts_text,
        )

        try:
            report_content = await asyncio.to_thread(
                self.llm.chat,
                messages=[
                    {"role": "system", "content": "You are a professional research report writer. Generate a high-quality report from the research results."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=4000,
            )
        except Exception as e:
            print(f"[ReportBuilder] 报告生成失败: {e}")
            report_content = self._generate_fallback_report(plan, knowledge_graph)

        # 生成摘要
        summary = await self._generate_summary(report_content, user_context)

        # 生成方法论说明
        methodology = await self._generate_methodology(iterations)

        # 收集所有来源
        all_sources = []
        for iteration in iterations:
            for source in iteration.sources_found:
                if source.quality.overall >= 0.5:  # 只包含质量达标的来源
                    all_sources.append(source)

        return DeepResearchReport(
            id=str(uuid.uuid4()),
            topic=plan.refined_topic,
            title=plan.suggested_title or f"{plan.refined_topic} Research Report",
            summary=summary,
            content=report_content,
            sources=all_sources[:20],  # 限制来源数量
            methodology=methodology,
            limitations=self._extract_limitations(report_content),
        )

    def _summarize_facts(self, knowledge_graph: KnowledgeGraph) -> str:
        """整理事实摘要，包含来源引用"""
        sections = []

        # 首先建立 URL 到引用编号的映射
        url_to_index = {}
        index = 1
        for facts in knowledge_graph.facts_by_subtopic.values():
            for f in facts:
                if f.source_url and f.source_url not in url_to_index:
                    url_to_index[f.source_url] = index
                    index += 1

        for subtopic, facts in knowledge_graph.facts_by_subtopic.items():
            if not facts:
                continue

            facts_text_list = []
            for f in facts[:10]:  # 每个子主题最多10条
                fact_text = f.summary or f.content[:150]
                # 添加来源引用
                if f.source_url and f.source_url in url_to_index:
                    fact_text += f" [{url_to_index[f.source_url]}]"
                facts_text_list.append(f"  - {fact_text}")

            facts_text = "\n".join(facts_text_list)
            sections.append(f"### {subtopic}\n{facts_text}")

        return "\n\n".join(sections) if sections else "No facts collected yet"

    def _format_sources(self, iterations: List[SearchIteration]) -> str:
        """格式化来源列表，包含发布日期和完整引用"""
        sources = []
        seen_urls = set()

        # 收集所有来源信息
        source_info = []
        for iteration in iterations:
            for source in iteration.sources_found:
                if source.url not in seen_urls and source.quality.overall >= 0.5:
                    seen_urls.add(source.url)
                    # 从提取的事实中获取发布日期
                    publish_date = ""
                    access_date = ""
                    for fact in iteration.new_facts:
                        if fact.source_url == source.url:
                            publish_date = fact.publish_date
                            access_date = fact.access_date
                            break

                    source_info.append({
                        "title": source.title,
                        "url": source.url,
                        "publish_date": publish_date,
                        "access_date": access_date,
                    })

                if len(source_info) >= 20:
                    break
            if len(source_info) >= 20:
                break

        # 格式化为引用格式
        for i, info in enumerate(source_info, 1):
            if info["publish_date"]:
                date_str = info["publish_date"]
            elif info["access_date"]:
                date_str = f"Accessed on {info['access_date']}"
            else:
                date_str = "Date unknown"

            sources.append(f"[{i}] {info['title']}. {date_str}. {info['url']}")

        return "\n".join(sources) if sources else "No sources"

    def _format_outline(self, plan: ResearchPlan) -> str:
        """格式化报告大纲"""
        sections = plan.outline.sections if plan.outline.sections else [
            "Overview",
            "Key Findings",
            "Detailed Analysis",
            "Conclusions and Recommendations",
        ]
        return "\n".join(f"- {s}" for s in sections)

    def _format_conflicts(self, knowledge_graph: KnowledgeGraph) -> str:
        """格式化冲突信息"""
        if not knowledge_graph.conflicts:
            return "No obvious conflicts were found"

        conflicts_text = []
        for c in knowledge_graph.conflicts[:5]:
            conflicts_text.append(
                f"- Viewpoint 1 ({c.source1}): {c.fact1[:100]}\n"
                f"  Viewpoint 2 ({c.source2}): {c.fact2[:100]}\n"
                f"  Analysis: {c.resolution}"
            )

        return "\n\n".join(conflicts_text)

    async def _generate_summary(
        self,
        report_content: str,
        user_context: str,
    ) -> str:
        """生成执行摘要"""
        prompt = DeepResearchPrompts.GENERATE_SUMMARY.format(
            report_content=report_content[:3000],
            user_profile=user_context,
        )

        try:
            summary = await asyncio.to_thread(
                self.llm.chat,
                messages=[
                    {"role": "system", "content": "You are a professional summary writer."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=0.5,
                max_tokens=500,
            )
            return summary
        except Exception as e:
            print(f"[ReportBuilder] 摘要生成失败: {e}")
            # 提取报告前 300 字作为摘要
            return report_content[:300] + "..."

    async def _generate_methodology(
        self,
        iterations: List[SearchIteration],
    ) -> str:
        """生成方法论说明"""
        total_sources = sum(it.sources_added for it in iterations)
        total_facts = sum(len(it.new_facts) for it in iterations)
        total_duration = sum(it.duration_seconds for it in iterations)

        prompt = DeepResearchPrompts.GENERATE_METHODOLOGY.format(
            iteration_count=len(iterations),
            source_count=total_sources,
            fact_count=total_facts,
            duration=f"{total_duration/60:.1f} minutes",
        )

        try:
            methodology = await asyncio.to_thread(
                self.llm.chat,
                messages=[
                    {"role": "system", "content": "Generate a brief research methodology statement."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=0.3,
                max_tokens=200,
            )
            return methodology
        except Exception as e:
            print(f"[ReportBuilder] 方法论生成失败: {e}")
            return f"This report is based on {len(iterations)} search iterations, {total_sources} information sources, and {total_facts} extracted key facts."

    def _extract_limitations(self, report_content: str) -> str:
        """提取/生成局限性说明"""
        # 尝试从报告中找到局限性部分
        lower_content = report_content.lower()
        if "局限" in lower_content or "limitation" in lower_content:
            # 报告中已包含局限性说明
            return ""

        # 生成通用局限性说明
        return (
            "This report is based on public web sources and may have the following limitations:\n"
            "1. Timeliness: some information may already be outdated\n"
            "2. Source bias: search results may reflect source-side bias\n"
            "3. Coverage: not every relevant viewpoint may have been captured"
        )

    def _generate_fallback_report(
        self,
        plan: ResearchPlan,
        knowledge_graph: KnowledgeGraph,
    ) -> str:
        """生成备用报告（当 LLM 调用失败时）"""
        sections = [
            f"# {plan.suggested_title or plan.refined_topic}",
            "",
            "## Overview",
            f"This report provides a research analysis of \"{plan.refined_topic}\".",
            "",
            "## Research Goals",
        ]

        for goal in plan.research_goals:
            sections.append(f"- {goal}")

        sections.extend(["", "## Key Findings", ""])

        for subtopic, facts in knowledge_graph.facts_by_subtopic.items():
            if facts:
                sections.append(f"### {subtopic}")
                for fact in facts[:5]:
                    sections.append(f"- {fact.summary or fact.content[:150]}")
                sections.append("")

        sections.extend([
            "## Conclusion",
            f"This research collected {len(knowledge_graph.facts)} relevant facts",
            f"and covered {len(knowledge_graph.facts_by_subtopic)} subtopics.",
        ])

        return "\n".join(sections)
