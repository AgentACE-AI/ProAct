"""
研究规划器。

负责分析主题、分解子主题、生成搜索计划。
"""

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.config import AgentModelConfig
from core.models import UserProfile
from services.deep_research.models import (
    ResearchConfig,
    ResearchOutline,
    ResearchPlan,
    SubTopic,
)
from services.deep_research.prompts import DeepResearchPrompts

if TYPE_CHECKING:
    from tools.llm_client import LLMClient


class ResearchPlanner:
    """
    研究规划器。

    职责:
    1. 分析主题复杂度和领域
    2. 分解为子主题
    3. 生成初始搜索查询
    4. 创建报告大纲
    """

    def __init__(
        self,
        config: AgentModelConfig,
        llm_client: "LLMClient",
    ):
        """
        初始化研究规划器。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        self.config = config
        self.llm = llm_client

    async def create_plan(
        self,
        topic: str,
        user_profile: UserProfile,
        config: ResearchConfig,
        existing_knowledge: str = "",
    ) -> ResearchPlan:
        """
        创建研究计划。

        Args:
            topic: 调研主题
            user_profile: 用户画像
            config: 研究配置
            existing_knowledge: 已有知识（用于增量搜索）

        Returns:
            研究计划
        """
        # 1. 分析主题
        analysis = await self._analyze_topic(
            topic=topic,
            user_profile=user_profile,
            existing_knowledge=existing_knowledge,
        )

        # 2. 生成子主题和查询
        subtopics_data = await self._generate_subtopics(
            topic=analysis.get("refined_topic", topic),
            research_goals=analysis.get("research_goals", []),
            user_interests=user_profile.interests,
            max_subtopics=config.max_subtopics,
        )

        # 3. 构建研究计划
        subtopics = [
            SubTopic(
                name=st.get("name", ""),
                priority=st.get("priority", 5),
                queries=st.get("queries", []),
            )
            for st in subtopics_data.get("subtopics", [])
        ]

        outline_data = subtopics_data.get("outline", {})
        outline = ResearchOutline(
            sections=outline_data.get("sections", []),
            key_questions=analysis.get("key_questions", []),
        )

        # 4. 收集初始查询
        initial_queries = []
        for st in subtopics[:3]:  # 优先处理前3个子主题
            for query in st.queries[:2]:
                initial_queries.append({
                    "query": query,
                    "purpose": f"调研子主题: {st.name}",
                    "priority": st.priority,
                })

        return ResearchPlan(
            original_topic=topic,
            refined_topic=analysis.get("refined_topic", topic),
            research_goals=analysis.get("research_goals", []),
            subtopics=subtopics,
            initial_queries=initial_queries,
            outline=outline,
            suggested_title=analysis.get("suggested_title", topic),
            complexity=analysis.get("complexity", "medium"),
            domain=analysis.get("domain", ""),
        )

    async def _analyze_topic(
        self,
        topic: str,
        user_profile: UserProfile,
        existing_knowledge: str = "",
    ) -> Dict[str, Any]:
        """分析主题（使用线程池避免阻塞事件循环）"""
        prompt = DeepResearchPrompts.TOPIC_ANALYSIS.format(
            topic=topic,
            user_profile=user_profile.to_prompt_text(),
            existing_knowledge=existing_knowledge or "None",
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat_json,
                messages=[
                    {"role": "system", "content": "You are a research planning expert. Analyze the research topic and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=self.config.temperature,
            )
            return response
        except Exception as e:
            print(f"[ResearchPlanner] 主题分析失败: {e}")
            return {
                "refined_topic": topic,
                "complexity": "medium",
                "domain": "",
                "research_goals": [f"Build a comprehensive understanding of {topic}"],
                "key_questions": [f"What is {topic}?", f"What are the main characteristics of {topic}?"],
                "suggested_title": f"{topic} Research Report",
            }

    async def _generate_subtopics(
        self,
        topic: str,
        research_goals: List[str],
        user_interests: List[str],
        max_subtopics: int = 6,
    ) -> Dict[str, Any]:
        """生成子主题（使用线程池避免阻塞事件循环）"""
        prompt = DeepResearchPrompts.SUBTOPIC_GENERATION.format(
            topic=topic,
            research_goals="\n".join(f"- {g}" for g in research_goals),
            user_interests=", ".join(user_interests) if user_interests else "No special user interests",
            max_subtopics=max_subtopics,
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat_json,
                messages=[
                    {"role": "system", "content": "You are a research planning expert. Break the topic into subtopics and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=self.config.temperature,
            )
            return response
        except Exception as e:
            print(f"[ResearchPlanner] 子主题生成失败: {e}")
            # 返回默认子主题
            return {
                "subtopics": [
                    {
                        "name": f"{topic} Overview",
                        "priority": 9,
                        "queries": [f"What is {topic}", f"{topic} overview"],
                    },
                    {
                        "name": f"{topic} Key Characteristics",
                        "priority": 8,
                        "queries": [f"{topic} characteristics", f"{topic} advantages"],
                    },
                    {
                        "name": f"{topic} Use Cases",
                        "priority": 7,
                        "queries": [f"{topic} applications", f"{topic} case studies"],
                    },
                ],
                "outline": {
                    "sections": [
                        "Overview",
                        "Key Characteristics",
                        "Use Cases",
                        "Summary",
                    ],
                },
            }

    async def analyze_incremental_gaps(
        self,
        new_topic: str,
        existing_topic: str,
        similarity: float,
        existing_facts: List[Dict[str, Any]],
        target_subtopics: List[str],
    ) -> Dict[str, Any]:
        """
        分析增量搜索的缺口。

        Args:
            new_topic: 新调研主题
            existing_topic: 已有相关调研主题
            similarity: 相似度
            existing_facts: 已有事实列表
            target_subtopics: 目标子主题

        Returns:
            缺口分析结果
        """
        # 格式化已有事实
        facts_text = "\n".join([
            f"- {f.get('summary_preview', f.get('content', '')[:100])}"
            for f in existing_facts[:20]
        ])

        prompt = DeepResearchPrompts.INCREMENTAL_GAP_ANALYSIS.format(
            new_topic=new_topic,
            existing_topic=existing_topic,
            similarity=f"{similarity:.2f}",
            fact_count=len(existing_facts),
            existing_facts=facts_text,
            target_subtopics="\n".join(f"- {st}" for st in target_subtopics),
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat_json,
                messages=[
                    {"role": "system", "content": "You are a research analysis expert. Analyze the research gaps and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=self.config.temperature,
            )
            return response
        except Exception as e:
            print(f"[ResearchPlanner] 缺口分析失败: {e}")
            return {
                "reusable_facts": [],
                "covered_subtopics": [],
                "missing_subtopics": target_subtopics,
                "supplementary_queries": [
                    {"query": new_topic, "purpose": "Primary research", "priority": 8}
                ],
                "estimated_coverage": similarity * 0.5,
            }
