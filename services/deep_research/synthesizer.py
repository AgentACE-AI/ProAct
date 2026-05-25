"""
知识综合器。

负责整合事实、识别冲突、计算覆盖度。
"""

import asyncio
from typing import Any, Dict, List, TYPE_CHECKING

from core.config import AgentModelConfig
from services.deep_research.models import (
    ExtractedFact,
    FactConflict,
    KnowledgeGraph,
    ResearchPlan,
    SearchIteration,
)
from services.deep_research.prompts import DeepResearchPrompts

if TYPE_CHECKING:
    from tools.llm_client import LLMClient


class KnowledgeSynthesizer:
    """
    知识综合器。

    职责:
    1. 合并相似事实
    2. 识别矛盾/冲突
    3. 按主题分类
    4. 计算覆盖度
    """

    def __init__(
        self,
        config: AgentModelConfig,
        llm_client: "LLMClient",
    ):
        """
        初始化知识综合器。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        self.config = config
        self.llm = llm_client

    async def synthesize(
        self,
        iterations: List[SearchIteration],
        plan: ResearchPlan,
    ) -> KnowledgeGraph:
        """
        综合所有搜索迭代的知识。

        Args:
            iterations: 搜索迭代列表
            plan: 研究计划

        Returns:
            知识图谱
        """
        # 收集所有事实
        all_facts = []
        for iteration in iterations:
            all_facts.extend(iteration.new_facts)

        if not all_facts:
            return KnowledgeGraph()

        # 按子主题分类
        facts_by_subtopic = self._categorize_facts(all_facts, plan)

        # 识别冲突
        conflicts = await self._identify_conflicts(all_facts, plan.refined_topic)

        # 计算覆盖度
        coverage = await self._calculate_coverage(
            topic=plan.refined_topic,
            subtopics=[st.name for st in plan.subtopics],
            facts_by_subtopic=facts_by_subtopic,
        )

        return KnowledgeGraph(
            facts=all_facts,
            facts_by_subtopic=facts_by_subtopic,
            conflicts=conflicts,
            coverage_by_subtopic=coverage.get("coverage_by_subtopic", {}),
            total_coverage=coverage.get("total_coverage", 0.0),
        )

    async def synthesize_from_facts(
        self,
        facts: List[Dict[str, Any]],
        plan: ResearchPlan,
    ) -> KnowledgeGraph:
        """
        从事实列表综合知识（用于增量搜索）。

        Args:
            facts: 事实列表（字典格式）
            plan: 研究计划

        Returns:
            知识图谱
        """
        # 转换为 ExtractedFact 对象
        extracted_facts = []
        for f in facts:
            if isinstance(f, ExtractedFact):
                extracted_facts.append(f)
            elif isinstance(f, dict):
                extracted_facts.append(ExtractedFact(
                    content=f.get("content", f.get("summary_preview", "")),
                    source_url=f.get("source_url", ""),
                    source_title=f.get("title", ""),
                    subtopic=f.get("topic", f.get("subtopic", "")),
                    confidence=f.get("fact_confidence", f.get("confidence", 0.8)),
                    summary=f.get("summary_preview", f.get("summary", "")),
                ))

        # 按子主题分类
        facts_by_subtopic = self._categorize_facts(extracted_facts, plan)

        # 识别冲突
        conflicts = await self._identify_conflicts(
            extracted_facts, plan.refined_topic
        )

        # 计算覆盖度
        coverage = await self._calculate_coverage(
            topic=plan.refined_topic,
            subtopics=[st.name for st in plan.subtopics],
            facts_by_subtopic=facts_by_subtopic,
        )

        return KnowledgeGraph(
            facts=extracted_facts,
            facts_by_subtopic=facts_by_subtopic,
            conflicts=conflicts,
            coverage_by_subtopic=coverage.get("coverage_by_subtopic", {}),
            total_coverage=coverage.get("total_coverage", 0.0),
        )

    def _categorize_facts(
        self,
        facts: List[ExtractedFact],
        plan: ResearchPlan,
    ) -> Dict[str, List[ExtractedFact]]:
        """按子主题分类事实"""
        result: Dict[str, List[ExtractedFact]] = {}

        subtopic_names = [st.name for st in plan.subtopics]

        for fact in facts:
            # 使用事实自带的子主题，或尝试匹配
            subtopic = fact.subtopic
            if not subtopic or subtopic not in subtopic_names:
                # 尝试通过内容匹配最相关的子主题
                subtopic = self._match_subtopic(fact.content, subtopic_names)

            if subtopic not in result:
                result[subtopic] = []
            result[subtopic].append(fact)

        return result

    def _match_subtopic(
        self,
        content: str,
        subtopic_names: List[str],
    ) -> str:
        """匹配最相关的子主题"""
        content_lower = content.lower()

        best_match = "其他"
        best_score = 0

        for name in subtopic_names:
            # 简单的关键词匹配
            keywords = name.lower().split()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > best_score:
                best_score = score
                best_match = name

        return best_match

    async def _identify_conflicts(
        self,
        facts: List[ExtractedFact],
        topic: str,
    ) -> List[FactConflict]:
        """识别事实之间的冲突"""
        if len(facts) < 2:
            return []

        # 只分析前 30 个事实（避免过长）
        facts_to_analyze = facts[:30]

        facts_text = "\n".join([
            f"[{i+1}] ({f.source_title}) {f.content[:200]}"
            for i, f in enumerate(facts_to_analyze)
        ])

        prompt = DeepResearchPrompts.IDENTIFY_CONFLICTS.format(
            topic=topic,
            facts=facts_text,
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat_json,
                messages=[
                    {"role": "system", "content": "You are an information analysis expert. Identify contradictions among facts and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=0.3,
            )

            conflicts = []
            for c in response.get("conflicts", []):
                conflicts.append(FactConflict(
                    fact1=c.get("fact1", ""),
                    fact2=c.get("fact2", ""),
                    source1=c.get("source1", ""),
                    source2=c.get("source2", ""),
                    resolution=c.get("resolution", ""),
                ))
            return conflicts

        except Exception as e:
            print(f"[KnowledgeSynthesizer] 冲突识别失败: {e}")
            return []

    async def _calculate_coverage(
        self,
        topic: str,
        subtopics: List[str],
        facts_by_subtopic: Dict[str, List[ExtractedFact]],
    ) -> Dict[str, Any]:
        """计算覆盖度"""
        # 格式化事实分布
        facts_distribution = {
            st: len(facts_by_subtopic.get(st, []))
            for st in subtopics
        }

        prompt = DeepResearchPrompts.CALCULATE_COVERAGE.format(
            topic=topic,
            planned_subtopics="\n".join(f"- {st}" for st in subtopics),
            facts_by_subtopic="\n".join(
                f"- {st}: {count} facts"
                for st, count in facts_distribution.items()
            ),
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat_json,
                messages=[
                    {"role": "system", "content": "You are a research analysis expert. Assess research coverage and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=0.3,
            )
            return response
        except Exception as e:
            print(f"[KnowledgeSynthesizer] 覆盖度计算失败: {e}")
            # 简单计算
            total_facts = sum(facts_distribution.values())
            coverage_by_subtopic = {}
            for st in subtopics:
                count = facts_distribution.get(st, 0)
                coverage_by_subtopic[st] = min(1.0, count / 5)  # 假设5条事实为满覆盖

            avg_coverage = sum(coverage_by_subtopic.values()) / len(subtopics) if subtopics else 0

            return {
                "coverage_by_subtopic": coverage_by_subtopic,
                "total_coverage": avg_coverage,
                "gaps": [st for st, c in coverage_by_subtopic.items() if c < 0.3],
            }
