"""
迭代搜索器。

负责执行搜索、评估来源质量、提取事实、发现新方向。
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.config import AgentModelConfig
from services.deep_research.models import (
    ExecutedQuery,
    ExtractedFact,
    ResearchConfig,
    SearchIteration,
    SourceQuality,
    SourceResult,
)
from services.deep_research.prompts import DeepResearchPrompts

if TYPE_CHECKING:
    from tools.llm_client import LLMClient
    from tools.search_tool import SearchTool
    from tools.web_fetcher import WebFetcher


class IterativeSearcher:
    """
    迭代搜索器。

    职责:
    1. 执行搜索查询
    2. 抓取网页内容
    3. 评估来源质量
    4. 提取事实
    5. 发现新问题/子主题
    6. 去重
    """

    def __init__(
        self,
        config: AgentModelConfig,
        llm_client: "LLMClient",
        search_tool: "SearchTool",
        web_fetcher: "WebFetcher",
    ):
        """
        初始化迭代搜索器。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
            search_tool: 搜索工具
            web_fetcher: 网页抓取工具
        """
        self.config = config
        self.llm = llm_client
        self.search_tool = search_tool
        self.web_fetcher = web_fetcher

    async def execute_iteration(
        self,
        iteration_number: int,
        queries: List[Dict[str, Any]],
        existing_sources: List[SourceResult],
        research_config: ResearchConfig,
        topic: str,
        subtopic: str = "",
    ) -> SearchIteration:
        """
        执行单次搜索迭代。

        Args:
            iteration_number: 迭代编号
            queries: 要执行的查询列表，每项包含 query, purpose, priority
            existing_sources: 已有来源（用于去重）
            research_config: 研究配置
            topic: 调研主题
            subtopic: 当前子主题

        Returns:
            搜索迭代结果
        """
        start_time = time.time()
        existing_urls = {s.url for s in existing_sources}

        iteration = SearchIteration(
            iteration_number=iteration_number,
        )

        # 执行每个查询
        for query_data in queries:
            query = query_data.get("query", "")
            purpose = query_data.get("purpose", "")
            priority = query_data.get("priority", 5)

            if not query:
                continue

            executed_query = ExecutedQuery(
                query=query,
                purpose=purpose,
                priority=priority,
            )

            try:
                # 执行搜索（使用线程池避免阻塞事件循环）
                search_results = await asyncio.to_thread(
                    self.search_tool.search,
                    query=query,
                    num_results=research_config.max_sources_per_iteration,
                )

                if not search_results:
                    executed_query.results_count = 0
                    iteration.queries_executed.append(executed_query)
                    continue

                executed_query.results_count = len(search_results)

                # 处理每个搜索结果
                for result in search_results:
                    # print("haha")
                    # url = result.get("link", result.get("url", ""))
                    # # print("haha2")
                    # title = result.get("title", "")
                    # snippet = result.get("snippet", "")
                    url = result.url
                    title = result.title
                    snippet = result.snippet

                    # 去重
                    if url in existing_urls:
                        continue

                    # 创建来源结果
                    source = SourceResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                    )

                    # 尝试抓取内容
                    try:
                        content = await self._fetch_content(url)
                        if content:
                            source.full_content = content

                            # 评估来源质量
                            quality = await self._assess_quality(
                                query=query,
                                title=title,
                                url=url,
                                snippet=snippet,
                                content_preview=content[:1000],
                            )
                            source.quality = quality

                            # 如果质量达标，提取事实
                            if quality.overall >= research_config.min_source_quality:
                                extraction_result = await self._extract_facts(
                                    topic=topic,
                                    subtopic=subtopic or purpose,
                                    title=title,
                                    url=url,
                                    content=content[:5000],  # 限制内容长度
                                )

                                facts = extraction_result.get("facts", [])
                                publish_date = extraction_result.get("publish_date", "")

                                source.extracted_facts = [f["content"] for f in facts]

                                for fact_data in facts:
                                    fact = ExtractedFact(
                                        content=fact_data.get("content", ""),
                                        source_url=url,
                                        source_title=title,
                                        subtopic=subtopic or purpose,
                                        confidence=fact_data.get("confidence", 0.8),
                                        summary=fact_data.get("summary", ""),
                                        publish_date=publish_date,
                                    )
                                    iteration.new_facts.append(fact)

                                iteration.sources_added += 1
                                existing_urls.add(url)

                    except Exception as e:
                        print(f"[IterativeSearcher] 抓取/处理 {url} 失败: {e}")

                    iteration.sources_found.append(source)

            except Exception as e:
                print(f"[IterativeSearcher] 执行查询 '{query}' 失败: {e}")
                executed_query.success = False
                executed_query.error = str(e)

            iteration.queries_executed.append(executed_query)

        # 发现新方向
        if iteration.new_facts:
            new_directions = await self._discover_new_directions(
                topic=topic,
                covered_subtopics=[subtopic] if subtopic else [],
                recent_facts=iteration.new_facts[-10:],
            )
            iteration.new_questions = new_directions.get("new_questions", [])
            iteration.discovered_subtopics = new_directions.get("new_subtopics", [])

        iteration.duration_seconds = time.time() - start_time
        return iteration

    async def _fetch_content(self, url: str) -> str:
        """抓取网页内容（使用线程池避免阻塞事件循环）"""
        try:
            content = await asyncio.to_thread(self.web_fetcher.fetch, url)
            return content or ""
        except Exception as e:
            print(f"[IterativeSearcher] 抓取失败: {url}, {e}")
            return ""

    async def _assess_quality(
        self,
        query: str,
        title: str,
        url: str,
        snippet: str,
        content_preview: str,
    ) -> SourceQuality:
        """评估来源质量（使用线程池避免阻塞事件循环）"""
        prompt = DeepResearchPrompts.SOURCE_QUALITY_ASSESSMENT.format(
            query=query,
            title=title,
            url=url,
            snippet=snippet,
            content_preview=content_preview,
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat_json,
                messages=[
                    {"role": "system", "content": "You are an information quality expert. Assess source quality and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=0.3,
                max_tokens=500,
            )

            relevance = response.get("relevance", 0.5)
            credibility = response.get("credibility", 0.5)
            freshness = response.get("freshness", 0.5)

            return SourceQuality(
                relevance=relevance,
                credibility=credibility,
                freshness=freshness,
                overall=(relevance * 0.5 + credibility * 0.3 + freshness * 0.2),
            )
        except Exception as e:
            print(f"[IterativeSearcher] 质量评估失败: {e}")
            return SourceQuality(
                relevance=0.5,
                credibility=0.5,
                freshness=0.5,
                overall=0.5,
            )

    async def _extract_facts(
        self,
        topic: str,
        subtopic: str,
        title: str,
        url: str,
        content: str,
    ) -> Dict[str, Any]:
        """从内容中提取事实（使用线程池避免阻塞事件循环）"""
        prompt = DeepResearchPrompts.FACT_EXTRACTION.format(
            topic=topic,
            subtopic=subtopic,
            title=title,
            url=url,
            content=content,
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat_json,
                messages=[
                    {"role": "system", "content": "You are an information extraction expert. Extract facts from the content and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=0.3,
            )
            return {
                "facts": response.get("facts", []),
                "publish_date": response.get("publish_date", ""),
            }
        except Exception as e:
            print(f"[IterativeSearcher] 事实提取失败: {e}")
            return {"facts": [], "publish_date": ""}

    async def _discover_new_directions(
        self,
        topic: str,
        covered_subtopics: List[str],
        recent_facts: List[ExtractedFact],
    ) -> Dict[str, Any]:
        """发现新的研究方向（使用线程池避免阻塞事件循环）"""
        facts_text = "\n".join([
            f"- {f.summary or f.content[:100]}"
            for f in recent_facts
        ])

        prompt = DeepResearchPrompts.DISCOVER_NEW_DIRECTIONS.format(
            topic=topic,
            covered_subtopics=", ".join(covered_subtopics) if covered_subtopics else "None",
            recent_facts=facts_text,
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat_json,
                messages=[
                    {"role": "system", "content": "You are a research analysis expert. Discover new research directions and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.model,
                temperature=0.5,
            )
            return response
        except Exception as e:
            print(f"[IterativeSearcher] 方向发现失败: {e}")
            return {
                "new_questions": [],
                "new_subtopics": [],
                "suggested_queries": [],
                "should_continue": True,
            }

    def check_convergence(
        self,
        iterations: List[SearchIteration],
        config: ResearchConfig,
    ) -> bool:
        """
        检查是否应该停止搜索。

        Args:
            iterations: 已完成的迭代列表
            config: 研究配置

        Returns:
            是否应该停止
        """
        if len(iterations) >= config.max_iterations:
            return True

        # 计算总来源数
        total_sources = sum(it.sources_added for it in iterations)
        if total_sources >= config.max_total_sources:
            return True

        # 检查最近几次迭代的收益
        if len(iterations) >= 3:
            recent = iterations[-3:]
            recent_added = sum(it.sources_added for it in recent)
            if recent_added < 3:  # 最近3次迭代只添加了不到3个来源
                return True

        return False
