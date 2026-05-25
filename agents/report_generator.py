"""
报告生成 Agent。

基于搜索结果生成调研报告。
"""

from typing import List, Optional

from core.config import AgentModelConfig
from core.models import SearchSource, UserProfile
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent


class ReportGeneratorAgent(BaseAgent):
    """
    报告生成 Agent。

    职责:
    - 基于搜索结果生成结构化的调研报告
    """

    def __init__(self, config: AgentModelConfig, llm_client: LLMClient):
        """
        初始化报告生成 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        super().__init__(config, llm_client)

    def run(
        self,
        topic: str,
        sources: List[SearchSource],
        user_profile: Optional[UserProfile] = None,
    ) -> str:
        """
        生成调研报告。

        Args:
            topic: 报告主题
            sources: 搜索来源列表
            user_profile: 用户画像（可选，用于个性化）

        Returns:
            报告内容 (Markdown 格式)
        """
        # 格式化搜索结果
        results_text = self._format_sources(sources)
        profile_text = user_profile.to_prompt_text() if user_profile else "Unknown"

        prompt = Prompts.REPORT_GENERATION.format(
            topic=topic,
            search_results=results_text if results_text else "No search results available.",
            user_profile=profile_text,
        )

        report = self._chat(
            [
                {
                    "role": "system",
                    "content": "You are a professional research report writer. Generate a well-structured, substantive research report in Markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        return report.strip()

    def _format_sources(self, sources: List[SearchSource]) -> str:
        """
        格式化搜索来源用于 prompt。

        Args:
            sources: 搜索来源列表

        Returns:
            格式化的来源文本
        """
        if not sources:
            return ""

        results_text = ""
        for i, source in enumerate(sources, 1):
            results_text += f"\n## Source {i}: {source.title}\n"
            results_text += f"URL: {source.url}\n\n"

            if source.full_content:
                # 截断过长的内容
                content = source.full_content
                if len(content) > 3000:
                    content = content[:3000] + "..."
                results_text += f"Content:\n{content}\n\n"
            else:
                results_text += f"Summary: {source.snippet}\n\n"

        return results_text
