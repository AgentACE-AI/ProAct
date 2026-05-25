"""
记忆批判 Agent。

检测知识缺口、逻辑问题，生成建议的搜索查询。
"""

from typing import Any, Dict, List

from core.config import AgentModelConfig
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent


class MemoryCriticAgent(BaseAgent):
    """
    记忆批判 Agent。

    职责:
    1. 检测知识缺口
    2. 检测逻辑问题
    3. 生成建议的搜索查询
    """

    def __init__(self, config: AgentModelConfig, llm_client: LLMClient):
        """
        初始化记忆批判 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        super().__init__(config, llm_client)

    def run(self, topic: str, memory_content: str) -> Dict[str, Any]:
        """
        验证记忆内容。

        Args:
            topic: 要分析的话题
            memory_content: 要验证的记忆内容

        Returns:
            {
                "status": "PASS" | "HAS_GAPS" | "HAS_ISSUES",
                "confidence_score": int,  # 0-100
                "knowledge_gaps": [
                    {"type": str, "description": str, "search_query": str}
                ],
                "logical_issues": [
                    {"type": str, "description": str, "search_query": str}
                ],
                "summary": str
            }
        """
        prompt = Prompts.MEMORY_CRITIC.format(
            topic=topic,
            memory_content=memory_content,
        )

        result = self._chat_json(
            [
                {"role": "system", "content": "You are a knowledge validation expert who identifies information gaps and logical weaknesses."},
                {"role": "user", "content": prompt},
            ]
        )

        return {
            "status": result.get("status", "PASS"),
            "confidence_score": result.get("confidence_score", 100),
            "knowledge_gaps": result.get("knowledge_gaps", []),
            "logical_issues": result.get("logical_issues", []),
            "summary": result.get("summary", ""),
        }
