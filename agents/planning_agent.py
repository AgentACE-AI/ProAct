"""
规划 Agent。

判断记忆是否足以回答用户问题，如果不足则生成搜索查询。
"""

from typing import TYPE_CHECKING, Any, Dict, List

from core.config import AgentModelConfig
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from memory.memory_system import MemorySystem


class PlanningAgent(BaseAgent):
    """
    规划 Agent。

    职责:
    1. 判断记忆是否足以回答用户问题
    2. 如果不足，生成搜索查询
    """

    def __init__(
        self,
        config: AgentModelConfig,
        llm_client: LLMClient,
        memory_system: "MemorySystem",
    ):
        """
        初始化规划 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
            memory_system: 记忆系统实例（用于检索相关记忆）
        """
        super().__init__(config, llm_client)
        self.memory = memory_system

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        规划如何回答用户问题。

        Args:
            user_query: 用户问题

        Returns:
            {
                "memory_sufficient": bool,
                "reason": str,
                "search_queries": List[str],  # 如果不足，需要搜索的查询
                "memory_context": str         # 检索到的记忆上下文
            }
        """
        # 1. 检索相关记忆
        memory_context = self.memory.format_context_for_query(user_query)

        # 2. 使用 LLM 判断记忆是否充足
        prompt = Prompts.PLANNING.format(
            user_query=user_query,
            memory_content=memory_context if memory_context else "There is no relevant information in memory.",
        )

        result = self._chat_json(
            [
                {"role": "system", "content": "You are a planning assistant. Analyze the problem and decide the next action."},
                {"role": "user", "content": prompt},
            ]
        )

        return {
            "memory_sufficient": result.get("memory_sufficient", False),
            "reason": result.get("reason", ""),
            "search_queries": result.get("search_queries", []),
            "memory_context": memory_context,
        }

    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        分析查询的复杂度。

        Args:
            query: 用户查询

        Returns:
            包含复杂度级别和理由的字典
        """
        response = self._chat_json(
            [
                {
                    "role": "system",
                    "content": "You analyze query complexity. Return JSON: {'complexity': 'low/medium/high', 'reason': 'reason'}",
                },
                {"role": "user", "content": f"Analyze the complexity of this query:\n\n{query}"},
            ]
        )

        return {
            "complexity": response.get("complexity", "medium"),
            "reason": response.get("reason", ""),
        }
