"""
搜索规划 Agent。

分析搜索任务复杂度，生成搜索计划，优化/扩展查询。
"""

from typing import Any, Dict, List, Optional

from core.config import AgentModelConfig
from core.models import UserProfile
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent


class SearchPlannerAgent(BaseAgent):
    """
    搜索规划 Agent。

    职责:
    1. 分析搜索任务复杂度
    2. 生成搜索计划
    3. 优化/扩展查询
    """

    def __init__(self, config: AgentModelConfig, llm_client: LLMClient):
        """
        初始化搜索规划 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        super().__init__(config, llm_client)

    def run(
        self,
        topic: str,
        purpose: str,
        user_profile: Optional[UserProfile] = None,
        existing_knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        规划搜索策略。

        Args:
            topic: 搜索主题
            purpose: 搜索目的
            user_profile: 用户画像（可选）
            existing_knowledge: 已有知识（可选）

        Returns:
            {
                "mode": "simple" | "comprehensive",
                "queries": [
                    {"query": str, "purpose": str, "priority": int}
                ],
                "overview": str
            }
        """
        profile_text = user_profile.to_prompt_text() if user_profile else "Unknown"

        prompt = Prompts.SEARCH_PLANNING.format(
            topic=topic,
            purpose=purpose or "Gather relevant information",
            user_profile=profile_text,
            existing_knowledge=existing_knowledge or "None",
        )

        result = self._chat_json(
            [
                {"role": "system", "content": "You are a search planning expert who creates efficient search strategies."},
                {"role": "user", "content": prompt},
            ]
        )

        return {
            "mode": result.get("mode", "simple"),
            "queries": result.get("queries", []),
            "overview": result.get("overview", ""),
        }
