"""
回复 Agent。

结合用户画像和上下文生成个性化回复。
"""

from typing import TYPE_CHECKING, Optional

from core.config import AgentModelConfig
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from memory.memory_system import MemorySystem


class AssistantAgent(BaseAgent):
    """
    回复 Agent。

    职责:
    - 结合用户画像和上下文生成个性化回复
    """

    def __init__(
        self,
        config: AgentModelConfig,
        llm_client: LLMClient,
        memory_system: "MemorySystem",
    ):
        """
        初始化回复 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
            memory_system: 记忆系统实例
        """
        super().__init__(config, llm_client)
        self.memory = memory_system

    def run(
        self,
        user_query: str,
        memory_context: str,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        生成回复。

        Args:
            user_query: 用户问题
            memory_context: 记忆上下文
            additional_context: 额外上下文 (如搜索结果)

        Returns:
            回复文本
        """
        # 获取用户画像
        user_profile = self.memory.get_user_profile()
        profile_text = user_profile.to_prompt_text()

        # 组合记忆内容
        memories_text = memory_context
        if additional_context:
            memories_text += f"\n\nAdditional reference information:\n{additional_context}"

        prompt = Prompts.ASSISTANT.format(
            user_profile=profile_text,
            relevant_memories=memories_text if memories_text else "No relevant memories.",
            user_query=user_query,
        )

        response = self._chat(
            [
                {"role": "system", "content": "You are a helpful personalized AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        return response.strip()
