"""
对话内需求预测 Agent。

基于当前对话上下文和用户画像，预测用户接下来可能需要的信息。
与 ResearchTopicPredictor 的区别：
- ResearchTopicPredictor: 预测研究主题 → 用于 idle-time web research（后台定时）
- ConversationNeedPredictor: 预测即时信息需求 → 用于对话内主动提供（实时）
"""

from typing import TYPE_CHECKING, Any, Dict, List

from core.config import AgentModelConfig
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from memory.memory_system import MemorySystem


class ConversationNeedPredictor(BaseAgent):
    """
    对话内需求预测 Agent。

    职责:
    - 分析当前对话消息和用户画像
    - 预测用户下一步可能需要的信息
    - 为每个预测提供置信度和检索查询

    设计原则:
    - 只预测高置信度的需求，宁缺毋滥
    - 预测结果交给规则门控（confidence 阈值）决定是否采纳
    - 不负责"是否主动"的决策，只负责"预测什么"
    """

    def __init__(
        self,
        config: AgentModelConfig,
        llm_client: LLMClient,
        memory_system: "MemorySystem",
    ):
        super().__init__(config, llm_client)
        self.memory = memory_system

    def run(
        self,
        current_message: str,
        conversation_history: str,
        user_profile: str,
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        预测用户接下来可能需要的信息。

        Args:
            current_message: 当前用户消息
            conversation_history: 对话历史文本
            user_profile: 用户画像文本
            memory_context: 当前检索到的记忆上下文

        Returns:
            {
                "predicted_needs": [
                    {
                        "need": str,           # 预测的信息需求描述
                        "reason": str,          # 预测理由
                        "confidence": float,    # 0-1 置信度
                        "lookup_query": str,    # 用于检索的查询
                    }
                ]
            }
        """
        prompt = Prompts.CONVERSATION_NEED_PREDICTION.format(
            current_message=current_message,
            conversation_history=conversation_history,
            user_profile=user_profile,
            memory_context=memory_context,
        )

        result = self._chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a need prediction expert. "
                        "Based on conversation context and user profile, "
                        "predict what information the user might need next. "
                        "Only predict high-confidence needs. Do not over-predict."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )

        return {
            "predicted_needs": result.get("predicted_needs", []),
        }
