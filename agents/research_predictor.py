"""
研究主题预测 Agent。

基于用户画像和近期交互记录，预测用户可能需要的研究主题。
"""

from typing import Any, Dict, List

from core.config import AgentModelConfig
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent


class ResearchTopicPredictor(BaseAgent):
    """
    研究主题预测 Agent。

    职责:
    - 分析用户画像和近期交互
    - 通过场景预测识别用户"正在做某事"可能遇到的问题
    - 通过关联扩展识别用户关注领域的相关话题
    """

    def __init__(self, config: AgentModelConfig, llm_client: LLMClient):
        """
        初始化研究主题预测 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        super().__init__(config, llm_client)

    def run(
        self,
        user_profile_text: str,
        recent_interactions: str,
        recent_topics: List[str],
        research_history: List[str],
    ) -> Dict[str, Any]:
        """
        预测研究主题。

        Args:
            user_profile_text: 用户画像文本
            recent_interactions: 近期交互记录（current_messages + 最近话题摘要）
            recent_topics: 最近讨论的话题列表
            research_history: 历史调研主题列表

        Returns:
            {
                "predictions": [
                    {
                        "topic": str,           # 预测的研究主题
                        "type": str,            # "scenario" | "related"
                        "reason": str,          # 预测理由
                        "confidence": float,    # 置信度 0-1
                        "trigger": str,         # 触发预测的关键信息
                    }
                ]
            }
        """
        # 格式化历史调研主题
        research_history_text = (
            "\n".join([f"- {t}" for t in research_history[-10:]])
            if research_history
            else "No research history"
        )

        # 格式化最近话题
        recent_topics_text = (
            "\n".join([f"- {t}" for t in recent_topics])
            if recent_topics
            else "No recent topics"
        )

        prompt = Prompts.RESEARCH_TOPIC_PREDICTION.format(
            user_profile=user_profile_text,
            recent_interactions=recent_interactions,
            recent_topics=recent_topics_text,
            research_history=research_history_text,
        )

        result = self._chat_json(
            [
                {
                    "role": "system",
                    "content": "You are a research topic prediction expert who infers latent information needs from user behavior.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        return {
            "predictions": result.get("predictions", []),
        }
