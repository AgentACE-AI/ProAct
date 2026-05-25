"""
推送评估 Agent。

评估信息价值和打断成本。
"""

from typing import Any, Dict, List

from core.config import AgentModelConfig
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent


class PushScoreAgent(BaseAgent):
    """
    推送评估 Agent。

    职责:
    - 评估信息价值 (value)
    - 评估打断成本 (cost)
    """

    def __init__(self, config: AgentModelConfig, llm_client: LLMClient):
        """
        初始化推送评估 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        super().__init__(config, llm_client)

    def run(
        self,
        report_topic: str,
        report_summary: str,
        user_interests: List[str],
        current_context: str,
        seconds_since_last_interaction: int,
        recent_interaction_count: int,
    ) -> Dict[str, Any]:
        """
        评估推送。

        Args:
            report_topic: 报告主题
            report_summary: 报告摘要
            user_interests: 用户兴趣列表
            current_context: 当前对话上下文
            seconds_since_last_interaction: 距离上次交互的秒数
            recent_interaction_count: 最近5分钟的交互次数

        Returns:
            {
                "value": int,    # 0-100
                "cost": int,     # 0-100
                "reason": str
            }
        """
        prompt = Prompts.PUSH_SCORE.format(
            report_topic=report_topic,
            report_summary=report_summary,
            user_interests=", ".join(user_interests) if user_interests else "Unknown",
            current_context=current_context[:500] if current_context else "None",
            seconds_since_last_interaction=seconds_since_last_interaction,
            recent_interaction_count=recent_interaction_count,
        )

        result = self._chat_json(
            [
                {"role": "system", "content": "You are a push-value evaluator responsible for scoring informational value and interruption cost."},
                {"role": "user", "content": prompt},
            ]
        )

        return {
            "value": result.get("value", 50),
            "cost": result.get("cost", 50),
            "reason": result.get("reason", ""),
        }
