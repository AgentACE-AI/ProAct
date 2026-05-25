"""
用户模拟器 Agent。

根据用户画像模拟用户行为，用于测试和验证系统功能。
"""

from typing import Any, Dict, List, Optional

from core.config import AgentModelConfig
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent


class UserSimulatorAgent(BaseAgent):
    """
    用户模拟器 Agent。

    根据用户画像模拟用户行为，生成符合用户特征的消息。
    主要用于：
    - 系统功能测试
    - 对话流程验证
    - 用户画像效果评估
    """

    def __init__(
        self,
        config: AgentModelConfig,
        llm_client: LLMClient,
    ):
        """
        初始化用户模拟器 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        super().__init__(
            config=config,
            llm_client=llm_client,
        )

    def run(
        self,
        user_profile: str,
        conversation_history: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> str:
        """
        生成模拟用户消息。

        Args:
            user_profile: 用户画像文本
            conversation_history: 对话历史
            context: 可选的额外上下文

        Returns:
            模拟的用户消息
        """
        history_text = self._format_history(conversation_history)

        prompt = Prompts.USER_SIMULATOR.format(
            user_profile=user_profile,
            conversation_history=history_text if history_text else "This is the start of the conversation."
        )

        if context:
            prompt += f"\n\nAdditional context: {context}"

        response = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a user simulator. Generate realistic user messages from the given user profile."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # 较高温度以产生更多样化的响应
        )

        return response.strip()

    def run_with_topic_hint(
        self,
        user_profile: str,
        conversation_history: List[Dict[str, str]],
        topic_hint: str,
    ) -> str:
        """
        根据话题提示生成模拟用户消息。

        Args:
            user_profile: 用户画像文本
            conversation_history: 对话历史
            topic_hint: 话题提示

        Returns:
            模拟的用户消息
        """
        context = f"The user wants to discuss: {topic_hint}"
        return self.run(user_profile, conversation_history, context)

    def _format_history(self, history: List[Dict[str, str]], limit: int = 10) -> str:
        """
        格式化对话历史。

        Args:
            history: 对话历史列表
            limit: 最大消息数量

        Returns:
            格式化后的对话历史文本
        """
        if not history:
            return ""

        recent = history[-limit:] if len(history) > limit else history
        lines = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            role_name = "User" if role == "user" else "Assistant" if role == "assistant" else role
            lines.append(f"{role_name}: {content}")

        return "\n".join(lines)

    @staticmethod
    def format_simulator_profile(profile_data: Dict[str, Any]) -> str:
        """
        将模拟器画像数据格式化为文本。

        Args:
            profile_data: 模拟器画像数据字典

        Returns:
            格式化后的画像文本
        """
        lines = []

        # 核心画像
        core = profile_data.get("core_profile", {})
        if core:
            lines.append(f"Role: {core.get('role', 'Unknown')}")
            if core.get("inferred_age"):
                lines.append(f"Age: {core.get('inferred_age')}")
            if core.get("location"):
                lines.append(f"Location: {core.get('location')}")
            if core.get("core_tags"):
                lines.append(f"Tags: {', '.join(core.get('core_tags', []))}")

        # 性格特点
        traits = profile_data.get("personality_and_traits", [])
        if traits:
            lines.append(f"\nTraits: {', '.join(traits)}")

        # 兴趣爱好
        interests = profile_data.get("interests_and_focus", [])
        if interests:
            lines.append(f"Interests: {', '.join(interests)}")

        # 目标动机
        goals = profile_data.get("goals_and_motivations", [])
        if goals:
            lines.append(f"Goals and motivations: {', '.join(goals)}")

        # 痛点挑战
        challenges = profile_data.get("pain_points_and_challenges", [])
        if challenges:
            lines.append(f"Challenges: {', '.join(challenges)}")

        # 沟通风格
        comm_style = profile_data.get("communication_style", [])
        if comm_style:
            lines.append(f"Communication style: {', '.join(comm_style)}")

        return "\n".join(lines)
