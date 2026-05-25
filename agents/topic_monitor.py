"""
话题检测 Agent。

检测用户消息是否切换了话题，是否返回之前讨论过的话题。
"""

from typing import Any, Dict, List, Optional

from core.config import AgentModelConfig
from core.models import Message
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent


class TopicMonitorAgent(BaseAgent):
    """
    话题检测 Agent。

    职责:
    1. 检测用户消息是否切换了话题
    2. 检测是否返回之前讨论过的话题
    3. 推断新话题名称
    """

    def __init__(self, config: AgentModelConfig, llm_client: LLMClient):
        """
        初始化话题检测 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        super().__init__(config, llm_client)

    def run(
        self,
        current_topic: Optional[str],
        conversation_history: List[Message],
        new_message: str,
        existing_topics: List[str],
    ) -> Dict[str, Any]:
        """
        检测话题变化。

        Args:
            current_topic: 当前话题 (可为 None)
            conversation_history: 当前对话历史
            new_message: 新用户消息
            existing_topics: 已存在的所有话题列表

        Returns:
            {
                "is_same_topic": bool,
                "is_returning_topic": bool,  # 是否返回之前的话题
                "returning_topic_name": str, # 返回的话题名 (如果 is_returning_topic)
                "new_topic": str,            # 新话题名 (如果 is_same_topic=False)
                "reason": str
            }
        """
        if not current_topic:
            # 没有当前话题，从消息推断话题
            new_topic = self._infer_topic(new_message)

            # 检查是否匹配已有话题
            if existing_topics:
                match = self._find_matching_topic(new_message, existing_topics)
                if match:
                    return {
                        "is_same_topic": False,
                        "is_returning_topic": True,
                        "returning_topic_name": match,
                        "new_topic": None,
                        "reason": f"返回之前讨论过的话题: {match}",
                    }

            return {
                "is_same_topic": False,
                "is_returning_topic": False,
                "returning_topic_name": None,
                "new_topic": new_topic,
                "reason": "没有先前话题",
            }

        # 格式化输入
        history_text = self._format_history(conversation_history)
        existing_topics_text = self._format_existing_topics(existing_topics, current_topic)

        prompt = Prompts.TOPIC_MONITOR.format(
            current_topic=current_topic,
            conversation_history=history_text,
            existing_topics=existing_topics_text,
            new_message=new_message,
        )

        result = self._chat_json(
            [
                {
                    "role": "system",
                    "content": "You are a topic monitoring assistant. Analyze the conversation, detect topic changes, and identify when the user returns to a previously discussed topic.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        # 处理结果
        is_same = result.get("is_same_topic", True)
        is_returning = result.get("is_returning_topic", False)
        returning_name = result.get("returning_topic_name")
        new_topic = result.get("new_topic")

        # 验证返回的话题名是否在已有话题中
        if is_returning and returning_name and existing_topics:
            matched = self._find_best_match(returning_name, existing_topics)
            if matched:
                returning_name = matched

        return {
            "is_same_topic": is_same,
            "is_returning_topic": is_returning and not is_same,
            "returning_topic_name": returning_name if (is_returning and not is_same) else None,
            "new_topic": new_topic if (not is_same and not is_returning) else None,
            "reason": result.get("reason", ""),
        }

    def _infer_topic(self, message: str) -> str:
        """
        从单条消息推断话题。

        Args:
            message: 用户消息

        Returns:
            推断出的话题名称
        """
        response = self._chat(
            [
                {
                    "role": "system",
                    "content": "Extract the main topic from the text. Reply with only a short topic name (2-5 words) in the same language as the message.",
                },
                {"role": "user", "content": f"What is the main topic of this message?\n\n{message}"},
            ],
            temperature=0.3,
        )
        return response.strip()

    def _find_matching_topic(self, message: str, existing_topics: List[str]) -> Optional[str]:
        """
        查找消息是否匹配已有话题。

        Args:
            message: 用户消息
            existing_topics: 已有话题列表

        Returns:
            匹配的话题名，如果没有匹配则返回 None
        """
        if not existing_topics:
            return None

        topics_text = "\n".join([f"- {t}" for t in existing_topics])

        response = self._chat_json(
            [
                {
                    "role": "system",
                    "content": "Determine whether the message belongs to one of the listed topics. Return JSON: {'matches': true/false, 'topic_name': matched topic or null}.",
                },
                {
                    "role": "user",
                    "content": f"Existing topics:\n{topics_text}\n\nMessage:\n{message}\n\nDoes this message match any existing topic?",
                },
            ]
        )

        if response.get("matches") and response.get("topic_name"):
            return response["topic_name"]
        return None

    def _find_best_match(self, topic_name: str, existing_topics: List[str]) -> Optional[str]:
        """
        从已有话题中找到最佳匹配。

        Args:
            topic_name: 要匹配的话题名
            existing_topics: 已有话题列表

        Returns:
            最佳匹配的话题名
        """
        # 精确匹配
        if topic_name in existing_topics:
            return topic_name

        # 大小写不敏感匹配
        topic_lower = topic_name.lower()
        for t in existing_topics:
            if t.lower() == topic_lower:
                return t

        # 部分匹配
        for t in existing_topics:
            if topic_lower in t.lower() or t.lower() in topic_lower:
                return t

        # 没找到匹配，返回原始名称
        return topic_name

    def _format_history(self, history: List[Message], limit: int = 5) -> str:
        """
        格式化对话历史用于 prompt。

        Args:
            history: 消息历史
            limit: 最大消息数

        Returns:
            格式化的历史文本
        """
        if not history:
            return "No conversation history."
        recent = history[-limit:] if len(history) > limit else history
        return "\n".join([f"{msg.role}: {msg.content}" for msg in recent])

    def _format_existing_topics(
        self, existing_topics: Optional[List[str]], current_topic: str
    ) -> str:
        """
        格式化已有话题列表用于 prompt。

        Args:
            existing_topics: 已有话题列表
            current_topic: 当前话题

        Returns:
            格式化的话题列表文本
        """
        if not existing_topics:
            return "No previously discussed topics."

        # 过滤掉当前话题
        other_topics = [t for t in existing_topics if t != current_topic]

        if not other_topics:
            return "No other previously discussed topics."

        return "\n".join([f"- {t}" for t in other_topics])
