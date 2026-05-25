"""
研究价值评估 Agent。

评估候选研究主题是否值得执行搜索。
"""

from typing import Any, Dict, List

from core.config import AgentModelConfig
from core.prompts import Prompts
from tools.llm_client import LLMClient

from .base_agent import BaseAgent


class ResearchValueEvaluator(BaseAgent):
    """
    研究价值评估 Agent。

    职责:
    - 评估候选研究主题的搜索价值
    - 综合考虑用户关注度、知识完整度、增量价值
    - 过滤低价值主题，确保 proactive search 的质量
    """

    def __init__(self, config: AgentModelConfig, llm_client: LLMClient):
        """
        初始化研究价值评估 Agent。

        Args:
            config: Agent 模型配置
            llm_client: LLM 客户端
        """
        super().__init__(config, llm_client)

    def evaluate(
        self,
        candidate_topic: Dict[str, Any],
        user_profile_text: str,
        existing_knowledge: str,
        research_history: List[str],
    ) -> Dict[str, Any]:
        """
        评估单个候选主题的研究价值。

        Args:
            candidate_topic: 候选主题信息
                {
                    "topic": str,
                    "source": str,      # "knowledge_gap" | "scenario" | "related"
                    "reason": str,
                    "confidence": float,
                }
            user_profile_text: 用户画像文本
            existing_knowledge: 该主题现有的知识内容
            research_history: 历史调研主题列表

        Returns:
            {
                "should_research": bool,    # 是否值得研究
                "score": int,               # 价值分数 0-100
                "reason": str,              # 评估理由
                "factors": {                # 各因素得分
                    "user_relevance": int,      # 用户相关性 0-100
                    "knowledge_gap": int,       # 知识缺口程度 0-100
                    "incremental_value": int,   # 增量价值 0-100
                    "timeliness": int,          # 时效性需求 0-100
                }
            }
        """
        # 格式化历史调研（用于去重判断）
        research_history_text = (
            "\n".join([f"- {t}" for t in research_history[-10:]])
            if research_history
            else "No research history"
        )

        prompt = Prompts.RESEARCH_VALUE_EVALUATION.format(
            topic=candidate_topic.get("topic", ""),
            topic_source=candidate_topic.get("source", "unknown"),
            topic_reason=candidate_topic.get("reason", ""),
            topic_confidence=candidate_topic.get("confidence", 0.5),
            user_profile=user_profile_text,
            existing_knowledge=existing_knowledge or "No existing knowledge",
            research_history=research_history_text,
        )

        result = self._chat_json(
            [
                {
                    "role": "system",
                    "content": "You are a research value evaluator responsible for deciding whether a research topic is worth spending resources to search.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        score = result.get("score", 0)

        return {
            "should_research": result.get("should_research", False),
            "score": score,
            "reason": result.get("reason", ""),
            "factors": result.get("factors", {}),
        }

    def evaluate_batch(
        self,
        candidates: List[Dict[str, Any]],
        user_profile_text: str,
        get_existing_knowledge: callable,
        research_history: List[str],
        threshold: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        批量评估候选主题并筛选。

        Args:
            candidates: 候选主题列表
            user_profile_text: 用户画像文本
            get_existing_knowledge: 获取现有知识的回调函数 (topic) -> str
            research_history: 历史调研主题列表
            threshold: 通过阈值，分数 >= threshold 才通过

        Returns:
            通过评估的主题列表（按分数降序排列）
        """
        evaluated = []

        for candidate in candidates:
            topic = candidate.get("topic", "")
            if not topic:
                continue

            # 获取该主题的现有知识
            existing_knowledge = get_existing_knowledge(topic) if get_existing_knowledge else ""

            # 评估
            result = self.evaluate(
                candidate_topic=candidate,
                user_profile_text=user_profile_text,
                existing_knowledge=existing_knowledge,
                research_history=research_history,
            )

            if result["should_research"] and result["score"] >= threshold:
                evaluated.append({
                    **candidate,
                    "evaluation": result,
                })

        # 按分数降序排序
        evaluated.sort(key=lambda x: x["evaluation"]["score"], reverse=True)

        return evaluated
