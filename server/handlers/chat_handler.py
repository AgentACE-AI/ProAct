"""
对话处理器。

处理用户消息的完整流程：话题检测 -> 规划 -> 搜索（如需要）-> 生成回复。
支持检测用户调研意图并启动深度调研。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from core.models import (
    MemoryUpdateRequest,
    Message,
    UpdateType,
)
from server.dependencies import UserDependencies
from services.deep_research import ResearchConfig
from services.proactive import DecisionContext, EligibilityContext, ProactiveCandidate


logger = logging.getLogger(__name__)

_RECENT_EXCHANGE_LIMIT = 5
_RECENT_DELIVERED_LOOKUP_LIMIT = 8


# 调研意图关键词
RESEARCH_INTENT_KEYWORDS = [
    # 中文
    "帮我调研", "调研一下", "研究一下", "深入了解",
    "详细分析", "全面调查", "系统了解", "深入分析",
    "给我一份关于", "写一份报告", "调研报告",
    "帮我研究", "深度调研", "全面研究",
    # 英文
    "research", "investigate", "analyze in depth",
    "comprehensive analysis", "detailed report",
    "deep dive", "in-depth study",
]


@dataclass
class ChatResult:
    """
    对话处理结果。

    包含回复内容和相关元数据。
    """
    reply: str
    topic: Optional[str] = None
    topic_changed: bool = False
    search_performed: bool = False
    research_task_id: Optional[str] = None  # 深度调研任务ID
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    notifications: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "reply": self.reply,
            "topic": self.topic,
            "topic_changed": self.topic_changed,
            "search_performed": self.search_performed,
            "research_task_id": self.research_task_id,
            "alerts": self.alerts,
            "notifications": self.notifications,
        }


class ChatHandler:
    """
    对话处理器。

    处理用户消息的完整流程：
    1. 话题检测
    2. 话题切换处理（如果需要）
    3. 规划回复
    4. 搜索（如果需要）
    5. 生成回复
    """

    def __init__(self, deps: UserDependencies):
        """
        初始化对话处理器。

        Args:
            deps: 用户依赖
        """
        self.deps = deps
        self.memory = deps.memory
        self._conversation_user_turn_count: int = self._count_user_turns(
            self.memory.current_messages
        )
        self._recent_exchanges = self._build_recent_exchanges(
            self.memory.current_messages
        )
        self._recent_delivered_lookup_keys: List[str] = []
        self._last_proactive_turn_by_topic: Dict[str, int] = {}
        self._last_proactive_trace: Dict[str, Any] = self._empty_proactive_trace()

    def preload_conversation_history(self, history: List[Dict[str, str]]) -> None:
        """
        预加载对话历史到 memory。

        用于模拟模式等场景，将客户端的对话历史同步到服务端。

        Args:
            history: 对话历史列表，每条消息包含 role 和 content
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        if not history:
            return

        # 清空当前会话，避免重复
        self.memory.clear_current_session()

        # 从第一条消息推断话题（如果还没有话题）
        if not self.memory.current_topic and history:
            first_user_msg = next(
                (msg["content"] for msg in history if msg.get("role") == "user"),
                None
            )
            if first_user_msg:
                # 简单推断话题（取前50个字符）
                inferred_topic = first_user_msg[:50].strip()
                if len(first_user_msg) > 50:
                    inferred_topic += "..."
                self.memory.set_current_topic(inferred_topic)

        # 加载对话历史
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                self.memory.add_message(role, content)
        self._conversation_user_turn_count = self._count_user_turns(
            self.memory.current_messages
        )
        self._recent_exchanges = self._build_recent_exchanges(
            self.memory.current_messages
        )
        self._recent_delivered_lookup_keys = []

    async def handle(self, message: str) -> ChatResult:
        """
        处理用户消息。

        完整流程：
        1. 检测调研意图（如果是调研请求，启动深度调研）
        2. 检测话题变化
        3. 处理话题切换（保存旧话题、提取信息）
        4. 规划如何回复
        5. 执行搜索（如需要）
        6. 生成最终回复

        Args:
            message: 用户消息

        Returns:
            对话处理结果
        """
        # 记录用户交互
        self.deps.record_interaction()
        self._last_proactive_trace = self._empty_proactive_trace()

        topic_changed = False
        search_performed = False
        research_task_id = None

        # 0. 检测调研意图
        if self._detect_research_intent(message):
            topic = self._extract_research_topic(message)
            if topic and self.deps._config.deep_research.enabled:
                # 启动深度调研
                try:
                    task = await self.deps.deep_research_orchestrator.start_research(
                        topic=topic,
                        config=ResearchConfig.standard(),
                        source="user_request",
                        incremental=True,
                    )
                    research_task_id = task.id

                    # 添加用户消息
                    self.memory.add_message("user", message)
                    self._register_user_turn()

                    # 生成调研启动回复
                    reply = (
                        f"好的，我开始为你深度调研「{topic}」。\n\n"
                        f"预计需要约 {task.config.max_duration_minutes} 分钟，"
                        f"完成后会通知你。\n\n"
                        f"任务ID: {task.id[:8]}..."
                    )

                    # 如果调研已完成（可能是复用已有调研）
                    if task.final_report:
                        reply = (
                            f"我已经为你完成了关于「{topic}」的调研。\n\n"
                            f"**{task.final_report.title}**\n\n"
                            f"{task.final_report.summary}\n\n"
                            f"完整报告已生成，你可以查看详情。"
                        )

                    self.memory.add_message("assistant", reply)
                    self._record_recent_exchange(message, reply)

                    return ChatResult(
                        reply=reply,
                        topic=self.memory.current_topic,
                        topic_changed=False,
                        search_performed=True,
                        research_task_id=research_task_id,
                    )
                except Exception as e:
                    print(f"[ChatHandler] 启动深度调研失败: {e}")
                    # 继续走普通流程

        # 1. 话题检测
        topic_result = self.deps.topic_monitor.run(
            current_topic=self.memory.current_topic,
            # conversation_history=[m.to_dict() for m in self.memory.current_messages],
            conversation_history=self.memory.current_messages,  # 直接传 Message 对象
            new_message=message,
            existing_topics=self.memory.get_all_topics(),
        )

        # 2. 话题切换处理
        if not topic_result.get("is_same_topic", True):
            topic_changed = True
            await self._handle_topic_change(topic_result)

        # 3. 添加用户消息到当前会话
        self.memory.add_message("user", message)
        self._register_user_turn()

        # 4. 规划回复
        plan = self.deps.planning_agent.run(message)

        # 5. 搜索（如果需要）
        additional_context = ""
        if not plan.get("memory_sufficient", True):
            search_queries = plan.get("search_queries", [])
            if search_queries:
                search_result = self.deps.search_service.search_for_query(
                    query=search_queries[0] if isinstance(search_queries[0], str) else search_queries[0].get("query", message),
                    purpose="回答用户问题",
                )
                if search_result.success and search_result.sources:
                    additional_context = self.deps.search_service.format_sources_for_context(
                        search_result.sources
                    )
                    search_performed = True

        # 6. 对话内主动预判
        memory_context = plan.get("memory_context", "")
        proactive_brief = ""
        proactive_fact_context = ""
        if self.deps._config.proactive.in_conversation_enabled:
            proactive_brief, proactive_fact_context = await self._proactive_anticipate(
                message=message,
                memory_context=memory_context,
            )

        # 7. 生成回复 — inject proactive fact context so the assistant can
        #    naturally incorporate anticipated information into its response.
        merged_additional = self._merge_contexts(
            additional_context, proactive_fact_context,
        )
        response = self.deps.assistant_agent.run(
            user_query=message,
            memory_context=memory_context,
            additional_context=merged_additional if merged_additional else None,
        )

        # 8. 添加回复消息到当前会话
        self.memory.add_message("assistant", response)
        self._record_recent_exchange(message, response)

        # 9. 增量画像提取和对话保存
        await self._incremental_extract_and_save(message, response)

        return ChatResult(
            reply=response,
            topic=self.memory.current_topic,
            topic_changed=topic_changed,
            search_performed=search_performed,
            research_task_id=research_task_id,
        )

    async def _proactive_anticipate(
        self,
        message: str,
        memory_context: str,
    ) -> tuple[str, str]:
        """
        主动预判用户接下来可能需要的信息。

        流程:
        1. NeedPredictor 预测需求（1 次 LLM 调用）
        2. 规则门控：confidence ≥ 阈值，取 top N
        3. 检索通过门控的需求对应的知识

        Args:
            message: 当前用户消息
            memory_context: 当前已检索的记忆上下文

        Returns:
            (proactive_brief, fact_context) — brief 用于日志/追踪，
            fact_context 作为 additional_context 注入 assistant 以自然融入回复。
        """
        _empty = ("", "")
        config = self.deps._config.proactive
        turn_count = self._current_turn_count()
        self._last_proactive_trace["admission_trace"] = {
            "stage": "start",
            "reason": "",
            "turn_count": turn_count,
            "min_turns": config.min_conversation_turns,
            "predictor_trace": {},
        }
        decision_entries: List[Dict[str, Any]] = []

        # 使用稳定的 conversation-level 轮次，而不是会被 topic 切换清空的当前 session。
        if self._conversation_user_turn_count <= config.min_conversation_turns:
            self._last_proactive_trace["admission_trace"].update(
                {
                    "stage": "skipped_before_prediction",
                    "reason": "below_min_conversation_turn_threshold",
                }
            )
            self._last_proactive_trace["decision_trace"] = self._build_decision_trace(
                decision_entries=decision_entries,
                candidate_count=0,
                approved_count=0,
                delivered_count=0,
            )
            return _empty

        try:
            if not self.deps.proactive_policy.allow_source("need_predictor"):
                self._last_proactive_trace["admission_trace"].update(
                    {
                        "stage": "skipped_before_prediction",
                        "reason": "policy_disallowed",
                    }
                )
                self._last_proactive_trace["decision_trace"] = self._build_decision_trace(
                    decision_entries=decision_entries,
                    candidate_count=0,
                    approved_count=0,
                    delivered_count=0,
                )
                return _empty

            # 获取上下文
            profile = self.memory.get_user_profile()
            profile_text = (
                profile.to_prompt_text() if profile else "No profile available"
            )
            history_text = self.memory.get_current_messages_formatted()
            proactive_history = self._build_proactive_history(history_text)
            turn_signals = self.deps.proactive_turn_signal_extractor.extract(
                message=message,
                conversation_history=proactive_history,
                current_topic=self.memory.current_topic or "",
                memory_context=memory_context,
            )

            # Step 6.5: NeedPredictor 预测需求
            prediction = self.deps.need_predictor.run(
                current_message=message,
                conversation_history=proactive_history,
                user_profile=profile_text,
                memory_context=memory_context,
            )
            predictor_trace = getattr(self.deps.need_predictor, "last_trace", {}) or {}
            self._last_proactive_trace["admission_trace"]["predictor_trace"] = dict(
                predictor_trace
            )

            predicted_needs = prediction.get("predicted_needs", [])
            self._last_proactive_trace["predictions"] = list(predicted_needs)
            if not predicted_needs:
                reason = "predictor_returned_no_candidates"
                if predictor_trace.get("repair_used"):
                    reason = "unsupported_candidates_after_repair"
                self._last_proactive_trace["admission_trace"].update(
                    {
                        "stage": "predictor_abstained",
                        "reason": reason,
                    }
                )
                self._last_proactive_trace["decision_trace"] = self._build_decision_trace(
                    decision_entries=decision_entries,
                    candidate_count=0,
                    approved_count=0,
                    delivered_count=0,
                )
                return _empty

            interaction_stats = self.deps.get_interaction_stats()
            active_items = self.deps.proactive_item_service.list_active()

            # Collect ALL approved & delivered facts so the assistant can
            # weave multiple proactive points into a single coherent reply.
            collected_briefs: list[str] = []
            collected_fact_snippets: list[str] = []

            for need in predicted_needs[:config.max_proactive_items]:
                need_text = (need.get("need") or "").strip()
                query = (need.get("lookup_query") or "").strip() or need_text or message
                reason = (need.get("reason") or "").strip() or need_text or message
                normalized_lookup = self._normalize_lookup_key(query)
                if (
                    normalized_lookup
                    and normalized_lookup in self._recent_delivered_lookup_keys
                ):
                    decision_entries.append(
                        {
                            "need": need_text,
                            "lookup_query": query,
                            "gate_allowed": False,
                            "delivered": False,
                            "reason": "recent_lookup_duplicate",
                        }
                    )
                    continue
                topic = self._candidate_topic(need_text or message)
                dedupe_key = self._build_dedupe_key(topic, query)
                active_items_per_topic = sum(
                    1 for item in active_items
                    if item.candidate.topic == topic
                )

                allowed = self.deps.proactive_eligibility_gate.allow(
                    source="need_predictor",
                    mode=config.mode,
                    context=EligibilityContext(
                        topic=topic,
                        reason=reason,
                        evidence=[need_text or reason],
                        dedupe_key=dedupe_key,
                        message=message,
                        turn_count=turn_count,
                        current_turn_index=turn_count,
                        has_new_information=turn_signals.has_new_information,
                        has_decision_pressure=turn_signals.has_decision_pressure,
                        is_smalltalk=turn_signals.is_smalltalk,
                        active_dedupe_keys=[
                            item.candidate.dedupe_key for item in active_items
                        ],
                        last_proactive_turn_by_topic=self._last_proactive_turn_by_topic,
                        active_items_per_user=len(active_items),
                        active_items_per_topic=active_items_per_topic,
                        is_briefable=True,
                        signal_fresh=True,
                        user_deliverable=True,
                    ),
                )
                if not allowed:
                    decision_entries.append(
                        {
                            "need": need_text,
                            "lookup_query": query,
                            "topic": topic,
                            "gate_allowed": False,
                            "delivered": False,
                            "reason": "eligibility_gate_blocked",
                        }
                    )
                    continue

                self._last_proactive_trace["approved"].append(dict(need))

                candidate = ProactiveCandidate(
                    candidate_id=f"pc_{uuid4().hex[:8]}",
                    source="need_predictor",
                    topic=topic,
                    candidate_confidence=float(need.get("confidence", 0.0)),
                    channel_hint="inline",
                    reason=reason,
                    evidence=[need_text or reason],
                    artifact_ref={"type": "direct_need", "id": query},
                    dedupe_key=dedupe_key,
                )
                memory_snippets = [
                    self._snippet_text(item)
                    for item in self.memory.search_knowledge(query, n_results=2)
                ]
                brief = self.deps.proactive_brief_service.build_for_candidate(
                    candidate,
                    memory_snippets=[snippet for snippet in memory_snippets if snippet],
                )
                decision = self.deps.proactive_decision_service.decide(
                    candidate=candidate,
                    brief=brief,
                    context=DecisionContext(
                        user_interests=profile.interests if profile else [],
                        current_context=proactive_history[-500:],
                        seconds_since_last_interaction=interaction_stats.get(
                            "seconds_since_last_interaction", 0
                        ),
                        recent_interaction_count=interaction_stats.get(
                            "recent_interaction_count", 0
                        ),
                        is_user_online=True,
                        has_active_dedupe=(
                            self.deps.proactive_item_service.find_active_by_dedupe_key(
                                dedupe_key
                            )
                            is not None
                        ),
                        topic_on_cooldown=False,
                        active_items_per_topic=active_items_per_topic,
                    ),
                )
                if decision.should_trigger and decision.channel == "inline":
                    self._record_delivered_lookup_key(query)
                    self._last_proactive_turn_by_topic[topic] = turn_count
                    delivered_entry = dict(need)
                    delivered_entry["delivered_fact_ids"] = self._delivered_fact_ids_for_snippets(
                        query=query,
                        memory_snippets=memory_snippets,
                    )
                    self._last_proactive_trace["delivered_inline"].append(delivered_entry)
                    item = self.deps.proactive_item_service.add_from_candidate(
                        candidate
                    )
                    self.deps.proactive_item_service.mark_delivered(
                        item.item_id, decision=decision,
                    )
                    collected_briefs.append(brief.summary)
                    for snippet in memory_snippets:
                        if snippet and snippet.strip():
                            collected_fact_snippets.append(snippet.strip())
                    decision_entries.append(
                        {
                            "need": need_text,
                            "lookup_query": query,
                            "topic": topic,
                            "gate_allowed": True,
                            "delivered": True,
                            "reason": decision.reason,
                            "channel": decision.channel,
                            "score": decision.score,
                        }
                    )
                else:
                    decision_entries.append(
                        {
                            "need": need_text,
                            "lookup_query": query,
                            "topic": topic,
                            "gate_allowed": True,
                            "delivered": False,
                            "reason": decision.reason,
                            "channel": decision.channel,
                            "score": decision.score,
                        }
                    )

            self._last_proactive_trace["decision_trace"] = self._build_decision_trace(
                decision_entries=decision_entries,
                candidate_count=len(predicted_needs),
                approved_count=len(self._last_proactive_trace["approved"]),
                delivered_count=len(self._last_proactive_trace["delivered_inline"]),
            )
            if collected_fact_snippets:
                self._last_proactive_trace["admission_trace"].update(
                    {
                        "stage": "delivered_inline",
                        "reason": "inline_delivery_ready",
                    }
                )
                fact_context = (
                    "=== Proactive context ===\n"
                    "The facts below may be useful for the user's NEXT step. "
                    "Follow this two-step structure strictly:\n"
                    "1. FIRST, fully and directly answer the user's current explicit question "
                    "using the main memory context. Do NOT skip or abbreviate this.\n"
                    "2. THEN, after the answer is complete, naturally introduce any "
                    "relevant proactive facts below using transitional phrases such as "
                    "\"By the way, you may also want to know...\", "
                    "\"For your next step...\", or \"It's also worth noting that...\". "
                    "Only include proactive facts that are genuinely useful — do not force them.\n"
                    "---\n"
                    + "\n".join(f"- {s}" for s in collected_fact_snippets)
                )
                return (" ".join(collected_briefs), fact_context)

            self._last_proactive_trace["admission_trace"].update(
                {
                    "stage": "predicted_but_not_delivered",
                    "reason": "no_inline_delivery_after_decision",
                }
            )

        except Exception as e:
            self._last_proactive_trace["admission_trace"].update(
                {
                    "stage": "error",
                    "reason": str(e),
                }
            )
            logger.exception("[ChatHandler] Proactive anticipation failed: %s", e)

        return _empty

    @staticmethod
    def _empty_proactive_trace() -> Dict[str, Any]:
        return {
            "predictions": [],
            "approved": [],
            "delivered_inline": [],
            "admission_trace": None,
            "decision_trace": None,
        }

    @staticmethod
    def _build_decision_trace(
        *,
        decision_entries: List[Dict[str, Any]],
        candidate_count: int,
        approved_count: int,
        delivered_count: int,
    ) -> Dict[str, Any]:
        return {
            "summary": {
                "candidate_count": candidate_count,
                "approved_count": approved_count,
                "delivered_count": delivered_count,
            },
            "candidates": list(decision_entries),
        }

    @staticmethod
    def _merge_contexts(*contexts: str) -> str:
        """合并多个 context 字符串，过滤空值。"""
        parts = [c for c in contexts if c and c.strip()]
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _append_inline_brief(reply: str, brief: str) -> str:
        if not brief.strip():
            return reply
        return f"{reply.rstrip()}\n\n{brief.strip()}"

    def _current_turn_count(self) -> int:
        return self._conversation_user_turn_count

    def _register_user_turn(self) -> None:
        self._conversation_user_turn_count += 1

    @staticmethod
    def _count_user_turns(messages: List[Message]) -> int:
        return sum(1 for msg in messages if msg.role == "user")

    def _build_proactive_history(self, current_history: str) -> str:
        parts: List[str] = []

        if self._count_user_turns(self.memory.current_messages) < self._conversation_user_turn_count:
            recent_text = self._format_recent_exchanges()
            if recent_text:
                parts.append(f"Recent cross-topic dialogue:\n{recent_text}")

        delivered_text = self._format_recent_delivered_lookup_keys()
        if delivered_text:
            parts.append(f"Recently delivered proactive lookups:\n{delivered_text}")

        if current_history.strip():
            parts.append(current_history)

        return "\n\n".join(parts)

    def _record_recent_exchange(self, user_message: str, assistant_response: str) -> None:
        if not user_message.strip() and not assistant_response.strip():
            return
        self._recent_exchanges.append(
            {
                "user": user_message.strip(),
                "assistant": assistant_response.strip(),
            }
        )
        self._recent_exchanges = self._recent_exchanges[-_RECENT_EXCHANGE_LIMIT:]

    @staticmethod
    def _build_recent_exchanges(messages: List[Message]) -> List[Dict[str, str]]:
        recent_exchanges: List[Dict[str, str]] = []
        pending_user: Optional[str] = None
        for msg in messages:
            if msg.role == "user":
                pending_user = msg.content
                continue
            if msg.role == "assistant" and pending_user is not None:
                recent_exchanges.append(
                    {
                        "user": pending_user.strip(),
                        "assistant": msg.content.strip(),
                    }
                )
                pending_user = None
        return recent_exchanges[-_RECENT_EXCHANGE_LIMIT:]

    def _format_recent_exchanges(self) -> str:
        parts: List[str] = []
        for exchange in self._recent_exchanges[-_RECENT_EXCHANGE_LIMIT:]:
            user = exchange.get("user", "").strip()
            assistant = exchange.get("assistant", "").strip()
            if user:
                parts.append(f"user: {user}")
            if assistant:
                parts.append(f"assistant: {assistant}")
        return "\n".join(parts)

    def _record_delivered_lookup_key(self, query: str) -> None:
        normalized = self._normalize_lookup_key(query)
        if not normalized:
            return
        remaining = [
            item for item in self._recent_delivered_lookup_keys
            if item != normalized
        ]
        remaining.append(normalized)
        self._recent_delivered_lookup_keys = remaining[-_RECENT_DELIVERED_LOOKUP_LIMIT:]

    def _format_recent_delivered_lookup_keys(self) -> str:
        if not self._recent_delivered_lookup_keys:
            return ""
        return "\n".join(f"- {item}" for item in self._recent_delivered_lookup_keys)

    @staticmethod
    def _normalize_lookup_key(query: str) -> str:
        return re.sub(r"\s+", "", (query or "").strip().lower())

    @staticmethod
    def _lookup_fact_ids(query: str) -> List[str]:
        query_text = (query or "").strip()
        factset_match = re.match(
            r"^factset:([A-Za-z0-9_, -]+)$",
            query_text,
            flags=re.IGNORECASE,
        )
        if factset_match:
            return [
                fact_id.strip().upper()
                for fact_id in factset_match.group(1).split(",")
                if fact_id.strip()
            ]

        fact_match = re.match(
            r"^fact:([A-Za-z0-9_-]+)$",
            query_text,
            flags=re.IGNORECASE,
        )
        if fact_match:
            return [fact_match.group(1).strip().upper()]

        return []

    def _delivered_fact_ids_for_snippets(
        self,
        *,
        query: str,
        memory_snippets: List[str],
    ) -> List[str]:
        fact_ids = self._lookup_fact_ids(query)
        if not fact_ids:
            return []

        delivered_count = sum(1 for snippet in memory_snippets if snippet and snippet.strip())
        return fact_ids[:delivered_count]

    def _candidate_topic(self, fallback: str) -> str:
        return (self.memory.current_topic or fallback).strip()

    @staticmethod
    def _build_dedupe_key(topic: str, query: str) -> str:
        normalized = re.sub(r"\s+", "-", f"{topic}:{query}".strip().lower())
        return normalized[:200]

    @staticmethod
    def _snippet_text(item: Any) -> str:
        if isinstance(item, dict):
            return item.get("summary") or item.get("content", "")[:200]
        return getattr(item, "summary", "") or getattr(item, "content", "")[:200]

    def _detect_research_intent(self, message: str) -> bool:
        """
        检测用户消息是否包含调研意图。

        Args:
            message: 用户消息

        Returns:
            是否包含调研意图
        """
        message_lower = message.lower()
        for keyword in RESEARCH_INTENT_KEYWORDS:
            if keyword.lower() in message_lower:
                return True
        return False

    def _extract_research_topic(self, message: str) -> Optional[str]:
        """
        从用户消息中提取调研主题。

        Args:
            message: 用户消息

        Returns:
            调研主题，或 None
        """
        # 尝试使用 LLM 提取主题
        try:
            prompt = f"""
Extract the research topic from the following user message.

User message: {message}

Return JSON in the following format:
{{"topic": "research topic"}}

If the topic cannot be extracted, return:
{{"topic": null}}
"""
            result = self.deps._llm.chat_json(
                messages=[
                    {"role": "system", "content": "You are a topic extraction assistant."},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4o-mini",
                temperature=0.3,
            )
            return result.get("topic")
        except Exception as e:
            print(f"[ChatHandler] 提取调研主题失败: {e}")
            # 简单的规则提取
            # 移除关键词，剩下的就是主题
            topic = message
            for keyword in RESEARCH_INTENT_KEYWORDS:
                topic = topic.replace(keyword, "").strip()
            # 清理标点和多余空格
            topic = re.sub(r'[，。？！、：；""''【】「」]', '', topic).strip()
            return topic if len(topic) > 2 else None

    async def _incremental_extract_and_save(
        self,
        user_query: str,
        assistant_response: str
    ) -> None:
        """
        增量画像提取和对话保存。

        每次交互后调用，基于 running_summary + 当前交互 进行画像提取，
        然后更新 running_summary 并保存消息到 conversations.json。

        Args:
            user_query: 用户消息
            assistant_response: 助手回复
        """
        from core.prompts import Prompts

        topic = self.memory.current_topic
        if not topic:
            return

        # 获取当前话题的 running_summary
        running_summary = self.memory.get_topic_running_summary(topic)

        # 默认情感值
        user_sentiment_label = None
        user_sentiment_score = None

        try:
            # 调用 LLM 进行增量提取
            prompt = Prompts.INCREMENTAL_PROFILE_EXTRACTION.format(
                topic=topic,
                running_summary=running_summary if running_summary else "(No prior summary; this is the first interaction on this topic.)",
                user_query=user_query,
                assistant_response=assistant_response[:1000],  # 限制长度
            )

            result = self.deps._llm.chat_json(
                messages=[
                    {"role": "system", "content": "You extract user profile information from the conversation and generate a summary. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                model="gpt-4o-mini",
            )

            # 更新用户画像
            profile_updates = result.get("profile_updates", {})
            if profile_updates:
                # 过滤空值
                filtered_updates = {
                    k: v for k, v in profile_updates.items()
                    if v is not None and v != [] and v != ""
                }
                if filtered_updates:
                    self.memory.submit_update(MemoryUpdateRequest(
                        update_type=UpdateType.UPDATE_PROFILE,
                        source="incremental_extraction",
                        user_id=self.memory.user_id,
                        data=filtered_updates,
                    ))

            # 提取情感信息
            user_sentiment = result.get("user_sentiment", {})
            if user_sentiment:
                user_sentiment_label = user_sentiment.get("label")
                user_sentiment_score = user_sentiment.get("score")
                # 验证情感标签有效性
                valid_labels = {"surprise", "anger", "sadness", "joy", "fear", "neutral", "disgust"}
                if user_sentiment_label and user_sentiment_label.lower() not in valid_labels:
                    user_sentiment_label = "neutral"  # 无效标签回退到 neutral
                elif user_sentiment_label:
                    user_sentiment_label = user_sentiment_label.lower()

            # 提取并存储用户事实
            extracted_facts = result.get("extracted_facts", [])
            if extracted_facts:
                self.memory.add_facts(extracted_facts, source_topic=topic)

            # 保存交互到 conversations.json
            updated_summary = result.get("updated_summary", running_summary)
            key_info = result.get("key_info", [])

            self.memory.append_interaction_to_topic(
                topic=topic,
                user_message=Message(
                    role="user",
                    content=user_query,
                    sentiment=user_sentiment_label,
                    sentiment_score=user_sentiment_score,
                ),
                assistant_message=Message(role="assistant", content=assistant_response),
                running_summary=updated_summary if updated_summary else f"Conversation about {topic}",
                key_info=key_info,
            )

        except Exception as e:
            print(f"[ChatHandler] 增量画像提取失败: {e}")
            # 即使提取失败，也要保存对话（不带情感信息）
            self.memory.append_interaction_to_topic(
                topic=topic,
                user_message=Message(role="user", content=user_query),
                assistant_message=Message(role="assistant", content=assistant_response),
                running_summary=running_summary if running_summary else f"Conversation about {topic}",
            )

    async def _handle_topic_change(self, topic_result: Dict[str, Any]) -> None:
        """
        处理话题切换。

        由于增量提取已经在每次交互后保存了对话和画像，
        话题切换时只需要：
        1. 设置新话题
        2. 清空当前会话缓存

        Args:
            topic_result: 话题检测结果
        """
        # 设置新话题
        if topic_result.get("is_returning_topic"):
            new_topic = topic_result.get("returning_topic_name", "General")
        else:
            new_topic = topic_result.get("new_topic", "General")

        self.memory.set_current_topic(new_topic)

        # 清空当前会话缓存（不影响 conversations.json 中已保存的数据）
        self.memory.clear_current_session()

    async def _extract_conversation_info(self) -> Dict[str, Any]:
        """
        从对话中提取信息。

        提取内容包括：
        - 对话摘要
        - 关键信息
        - 用户偏好
        - 画像更新建议

        Returns:
            提取的信息字典
        """
        from core.prompts import Prompts
        from tools.llm_client import LLMClient

        # 格式化对话内容
        messages_text = self.memory.get_current_messages_formatted()

        if not messages_text:
            return {
                "summary": "",
                "key_info": [],
                "preferences": [],
                "profile_updates": {},
            }

        try:
            llm = self.deps._llm

            # 1. 提取对话摘要和关键信息
            memory_prompt = Prompts.MEMORY_EXTRACTION.format(
                topic=self.memory.current_topic or "General",
                conversation=messages_text,
            )

            memory_result = llm.chat_json(
                messages=[
                    {"role": "system", "content": "You extract key information from conversations."},
                    {"role": "user", "content": memory_prompt},
                ],
                temperature=0.3,
                model="gpt-4o-mini",
            )

            # 2. 提取用户画像更新
            profile_prompt = Prompts.PROFILE_EXTRACTION.format(
                conversation=messages_text,
            )

            profile_result = llm.chat_json(
                messages=[
                    {"role": "system", "content": "You extract user profile information from conversations."},
                    {"role": "user", "content": profile_prompt},
                ],
                temperature=0.3,
                model="gpt-4o-mini",
            )

            # 构建画像更新（过滤掉 null 和空值）
            profile_updates = {}
            for key in ["role", "age", "location", "interests", "goals", "challenges", "traits", "communication_style"]:
                value = profile_result.get(key)
                if value is not None and value != [] and value != "":
                    profile_updates[key] = value

            return {
                "summary": memory_result.get("topic_summary", ""),
                "key_info": memory_result.get("key_information", []),
                "preferences": memory_result.get("user_preferences", []),
                "profile_updates": profile_updates,
            }
        except Exception as e:
            print(f"[ChatHandler] 提取对话信息失败: {e}")
            # 返回基本摘要
            return {
                "summary": f"Conversation about {self.memory.current_topic}",
                "key_info": [],
                "preferences": [],
                "profile_updates": {},
            }

    def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取当前对话历史。

        Args:
            limit: 返回消息数量限制

        Returns:
            对话历史列表
        """
        messages = self.memory.current_messages[-limit:]
        return [m.to_dict() for m in messages]

    def reset_session(self) -> None:
        """
        重置当前会话。

        保存当前对话并清空会话状态。
        """
        if self.memory.current_topic and self.memory.current_messages:
            # 保存当前对话
            self.memory.submit_update(MemoryUpdateRequest(
                update_type=UpdateType.ADD_CONVERSATION,
                source="session_reset",
                user_id=self.memory.user_id,
                data={
                    "topic": self.memory.current_topic,
                    "messages": [m.to_dict() for m in self.memory.current_messages],
                    "summary": f"关于 {self.memory.current_topic} 的对话（会话重置）",
                    "key_info": [],
                    "user_preferences": [],
                },
            ))

        self.memory.current_topic = None
        self.memory.clear_current_session()
