"""
记忆系统门面。

提供统一的记忆访问接口，封装底层存储细节。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.config import Config
from core.models import (
    Conversation,
    KnowledgeItem,
    KnowledgeStatus,
    MemoryUpdateRequest,
    MemoryUpdateResult,
    Message,
    UpdateType,
    UserFact,
    UserProfile,
)

from memory.fact_store import FactStore
from memory.knowledge_index import KnowledgeIndex
from memory.memory_update_agent import MemoryUpdateAgent
from memory.profile_store import ProfileStore
from memory.vector_store import VectorStore


class MemorySystem:
    """
    记忆系统门面。

    职责:
    1. 提供统一的记忆访问接口
    2. 封装底层存储细节
    3. 协调 MemoryUpdateAgent 处理写入
    4. 管理当前会话状态
    """

    def __init__(self, user_id: str, config: Config, llm_client: Optional[Any] = None):
        """
        初始化记忆系统。

        Args:
            user_id: 用户 ID
            config: 全局配置
            llm_client: LLM 客户端（可选，延迟注入）
        """
        self.user_id = user_id
        self.config = config

        # 初始化存储层
        self._vector_store = VectorStore(user_id, config.storage)
        self._knowledge_index = KnowledgeIndex(user_id, config.storage)
        self._profile_store = ProfileStore(user_id, config.storage)
        self._fact_store = FactStore(user_id, config.storage)

        # 对话存储路径
        self._conversations_path = config.storage.get_conversations_path(user_id)
        self._conversations: Dict[str, Conversation] = {}
        self._conversations_loaded = False  # 标记是否成功加载
        self._load_conversations()

        # 当前会话状态
        self.current_topic: Optional[str] = None
        self.current_messages: List[Message] = []

        # LLM 客户端和更新 Agent（延迟初始化）
        self._llm_client = llm_client
        self._update_agent: Optional[MemoryUpdateAgent] = None

    def set_llm_client(self, llm_client: Any) -> None:
        """
        设置 LLM 客户端（依赖注入）。

        Args:
            llm_client: LLM 客户端实例
        """
        self._llm_client = llm_client
        self._init_update_agent()

    def _init_update_agent(self) -> None:
        """初始化更新 Agent"""
        if self._llm_client and not self._update_agent:
            self._update_agent = MemoryUpdateAgent(
                vector_store=self._vector_store,
                knowledge_index=self._knowledge_index,
                profile_store=self._profile_store,
                llm_client=self._llm_client,
                config=self.config.agents.memory_update,
            )

    # ==================== 查询接口 (只读) ====================

    def search_knowledge(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[KnowledgeItem]:
        """
        语义搜索知识库。

        Args:
            query: 搜索查询
            n_results: 返回数量

        Returns:
            匹配的知识条目列表
        """
        results = self._vector_store.search_knowledge(query, n_results * 2)

        # 过滤掉非活跃状态的记录
        items = []
        for r in results:
            doc_id = r.get("id")
            if doc_id:
                index_record = self._knowledge_index.get_by_id(doc_id)
                if index_record and index_record.get("status") == KnowledgeStatus.ACTIVE.value:
                    items.append(self._result_to_knowledge_item(r, index_record))
            else:
                # 兼容旧数据
                items.append(self._result_to_knowledge_item(r, {}))

            if len(items) >= n_results:
                break

        return items

    def search_conversations(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Conversation]:
        """
        搜索历史对话。

        Args:
            query: 搜索查询
            n_results: 返回数量

        Returns:
            匹配的对话列表
        """
        results = self._vector_store.search_conversations(query, n_results)

        conversations = []
        for r in results:
            metadata = r.get("metadata", {})
            topic = metadata.get("topic", "")
            if topic and topic in self._conversations:
                conversations.append(self._conversations[topic])
            else:
                # 创建临时对话对象
                conversations.append(Conversation(
                    topic=topic,
                    summary=r.get("content", ""),
                ))

        return conversations

    def get_user_profile(self) -> UserProfile:
        """
        获取用户画像。

        Returns:
            用户画像
        """
        return self._profile_store.get()

    def get_recent_topics(self, limit: int = 5) -> List[str]:
        """
        获取最近话题列表。

        Args:
            limit: 返回数量

        Returns:
            最近话题列表
        """
        sorted_convs = sorted(
            self._conversations.values(),
            key=lambda x: x.updated_at,
            reverse=True,
        )
        return [c.topic for c in sorted_convs[:limit]]

    def get_all_topics(self) -> List[str]:
        """
        获取所有话题列表。

        Returns:
            所有话题列表
        """
        return list(self._conversations.keys())

    def get_topic_summary(self, topic: str) -> Optional[str]:
        """
        获取话题摘要。

        Args:
            topic: 话题名

        Returns:
            话题摘要，或 None
        """
        conv = self._conversations.get(topic)
        return conv.summary if conv else None

    def get_conversation(self, topic: str) -> Optional[Conversation]:
        """
        获取指定话题的对话。

        Args:
            topic: 话题名

        Returns:
            对话对象，或 None
        """
        return self._conversations.get(topic)

    def find_similar_knowledge(
        self,
        content: str,
        threshold: float = 0.15,
    ) -> List[KnowledgeItem]:
        """
        查找相似知识（用于查重）。

        Args:
            content: 待检查内容
            threshold: 相似度阈值 (距离，越小越相似)

        Returns:
            相似的知识条目列表
        """
        results = self._vector_store.search_knowledge(content[:2000], 5)

        items = []
        for r in results:
            distance = r.get("distance", 1.0)
            if distance < threshold:
                items.append(self._result_to_knowledge_item(r, {}))

        return items

    def format_context_for_query(self, query: str) -> str:
        """
        为回复生成格式化上下文。

        结合用户事实、知识库、历史对话、用户偏好。
        自动检测时间相关查询并补充时序信息。

        Args:
            query: 用户查询

        Returns:
            格式化的上下文文本
        """
        import re

        sections = []
        is_sentiment_query = bool(re.search(r"sentiment", query, re.IGNORECASE))

        # 全量提供用户事实信息（情感查询仅依赖时间窗口上下文，跳过 facts）
        if not is_sentiment_query:
            facts_text = self.format_facts_for_prompt()
            if facts_text:
                sections.append(f"User-mentioned facts:\n{facts_text}")

        # 检测是否为时间相关查询
        time_match = re.search(
            r"(?:around|at|during|about)\s*['\"]?(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2})",
            query,
            re.IGNORECASE
        )

        # 如果是时间相关查询，优先检索该时间段的消息
        if time_match:
            timestamp = time_match.group(1)
            time_context = self._get_temporal_context(
                timestamp,
                window_minutes=2 if is_sentiment_query else 10,
                user_only=is_sentiment_query,
                max_content_len=300 if is_sentiment_query else 100,
            )
            if time_context:
                sections.append(time_context)

        # 情感查询：仅返回时序上下文，跳过 knowledge/conversations/preferences
        if is_sentiment_query:
            return "\n\n".join(sections) if sections else "No relevant memory available"

        # 搜索相关知识
        knowledge = self.search_knowledge(query, 3)
        if knowledge:
            knowledge_text = "\n".join([
                f"- {k.summary or k.content[:200]}"
                for k in knowledge
            ])
            sections.append(f"Relevant knowledge:\n{knowledge_text}")

        # 搜索相关对话
        conversations = self.search_conversations(query, 2)
        if conversations:
            conv_text = "\n".join([
                f"- [{c.topic}] {c.summary[:150] if c.summary else 'No summary'}"
                for c in conversations
            ])
            sections.append(f"Relevant conversations:\n{conv_text}")

        # 搜索用户偏好
        preferences = self._vector_store.search_preferences(query, 3)
        if preferences:
            pref_text = "\n".join([
                f"- {p.get('content', '')}"
                for p in preferences
            ])
            sections.append(f"User preferences:\n{pref_text}")

        return "\n\n".join(sections) if sections else "No relevant memory available"

    def _get_temporal_context(
        self,
        timestamp: str,
        window_minutes: int = 10,
        user_only: bool = False,
        max_content_len: int = 100,
    ) -> Optional[str]:
        """
        获取特定时间段的对话上下文。

        Args:
            timestamp: 目标时间点
            window_minutes: 时间窗口（前后各 N 分钟）

        Returns:
            格式化的时间段上下文，包含消息内容和情感信息
        """
        from datetime import timedelta

        target_dt = self._parse_flexible_timestamp(timestamp)
        if not target_dt:
            return None

        # 计算时间窗口
        start_dt = target_dt - timedelta(minutes=window_minutes)
        end_dt = target_dt + timedelta(minutes=window_minutes)

        # 检索时间范围内的消息
        self._ensure_conversations_loaded()
        messages = []

        for conv in self._conversations.values():
            for msg in conv.messages:
                if user_only and msg.role != "user":
                    continue
                msg_dt = self._parse_flexible_timestamp(msg.timestamp)
                if msg_dt and start_dt <= msg_dt <= end_dt:
                    messages.append((msg_dt, msg, conv.topic))

        if not messages:
            return None

        # 按时间排序
        messages.sort(key=lambda x: x[0])

        # 标记最接近目标时间点的消息
        closest_delta = min(abs(msg_dt - target_dt) for msg_dt, _, _ in messages)

        # 格式化输出
        lines = [
            f"Conversation records from this time window ({timestamp} ±{window_minutes} minutes):",
            "Note: the >>> prefix marks the message closest to the target timestamp",
        ]
        for msg_dt, msg, topic in messages:
            time_str = msg_dt.strftime("%H:%M")
            sentiment_info = ""
            if msg.sentiment:
                sentiment_info = f" [Sentiment: {msg.sentiment}"
                if msg.sentiment_score is not None:
                    sentiment_info += f", intensity: {msg.sentiment_score}"
                sentiment_info += "]"
            marker = ">>>" if abs(msg_dt - target_dt) == closest_delta else "   "
            lines.append(f"{marker} [{time_str}] {msg.role}: {msg.content[:max_content_len]}{'...' if len(msg.content) > max_content_len else ''}{sentiment_info}")

        return "\n".join(lines)

    # ==================== 更新接口 ====================

    def submit_update(self, request: MemoryUpdateRequest) -> MemoryUpdateResult:
        """
        提交记忆更新请求。

        所有写入操作都通过此方法。

        Args:
            request: 更新请求

        Returns:
            更新结果
        """
        if not self._update_agent:
            self._init_update_agent()

        if not self._update_agent:
            # 无法处理，直接处理简单情况
            return self._simple_update(request)

        result = self._update_agent.process(request)

        if result.success and request.update_type == UpdateType.ADD_CONVERSATION:
            # 成功添加对话后，保存对话记录
            self._save_conversation_direct(request.data)

        return result

    def handle_topic_change(
        self,
        returning_to_topic: Optional[str] = None,
        extraction_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        处理话题切换：保存当前对话并提取信息。
        支持新话题和返回之前讨论过的话题两种情况。

        Args:
            returning_to_topic: 如果用户返回之前的话题，传入该话题名
            extraction_data: LLM 提取的对话信息（summary, key_info, user_preferences等）

        Returns:
            处理结果，包含 saved_topic, was_merge 等信息
        """
        if not self.current_messages or not self.current_topic:
            return {"saved_topic": None, "was_merge": False}

        extraction_data = extraction_data or {}

        if returning_to_topic and returning_to_topic in self._conversations:
            # 返回之前的话题：合并对话
            result = self.merge_into_topic(
                topic=returning_to_topic,
                messages=self.current_messages.copy(),
                extraction_data=extraction_data,
            )
            saved_topic = returning_to_topic
            was_merge = True
        else:
            # 新话题或返回的话题不存在：创建/更新对话
            self._update_conversation(
                topic=self.current_topic,
                messages=self.current_messages.copy(),
                extraction_data=extraction_data,
            )
            saved_topic = self.current_topic
            was_merge = False

        # 清空当前会话
        old_topic = self.current_topic
        old_messages = self.current_messages.copy()
        self.current_topic = None
        self.current_messages = []

        return {
            "saved_topic": saved_topic,
            "previous_topic": old_topic,
            "message_count": len(old_messages),
            "was_merge": was_merge,
            "extraction": extraction_data,
        }

    def merge_into_topic(
        self,
        topic: str,
        messages: List[Message],
        extraction_data: Optional[Dict[str, Any]] = None,
        summary_updater: Optional[Callable[[str, str, str], str]] = None,
    ) -> bool:
        """
        将新消息合并到已存在的话题对话中。

        Args:
            topic: 要合并到的话题名
            messages: 新消息列表
            extraction_data: 提取的信息
            summary_updater: 可选的摘要更新函数 (topic, old_summary, new_conversation) -> new_summary

        Returns:
            是否合并成功
        """
        import uuid

        if topic not in self._conversations:
            return False

        conv = self._conversations[topic]

        # 追加消息
        conv.messages.extend(messages)
        conv.updated_at = datetime.now().isoformat()

        # 更新摘要
        if summary_updater and messages:
            new_conv_text = "\n".join([
                f"{m.role}: {m.content}" for m in messages
            ])
            try:
                updated_summary = summary_updater(topic, conv.summary, new_conv_text)
                conv.summary = updated_summary
            except Exception as e:
                print(f"[MemorySystem] 更新摘要失败: {e}")
        elif extraction_data and extraction_data.get("summary"):
            # 如果没有 summary_updater 但有新摘要，追加到现有摘要
            if conv.summary:
                conv.summary = f"{conv.summary}\n\n[续] {extraction_data['summary']}"
            else:
                conv.summary = extraction_data["summary"]

        # 合并提取的信息
        if extraction_data:
            # 追加并去重
            conv.key_info = list(set(
                conv.key_info + extraction_data.get("key_info", [])
            ))
            conv.user_preferences = list(set(
                conv.user_preferences + extraction_data.get("user_preferences", [])
            ))

        # 添加到向量库
        doc_id = str(uuid.uuid4())
        summary_text = extraction_data.get("summary", "") if extraction_data else ""
        self._vector_store.add_conversation(
            doc_id=doc_id,
            content=summary_text or f"关于{topic}的后续对话",
            metadata={"topic": topic, "is_merge": True},
        )

        self._save_conversations()
        return True

    def _simple_update(self, request: MemoryUpdateRequest) -> MemoryUpdateResult:
        """
        简单更新（无 LLM 支持时的降级处理）。

        Args:
            request: 更新请求

        Returns:
            更新结果
        """
        from core.models import UpdateAction

        if request.update_type == UpdateType.UPDATE_PROFILE:
            self._profile_store.update(request.data)
            return MemoryUpdateResult(
                success=True,
                action=UpdateAction.ADDED,
                message="用户画像已更新（简单模式）",
            )

        if request.update_type == UpdateType.ADD_CONVERSATION:
            return self._save_conversation_direct(request.data)

        return MemoryUpdateResult(
            success=False,
            action=UpdateAction.SKIPPED,
            message="需要 LLM 支持的更新操作",
        )

    def _save_conversation_direct(self, data: Dict[str, Any]) -> MemoryUpdateResult:
        """直接保存对话（无需 LLM），使用统一的 _update_conversation 方法"""
        from core.models import UpdateAction

        topic = data.get("topic", "")
        messages = data.get("messages", [])
        summary = data.get("summary", "")

        if not topic:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message="话题为空",
            )

        # 转换消息格式
        msg_list = []
        for m in messages:
            if isinstance(m, dict):
                msg_list.append(Message.from_dict(m))
            elif isinstance(m, Message):
                msg_list.append(m)

        # 使用 _update_conversation 方法统一处理
        self._update_conversation(
            topic=topic,
            messages=msg_list,
            extraction_data={
                "summary": summary,
                "key_info": data.get("key_info", []),
                "user_preferences": data.get("user_preferences", []),
            },
        )

        return MemoryUpdateResult(
            success=True,
            action=UpdateAction.ADDED,
            message="对话已保存",
        )

    # ==================== 会话管理 ====================

    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到当前会话。

        Args:
            role: 角色 ("user" | "assistant")
            content: 消息内容
        """
        msg = Message(role=role, content=content)
        self.current_messages.append(msg)

    def set_current_topic(self, topic: str) -> None:
        """
        设置当前话题。

        Args:
            topic: 话题名
        """
        self.current_topic = topic

    def clear_current_session(self) -> None:
        """清空当前会话"""
        self.current_messages = []

    def get_current_messages_formatted(self) -> str:
        """
        获取当前会话消息的格式化文本。

        Returns:
            格式化的消息文本
        """
        return "\n".join([
            f"{m.role}: {m.content}"
            for m in self.current_messages
        ])

    def get_topic_running_summary(self, topic: str) -> str:
        """
        获取话题的滚动摘要。

        Args:
            topic: 话题名

        Returns:
            滚动摘要，如果话题不存在则返回空字符串
        """
        self._ensure_conversations_loaded()
        if topic in self._conversations:
            return self._conversations[topic].running_summary
        return ""

    def append_interaction_to_topic(
        self,
        topic: str,
        user_message: Message,
        assistant_message: Message,
        running_summary: str,
        key_info: Optional[List[str]] = None,
        user_preferences: Optional[List[str]] = None,
    ) -> None:
        """
        向话题追加一次交互记录。

        这是增量画像提取方案的核心方法，每次对话后调用，
        保存消息并更新 running_summary。

        Args:
            topic: 话题名
            user_message: 用户消息
            assistant_message: 助手消息
            running_summary: 更新后的滚动摘要
            key_info: 关键信息列表
            user_preferences: 用户偏好列表
        """
        self._ensure_conversations_loaded()

        if topic not in self._conversations:
            # 创建新对话
            self._conversations[topic] = Conversation(topic=topic)

        conv = self._conversations[topic]
        conv.messages.append(user_message)
        conv.messages.append(assistant_message)
        conv.running_summary = running_summary
        conv.summary = running_summary  # 同步更新 summary
        conv.updated_at = datetime.now().isoformat()

        if key_info:
            conv.key_info = list(set(conv.key_info + key_info))
        if user_preferences:
            conv.user_preferences = list(set(conv.user_preferences + user_preferences))

        self._save_conversations()

    # ==================== 时序查询接口 ====================

    @staticmethod
    def _parse_flexible_timestamp(timestamp_str: str) -> Optional[datetime]:
        """
        解析多种格式的时间戳。

        支持格式:
        - ISO 8601: "2024-10-01T08:00:00"
        - 测试格式: "'2024-10-01 08:00' Tuesday"
        - 简化格式: "2024-10-01 08:00"

        Args:
            timestamp_str: 时间字符串

        Returns:
            datetime 对象，解析失败返回 None
        """
        import re
        from datetime import datetime as dt

        if not timestamp_str:
            return None

        # 清理字符串
        clean_str = timestamp_str.strip()

        # 尝试多种格式
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f",  # ISO with microseconds
            "%Y-%m-%dT%H:%M:%S",      # ISO without microseconds
            "%Y-%m-%d %H:%M:%S",      # Standard datetime
            "%Y-%m-%d %H:%M",         # Without seconds
            "%Y-%m-%d",               # Date only
        ]

        # 处理测试数据格式: "'2024-10-01 08:00' Tuesday"
        # 提取引号内的时间部分
        quoted_match = re.search(r"'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)'", clean_str)
        if quoted_match:
            clean_str = quoted_match.group(1)

        for fmt in formats:
            try:
                return dt.strptime(clean_str, fmt)
            except ValueError:
                continue

        return None

    def get_messages_by_time_range(
        self,
        start_time: str,
        end_time: str,
        role_filter: Optional[str] = None,
    ) -> List[Message]:
        """
        按时间范围检索消息。

        Args:
            start_time: 开始时间 (支持多种格式)
            end_time: 结束时间
            role_filter: 可选的角色过滤 ("user" | "assistant" | None)

        Returns:
            时间范围内的消息列表（按时间排序）
        """
        self._ensure_conversations_loaded()

        start_dt = self._parse_flexible_timestamp(start_time)
        end_dt = self._parse_flexible_timestamp(end_time)

        if not start_dt or not end_dt:
            return []

        messages = []
        for conv in self._conversations.values():
            for msg in conv.messages:
                msg_dt = self._parse_flexible_timestamp(msg.timestamp)
                if not msg_dt:
                    continue

                if start_dt <= msg_dt <= end_dt:
                    if role_filter is None or msg.role == role_filter:
                        messages.append(msg)

        # 按时间排序
        messages.sort(key=lambda m: m.timestamp)
        return messages

    # ==================== 统计接口 ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计。

        Returns:
            统计信息
        """
        vector_stats = self._vector_store.get_stats()
        index_stats = self._knowledge_index.get_stats()
        profile = self._profile_store.get()

        return {
            "user_id": self.user_id,
            "vector_store": vector_stats,
            "knowledge_index": index_stats,
            "conversations_count": len(self._conversations),
            "current_topic": self.current_topic,
            "current_messages_count": len(self.current_messages),
            "profile_exists": self._profile_store.exists(),
            "profile_interests_count": len(profile.interests),
            "all_topics": self.get_all_topics(),
        }

    # ==================== 私有方法 ====================

    def _ensure_conversations_loaded(self) -> None:
        """确保对话记录已加载"""
        if not self._conversations_loaded:
            self._load_conversations()

    def _load_conversations(self) -> None:
        """从文件加载对话记录"""
        if self._conversations_path.exists():
            try:
                with open(self._conversations_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for topic, conv_data in data.get("conversations", {}).items():
                        self._conversations[topic] = Conversation.from_dict(conv_data)
                self._conversations_loaded = True  # 标记加载成功
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[MemorySystem] 加载对话记录失败: {e}")
                # 备份损坏的文件
                self._backup_corrupted_file()
                self._conversations_loaded = False
        else:
            # 文件不存在是正常情况（新用户）
            self._conversations_loaded = True

    def _backup_corrupted_file(self) -> None:
        """备份损坏的对话文件"""
        if self._conversations_path.exists():
            backup_path = self._conversations_path.with_suffix(
                f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            try:
                import shutil
                shutil.copy(self._conversations_path, backup_path)
                print(f"[MemorySystem] 已备份损坏文件到: {backup_path}")
            except Exception as e:
                print(f"[MemorySystem] 备份文件失败: {e}")

    def _save_conversations(self) -> None:
        """保存对话记录到文件（带安全检查和原子写入）"""
        # 安全检查：如果加载失败且当前对话为空，拒绝保存以防覆盖
        if not self._conversations_loaded and not self._conversations:
            print("[MemorySystem] 警告: 加载未成功，拒绝空写入以防止数据丢失")
            return

        self._conversations_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "user_id": self.user_id,
            "conversations": {
                topic: conv.to_dict()
                for topic, conv in self._conversations.items()
            },
            "last_updated": datetime.now().isoformat(),
        }

        # 使用原子写入：先写临时文件，再重命名
        temp_path = self._conversations_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # 原子替换
            temp_path.replace(self._conversations_path)
            self._conversations_loaded = True  # 更新状态
        except Exception as e:
            print(f"[MemorySystem] 保存对话失败: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def _update_conversation(
        self,
        topic: str,
        messages: List[Message],
        extraction_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        更新或创建话题对话。

        Args:
            topic: 话题名
            messages: 消息列表
            extraction_data: 提取的信息（summary, key_info, user_preferences等）
        """
        import uuid

        extraction_data = extraction_data or {}

        if topic in self._conversations:
            # 已存在：追加消息和信息
            conv = self._conversations[topic]
            conv.messages.extend(messages)
            conv.updated_at = datetime.now().isoformat()
            if extraction_data.get("summary"):
                conv.summary = extraction_data["summary"]
            conv.key_info = list(set(
                conv.key_info + extraction_data.get("key_info", [])
            ))
            conv.user_preferences = list(set(
                conv.user_preferences + extraction_data.get("user_preferences", [])
            ))
        else:
            # 新建
            conv = Conversation(
                topic=topic,
                messages=messages,
                summary=extraction_data.get("summary", ""),
                key_info=extraction_data.get("key_info", []),
                user_preferences=extraction_data.get("user_preferences", []),
            )
            self._conversations[topic] = conv

        # 添加到向量库
        doc_id = str(uuid.uuid4())
        self._vector_store.add_conversation(
            doc_id=doc_id,
            content=extraction_data.get("summary", f"关于{topic}的对话"),
            metadata={"topic": topic},
        )

        self._save_conversations()

    def _result_to_knowledge_item(
        self,
        result: Dict[str, Any],
        index_record: Dict[str, Any],
    ) -> KnowledgeItem:
        """将搜索结果转换为 KnowledgeItem"""
        metadata = result.get("metadata", {})
        return KnowledgeItem(
            id=result.get("id", index_record.get("id", "")),
            content=metadata.get("full_content", result.get("content", "")),
            summary=result.get("content", ""),
            topic=metadata.get("topic", index_record.get("topic", "")),
            source_url=metadata.get("source_url", index_record.get("source_url", "")),
            title=metadata.get("title", index_record.get("title", "")),
            status=KnowledgeStatus(index_record.get("status", "active")),
            content_hash=index_record.get("content_hash", ""),
        )

    # ==================== 存储层访问（供高级使用）====================

    @property
    def vector_store(self) -> VectorStore:
        """获取向量存储实例"""
        return self._vector_store

    @property
    def knowledge_index(self) -> KnowledgeIndex:
        """获取知识索引实例"""
        return self._knowledge_index

    @property
    def profile_store(self) -> ProfileStore:
        """获取画像存储实例"""
        return self._profile_store

    @property
    def fact_store(self) -> FactStore:
        """获取事实存储实例"""
        return self._fact_store

    # ==================== 用户事实相关方法 ====================

    def add_facts(self, facts_data: List[Dict[str, Any]], source_topic: str = "") -> int:
        """
        批量添加用户事实。

        Args:
            facts_data: 事实数据列表
            source_topic: 来源话题

        Returns:
            添加的事实数量
        """
        return self._fact_store.add_facts_from_extraction(facts_data, source_topic)

    def get_all_facts(self) -> List[UserFact]:
        """
        获取所有用户事实。

        Returns:
            所有事实列表
        """
        return self._fact_store.get_all_facts()

    def format_facts_for_prompt(self) -> str:
        """
        格式化用户事实为 Prompt 文本。

        Returns:
            格式化的事实文本
        """
        return self._fact_store.format_for_prompt()

    # ==================== 调研相关方法 ====================

    def get_existing_research(
        self,
        topic: str,
        similarity_threshold: float = 0.75,
    ) -> Optional[Dict[str, Any]]:
        """
        查找与主题相关的已有调研。

        通过向量相似度匹配已有的调研主题，如果找到相关调研，
        返回调研事实和覆盖度信息。

        Args:
            topic: 调研主题
            similarity_threshold: 相似度阈值

        Returns:
            调研信息字典，包含:
            - related_topic: 相关的已有调研主题
            - similarity: 相似度分数
            - facts: 已有事实列表
            - coverage_score: 覆盖度分数 (0-100)
            - subtopics_covered: 已覆盖的子主题
            如果没有找到相关调研，返回 None
        """
        # 获取所有已调研主题
        research_topics = self._knowledge_index.get_research_topics()

        if not research_topics:
            return None

        # 通过向量搜索找到最相似的调研主题
        # 使用主题名作为查询
        similar_results = self._vector_store.search_knowledge(topic, n_results=5)

        # 从搜索结果中匹配调研主题
        research_topic_set = {rt["topic"] for rt in research_topics}
        best_match = None
        best_similarity = 0.0

        for result in similar_results:
            metadata = result.get("metadata", {})
            result_topic = metadata.get("research_topic") or metadata.get("topic", "")

            # 检查是否是调研主题
            if result_topic in research_topic_set:
                # 计算相似度（1 - distance）
                distance = result.get("distance", 1.0)
                similarity = 1.0 - min(distance, 1.0)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = result_topic

        # 也检查主题名的直接匹配
        for rt in research_topics:
            if topic.lower() in rt["topic"].lower() or rt["topic"].lower() in topic.lower():
                if best_match is None or best_similarity < 0.9:
                    best_match = rt["topic"]
                    best_similarity = 0.9
                    break

        if not best_match or best_similarity < similarity_threshold:
            return None

        # 获取该主题的调研事实和覆盖度
        facts = self._knowledge_index.get_facts_by_research_topic(best_match)
        coverage = self._knowledge_index.get_topic_coverage(best_match)

        return {
            "related_topic": best_match,
            "similarity": best_similarity,
            "facts": facts,
            "coverage_score": coverage["coverage_score"],
            "subtopics_covered": coverage["subtopics_covered"],
            "fact_count": coverage["fact_count"],
            "source_count": coverage["source_count"],
            "last_updated": coverage["last_updated"],
        }

    def add_research_knowledge(
        self,
        topic: str,
        task_id: str,
        facts: List[Dict[str, Any]],
        sources: List[Any],
    ) -> Dict[str, int]:
        """
        添加调研知识（带调研标记）。

        通过 submit_update 统一写入，但附加调研元数据。

        Args:
            topic: 调研主题
            task_id: 调研任务ID
            facts: 事实列表，每项包含 content, summary, source_url, title, confidence
            sources: 来源列表

        Returns:
            统计信息 {"added": int, "skipped": int, "merged": int}
        """
        import hashlib
        import uuid

        stats = {"added": 0, "skipped": 0, "merged": 0}

        for fact in facts:
            content = fact.get("content", "")
            if not content:
                continue

            # 计算内容哈希
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # 检查是否已存在
            if self._knowledge_index.exists_by_hash(content_hash):
                stats["skipped"] += 1
                continue

            # 创建文档ID
            doc_id = str(uuid.uuid4())

            # 插入到知识索引
            self._knowledge_index.insert_research_fact({
                "id": doc_id,
                "content_hash": content_hash,
                "topic": fact.get("subtopic", topic),
                "research_topic": topic,
                "research_task_id": task_id,
                "source_url": fact.get("source_url", ""),
                "source_title": fact.get("source_title", fact.get("title", "")),
                "summary_preview": fact.get("summary", content[:200]),
                "fact_confidence": fact.get("confidence", 0.8),
                "publish_date": fact.get("publish_date", ""),
                "access_date": fact.get("access_date", ""),
            })

            # 插入到向量库
            self._vector_store.add_knowledge(
                doc_id=doc_id,
                content=content[:2000],  # 限制长度
                metadata={
                    "topic": fact.get("subtopic", topic),
                    "research_topic": topic,
                    "research_task_id": task_id,
                    "source_url": fact.get("source_url", ""),
                    "source_title": fact.get("source_title", fact.get("title", "")),
                    "full_content": content,
                    "publish_date": fact.get("publish_date", ""),
                    "access_date": fact.get("access_date", ""),
                },
            )

            stats["added"] += 1

        return stats

    def get_research_topics(self) -> List[Dict[str, Any]]:
        """
        获取所有已调研过的主题。

        Returns:
            调研主题列表
        """
        return self._knowledge_index.get_research_topics()

    def get_research_coverage(self, topic: str) -> Dict[str, Any]:
        """
        获取某主题的调研覆盖情况。

        Args:
            topic: 调研主题

        Returns:
            覆盖情况字典
        """
        return self._knowledge_index.get_topic_coverage(topic)

    def get_research_facts(self, topic: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取某调研主题下的所有事实。

        Args:
            topic: 调研主题
            limit: 最大返回数量

        Returns:
            事实列表
        """
        return self._knowledge_index.get_facts_by_research_topic(topic, limit)
