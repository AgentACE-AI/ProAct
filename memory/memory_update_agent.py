"""
记忆更新 Agent。

统一处理所有记忆写入请求，包括：
- 知识添加（带查重）
- 对话记录添加
- 用户画像更新
- 知识合并/废弃
"""

import hashlib
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.config import AgentModelConfig
from core.models import (
    Conversation,
    KnowledgeItem,
    KnowledgeStatus,
    MemoryUpdateRequest,
    MemoryUpdateResult,
    Message,
    UpdateAction,
    UpdateType,
)

if TYPE_CHECKING:
    from memory.knowledge_index import KnowledgeIndex
    from memory.profile_store import ProfileStore
    from memory.vector_store import VectorStore
    from tools.llm_client import LLMClient


class MemoryUpdateAgent:
    """
    记忆更新 Agent。

    职责:
    1. 接收来自各模块的更新请求
    2. 处理知识查重 (调用 LLM 决策)
    3. 执行实际的存储操作

    注意: 此 Agent 是一个纯粹的执行者，
    不具备主动整理/压缩记忆的能力。
    """

    # 相似度阈值（向量距离，越小越相似）
    SIMILARITY_THRESHOLD = 0.15

    def __init__(
        self,
        vector_store: "VectorStore",
        knowledge_index: "KnowledgeIndex",
        profile_store: "ProfileStore",
        llm_client: "LLMClient",
        config: AgentModelConfig,
    ):
        """
        初始化记忆更新 Agent。

        Args:
            vector_store: 向量存储实例
            knowledge_index: 知识索引实例
            profile_store: 画像存储实例
            llm_client: LLM 客户端
            config: Agent 模型配置
        """
        self._vector_store = vector_store
        self._knowledge_index = knowledge_index
        self._profile_store = profile_store
        self._llm = llm_client
        self._config = config

    def process(self, request: MemoryUpdateRequest) -> MemoryUpdateResult:
        """
        处理更新请求。

        根据 request.update_type 分发到对应处理方法。

        Args:
            request: 更新请求

        Returns:
            更新结果
        """
        try:
            if request.update_type == UpdateType.ADD_KNOWLEDGE:
                return self._add_knowledge(request)
            elif request.update_type == UpdateType.ADD_CONVERSATION:
                return self._add_conversation(request)
            elif request.update_type == UpdateType.UPDATE_PROFILE:
                return self._update_profile(request)
            elif request.update_type == UpdateType.MERGE_KNOWLEDGE:
                return self._merge_knowledge(request)
            elif request.update_type == UpdateType.DEPRECATE_KNOWLEDGE:
                return self._deprecate_knowledge(request)
            else:
                return MemoryUpdateResult(
                    success=False,
                    action=UpdateAction.SKIPPED,
                    message=f"未知的更新类型: {request.update_type}",
                )
        except Exception as e:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message=f"处理请求时出错: {str(e)}",
            )

    def _add_knowledge(self, request: MemoryUpdateRequest) -> MemoryUpdateResult:
        """
        添加知识。

        流程:
        1. 计算 content hash，快速去重
        2. 如果 hash 不存在，语义相似度查重
        3. 如果发现相似，调用 LLM 决策 (skip/replace/merge)
        4. 执行决策

        Args:
            request: 更新请求

        Returns:
            更新结果
        """
        data = request.data
        content = data.get("content", "")
        summary = data.get("summary", content)
        topic = data.get("topic", "")
        source_url = data.get("source_url", "")
        title = data.get("title", "")

        if not content:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message="内容为空",
            )

        # 1. Hash 快速去重
        content_hash = self._compute_hash(content)
        if self._knowledge_index.exists_by_hash(content_hash):
            return MemoryUpdateResult(
                success=True,
                action=UpdateAction.SKIPPED,
                message="完全重复内容 (哈希匹配)",
            )

        # 2. 语义相似度查重
        similar_docs = self._vector_store.search_knowledge(
            query=content[:2000],  # 限制查询长度
            n_results=1,
        )

        if similar_docs and similar_docs[0].get("distance", 1.0) < self.SIMILARITY_THRESHOLD:
            existing_doc = similar_docs[0]

            # 3. LLM 决策
            decision = self._llm_decide_duplicate(
                existing=existing_doc,
                new_content=content,
            )

            if decision == "skip":
                return MemoryUpdateResult(
                    success=True,
                    action=UpdateAction.SKIPPED,
                    message="LLM 决策跳过: 内容重复或无新增价值",
                )
            elif decision == "replace":
                # 替换旧内容
                return self._do_replace(
                    existing_doc=existing_doc,
                    new_content=content,
                    new_summary=summary,
                    new_topic=topic,
                    new_source_url=source_url,
                    new_title=title,
                    content_hash=content_hash,
                )
            elif decision == "merge":
                # 合并内容
                merged_content = self._llm_merge(
                    existing=existing_doc.get("content", ""),
                    new_content=content,
                    topic=topic,
                )
                return self._do_replace(
                    existing_doc=existing_doc,
                    new_content=merged_content,
                    new_summary=merged_content[:500],
                    new_topic=topic,
                    new_source_url=f"{existing_doc.get('metadata', {}).get('source_url', '')}; {source_url}".strip("; "),
                    new_title=title or existing_doc.get("metadata", {}).get("title", ""),
                    content_hash=self._compute_hash(merged_content),
                    is_merge=True,
                )

        # 4. 无重复，直接添加
        return self._do_add(
            content=content,
            summary=summary,
            topic=topic,
            source_url=source_url,
            title=title,
            content_hash=content_hash,
        )

    def _add_conversation(self, request: MemoryUpdateRequest) -> MemoryUpdateResult:
        """
        添加对话记录。

        Args:
            request: 更新请求

        Returns:
            更新结果
        """
        data = request.data
        print("----------------notice`-------------------")
        print(data)
        print(type(data))
        print("-----------------------------------------")
        topic = data.get("topic", "")
        messages = data.get("messages", [])
        summary = data.get("summary", "")
        key_info = data.get("key_info", [])
        user_preferences = data.get("user_preferences", [])

        if not topic or not messages:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message="话题或消息为空",
            )

        # 生成文档 ID
        doc_id = str(uuid.uuid4())

        # 准备内容（用于 embedding）
        content = summary if summary else self._format_messages_for_embedding(messages)

        # 添加到向量库
        self._vector_store.add_conversation(
            doc_id=doc_id,
            content=content,
            metadata={
                "topic": topic,
                "message_count": len(messages) if isinstance(messages, list) else 0,
                "key_info": "; ".join(key_info) if key_info else "",
                "user_preferences": "; ".join(user_preferences) if user_preferences else "",
            },
        )

        return MemoryUpdateResult(
            success=True,
            action=UpdateAction.ADDED,
            affected_ids=[doc_id],
            message="对话记录已添加",
        )

    def _update_profile(self, request: MemoryUpdateRequest) -> MemoryUpdateResult:
        """
        更新用户画像。

        Args:
            request: 更新请求

        Returns:
            更新结果
        """
        data = request.data

        if not data:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message="更新数据为空",
            )

        # 增量更新画像
        self._profile_store.update(data)

        return MemoryUpdateResult(
            success=True,
            action=UpdateAction.ADDED,
            message="用户画像已更新",
        )

    def _merge_knowledge(self, request: MemoryUpdateRequest) -> MemoryUpdateResult:
        """
        合并指定的知识条目。

        Args:
            request: 更新请求

        Returns:
            更新结果
        """
        data = request.data
        source_ids = data.get("source_ids", [])
        merged_content = data.get("merged_content", "")

        if len(source_ids) < 2:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message="至少需要2条记录才能合并",
            )

        # 获取源记录内容
        contents = []
        sources = []
        topic = ""

        for doc_id in source_ids:
            doc = self._vector_store.get_knowledge_by_id(doc_id)
            if doc:
                contents.append(doc.get("content", ""))
                metadata = doc.get("metadata", {})
                sources.append(metadata.get("source_url", ""))
                if not topic:
                    topic = metadata.get("topic", "")

        if len(contents) < 2:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message="无法获取足够的记录内容",
            )

        # 如果没有提供合并内容，用 LLM 合并
        if not merged_content:
            merged_content = self._llm_merge_multiple(contents, topic)

        # 创建新记录
        new_doc_id = str(uuid.uuid4())
        combined_sources = "; ".join(filter(None, sources))
        content_hash = self._compute_hash(merged_content)

        # 添加到向量库
        self._vector_store.add_knowledge(
            doc_id=new_doc_id,
            content=merged_content,
            metadata={
                "topic": topic,
                "source_url": combined_sources,
                "source_type": "merged",
                "merged_from": source_ids,
            },
        )

        # 添加到索引
        self._knowledge_index.insert({
            "id": new_doc_id,
            "content_hash": content_hash,
            "topic": topic,
            "source_url": combined_sources,
            "source_type": "merged",
            "summary_preview": merged_content[:200],
        })

        # 标记旧记录为已合并
        for doc_id in source_ids:
            self._knowledge_index.update_status(
                doc_id=doc_id,
                status=KnowledgeStatus.MERGED.value,
                merged_into=new_doc_id,
            )

        return MemoryUpdateResult(
            success=True,
            action=UpdateAction.MERGED,
            affected_ids=[new_doc_id] + source_ids,
            message=f"合并了 {len(source_ids)} 条记录",
        )

    def _deprecate_knowledge(self, request: MemoryUpdateRequest) -> MemoryUpdateResult:
        """
        废弃知识条目。

        Args:
            request: 更新请求

        Returns:
            更新结果
        """
        data = request.data
        knowledge_id = data.get("knowledge_id", "")
        reason = data.get("reason", "")

        if not knowledge_id:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message="知识 ID 为空",
            )

        # 更新索引状态
        success = self._knowledge_index.update_status(
            doc_id=knowledge_id,
            status=KnowledgeStatus.DEPRECATED.value,
        )

        if success:
            return MemoryUpdateResult(
                success=True,
                action=UpdateAction.REPLACED,
                affected_ids=[knowledge_id],
                message=f"知识已废弃: {reason}",
            )
        else:
            return MemoryUpdateResult(
                success=False,
                action=UpdateAction.SKIPPED,
                message="未找到指定的知识记录",
            )

    # ==================== 私有方法 ====================

    def _compute_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _do_add(
        self,
        content: str,
        summary: str,
        topic: str,
        source_url: str,
        title: str,
        content_hash: str,
    ) -> MemoryUpdateResult:
        """执行添加操作"""
        doc_id = str(uuid.uuid4())

        # 添加到向量库
        self._vector_store.add_knowledge(
            doc_id=doc_id,
            content=summary if summary else content[:2000],
            metadata={
                "topic": topic,
                "source_url": source_url,
                "title": title,
                "source_type": "web_search",
                "full_content": content[:10000] if len(content) > 2000 else "",
            },
        )

        # 添加到索引
        self._knowledge_index.insert({
            "id": doc_id,
            "content_hash": content_hash,
            "topic": topic,
            "source_url": source_url,
            "source_type": "web_search",
            "title": title,
            "summary_preview": (summary if summary else content)[:200],
        })

        return MemoryUpdateResult(
            success=True,
            action=UpdateAction.ADDED,
            affected_ids=[doc_id],
            message="新知识已添加",
        )

    def _do_replace(
        self,
        existing_doc: Dict[str, Any],
        new_content: str,
        new_summary: str,
        new_topic: str,
        new_source_url: str,
        new_title: str,
        content_hash: str,
        is_merge: bool = False,
    ) -> MemoryUpdateResult:
        """执行替换操作"""
        existing_id = existing_doc.get("id")

        if existing_id:
            # 更新向量库
            self._vector_store.update(
                doc_id=existing_id,
                content=new_summary if new_summary else new_content[:2000],
                metadata={
                    "topic": new_topic,
                    "source_url": new_source_url,
                    "title": new_title,
                    "source_type": "merged" if is_merge else "web_search",
                    "full_content": new_content[:10000] if len(new_content) > 2000 else "",
                },
                collection="knowledge",
            )

            # 更新索引
            self._knowledge_index.update_record(
                doc_id=existing_id,
                content_hash=content_hash,
                summary_preview=(new_summary if new_summary else new_content)[:200],
                source_url=new_source_url,
            )

            action = UpdateAction.MERGED if is_merge else UpdateAction.REPLACED
            message = "内容已合并" if is_merge else "内容已替换"

            return MemoryUpdateResult(
                success=True,
                action=action,
                affected_ids=[existing_id],
                message=message,
            )
        else:
            # 旧数据没有 ID，创建新记录
            return self._do_add(
                content=new_content,
                summary=new_summary,
                topic=new_topic,
                source_url=new_source_url,
                title=new_title,
                content_hash=content_hash,
            )

    def _llm_decide_duplicate(
        self,
        existing: Dict[str, Any],
        new_content: str,
    ) -> str:
        """
        LLM 决策如何处理重复内容。

        Args:
            existing: 现有文档
            new_content: 新内容

        Returns:
            决策结果: "skip" | "replace" | "merge"
        """
        existing_content = existing.get("content", "")
        existing_metadata = existing.get("metadata", {})
        existing_source = existing_metadata.get("source_url", "")
        existing_time = existing_metadata.get("timestamp", "")

        prompt = f"""You are a knowledge base manager. The knowledge base already contains one record, and a new piece of highly similar content is about to be added.
Determine how it should be handled.

[Existing Record]
Content: {existing_content[:1500]}
Source: {existing_source}
Stored at: {existing_time}

[New Content]
Content: {new_content[:1500]}

Analyze the two items and choose a strategy:
1. skip - The new content is duplicate or adds no meaningful value, so do not add it
2. replace - The new content is more accurate, newer, or more complete, so it should replace the existing record
3. merge - Both contain useful information (for example, complementary perspectives) and should be merged into a more complete record

Reply in JSON format:
{{
    "strategy": "skip|replace|merge",
    "reason": "Brief reason (within 20 words)"
}}"""

        try:
            result = self._llm.chat_json(
                messages=[
                    {"role": "system", "content": "You are a knowledge base manager helping decide how to handle duplicate content. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                model=self._config.model,
                temperature=self._config.temperature,
            )
            return result.get("strategy", "skip")
        except Exception as e:
            print(f"[MemoryUpdateAgent] LLM 决策失败: {e}")
            return "skip"

    def _llm_merge(
        self,
        existing: str,
        new_content: str,
        topic: str,
    ) -> str:
        """LLM 合并两段内容"""
        prompt = f"""Merge the following two pieces of content about "{topic}" into one complete, non-redundant knowledge record.

[Content A]
{existing[:2000]}

 [Content B]
{new_content[:2000]}

Requirements:
1. Preserve the unique information from both pieces and remove duplication
2. If there is a conflict, keep the more accurate or newer information
3. Keep the writing fluent and the structure clear
4. Output only the merged content, with no extra explanation"""

        try:
            merged = self._llm.chat(
                messages=[
                    {"role": "system", "content": "You are an information organization assistant who excels at merging and structuring content."},
                    {"role": "user", "content": prompt},
                ],
                model=self._config.model,
                temperature=0.3,
            )
            return merged.strip()
        except Exception as e:
            print(f"[MemoryUpdateAgent] LLM 合并失败: {e}")
            return f"{existing}\n\n---\n\n{new_content}"

    def _llm_merge_multiple(self, contents: List[str], topic: str) -> str:
        """LLM 合并多条内容"""
        contents_text = "\n\n---\n\n".join([
            f"[Content {i+1}]\n{c[:1500]}" for i, c in enumerate(contents)
        ])

        prompt = f"""Merge the following {len(contents)} pieces of content about "{topic}" into one complete, non-redundant knowledge record.

{contents_text}

Requirements:
1. Preserve all unique information and remove duplication
2. If there is a conflict, keep the more accurate or newer information
3. Keep the writing fluent and the structure clear
4. Output only the merged content"""

        try:
            merged = self._llm.chat(
                messages=[
                    {"role": "system", "content": "You are an information organization assistant who excels at merging and structuring content."},
                    {"role": "user", "content": prompt},
                ],
                model=self._config.model,
                temperature=0.3,
            )
            return merged.strip()
        except Exception as e:
            print(f"[MemoryUpdateAgent] LLM 多条合并失败: {e}")
            return "\n\n".join(contents)

    def _format_messages_for_embedding(self, messages: List[Any]) -> str:
        """将消息列表格式化为用于 embedding 的文本"""
        if not messages:
            return ""

        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            elif isinstance(msg, Message):
                role = msg.role
                content = msg.content
            else:
                continue

            formatted.append(f"{role}: {content}")

        # 限制长度
        result = "\n".join(formatted)
        return result[:2000]
