"""
ChromaDB 向量存储。

提供基于 ChromaDB 的向量存储功能，支持三种类型的集合：
- knowledge: 知识库
- conversations: 对话摘要
- preferences: 用户偏好
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import StorageConfig


class VectorStore:
    """
    ChromaDB 向量存储。

    使用 ChromaDB 的默认 embedding 函数 (sentence-transformers)。
    支持知识库、对话记录、用户偏好三种类型的向量存储。
    """

    def __init__(self, user_id: str, storage_config: StorageConfig):
        """
        初始化向量存储。

        Args:
            user_id: 用户 ID
            storage_config: 存储配置
        """
        self.user_id = user_id
        self.storage_config = storage_config
        self.persist_dir = str(storage_config.get_vector_db_path(user_id))

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 创建/获取集合
        self._knowledge = self.client.get_or_create_collection(
            name="knowledge",
            metadata={"description": "知识库存储"},
        )

        self._conversations = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "对话记录存储"},
        )

        self._preferences = self.client.get_or_create_collection(
            name="preferences",
            metadata={"description": "用户偏好存储"},
        )

    # ==================== 查询方法 ====================

    def search_knowledge(
        self,
        query: str,
        n_results: int = 5,
        topic_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索知识库。

        Args:
            query: 搜索查询
            n_results: 返回结果数量
            topic_filter: 可选的话题过滤

        Returns:
            匹配的知识列表，每项包含 content, metadata, distance
        """
        if self._knowledge.count() == 0:
            return []

        where_filter = None
        if topic_filter:
            where_filter = {"topic": topic_filter}

        results = self._knowledge.query(
            query_texts=[query],
            n_results=min(n_results, self._knowledge.count()),
            where=where_filter,
        )

        return self._format_results(results)

    def search_conversations(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        搜索对话记录。

        Args:
            query: 搜索查询
            n_results: 返回结果数量

        Returns:
            匹配的对话列表
        """
        if self._conversations.count() == 0:
            return []

        results = self._conversations.query(
            query_texts=[query],
            n_results=min(n_results, self._conversations.count()),
        )

        return self._format_results(results)

    def search_preferences(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        搜索用户偏好。

        Args:
            query: 搜索查询
            n_results: 返回结果数量

        Returns:
            匹配的偏好列表
        """
        if self._preferences.count() == 0:
            return []

        results = self._preferences.query(
            query_texts=[query],
            n_results=min(n_results, self._preferences.count()),
        )

        return self._format_results(results)

    def get_knowledge_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取知识条目。

        Args:
            doc_id: 文档 ID

        Returns:
            知识条目，或 None
        """
        try:
            results = self._knowledge.get(ids=[doc_id])
            if results and results.get("documents"):
                return {
                    "id": doc_id,
                    "content": results["documents"][0],
                    "metadata": results.get("metadatas", [{}])[0],
                }
        except Exception:
            pass
        return None

    # ==================== 写入方法 (仅供 MemoryUpdateAgent 调用) ====================

    def add_knowledge(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        添加知识到知识库。

        Args:
            doc_id: 文档 ID
            content: 文档内容 (用于 embedding)
            metadata: 元数据
        """
        # 过滤 metadata，只保留标量类型
        filtered_metadata = self._filter_metadata(metadata)
        filtered_metadata["timestamp"] = datetime.now().isoformat()
        filtered_metadata["type"] = "knowledge"

        self._knowledge.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[filtered_metadata],
        )

    def add_conversation(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        添加对话记录。

        Args:
            doc_id: 文档 ID
            content: 对话摘要内容
            metadata: 元数据
        """
        filtered_metadata = self._filter_metadata(metadata)
        filtered_metadata["timestamp"] = datetime.now().isoformat()
        filtered_metadata["type"] = "conversation"

        self._conversations.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[filtered_metadata],
        )

    def add_preference(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        添加用户偏好。

        Args:
            doc_id: 文档 ID
            content: 偏好描述
            metadata: 元数据
        """
        filtered_metadata = self._filter_metadata(metadata)
        filtered_metadata["timestamp"] = datetime.now().isoformat()
        filtered_metadata["type"] = "preference"

        self._preferences.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[filtered_metadata],
        )

    def delete(self, doc_id: str, collection: str) -> bool:
        """
        从指定集合中删除文档。

        Args:
            doc_id: 文档 ID
            collection: 集合名称 ("knowledge" | "conversations" | "preferences")

        Returns:
            是否删除成功
        """
        try:
            col = self._get_collection(collection)
            if col:
                col.delete(ids=[doc_id])
                return True
        except Exception:
            pass
        return False

    def update(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        collection: str,
    ) -> bool:
        """
        更新文档。

        由于 ChromaDB 不支持直接更新，采用删除+添加的方式。

        Args:
            doc_id: 文档 ID
            content: 新内容
            metadata: 新元数据
            collection: 集合名称

        Returns:
            是否更新成功
        """
        try:
            col = self._get_collection(collection)
            if col:
                # 删除旧文档
                try:
                    col.delete(ids=[doc_id])
                except Exception:
                    pass

                # 添加新文档
                filtered_metadata = self._filter_metadata(metadata)
                filtered_metadata["timestamp"] = datetime.now().isoformat()
                filtered_metadata["updated"] = True

                col.add(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[filtered_metadata],
                )
                return True
        except Exception:
            pass
        return False

    # ==================== 统计方法 ====================

    def get_stats(self) -> Dict[str, int]:
        """
        获取各集合的文档数量统计。

        Returns:
            各集合的文档数量
        """
        return {
            "knowledge": self._knowledge.count(),
            "conversations": self._conversations.count(),
            "preferences": self._preferences.count(),
        }

    # ==================== 私有方法 ====================

    def _get_collection(self, name: str):
        """根据名称获取集合"""
        if name == "knowledge":
            return self._knowledge
        elif name == "conversations":
            return self._conversations
        elif name == "preferences":
            return self._preferences
        return None

    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤 metadata，只保留 ChromaDB 支持的标量类型。

        Args:
            metadata: 原始元数据

        Returns:
            过滤后的元数据
        """
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                # 限制字符串长度以避免 ChromaDB 限制
                if isinstance(value, str) and len(value) > 10000:
                    filtered[key] = value[:10000]
                    filtered[f"{key}_truncated"] = True
                else:
                    filtered[key] = value
        return filtered

    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """
        格式化 ChromaDB 查询结果。

        Args:
            results: ChromaDB 原始查询结果

        Returns:
            格式化后的结果列表
        """
        formatted = []
        if results and results.get("documents"):
            docs = results["documents"][0] if results["documents"] else []
            ids = results["ids"][0] if results.get("ids") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i, doc in enumerate(docs):
                formatted.append({
                    "id": ids[i] if i < len(ids) else None,
                    "content": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                })

        return formatted
