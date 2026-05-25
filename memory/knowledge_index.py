"""
SQLite 知识索引。

配合向量库使用，提供：
- 知识记录的元数据管理
- 状态追踪（active/merged/deprecated）
- 内容哈希快速去重
- 合并历史记录
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from core.config import StorageConfig
from core.models import KnowledgeStatus


class KnowledgeIndex:
    """
    SQLite 知识索引。

    表结构:
    - id: 主键（与向量库 doc_id 对应）
    - content_hash: 内容哈希（用于快速去重）
    - topic: 话题
    - source_url: 来源 URL
    - source_type: 来源类型 ("web_search" | "conversation" | "merged" | "deep_research")
    - title: 标题
    - summary_preview: 摘要预览（前 200 字）
    - status: 状态 (active/merged/deprecated)
    - merged_into: 合并目标 ID（如果 status=merged）
    - merged_from: 合并来源 ID 列表（JSON，如果是合并产生的）
    - created_at: 创建时间
    - updated_at: 更新时间
    - update_count: 更新次数
    - research_topic: 所属调研主题（可为空）
    - research_task_id: 调研任务ID（可为空）
    - is_research_fact: 是否为调研提取的事实
    - fact_confidence: 事实置信度 (0-1)
    """

    def __init__(self, user_id: str, storage_config: StorageConfig):
        """
        初始化知识索引。

        Args:
            user_id: 用户 ID
            storage_config: 存储配置
        """
        self.user_id = user_id
        self.storage_config = storage_config
        self.db_path = storage_config.get_knowledge_index_path(user_id)
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    source_url TEXT,
                    source_type TEXT NOT NULL DEFAULT 'web_search',
                    title TEXT,
                    summary_preview TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    merged_into TEXT,
                    merged_from TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    update_count INTEGER DEFAULT 0,
                    research_topic TEXT,
                    research_task_id TEXT,
                    is_research_fact INTEGER DEFAULT 0,
                    fact_confidence REAL DEFAULT 0.8
                )
            """)

            # 创建索引
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_hash ON knowledge(content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_topic ON knowledge(topic)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON knowledge(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source_url ON knowledge(source_url)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_updated_at ON knowledge(updated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_topic ON knowledge(research_topic)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_is_research_fact ON knowledge(is_research_fact)"
            )

            conn.commit()

        # 确保调研相关列存在（兼容已有数据库）
        self._ensure_research_columns()

    # ==================== 查询方法 ====================

    def exists_by_hash(self, content_hash: str) -> bool:
        """
        检查内容哈希是否已存在（活跃状态）。

        Args:
            content_hash: 内容哈希

        Returns:
            是否存在
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM knowledge WHERE content_hash = ? AND status = ? LIMIT 1",
                (content_hash, KnowledgeStatus.ACTIVE.value),
            )
            return cursor.fetchone() is not None

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取记录。

        Args:
            doc_id: 文档 ID

        Returns:
            记录字典，或 None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM knowledge WHERE id = ?",
                (doc_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def get_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        根据内容哈希获取活跃记录。

        Args:
            content_hash: 内容哈希

        Returns:
            记录字典，或 None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM knowledge WHERE content_hash = ? AND status = ?",
                (content_hash, KnowledgeStatus.ACTIVE.value),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def get_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        根据 URL 获取活跃记录。

        Args:
            url: 来源 URL

        Returns:
            记录字典，或 None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM knowledge WHERE source_url = ? AND status = ?",
                (url, KnowledgeStatus.ACTIVE.value),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def get_active_count(self) -> int:
        """
        获取活跃记录数量。

        Returns:
            活跃记录数
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM knowledge WHERE status = ?",
                (KnowledgeStatus.ACTIVE.value,),
            )
            return cursor.fetchone()[0]

    def get_all(
        self,
        status: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        获取所有记录（支持过滤）。

        Args:
            status: 状态过滤
            topic: 话题过滤（模糊匹配）
            limit: 最大返回数量

        Returns:
            记录列表
        """
        conditions = []
        values = []

        if status:
            conditions.append("status = ?")
            values.append(status)

        if topic:
            conditions.append("topic LIKE ?")
            values.append(f"%{topic}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        values.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"SELECT * FROM knowledge WHERE {where_clause} ORDER BY updated_at DESC LIMIT ?",
                values,
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_topics(self) -> List[str]:
        """
        获取所有活跃知识的话题列表。

        Returns:
            话题列表（去重）
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT topic FROM knowledge WHERE status = ? ORDER BY topic",
                (KnowledgeStatus.ACTIVE.value,),
            )
            return [row[0] for row in cursor.fetchall()]

    # ==================== 写入方法 (仅供 MemoryUpdateAgent 调用) ====================

    def insert(self, record: Dict[str, Any]) -> str:
        """
        插入新记录。

        Args:
            record: 记录字典，必须包含:
                - id: 文档 ID
                - content_hash: 内容哈希
                - topic: 话题
                - source_type: 来源类型

        Returns:
            插入的记录 ID
        """
        now = datetime.now().isoformat()
        doc_id = record["id"]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO knowledge
                (id, content_hash, topic, source_url, source_type, title,
                 summary_preview, status, created_at, updated_at, update_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    record["content_hash"],
                    record["topic"],
                    record.get("source_url", ""),
                    record.get("source_type", "web_search"),
                    record.get("title", ""),
                    record.get("summary_preview", "")[:200],
                    KnowledgeStatus.ACTIVE.value,
                    now,
                    now,
                    0,
                ),
            )
            conn.commit()

        return doc_id

    def update_status(
        self,
        doc_id: str,
        status: str,
        merged_into: Optional[str] = None,
    ) -> bool:
        """
        更新记录状态。

        Args:
            doc_id: 文档 ID
            status: 新状态
            merged_into: 如果是合并状态，指向的目标 ID

        Returns:
            是否更新成功
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            if merged_into:
                cursor = conn.execute(
                    """
                    UPDATE knowledge
                    SET status = ?, merged_into = ?, updated_at = ?, update_count = update_count + 1
                    WHERE id = ?
                    """,
                    (status, merged_into, now, doc_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE knowledge
                    SET status = ?, updated_at = ?, update_count = update_count + 1
                    WHERE id = ?
                    """,
                    (status, now, doc_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def update_record(
        self,
        doc_id: str,
        content_hash: Optional[str] = None,
        summary_preview: Optional[str] = None,
        source_url: Optional[str] = None,
        merged_from: Optional[List[str]] = None,
    ) -> bool:
        """
        更新记录字段。

        Args:
            doc_id: 文档 ID
            content_hash: 新内容哈希
            summary_preview: 新摘要预览
            source_url: 新来源 URL
            merged_from: 合并来源 ID 列表

        Returns:
            是否更新成功
        """
        updates = []
        values = []

        if content_hash is not None:
            updates.append("content_hash = ?")
            values.append(content_hash)

        if summary_preview is not None:
            updates.append("summary_preview = ?")
            values.append(summary_preview[:200])

        if source_url is not None:
            updates.append("source_url = ?")
            values.append(source_url)

        if merged_from is not None:
            updates.append("merged_from = ?")
            values.append(json.dumps(merged_from))

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        updates.append("update_count = update_count + 1")

        values.append(doc_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE knowledge SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, doc_id: str) -> bool:
        """
        删除记录（慎用，建议使用 update_status 标记为 deprecated）。

        Args:
            doc_id: 文档 ID

        Returns:
            是否删除成功
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM knowledge WHERE id = ?",
                (doc_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    # ==================== 统计方法 ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息。

        Returns:
            统计信息字典
        """
        with sqlite3.connect(self.db_path) as conn:
            # 按状态统计
            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM knowledge GROUP BY status"
            )
            by_status = {row[0]: row[1] for row in cursor.fetchall()}

            # 按来源类型统计（仅活跃）
            cursor = conn.execute(
                """
                SELECT source_type, COUNT(*) FROM knowledge
                WHERE status = ? GROUP BY source_type
                """,
                (KnowledgeStatus.ACTIVE.value,),
            )
            by_source = {row[0]: row[1] for row in cursor.fetchall()}

            # 按话题统计 top 10（仅活跃）
            cursor = conn.execute(
                """
                SELECT topic, COUNT(*) as cnt FROM knowledge
                WHERE status = ? GROUP BY topic ORDER BY cnt DESC LIMIT 10
                """,
                (KnowledgeStatus.ACTIVE.value,),
            )
            by_topic = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total": sum(by_status.values()),
                "active": by_status.get(KnowledgeStatus.ACTIVE.value, 0),
                "by_status": by_status,
                "by_source_type": by_source,
                "top_topics": by_topic,
            }

    def find_stale_records(self, days_threshold: int = 30) -> List[Dict[str, Any]]:
        """
        查找可能过时的记录。

        Args:
            days_threshold: 天数阈值

        Returns:
            可能过时的记录列表
        """
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM knowledge
                WHERE status = ? AND updated_at < ?
                ORDER BY updated_at ASC LIMIT 50
                """,
                (KnowledgeStatus.ACTIVE.value, threshold_date),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ==================== 私有方法 ====================

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转换为字典"""
        result = dict(row)
        # 解析 JSON 字段
        if result.get("merged_from"):
            try:
                result["merged_from"] = json.loads(result["merged_from"])
            except (json.JSONDecodeError, TypeError):
                result["merged_from"] = []
        return result

    def _ensure_research_columns(self) -> None:
        """确保调研相关列存在（兼容已有数据库）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(knowledge)")
            columns = {row[1] for row in cursor.fetchall()}

            new_columns = [
                ("research_topic", "TEXT"),
                ("research_task_id", "TEXT"),
                ("is_research_fact", "INTEGER DEFAULT 0"),
                ("fact_confidence", "REAL DEFAULT 0.8"),
            ]

            for col_name, col_type in new_columns:
                if col_name not in columns:
                    try:
                        conn.execute(
                            f"ALTER TABLE knowledge ADD COLUMN {col_name} {col_type}"
                        )
                    except sqlite3.OperationalError:
                        pass  # 列已存在

            conn.commit()

    # ==================== 调研相关查询方法 ====================

    def get_research_topics(self) -> List[Dict[str, Any]]:
        """
        获取所有已调研过的主题。

        Returns:
            调研主题列表，每项包含 topic, fact_count, source_count, last_updated
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    research_topic,
                    COUNT(*) as fact_count,
                    COUNT(DISTINCT source_url) as source_count,
                    MAX(updated_at) as last_updated
                FROM knowledge
                WHERE research_topic IS NOT NULL
                    AND research_topic != ''
                    AND status = ?
                GROUP BY research_topic
                ORDER BY last_updated DESC
                """,
                (KnowledgeStatus.ACTIVE.value,),
            )
            return [
                {
                    "topic": row[0],
                    "fact_count": row[1],
                    "source_count": row[2],
                    "last_updated": row[3],
                }
                for row in cursor.fetchall()
            ]

    def get_topic_coverage(self, topic: str) -> Dict[str, Any]:
        """
        获取某主题的调研覆盖情况。

        Args:
            topic: 调研主题

        Returns:
            覆盖情况字典
        """
        with sqlite3.connect(self.db_path) as conn:
            # 获取事实统计
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as fact_count,
                    COUNT(DISTINCT source_url) as source_count,
                    AVG(fact_confidence) as avg_confidence,
                    MIN(created_at) as first_researched,
                    MAX(updated_at) as last_updated
                FROM knowledge
                WHERE research_topic = ?
                    AND status = ?
                    AND is_research_fact = 1
                """,
                (topic, KnowledgeStatus.ACTIVE.value),
            )
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return {
                    "topic": topic,
                    "fact_count": 0,
                    "source_count": 0,
                    "coverage_score": 0.0,
                    "avg_confidence": 0.0,
                    "subtopics_covered": [],
                    "first_researched": None,
                    "last_updated": None,
                }

            # 获取相关子主题（通过普通 topic 字段）
            cursor = conn.execute(
                """
                SELECT DISTINCT topic
                FROM knowledge
                WHERE research_topic = ?
                    AND status = ?
                """,
                (topic, KnowledgeStatus.ACTIVE.value),
            )
            subtopics = [r[0] for r in cursor.fetchall()]

            # 计算覆盖度分数（基于事实数量和来源数量）
            fact_count = row[0]
            source_count = row[1]
            # 简单的覆盖度计算：事实数量 * 0.6 + 来源数量 * 0.4，上限100
            coverage_score = min(100.0, (fact_count * 2 + source_count * 5))

            return {
                "topic": topic,
                "fact_count": fact_count,
                "source_count": source_count,
                "coverage_score": coverage_score,
                "avg_confidence": row[2] or 0.0,
                "subtopics_covered": subtopics,
                "first_researched": row[3],
                "last_updated": row[4],
            }

    def get_facts_by_research_topic(
        self,
        topic: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        获取某调研主题下的所有事实。

        Args:
            topic: 调研主题
            limit: 最大返回数量

        Returns:
            事实列表
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM knowledge
                WHERE research_topic = ?
                    AND status = ?
                    AND is_research_fact = 1
                ORDER BY fact_confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (topic, KnowledgeStatus.ACTIVE.value, limit),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def insert_research_fact(self, record: Dict[str, Any]) -> str:
        """
        插入调研事实记录。

        Args:
            record: 记录字典，必须包含:
                - id: 文档 ID
                - content_hash: 内容哈希
                - topic: 子话题
                - research_topic: 调研主题
                - research_task_id: 调研任务ID

        Returns:
            插入的记录 ID
        """
        now = datetime.now().isoformat()
        doc_id = record["id"]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO knowledge
                (id, content_hash, topic, source_url, source_type, title,
                 summary_preview, status, created_at, updated_at, update_count,
                 research_topic, research_task_id, is_research_fact, fact_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    record["content_hash"],
                    record.get("topic", record.get("research_topic", "")),
                    record.get("source_url", ""),
                    "deep_research",
                    record.get("title", ""),
                    record.get("summary_preview", "")[:200],
                    KnowledgeStatus.ACTIVE.value,
                    now,
                    now,
                    0,
                    record.get("research_topic", ""),
                    record.get("research_task_id", ""),
                    1,  # is_research_fact
                    record.get("fact_confidence", 0.8),
                ),
            )
            conn.commit()

        return doc_id

    def get_research_stats(self) -> Dict[str, Any]:
        """
        获取调研相关统计信息。

        Returns:
            统计信息字典
        """
        with sqlite3.connect(self.db_path) as conn:
            # 调研主题数量
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT research_topic)
                FROM knowledge
                WHERE research_topic IS NOT NULL AND research_topic != ''
                """
            )
            topic_count = cursor.fetchone()[0]

            # 调研事实数量
            cursor = conn.execute(
                "SELECT COUNT(*) FROM knowledge WHERE is_research_fact = 1 AND status = ?",
                (KnowledgeStatus.ACTIVE.value,),
            )
            fact_count = cursor.fetchone()[0]

            # 按调研主题的事实分布
            cursor = conn.execute(
                """
                SELECT research_topic, COUNT(*) as cnt
                FROM knowledge
                WHERE research_topic IS NOT NULL
                    AND research_topic != ''
                    AND status = ?
                GROUP BY research_topic
                ORDER BY cnt DESC
                LIMIT 10
                """,
                (KnowledgeStatus.ACTIVE.value,),
            )
            by_topic = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "research_topic_count": topic_count,
                "research_fact_count": fact_count,
                "facts_by_topic": by_topic,
            }
