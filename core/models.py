"""
核心数据模型定义。

所有模块共享这些数据结构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ==================== 用户相关 ====================

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    role: str = ""
    age: str = ""
    location: str = ""
    interests: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    communication_style: List[str] = field(default_factory=list)
    research_history: List[str] = field(default_factory=list)  # 历史调研主题
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_prompt_text(self) -> str:
        """格式化为 Prompt 文本"""
        sections = []
        if self.role:
            sections.append(f"Role: {self.role}")
        if self.age:
            sections.append(f"Age: {self.age}")
        if self.location:
            sections.append(f"Location: {self.location}")
        if self.interests:
            sections.append(f"Interests: {', '.join(self.interests)}")
        if self.goals:
            sections.append(f"Goals: {', '.join(self.goals)}")
        if self.challenges:
            sections.append(f"Challenges: {', '.join(self.challenges)}")
        if self.traits:
            sections.append(f"Traits: {', '.join(self.traits)}")
        if self.communication_style:
            sections.append(f"Communication style: {', '.join(self.communication_style)}")
        if self.research_history:
            # 只显示最近5个调研主题
            recent_research = self.research_history[-5:]
            sections.append(f"Research history: {', '.join(recent_research)}")
        return "\n".join(sections) if sections else "No user profile information available"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "role": self.role,
            "age": self.age,
            "location": self.location,
            "interests": self.interests,
            "goals": self.goals,
            "challenges": self.challenges,
            "traits": self.traits,
            "communication_style": self.communication_style,
            "research_history": self.research_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """从字典创建"""
        return cls(
            user_id=data.get("user_id", ""),
            role=data.get("role", ""),
            age=data.get("age", ""),
            location=data.get("location", ""),
            interests=data.get("interests", []),
            goals=data.get("goals", []),
            challenges=data.get("challenges", []),
            traits=data.get("traits", []),
            communication_style=data.get("communication_style", []),
            research_history=data.get("research_history", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )

    def add_research_topic(self, topic: str) -> None:
        """添加调研主题到历史记录"""
        if topic and topic not in self.research_history:
            self.research_history.append(topic)
            self.updated_at = datetime.now().isoformat()


# ==================== 用户事实 ====================

@dataclass
class UserFact:
    """
    用户提到的事实/实体信息。

    用于存储用户在对话中提到的人物、地点、组织等实体信息，
    以及这些实体的属性（如年龄、职业、关系等）。
    """
    entity: str                      # 实体名称，如 "妈妈", "弟弟", "朋友Alice"
    entity_type: str                 # 类型: person/place/organization/event/other
    attributes: Dict[str, str] = field(default_factory=dict)  # 属性，如 {"职业": "医生", "年龄": "50岁"}
    relationship: str = ""           # 与用户的关系，如 "母亲", "弟弟", "朋友"
    source_topic: str = ""           # 来源话题
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "entity": self.entity,
            "entity_type": self.entity_type,
            "attributes": self.attributes,
            "relationship": self.relationship,
            "source_topic": self.source_topic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserFact":
        """从字典创建"""
        return cls(
            entity=data.get("entity", ""),
            entity_type=data.get("entity_type", "other"),
            attributes=data.get("attributes", {}),
            relationship=data.get("relationship", ""),
            source_topic=data.get("source_topic", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )

    def to_prompt_text(self) -> str:
        """格式化为 Prompt 文本"""
        parts = [self.entity]
        if self.relationship:
            parts.append(f"({self.relationship})")
        if self.attributes:
            attr_text = ", ".join([f"{k}: {v}" for k, v in self.attributes.items()])
            parts.append(f"- {attr_text}")
        return " ".join(parts)

    def merge_attributes(self, new_attributes: Dict[str, str]) -> None:
        """合并新属性到现有属性"""
        self.attributes.update(new_attributes)
        self.updated_at = datetime.now().isoformat()


# ==================== 消息相关 ====================

@dataclass
class Message:
    """对话消息"""
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # 情感分析字段
    sentiment: Optional[str] = None  # surprise/anger/sadness/joy/fear/neutral/disgust
    sentiment_score: Optional[float] = None  # -1.0 (负面) 到 1.0 (正面)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        # 只在有值时添加情感字段，保持向后兼容
        if self.sentiment is not None:
            result["sentiment"] = self.sentiment
        if self.sentiment_score is not None:
            result["sentiment_score"] = self.sentiment_score
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建"""
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            sentiment=data.get("sentiment"),
            sentiment_score=data.get("sentiment_score"),
        )


@dataclass
class Conversation:
    """一段对话（按话题组织）"""
    topic: str
    messages: List[Message] = field(default_factory=list)
    summary: str = ""
    running_summary: str = ""  # 增量摘要，用于实时画像提取
    key_info: List[str] = field(default_factory=list)
    user_preferences: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "topic": self.topic,
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
            "running_summary": self.running_summary,
            "key_info": self.key_info,
            "user_preferences": self.user_preferences,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """从字典创建"""
        return cls(
            topic=data.get("topic", ""),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            summary=data.get("summary", ""),
            running_summary=data.get("running_summary", ""),
            key_info=data.get("key_info", []),
            user_preferences=data.get("user_preferences", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


# ==================== 知识相关 ====================

class KnowledgeStatus(Enum):
    """知识状态"""
    ACTIVE = "active"
    MERGED = "merged"
    DEPRECATED = "deprecated"


@dataclass
class KnowledgeItem:
    """知识条目"""
    id: str
    content: str
    summary: str
    topic: str
    source_url: str = ""
    title: str = ""
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    content_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "topic": self.topic,
            "source_url": self.source_url,
            "title": self.title,
            "status": self.status.value,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeItem":
        """从字典创建"""
        status = data.get("status", "active")
        if isinstance(status, str):
            status = KnowledgeStatus(status)
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            topic=data.get("topic", ""),
            source_url=data.get("source_url", ""),
            title=data.get("title", ""),
            status=status,
            content_hash=data.get("content_hash", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


# ==================== 记忆更新相关 ====================

class UpdateType(Enum):
    """记忆更新类型"""
    ADD_KNOWLEDGE = "add_knowledge"
    ADD_CONVERSATION = "add_conversation"
    UPDATE_PROFILE = "update_profile"
    MERGE_KNOWLEDGE = "merge_knowledge"
    DEPRECATE_KNOWLEDGE = "deprecate_knowledge"


class UpdateAction(Enum):
    """更新执行的动作"""
    ADDED = "added"
    SKIPPED = "skipped"
    MERGED = "merged"
    REPLACED = "replaced"


@dataclass
class MemoryUpdateRequest:
    """
    记忆更新请求。

    data 结构说明:
    - ADD_KNOWLEDGE:
        {
            "content": str,
            "summary": str,
            "topic": str,
            "source_url": str,
            "title": str
        }
    - ADD_CONVERSATION:
        {
            "topic": str,
            "messages": List[Message],
            "summary": str,
            "key_info": List[str],
            "user_preferences": List[str]
        }
    - UPDATE_PROFILE:
        {
            "interests": List[str],    # 新增的兴趣
            "goals": List[str],        # 新增的目标
            "traits": List[str],       # 新增的特点
            ...
        }
    - MERGE_KNOWLEDGE:
        {
            "source_ids": List[str],   # 要合并的知识 ID 列表
            "merged_content": str      # 合并后的内容
        }
    - DEPRECATE_KNOWLEDGE:
        {
            "knowledge_id": str,       # 要废弃的知识 ID
            "reason": str              # 废弃原因
        }
    """
    update_type: UpdateType
    source: str  # 来源模块: "search_service", "topic_change", "proactive", etc.
    user_id: str
    data: Dict[str, Any]


@dataclass
class MemoryUpdateResult:
    """记忆更新结果"""
    success: bool
    action: UpdateAction
    affected_ids: List[str] = field(default_factory=list)
    message: str = ""


# ==================== 搜索相关 ====================

@dataclass
class SearchQuery:
    """搜索查询"""
    query: str
    purpose: str = ""
    priority: int = 5  # 1-10


@dataclass
class SearchSource:
    """搜索来源"""
    url: str
    title: str
    snippet: str
    full_content: str = ""

    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "full_content": self.full_content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "SearchSource":
        """从字典创建"""
        return cls(
            url=data.get("url", ""),
            title=data.get("title", ""),
            snippet=data.get("snippet", ""),
            full_content=data.get("full_content", ""),
        )


@dataclass
class SearchResult:
    """搜索结果"""
    success: bool
    query: str
    sources: List[SearchSource] = field(default_factory=list)
    error: str = ""


# ==================== 报告相关 ====================

@dataclass
class Report:
    """调研报告"""
    id: str
    topic: str
    title: str
    summary: str
    content: str
    sources: List[SearchSource] = field(default_factory=list)
    relevance: int = 50  # 0-100
    urgency: int = 50    # 0-100
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "topic": self.topic,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "relevance": self.relevance,
            "urgency": self.urgency,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Report":
        """从字典创建"""
        return cls(
            id=data.get("id", ""),
            topic=data.get("topic", ""),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            content=data.get("content", ""),
            sources=[SearchSource.from_dict(s) for s in data.get("sources", [])],
            relevance=data.get("relevance", 50),
            urgency=data.get("urgency", 50),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


# ==================== 推送相关 ====================

class PushAction(Enum):
    """推送动作"""
    NOTIFY = "notify"    # 提醒用户是否需要查看报告
    SILENT = "silent"    # 静默存储


@dataclass
class PushDecision:
    """推送决策"""
    action: PushAction
    score: int  # 0-100
    value: int  # 信息价值 0-100
    cost: int   # 打断成本 0-100
    reason: str = ""
