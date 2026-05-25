"""
Core 模块。

提供全局配置、数据模型和 Prompt 模板。
"""

from .config import (
    Config,
    config,
    LLMConfig,
    AgentModelConfig,
    AgentsConfig,
    TasksConfig,
    StorageConfig,
    ServerConfig,
    SearchConfig,
    LoggingConfig,
)

from .models import (
    # 用户相关
    UserProfile,
    # 消息相关
    Message,
    Conversation,
    # 知识相关
    KnowledgeStatus,
    KnowledgeItem,
    # 记忆更新相关
    UpdateType,
    UpdateAction,
    MemoryUpdateRequest,
    MemoryUpdateResult,
    # 搜索相关
    SearchQuery,
    SearchSource,
    SearchResult,
    # 报告相关
    Report,
    # 推送相关
    PushAction,
    PushDecision,
)

from .prompts import Prompts


__all__ = [
    # 配置
    "Config",
    "config",
    "LLMConfig",
    "AgentModelConfig",
    "AgentsConfig",
    "TasksConfig",
    "StorageConfig",
    "ServerConfig",
    "SearchConfig",
    "LoggingConfig",
    # 数据模型
    "UserProfile",
    "Message",
    "Conversation",
    "KnowledgeStatus",
    "KnowledgeItem",
    "UpdateType",
    "UpdateAction",
    "MemoryUpdateRequest",
    "MemoryUpdateResult",
    "SearchQuery",
    "SearchSource",
    "SearchResult",
    "Report",
    "PushAction",
    "PushDecision",
    # Prompts
    "Prompts",
]
