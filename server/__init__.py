"""
Server 模块。

提供 HTTP API、WebSocket 通信和后台任务管理。
"""

from .app import app, create_app
from .dependencies import (
    UserDependencies,
    get_config,
    get_llm_client,
    get_search_tool,
    get_web_fetcher,
    get_user_dependencies,
    clear_user_dependencies,
)

__all__ = [
    "app",
    "create_app",
    "UserDependencies",
    "get_config",
    "get_llm_client",
    "get_search_tool",
    "get_web_fetcher",
    "get_user_dependencies",
    "clear_user_dependencies",
]
