"""
Tools 模块。

封装外部依赖，提供 LLM 调用、网络搜索、网页抓取等功能。
"""

from .llm_client import LLMClient
from .search_tool import SearchTool, DuckDuckGoSearch, SerperSearch
from .web_fetcher import WebFetcher


__all__ = [
    "LLMClient",
    "SearchTool",
    "DuckDuckGoSearch",
    "SerperSearch",
    "WebFetcher",
]
