"""
搜索工具。

封装网络搜索功能，支持 DuckDuckGo 和 Serper。
"""

from typing import List, Optional

from core.config import SearchConfig
from core.models import SearchSource


class DuckDuckGoSearch:
    """DuckDuckGo 搜索实现"""

    def __init__(self):
        """初始化 DuckDuckGo 搜索"""
        self.available = False
        self.ddgs = None

        try:
            from ddgs import DDGS

            self.ddgs = DDGS()
            self.available = True
        except ImportError:
            pass

    def search(self, query: str, num_results: int = 5) -> List[SearchSource]:
        """
        执行 DuckDuckGo 搜索。

        Args:
            query: 搜索查询
            num_results: 最大结果数

        Returns:
            搜索结果列表
        """
        if not self.available or not self.ddgs:
            return []

        try:
            results = self.ddgs.text(query, max_results=num_results)
            return [
                SearchSource(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
                for r in results
            ]
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []


class SerperSearch:
    """Serper 搜索实现"""

    def __init__(self, api_key: str = ""):
        """
        初始化 Serper 搜索。

        Args:
            api_key: Serper API key
        """
        self.api_key = api_key
        self.available = bool(api_key)
        self.base_url = "https://google.serper.dev/search"

    def search(self, query: str, num_results: int = 5) -> List[SearchSource]:
        """
        执行 Serper Google 搜索。

        Args:
            query: 搜索查询
            num_results: 最大结果数

        Returns:
            搜索结果列表
        """
        if not self.available:
            return []

        try:
            import requests
            import json

            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "q": query,
                "num": num_results,
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()

            # 以json格式打印原始结果以供调试
            # print(json.dumps(results, indent=2, ensure_ascii=False))

            organic_results = results.get("organic", [])
            return [
                SearchSource(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", "")
                )
                for r in organic_results[:num_results]
            ]
        except Exception as e:
            print(f"Serper search error: {e}")
            return []


class SearchTool:
    """
    统一搜索工具。

    支持多个搜索后端，自动回退。
    """

    def __init__(self, config: SearchConfig):
        """
        初始化搜索工具。

        Args:
            config: 搜索配置
        """
        self.config = config
        self.duckduckgo = DuckDuckGoSearch()
        self.serper = SerperSearch(api_key=config.serper_api_key)

    def search(self, query: str, num_results: Optional[int] = None) -> List[SearchSource]:
        """
        执行搜索。

        根据配置选择搜索后端，支持自动回退。

        Args:
            query: 搜索查询
            num_results: 最大结果数，默认使用配置值

        Returns:
            搜索结果列表
        """
        if num_results is None:
            num_results = self.config.max_results_per_query

        # 根据配置选择首选后端
        if self.config.prefer_serper and self.serper.available:
            print("------------Using Serper as primary search backend.------------")
            results = self.serper.search(query, num_results)
            if results:
                return results
            # 回退到 DuckDuckGo
            return self.duckduckgo.search(query, num_results)
        else:
            # 默认使用 DuckDuckGo
            results = self.duckduckgo.search(query, num_results)
            if results:
                return results
            # 回退到 Serper
            if self.serper.available:
                return self.serper.search(query, num_results)
            return []

    def search_and_format(self, query: str, num_results: Optional[int] = None) -> str:
        """
        执行搜索并返回格式化的字符串结果。

        Args:
            query: 搜索查询
            num_results: 最大结果数

        Returns:
            格式化的搜索结果字符串
        """
        results = self.search(query, num_results)

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result.title}\n"
                f"   URL: {result.url}\n"
                f"   {result.snippet}\n"
            )
        return "\n".join(formatted)

    @property
    def is_available(self) -> bool:
        """检查是否有可用的搜索后端"""
        return self.duckduckgo.available or self.serper.available

if __name__ == "__main__":
    # 简单测试
    config = SearchConfig(
        prefer_serper=True
    )
    search_tool = SearchTool(config)
    query = "OpenAI GPT-4"
    results = search_tool.search_and_format(query, num_results=3)
    print(results)