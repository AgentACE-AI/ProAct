"""
网页抓取工具。

支持使用 Jina Reader 或直接抓取网页内容。
"""

from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from core.config import SearchConfig


class WebFetcher:
    """
    网页抓取工具。

    支持使用 Jina Reader API 将网页转换为 Markdown 格式，
    或直接抓取网页内容。
    """

    def __init__(self, config: SearchConfig):
        """
        初始化网页抓取工具。

        Args:
            config: 搜索配置
        """
        self.config = config
        self.jina_base_url = "https://r.jina.ai/"
        self.timeout = config.fetch_timeout

    def fetch(self, url: str, use_jina: bool = True, user_serper: bool = True) -> Optional[str]:
        """
        抓取单个网页内容，支持多后端自动回退。

        Args:
            url: 要抓取的 URL
            use_jina: 是否使用 Jina Reader（默认 True）
            user_serper: 是否使用 Serper 抓取（默认 True）

        Returns:
            网页内容（Markdown 格式），失败返回 None
        """
        # Serper scraping (优先)
        if user_serper and self.config.prefer_serper and self.config.serper_api_key:
            result = self._fetch_with_serper(url)
            if result:
                return result
        # Jina Reader (回退)
        if use_jina and self.config.jina_api_key:
            result = self._fetch_with_jina(url)
            if result:
                return result
        # 直接抓取 (最终回退)
        return self._fetch_direct(url)

    def fetch_multiple(
        self,
        urls: List[str],
        use_jina: bool = True,
        max_workers: int = 5,
    ) -> Dict[str, Optional[str]]:
        """
        批量抓取多个网页内容。

        Args:
            urls: URL 列表
            use_jina: 是否使用 Jina Reader
            max_workers: 最大并发数

        Returns:
            URL 到内容的映射字典
        """
        results: Dict[str, Optional[str]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.fetch, url, use_jina): url for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                except Exception as e:
                    print(f"Error fetching {url}: {e}")
                    results[url] = None

        return results

    def _fetch_with_serper(self, url: str) -> Optional[str]:
        import http.client
        import json

        conn = http.client.HTTPSConnection("scrape.serper.dev")
        payload = json.dumps({
            "url": url
        })
        headers = {
            'X-API-KEY': self.config.serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/", payload, headers)
        res = conn.getresponse()
        data = res.read()

        json_result = json.loads(data.decode("utf-8"))
        try:
            text_to_return = json_result['text']
            return text_to_return
        except KeyError:
            print(f"Serper fetch error for {url}: {json_result}")
            return None

    def _fetch_with_jina(self, url: str) -> Optional[str]:
        """
        使用 Jina Reader 抓取网页。

        Args:
            url: 要抓取的 URL

        Returns:
            Markdown 格式的网页内容
        """
        jina_url = self.jina_base_url + url
        headers = {}

        if self.config.jina_api_key:
            headers["Authorization"] = f"Bearer {self.config.jina_api_key}"

        try:
            response = requests.get(jina_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.Timeout:
            print(f"Jina fetch timeout for: {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Jina fetch error for {url}: {e}")
            return None

    def _fetch_direct(self, url: str) -> Optional[str]:
        """
        直接抓取网页内容。

        Args:
            url: 要抓取的 URL

        Returns:
            网页文本内容
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            # 尝试使用 BeautifulSoup 提取文本
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "html.parser")

                # 移除脚本和样式
                for script in soup(["script", "style"]):
                    script.decompose()

                # 获取文本
                text = soup.get_text(separator="\n", strip=True)
                return text
            except ImportError:
                # 如果没有 BeautifulSoup，返回原始 HTML
                return response.text

        except requests.exceptions.Timeout:
            print(f"Direct fetch timeout for: {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Direct fetch error for {url}: {e}")
            return None
