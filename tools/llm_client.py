"""
LLM 客户端。

封装 OpenAI-compatible API 调用，提供重试、超时、日志等生产级特性。
"""

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from core.config import LLMConfig

logger = logging.getLogger(__name__)

# 可重试的 API 异常类型
_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class LLMClient:
    """
    LLM 客户端。

    封装 LLM API 调用，提供统一的接口。
    """

    def __init__(self, config: LLMConfig):
        """
        初始化 LLM 客户端。

        Args:
            config: LLM 配置
        """
        self.config = config
        self.request_timeout = config.timeout
        self.max_retries = 3
        client_kwargs: Dict[str, Any] = {
            "api_key": config.api_key,
            "base_url": config.base_url if config.base_url else None,
            "max_retries": 0,  # 由我们自己管理重试，禁用 SDK 内置重试
        }
        if self.request_timeout is not None:
            client_kwargs["timeout"] = self.request_timeout

        self.client = OpenAI(
            **client_kwargs,
        )
        self._usage_stats: Dict[str, Dict[str, int]] = {}  # model -> {calls, prompt_tokens, ...}

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求，获取文本响应。带指数退避重试。

        Args:
            messages: 消息列表，每条消息包含 'role' 和 'content'
            model: 使用的模型名称
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            response_format: 响应格式配置

        Returns:
            助手的回复内容

        Raises:
            RuntimeError: 重试耗尽后仍然失败
        """
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if seed is not None:
            kwargs["seed"] = seed

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if response_format:
            kwargs["response_format"] = response_format

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(**kwargs)

                # 记录 token 使用量
                usage = getattr(completion, "usage", None)
                if usage:
                    logger.info(
                        "LLM usage model=%s prompt=%s completion=%s total=%s",
                        model,
                        getattr(usage, "prompt_tokens", "?"),
                        getattr(usage, "completion_tokens", "?"),
                        getattr(usage, "total_tokens", "?"),
                    )

                    # 累计 token 用量
                    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                    ct = int(getattr(usage, "completion_tokens", 0) or 0)
                    tt = int(getattr(usage, "total_tokens", 0) or 0) or (pt + ct)
                    bucket = self._usage_stats.setdefault(
                        model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    )
                    bucket["calls"] += 1
                    bucket["prompt_tokens"] += pt
                    bucket["completion_tokens"] += ct
                    bucket["total_tokens"] += tt

                return completion.choices[0].message.content or ""

            except _RETRYABLE_ERRORS as e:
                last_error = e
                logger.warning(
                    "LLM chat attempt %d/%d failed model=%s: %s",
                    attempt, self.max_retries, model, e,
                )
                if attempt >= self.max_retries:
                    break
                backoff = (2 ** (attempt - 1)) + random.random()
                logger.info("Retrying in %.2fs (attempt %d/%d)", backoff, attempt + 1, self.max_retries)
                time.sleep(backoff)

        raise RuntimeError(
            f"LLM chat failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        发送聊天请求，获取 JSON 响应。

        Args:
            messages: 消息列表
            model: 使用的模型名称
            temperature: 采样温度（默认较低以获得更稳定的 JSON）
            max_tokens: 最大生成 token 数

        Returns:
            解析后的 JSON 响应

        Raises:
            ValueError: JSON 解析失败时抛出
        """
        response = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            seed=seed,
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 使用 raw_decode 尝试从文本中精确提取 JSON 对象
            decoder = json.JSONDecoder()
            candidate = response.strip()

            # 先尝试直接解析（去掉首尾空白后）
            try:
                parsed, _ = decoder.raw_decode(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            # 逐字符查找首个 JSON 对象起始位置
            for idx, ch in enumerate(candidate):
                if ch != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[idx:])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

            raise ValueError(f"Failed to parse JSON from response: {response[:200]}")

    # ==================== Token 用量统计 ====================

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """获取当前 token 使用统计（按模型分组）。"""
        return {model: stats.copy() for model, stats in self._usage_stats.items()}

    def reset_usage_stats(self) -> None:
        """重置 token 使用统计。"""
        self._usage_stats.clear()

    def format_usage_report(self) -> str:
        """格式化 token 使用报告（含预估费用）。"""
        if not self._usage_stats:
            return "Token Usage Report\nNo usage data recorded."

        # OpenAI pricing (per 1M tokens)
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }

        def _resolve_pricing(model_name: str) -> Dict[str, float]:
            name = model_name.lower()
            if name.startswith("gpt-4o-mini"):
                return pricing["gpt-4o-mini"]
            if name.startswith("gpt-4o"):
                return pricing["gpt-4o"]
            return {"input": 0.0, "output": 0.0}  # 未知/本地模型不计费

        lines = ["Token Usage Report", "=" * 60]
        grand = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        grand_cost = 0.0

        for model_name in sorted(self._usage_stats):
            s = self._usage_stats[model_name]
            rates = _resolve_pricing(model_name)
            cost = (s["prompt_tokens"] / 1_000_000) * rates["input"] + \
                   (s["completion_tokens"] / 1_000_000) * rates["output"]

            lines.extend([
                f"Model: {model_name}",
                f"  Calls:              {s['calls']}",
                f"  Prompt tokens:      {s['prompt_tokens']:,}",
                f"  Completion tokens:  {s['completion_tokens']:,}",
                f"  Total tokens:       {s['total_tokens']:,}",
                f"  Estimated cost:     ${cost:.4f}",
                "-" * 60,
            ])

            for k in grand:
                grand[k] += s[k]
            grand_cost += cost

        lines.extend([
            "GRAND TOTAL",
            f"  Calls:              {grand['calls']}",
            f"  Prompt tokens:      {grand['prompt_tokens']:,}",
            f"  Completion tokens:  {grand['completion_tokens']:,}",
            f"  Total tokens:       {grand['total_tokens']:,}",
            f"  Estimated cost:     ${grand_cost:.4f}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的 embedding 向量。

        Args:
            text: 要嵌入的文本

        Returns:
            embedding 向量（浮点数列表）
        """
        if len(text) > 8000:
            logger.warning(
                "Embedding input too long (%d chars), truncating to 8000",
                len(text),
            )
            text = text[:8000]

        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的 embedding 向量。

        Args:
            texts: 要嵌入的文本列表

        Returns:
            embedding 向量列表
        """
        if not texts:
            return []

        # 截断过长的文本
        truncated = []
        for t in texts:
            if len(t) > 8000:
                logger.warning("Batch embedding input truncated (%d chars)", len(t))
                truncated.append(t[:8000])
            else:
                truncated.append(t)

        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=truncated,
        )
        return [item.embedding for item in response.data]
