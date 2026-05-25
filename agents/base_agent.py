"""
Agent 基类。

提供 LLM 调用封装、日志记录和配置管理。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from core.config import AgentModelConfig
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Agent 基类。

    提供:
    - LLM 调用封装（含错误处理和日志）
    - 配置管理
    """

    def __init__(self, config: AgentModelConfig, llm_client: LLMClient):
        """
        初始化 Agent。

        Args:
            config: 该 Agent 的模型配置
            llm_client: LLM 客户端
        """
        self.config = config
        self.llm = llm_client

    def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """
        调用 LLM 获取文本响应。

        Args:
            messages: 消息列表
            temperature: 可选的温度参数，默认使用配置中的温度

        Returns:
            LLM 的文本响应
        """
        start = time.perf_counter()
        try:
            response = self.llm.chat(
                messages=messages,
                model=self.config.model,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
            )
            elapsed = time.perf_counter() - start
            logger.info(
                "%s._chat ok model=%s elapsed=%.3fs",
                self.__class__.__name__, self.config.model, elapsed,
            )
            return response
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "%s._chat failed model=%s elapsed=%.3fs",
                self.__class__.__name__, self.config.model, elapsed,
            )
            raise

    def _chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        调用 LLM 获取 JSON 响应。

        Args:
            messages: 消息列表
            temperature: 可选的温度参数，默认使用配置中的温度

        Returns:
            解析后的 JSON 响应（保证为 dict 类型）
        """
        start = time.perf_counter()
        try:
            response = self.llm.chat_json(
                messages=messages,
                model=self.config.model,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
            )
            if not isinstance(response, dict):
                raise TypeError(
                    f"{self.__class__.__name__} expected dict JSON response, "
                    f"got {type(response).__name__}"
                )

            elapsed = time.perf_counter() - start
            logger.info(
                "%s._chat_json ok model=%s elapsed=%.3fs",
                self.__class__.__name__, self.config.model, elapsed,
            )
            return response
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "%s._chat_json failed model=%s elapsed=%.3fs",
                self.__class__.__name__, self.config.model, elapsed,
            )
            raise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"
