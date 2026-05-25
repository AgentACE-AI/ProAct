"""
本地 LLM 客户端。

使用 HuggingFace transformers 加载本地因果语言模型，
提供与 LLMClient 完全对齐的聊天与 token 统计接口。
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# 禁用 tokenizers Rust 后端的并行线程池，防止 tokio-runtime-worker 线程泄漏。
# 不设此项时，每次 tokenize 调用都会累积新线程，数千次调用后进程将因线程数
# 耗尽而挂死。单线程 tokenize 对端到端推理耗时几乎无影响。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# 模块级缓存：避免重复加载同一模型
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


class LocalLLMClient:
    """
    本地 LLM 客户端。

    使用 HuggingFace 因果语言模型提供与 LLMClient 对齐的聊天接口。
    模型按 model_name_or_path 缓存，跨实例复用。
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        """
        初始化本地 LLM 客户端。

        Args:
            model_name_or_path: HuggingFace 模型名或本地路径
            device: 推理设备 ('cuda', 'cpu', 'auto'). None 则自动检测。
            torch_dtype: 模型 dtype. None 则使用模型默认配置。
        """
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        self._usage_stats: Dict[str, Dict[str, int]] = {}

        self.tokenizer, self.model = self._load_model()
        logger.info(
            "LocalLLMClient ready: model=%s device=%s dtype=%s",
            model_name_or_path, self.device, self.torch_dtype,
        )

    @staticmethod
    def _resolve_torch_dtype(
        dtype: Optional[Union[str, torch.dtype]],
    ) -> Optional[Union[str, torch.dtype]]:
        """将字符串 dtype 转换为 torch.dtype。"""
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            "auto": "auto",
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        key = str(dtype).strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {dtype!r}")
        return mapping[key]

    def _load_model(self) -> Tuple[Any, Any]:
        """加载模型和 tokenizer，命中缓存则复用。"""
        cache_key = self.model_name_or_path
        if cache_key in _MODEL_CACHE:
            logger.info("Reusing cached model: %s", cache_key)
            return _MODEL_CACHE[cache_key]

        logger.info("Loading model: %s ...", self.model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if self.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.torch_dtype

        # device_map="auto" 自动分配 GPU/CPU，支持多卡
        if self.device == "auto" or self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **load_kwargs
        )
        model.eval()

        # 确保 pad_token 存在
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.generation_config, "pad_token_id", None) is None:
            if tokenizer.pad_token_id is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id

        _MODEL_CACHE[cache_key] = (tokenizer, model)
        logger.info("Model loaded successfully: %s", self.model_name_or_path)
        return tokenizer, model

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        发送聊天请求，获取文本响应。接口与 LLMClient.chat() 完全一致。

        Args:
            messages: 消息列表，每条消息包含 'role' 和 'content'
            model: 忽略（已加载固定模型），但写入 usage stats
            temperature: 采样温度；0 时使用 greedy decoding
            max_tokens: 最大生成 token 数
            response_format: 忽略（本地模型不支持 JSON mode）

        Returns:
            助手的回复内容
        """
        # 构造 prompt
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_device = next(self.model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        prompt_tokens = inputs["input_ids"].shape[-1]

        # 构造生成参数
        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens or 2048,
        }

        if temperature == 0 or temperature is None:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        # 推理
        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)

        # 只解码新生成的 token
        generated_ids = output_ids[0, prompt_tokens:]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        # 统计 token 用量
        completion_tokens = int(generated_ids.shape[-1])
        total_tokens = int(prompt_tokens) + completion_tokens
        usage_model = model or self.model_name_or_path

        bucket = self._usage_stats.setdefault(
            usage_model,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += int(prompt_tokens)
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += total_tokens

        logger.info(
            "LocalLLM usage model=%s prompt=%s completion=%s total=%s",
            usage_model, prompt_tokens, completion_tokens, total_tokens,
        )

        return response

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        发送聊天请求，获取 JSON 响应。
        JSON 提取逻辑与 LLMClient.chat_json() 完全一致。

        Args:
            messages: 消息列表
            model: 忽略，但写入 usage stats
            temperature: 采样温度
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
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 使用 raw_decode 尝试从文本中精确提取 JSON 对象
            decoder = json.JSONDecoder()
            candidate = response.strip()

            # 先尝试直接解析
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
        """格式化 token 使用报告（本地模型成本为 $0）。"""
        if not self._usage_stats:
            return "Token Usage Report\nNo usage data recorded."

        lines = ["Token Usage Report", "=" * 60]
        grand = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for model_name in sorted(self._usage_stats):
            s = self._usage_stats[model_name]
            lines.extend([
                f"Model: {model_name}",
                f"  Calls:              {s['calls']}",
                f"  Prompt tokens:      {s['prompt_tokens']:,}",
                f"  Completion tokens:  {s['completion_tokens']:,}",
                f"  Total tokens:       {s['total_tokens']:,}",
                "  Estimated cost:     $0.0000 (local)",
                "-" * 60,
            ])
            for k in grand:
                grand[k] += s[k]

        lines.extend([
            "GRAND TOTAL",
            f"  Calls:              {grand['calls']}",
            f"  Prompt tokens:      {grand['prompt_tokens']:,}",
            f"  Completion tokens:  {grand['completion_tokens']:,}",
            f"  Total tokens:       {grand['total_tokens']:,}",
            "  Estimated cost:     $0.0000 (local)",
            "=" * 60,
        ])
        return "\n".join(lines)
