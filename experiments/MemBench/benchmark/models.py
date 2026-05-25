"""
MemBench Benchmark 数据模型。

定义评估过程中使用的所有数据结构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    """Benchmark 配置"""

    # 数据路径基础目录
    data_base_dir: str = "./experiments/MemBench/Membenchdata"

    # 数据集选择: "10k", "100k", 或 "raw" (原始完整数据)
    # - "10k": 使用 data2test/0-10k/ 目录下论文采样的 0-10k token 数据
    # - "100k": 使用 data2test/100k/ 目录下论文采样的 100k token 数据
    # - "raw": 使用 data/ 目录下的原始完整数据
    dataset_variant: str = "10k"

    # 评估设置
    question_types: Optional[List[str]] = None  # None = 全部
    max_trajectories_per_type: Optional[int] = None  # None = 全部
    max_words: int = 2000  # 召回上下文最大词数

    # 容量测试设置 (100k tokens)
    capacity_test: bool = False  # 是否进行 100k 容量测试
    min_tokens_for_capacity: int = 8000  # 100k 测试时，每个轨迹的最小 token 数

    # 输出设置
    output_dir: str = "./experiments/MemBench/results"
    checkpoint_file: str = "checkpoint.json"

    # LLM 设置
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    def get_data_path(self, agent_type: str, level: str) -> Path:
        """
        获取数据文件路径。

        根据 dataset_variant 选择正确的数据目录和文件名：
        - "10k": data2test/0-10k/XXX_multiple_0.json
        - "100k": data2test/100k/XXX_multiple_100.json
        - "raw": data/XXX.json
        """
        base_names = {
            ("FirstAgent", "LowLevel"): "FirstAgentDataLowLevel",
            ("FirstAgent", "HighLevel"): "FirstAgentDataHighLevel",
            ("ThirdAgent", "LowLevel"): "ThirdAgentDataLowLevel",
            ("ThirdAgent", "HighLevel"): "ThirdAgentDataHighLevel",
        }

        base_name = base_names.get((agent_type, level))
        if not base_name:
            raise ValueError(f"Invalid agent_type={agent_type}, level={level}")

        if self.dataset_variant == "10k":
            # 使用论文中的 0-10k 数据
            subdir = "data2test/0-10k"
            filename = f"{base_name}_multiple_0.json"
        elif self.dataset_variant == "100k":
            # 使用论文中的 100k 数据
            subdir = "data2test/100k"
            filename = f"{base_name}_multiple_100.json"
        elif self.dataset_variant == "raw":
            # 使用原始完整数据
            subdir = "data"
            filename = f"{base_name}.json"
        else:
            raise ValueError(f"Invalid dataset_variant: {self.dataset_variant}. Must be '10k', '100k', or 'raw'")

        return Path(self.data_base_dir) / subdir / filename


@dataclass
class QuestionAnswer:
    """问答数据"""

    qid: int
    question: str
    answer: str
    choices: Dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    ground_truth: str  # "A", "B", "C", or "D"
    target_step_id: List[List[int]]  # [[session_idx, msg_idx], ...]
    time: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionAnswer":
        """从字典创建"""
        return cls(
            qid=data.get("qid", 0),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            choices=data.get("choices", {}),
            ground_truth=data.get("ground_truth", ""),
            target_step_id=data.get("target_step_id", []),
            time=data.get("time", ""),
        )


@dataclass
class MemBenchMessage:
    """统一消息格式"""

    step_id: int  # 全局步骤 ID
    session_id: str  # 会话 ID
    user_content: str  # 用户消息
    assistant_content: str  # 助手消息
    time: str = ""  # 时间戳
    place: str = ""  # 地点

    def to_observation(self) -> str:
        """转换为 MemBench 观察格式（包含时间信息）"""
        # 构建基础内容
        if self.assistant_content:
            content = f"'user': {self.user_content}; 'agent': {self.assistant_content}"
        else:
            content = self.user_content

        # 添加时间信息（如果有）
        if self.time:
            content = f"[time: {self.time}] {content}"

        return f"{self.step_id}[|]{content}"


@dataclass
class Trajectory:
    """轨迹数据（一个完整的对话 + 问答）"""

    tid: int
    question_type: str
    scenario: str
    message_list: List[List[Dict[str, Any]]]  # 嵌套的会话消息
    qa: QuestionAnswer

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], question_type: str, scenario: str
    ) -> "Trajectory":
        """从字典创建"""
        return cls(
            tid=data.get("tid", 0),
            question_type=question_type,
            scenario=scenario,
            message_list=data.get("message_list", []),
            qa=QuestionAnswer.from_dict(data.get("QA", {})),
        )


@dataclass
class EvaluationResult:
    """单个轨迹的评估结果"""

    tid: int
    question_type: str
    scenario: str
    question: str
    is_correct: bool
    predicted_answer: str
    ground_truth: str
    user_id: str = ""
    context_preview: str = ""
    run_id: str = ""
    recall_score: Optional[float] = None
    write_time_avg_ms: float = 0.0
    read_time_ms: float = 0.0
    memory_size_tokens: int = 0
    retrieved_step_ids: List[int] = field(default_factory=list)
    target_step_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "tid": self.tid,
            "question_type": self.question_type,
            "scenario": self.scenario,
            "question": self.question,
            "is_correct": self.is_correct,
            "predicted_answer": self.predicted_answer,
            "ground_truth": self.ground_truth,
            "user_id": self.user_id,
            "context_preview": self.context_preview,
            "run_id": self.run_id,
            "recall_score": self.recall_score,
            "write_time_avg_ms": self.write_time_avg_ms,
            "read_time_ms": self.read_time_ms,
            "memory_size_tokens": self.memory_size_tokens,
            "retrieved_step_ids": self.retrieved_step_ids,
            "target_step_ids": self.target_step_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """从字典创建"""
        return cls(
            tid=data.get("tid", 0),
            question_type=data.get("question_type", ""),
            scenario=data.get("scenario", ""),
            question=data.get("question", ""),
            is_correct=data.get("is_correct", False),
            predicted_answer=data.get("predicted_answer", ""),
            ground_truth=data.get("ground_truth", ""),
            user_id=data.get("user_id", ""),
            context_preview=data.get("context_preview", ""),
            run_id=data.get("run_id", ""),
            recall_score=data.get("recall_score"),
            write_time_avg_ms=data.get("write_time_avg_ms", 0.0),
            read_time_ms=data.get("read_time_ms", 0.0),
            memory_size_tokens=data.get("memory_size_tokens", 0),
            retrieved_step_ids=data.get("retrieved_step_ids", []),
            target_step_ids=data.get("target_step_ids", []),
        )


@dataclass
class AggregateMetrics:
    """聚合评估指标"""

    # 准确率
    accuracy: float = 0.0
    accuracy_by_type: Dict[str, float] = field(default_factory=dict)
    accuracy_by_scenario: Dict[str, float] = field(default_factory=dict)

    # 召回率
    avg_recall_at_10: float = 0.0
    recall_by_type: Dict[str, float] = field(default_factory=dict)

    # 效率
    avg_write_time_ms: float = 0.0
    avg_read_time_ms: float = 0.0
    write_time_by_type: Dict[str, float] = field(default_factory=dict)
    read_time_by_type: Dict[str, float] = field(default_factory=dict)

    # 容量
    capacity_curve: List[Tuple[int, float]] = field(default_factory=list)

    # 统计
    total_trajectories: int = 0
    correct_count: int = 0
    by_type_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "accuracy": self.accuracy,
            "accuracy_by_type": self.accuracy_by_type,
            "accuracy_by_scenario": self.accuracy_by_scenario,
            "avg_recall_at_10": self.avg_recall_at_10,
            "recall_by_type": self.recall_by_type,
            "avg_write_time_ms": self.avg_write_time_ms,
            "avg_read_time_ms": self.avg_read_time_ms,
            "write_time_by_type": self.write_time_by_type,
            "read_time_by_type": self.read_time_by_type,
            "capacity_curve": self.capacity_curve,
            "total_trajectories": self.total_trajectories,
            "correct_count": self.correct_count,
            "by_type_stats": self.by_type_stats,
        }


# 问题类型常量
FACTUAL_QUESTION_TYPES = [
    "Single-hop",
    "Multi-hop",
    "Comparative",
    "Aggregative",
    "Post_processing",
    "Knowledge_updating",
    "Single-session-assistant",
    "Multi-session-assistant",
]

REFLECTIVE_QUESTION_TYPES = [
    "Emotion",
    "Preference",
]

ALL_QUESTION_TYPES = FACTUAL_QUESTION_TYPES + REFLECTIVE_QUESTION_TYPES
