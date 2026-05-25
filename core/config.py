"""
统一配置管理。

提供全局配置、Agent 配置、存储配置等统一管理。
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import re

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM API 配置"""
    api_key: str = ""
    base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    timeout: float | None = None

    def __post_init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", self.api_key)
        self.base_url = os.getenv("OPENAI_BASE_URL", self.base_url) or "https://api.openai.com/v1"
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        timeout = os.getenv("OPENAI_TIMEOUT")
        if timeout not in (None, ""):
            self.timeout = float(timeout)
        elif self.timeout is not None:
            self.timeout = float(self.timeout)


@dataclass
class AgentModelConfig:
    """单个 Agent 的模型配置"""
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    seed: int | None = None


@dataclass
class AgentsConfig:
    """所有 Agent 的配置"""

    # 对话相关 (需要快速响应)
    topic_monitor: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o-mini", temperature=0.3)
    )
    planning: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o-mini", temperature=0.3)
    )
    assistant: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o", temperature=0.4)
    )

    # 主动服务 (后台运行)
    memory_critic: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o", temperature=0.3)
    )
    research_predictor: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o-mini", temperature=0.3)
    )
    research_evaluator: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o-mini", temperature=0.3)
    )
    push_score: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o-mini", temperature=0.3)
    )

    # 工具 Agent
    search_planner: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o", temperature=0.3)
    )
    memory_update: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o-mini", temperature=0.3)
    )
    report_generator: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o", temperature=0.7)
    )

    # 对话内主动预判
    need_predictor: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o-mini", temperature=0.3)
    )

    # 用户模拟器
    user_simulator: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(model="gpt-4o-mini", temperature=0.5)
    )

    def set_seed_all(self, seed: int) -> None:
        """Set seed on all agent configs for reproducibility."""
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if isinstance(val, AgentModelConfig):
                val.seed = seed


@dataclass
class TasksConfig:
    """
    后台任务配置。

    任务分为两类：
    1. 全局任务（对所有用户执行）：proactive_search, memory_maintenance
    2. 在线任务（仅对在线用户执行）：initiative_check

    注：idle_check 功能已移至 WebSocket 断开事件处理
    注：memory_validation 和 stale_check 已合并为 memory_maintenance
    """
    enabled: bool = True

    # 全局任务间隔
    proactive_search_interval: int = 300      # 秒 - 主动搜索
    memory_maintenance_interval: int = 900    # 秒 - 记忆维护（合并验证+过期检查）

    # 在线任务间隔
    initiative_check_interval: int = 180      # 秒 - 主动对话检查

    # 推送阈值
    push_threshold_blocking: int = 70
    push_threshold_notify: int = 40

    # 研究价值评估阈值
    research_value_threshold: int = 60  # 分数 >= 此值才执行搜索

    # 开发模式下的快速任务间隔
    dev_proactive_search_interval: int = 5
    dev_memory_maintenance_interval: int = 120
    dev_initiative_check_interval: int = 150

    def __post_init__(self):
        self.enabled = os.getenv("ENABLE_PROACTIVE_TASKS", "true").lower() == "true"

        for name in ("push_threshold_blocking", "push_threshold_notify", "research_value_threshold"):
            value = getattr(self, name)
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100], got {value}")

        if self.push_threshold_blocking < self.push_threshold_notify:
            raise ValueError(
                f"push_threshold_blocking ({self.push_threshold_blocking}) "
                f"must be >= push_threshold_notify ({self.push_threshold_notify})"
            )

    def get_intervals(self, debug_mode: bool = False) -> dict:
        """
        获取任务间隔配置。

        Args:
            debug_mode: 是否为调试模式

        Returns:
            任务间隔配置字典
        """
        if debug_mode:
            return {
                "proactive_search": self.dev_proactive_search_interval,
                "memory_maintenance": self.dev_memory_maintenance_interval,
                "initiative_check": self.dev_initiative_check_interval,
            }
        else:
            return {
                "proactive_search": self.proactive_search_interval,
                "memory_maintenance": self.memory_maintenance_interval,
                "initiative_check": self.initiative_check_interval,
            }


@dataclass
class StorageConfig:
    """存储配置"""
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # 合法 user_id 的正则：1-64 个字母、数字、下划线或连字符
    _USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

    def __post_init__(self):
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)

    def validate_user_id(self, user_id: str) -> None:
        """校验 user_id 合法性，防止路径遍历攻击。"""
        if not self._USER_ID_PATTERN.match(user_id):
            raise ValueError(
                f"Invalid user_id: {user_id!r}. "
                f"Must match [A-Za-z0-9_-]{{1,64}}."
            )

    def get_user_dir(self, user_id: str) -> Path:
        """
        获取用户数据目录。

        Args:
            user_id: 用户 ID

        Returns:
            用户数据目录路径

        Raises:
            ValueError: user_id 不合法或存在路径遍历风险
        """
        self.validate_user_id(user_id)

        users_base = (self.data_dir / "users").resolve()
        path = (users_base / user_id).resolve()

        # 双重防护：即使正则通过，仍校验解析后的路径在预期目录内
        try:
            path.relative_to(users_base)
        except ValueError as e:
            raise ValueError(f"Unsafe user_id path traversal detected: {user_id!r}") from e

        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_vector_db_path(self, user_id: str) -> Path:
        """获取用户向量数据库路径"""
        return self.get_user_dir(user_id) / "chroma"

    def get_knowledge_index_path(self, user_id: str) -> Path:
        """获取用户知识索引数据库路径"""
        return self.get_user_dir(user_id) / "knowledge_index.db"

    def get_profile_path(self, user_id: str) -> Path:
        """获取用户画像文件路径"""
        return self.get_user_dir(user_id) / "profile.json"

    def get_conversations_path(self, user_id: str) -> Path:
        """获取用户对话记录文件路径"""
        return self.get_user_dir(user_id) / "conversations.json"

    def get_facts_path(self, user_id: str) -> Path:
        """获取用户事实文件路径"""
        return self.get_user_dir(user_id) / "facts.json"

    def get_reports_dir(self, user_id: str) -> Path:
        """获取用户报告目录路径"""
        path = self.get_user_dir(user_id) / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_pending_reports_path(self, user_id: str) -> Path:
        """获取用户待推送报告文件路径"""
        return self.get_user_dir(user_id) / "pending_reports.json"

    def get_proactive_items_path(self, user_id: str) -> Path:
        """获取用户主动项存储文件路径"""
        return self.get_user_dir(user_id) / "proactive_items.json"

    def user_exists(self, user_id: str) -> bool:
        """检查用户是否存在"""
        return self.get_profile_path(user_id).exists()

    def get_all_existing_user_ids(self) -> list:
        """
        获取磁盘上所有已存在的用户 ID。

        扫描 data/users/ 目录下的所有用户文件夹，
        只返回有效的用户（至少有 profile.json 文件）。

        Returns:
            所有存在的用户 ID 列表
        """
        users_dir = self.data_dir / "users"
        if not users_dir.exists():
            return []

        user_ids = []
        for user_dir in users_dir.iterdir():
            if user_dir.is_dir():
                # 验证是有效的用户目录（至少有 profile.json）
                if (user_dir / "profile.json").exists():
                    user_ids.append(user_dir.name)

        return user_ids


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    def __post_init__(self):
        self.host = os.getenv("SERVER_HOST", self.host)
        self.port = int(os.getenv("SERVER_PORT", str(self.port)))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        if not 1 <= self.port <= 65535:
            raise ValueError(f"SERVER_PORT must be in [1, 65535], got {self.port}")


@dataclass
class SearchConfig:
    """搜索配置"""
    max_queries_per_search: int = 5
    max_results_per_query: int = 3
    serper_api_key: str = ""
    jina_api_key: str = ""
    prefer_serper: bool = False
    fetch_timeout: int = 30

    def __post_init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY", self.serper_api_key)
        self.jina_api_key = os.getenv("JINA_API_KEY", self.jina_api_key)
        self.prefer_serper = os.getenv("PREFER_SERPER", "false").lower() == "true"


@dataclass
class DeepResearchConfig:
    """深度调研配置"""
    enabled: bool = True
    default_depth: str = "standard"  # quick | standard | deep

    # Standard 模式参数
    standard_max_iterations: int = 10
    standard_max_sources: int = 50
    standard_max_duration_minutes: int = 15

    # Quick 模式参数
    quick_max_iterations: int = 3
    quick_max_sources: int = 20
    quick_max_duration_minutes: int = 5

    # Deep 模式参数
    deep_max_iterations: int = 20
    deep_max_sources: int = 100
    deep_max_duration_minutes: int = 30

    # 增量搜索配置
    incremental_search_enabled: bool = True
    reuse_threshold: float = 0.8       # 覆盖度超过此值时直接复用已有调研
    incremental_threshold: float = 0.5  # 覆盖度超过此值时进行增量搜索
    topic_similarity_threshold: float = 0.75  # 主题相似度阈值

    # 质量控制
    min_source_quality: float = 0.6
    dedup_similarity_threshold: float = 0.85

    # 子主题控制
    max_subtopics: int = 6
    subtopic_depth: int = 2

    def __post_init__(self):
        self.enabled = os.getenv("DEEP_RESEARCH_ENABLED", "true").lower() == "true"
        self.default_depth = os.getenv("DEEP_RESEARCH_DEPTH", self.default_depth)

        for name in ("reuse_threshold", "incremental_threshold",
                     "topic_similarity_threshold", "min_source_quality",
                     "dedup_similarity_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")

        if self.reuse_threshold < self.incremental_threshold:
            raise ValueError(
                f"reuse_threshold ({self.reuse_threshold}) "
                f"must be >= incremental_threshold ({self.incremental_threshold})"
            )

    def get_config_for_depth(self, depth: str = None) -> dict:
        """
        获取指定深度的配置参数。

        Args:
            depth: 深度级别 (quick | standard | deep)

        Returns:
            配置参数字典
        """
        depth = depth or self.default_depth

        if depth == "quick":
            return {
                "max_iterations": self.quick_max_iterations,
                "max_sources": self.quick_max_sources,
                "max_duration_minutes": self.quick_max_duration_minutes,
            }
        elif depth == "deep":
            return {
                "max_iterations": self.deep_max_iterations,
                "max_sources": self.deep_max_sources,
                "max_duration_minutes": self.deep_max_duration_minutes,
            }
        else:  # standard
            return {
                "max_iterations": self.standard_max_iterations,
                "max_sources": self.standard_max_sources,
                "max_duration_minutes": self.standard_max_duration_minutes,
            }


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    to_file: bool = True
    log_dir: str = "logs"
    retention_days: int = 7

    def __post_init__(self):
        self.level = os.getenv("LOG_LEVEL", self.level)
        self.to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
        self.log_dir = os.getenv("LOG_DIR", self.log_dir)
        self.retention_days = int(os.getenv("LOG_RETENTION_DAYS", str(self.retention_days)))


@dataclass
class ProactiveConfig:
    """对话内主动预判配置"""
    mode: str = "hybrid"
    turn_signal_backend: str = "llm"
    turn_signal_model: str = "gpt-4o-mini"
    enable_llm_turn_signals_in_need_only: bool = False
    in_conversation_enabled: bool = True
    confidence_threshold: float = 0.7
    max_proactive_items: int = 2
    min_conversation_turns: int = 1
    inline_min_turns: int = 1
    min_message_chars_for_need_prediction: int = 20
    inline_same_topic_cooldown_turns: int = 2
    same_dedupe_cooldown_minutes: int = 30
    background_topic_cooldown_minutes: int = 180
    max_active_items_per_user: int = 3
    max_active_items_per_topic: int = 1
    inline_trigger_threshold: float = 0.7
    push_trigger_threshold: int = 70
    queue_threshold: int = 40
    topic_cooldown_minutes: int = 180

    def __post_init__(self):
        self.mode = os.getenv("PROACTIVE_MODE", self.mode)
        self.turn_signal_backend = os.getenv(
            "PROACTIVE_TURN_SIGNAL_BACKEND",
            self.turn_signal_backend,
        )
        self.turn_signal_model = os.getenv(
            "PROACTIVE_TURN_SIGNAL_MODEL",
            self.turn_signal_model,
        )
        self.enable_llm_turn_signals_in_need_only = os.getenv(
            "PROACTIVE_ENABLE_LLM_TURN_SIGNALS_IN_NEED_ONLY",
            str(self.enable_llm_turn_signals_in_need_only),
        ).lower() in ("true", "1", "yes")
        self.in_conversation_enabled = os.getenv(
            "PROACTIVE_IN_CONVERSATION_ENABLED", str(self.in_conversation_enabled)
        ).lower() in ("true", "1", "yes")
        self.confidence_threshold = float(
            os.getenv("PROACTIVE_CONFIDENCE_THRESHOLD", str(self.confidence_threshold))
        )
        self.max_proactive_items = int(
            os.getenv("PROACTIVE_MAX_ITEMS", str(self.max_proactive_items))
        )
        self.min_conversation_turns = int(
            os.getenv("PROACTIVE_MIN_CONVERSATION_TURNS", str(self.min_conversation_turns))
        )
        self.inline_min_turns = int(
            os.getenv("PROACTIVE_INLINE_MIN_TURNS", str(self.inline_min_turns))
        )
        self.min_message_chars_for_need_prediction = int(
            os.getenv(
                "PROACTIVE_MIN_MESSAGE_CHARS_FOR_NEED_PREDICTION",
                str(self.min_message_chars_for_need_prediction),
            )
        )
        self.inline_same_topic_cooldown_turns = int(
            os.getenv(
                "PROACTIVE_INLINE_SAME_TOPIC_COOLDOWN_TURNS",
                str(self.inline_same_topic_cooldown_turns),
            )
        )
        self.same_dedupe_cooldown_minutes = int(
            os.getenv(
                "PROACTIVE_SAME_DEDUPE_COOLDOWN_MINUTES",
                str(self.same_dedupe_cooldown_minutes),
            )
        )
        self.background_topic_cooldown_minutes = int(
            os.getenv(
                "PROACTIVE_BACKGROUND_TOPIC_COOLDOWN_MINUTES",
                str(self.background_topic_cooldown_minutes),
            )
        )
        self.max_active_items_per_user = int(
            os.getenv(
                "PROACTIVE_MAX_ACTIVE_ITEMS_PER_USER",
                str(self.max_active_items_per_user),
            )
        )
        self.max_active_items_per_topic = int(
            os.getenv(
                "PROACTIVE_MAX_ACTIVE_ITEMS_PER_TOPIC",
                str(self.max_active_items_per_topic),
            )
        )
        self.inline_trigger_threshold = float(
            os.getenv(
                "PROACTIVE_INLINE_TRIGGER_THRESHOLD",
                str(self.inline_trigger_threshold),
            )
        )
        self.push_trigger_threshold = int(
            os.getenv(
                "PROACTIVE_PUSH_TRIGGER_THRESHOLD",
                str(self.push_trigger_threshold),
            )
        )
        self.queue_threshold = int(
            os.getenv(
                "PROACTIVE_QUEUE_THRESHOLD",
                str(self.queue_threshold),
            )
        )
        self.topic_cooldown_minutes = int(
            os.getenv(
                "PROACTIVE_TOPIC_COOLDOWN_MINUTES",
                str(self.topic_cooldown_minutes),
            )
        )

        if self.mode not in {"need_only", "research", "hybrid"}:
            raise ValueError(
                f"PROACTIVE_MODE must be one of need_only/research/hybrid, got {self.mode!r}"
            )
        if self.turn_signal_backend not in {"rule", "llm"}:
            raise ValueError(
                "PROACTIVE_TURN_SIGNAL_BACKEND must be one of rule/llm, "
                f"got {self.turn_signal_backend!r}"
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"PROACTIVE_CONFIDENCE_THRESHOLD must be in [0.0, 1.0], got {self.confidence_threshold}"
            )
        if not 0.0 <= self.inline_trigger_threshold <= 1.0:
            raise ValueError(
                f"PROACTIVE_INLINE_TRIGGER_THRESHOLD must be in [0.0, 1.0], got {self.inline_trigger_threshold}"
            )
        for name in ("push_trigger_threshold", "queue_threshold"):
            value = getattr(self, name)
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100], got {value}")

        for name in (
            "max_proactive_items",
            "min_conversation_turns",
            "inline_min_turns",
            "min_message_chars_for_need_prediction",
            "inline_same_topic_cooldown_turns",
            "same_dedupe_cooldown_minutes",
            "background_topic_cooldown_minutes",
            "max_active_items_per_user",
            "max_active_items_per_topic",
            "topic_cooldown_minutes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


@dataclass
class Config:
    """总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    deep_research: DeepResearchConfig = field(default_factory=DeepResearchConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)

    @property
    def debug_mode(self) -> bool:
        """是否为调试模式"""
        return self.server.debug


# 全局配置实例
config = Config()
