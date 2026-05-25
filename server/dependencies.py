"""
依赖注入配置。

在这里组装所有依赖关系，为每个用户创建独立的依赖实例。
"""

from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

from core.config import Config, config
from tools.llm_client import LLMClient
from tools.search_tool import SearchTool
from tools.web_fetcher import WebFetcher
from memory.memory_system import MemorySystem
from agents.topic_monitor import TopicMonitorAgent
from agents.planning_agent import PlanningAgent
from agents.assistant_agent import AssistantAgent
from agents.search_planner import SearchPlannerAgent
from agents.report_generator import ReportGeneratorAgent
from agents.memory_critic import MemoryCriticAgent
from agents.research_predictor import ResearchTopicPredictor
from agents.research_evaluator import ResearchValueEvaluator
from agents.push_score import PushScoreAgent
from agents.need_predictor import ConversationNeedPredictor
from services.search_service import SearchService
from services.report_service import ReportService
from services.push_service import PushService
from services.proactive.brief_service import ProactiveBriefService
from services.proactive.decision_service import ProactiveDecisionService
from services.proactive.eligibility import ProactiveEligibilityGate
from services.proactive.item_service import ProactiveItemService
from services.proactive.policy import ProactivePolicy
from services.proactive.turn_signals import build_turn_signal_extractor
from services.deep_research import (
    DeepResearchOrchestrator,
    ResearchPlanner,
    IterativeSearcher,
    KnowledgeSynthesizer,
    ReportBuilder,
)


@lru_cache()
def get_config() -> Config:
    """
    获取全局配置。

    Returns:
        全局配置实例
    """
    return config


@lru_cache()
def get_llm_client() -> LLMClient:
    """
    获取 LLM 客户端。

    Returns:
        LLM 客户端实例
    """
    return LLMClient(get_config().llm)


@lru_cache()
def get_search_tool() -> SearchTool:
    """
    获取搜索工具。

    Returns:
        搜索工具实例
    """
    return SearchTool(get_config().search)


@lru_cache()
def get_web_fetcher() -> WebFetcher:
    """
    获取网页抓取工具。

    Returns:
        网页抓取工具实例
    """
    return WebFetcher(get_config().search)


class UserDependencies:
    """
    用户级别的依赖。

    每个用户有独立的实例，包括记忆系统、Agent 和 Service。
    这确保了多用户之间的数据隔离。
    """

    def __init__(self, user_id: str):
        """
        初始化用户依赖。

        Args:
            user_id: 用户 ID
        """
        self.user_id = user_id
        self._config = get_config()
        self._llm = get_llm_client()
        self._search_tool = get_search_tool()
        self._web_fetcher = get_web_fetcher()

        # 初始化记忆系统
        self.memory = MemorySystem(user_id, self._config, self._llm)

        # 会话状态跟踪
        self.last_interaction: datetime = datetime.now()
        self._message_timestamps: List[datetime] = []
        self._max_timestamps: int = 100

        # 初始化 Agent 实例
        self._init_agents()

        # 初始化 Service 实例
        self._init_services()

    def _init_agents(self) -> None:
        """初始化所有 Agent"""
        # 话题监控 Agent
        self.topic_monitor = TopicMonitorAgent(
            config=self._config.agents.topic_monitor,
            llm_client=self._llm,
        )

        # 规划 Agent（需要记忆系统）
        self.planning_agent = PlanningAgent(
            config=self._config.agents.planning,
            llm_client=self._llm,
            memory_system=self.memory,
        )

        # 助手 Agent（需要记忆系统）
        self.assistant_agent = AssistantAgent(
            config=self._config.agents.assistant,
            llm_client=self._llm,
            memory_system=self.memory,
        )

        # 搜索规划 Agent
        self.search_planner = SearchPlannerAgent(
            config=self._config.agents.search_planner,
            llm_client=self._llm,
        )

        # 报告生成 Agent
        self.report_generator = ReportGeneratorAgent(
            config=self._config.agents.report_generator,
            llm_client=self._llm,
        )

        # 记忆批判 Agent
        self.memory_critic = MemoryCriticAgent(
            config=self._config.agents.memory_critic,
            llm_client=self._llm,
        )

        # 研究主题预测 Agent
        self.research_predictor = ResearchTopicPredictor(
            config=self._config.agents.research_predictor,
            llm_client=self._llm,
        )

        # 研究价值评估 Agent
        self.research_evaluator = ResearchValueEvaluator(
            config=self._config.agents.research_evaluator,
            llm_client=self._llm,
        )

        # 推送评分 Agent
        self.push_score_agent = PushScoreAgent(
            config=self._config.agents.push_score,
            llm_client=self._llm,
        )

        # 对话内需求预测 Agent（需要记忆系统）
        self.need_predictor = ConversationNeedPredictor(
            config=self._config.agents.need_predictor,
            llm_client=self._llm,
            memory_system=self.memory,
        )

    def _init_services(self) -> None:
        """初始化所有 Service"""
        # 搜索服务
        self.search_service = SearchService(
            memory_system=self.memory,
            search_planner=self.search_planner,
            report_generator=self.report_generator,
            search_tool=self._search_tool,
            web_fetcher=self._web_fetcher,
            llm_client=self._llm,
            config=self._config,
        )

        # 报告服务
        self.report_service = ReportService(
            user_id=self.user_id,
            storage_config=self._config.storage,
        )

        self.proactive_item_service = ProactiveItemService(
            user_id=self.user_id,
            storage_config=self._config.storage,
        )

        # 推送服务
        self.push_service = PushService(
            push_score_agent=self.push_score_agent,
            config=self._config.tasks,
        )

        self.proactive_policy = ProactivePolicy(self._config.proactive.mode)
        self.proactive_eligibility_gate = ProactiveEligibilityGate(
            self._config.proactive
        )
        self.proactive_brief_service = ProactiveBriefService()
        self.proactive_turn_signal_extractor = build_turn_signal_extractor(
            config=self._config.proactive,
            llm_client=self._llm,
        )
        self.proactive_decision_service = ProactiveDecisionService(
            config=self._config.proactive,
            push_score_agent=self.push_score_agent,
        )

        # 深度调研服务
        self._init_deep_research()

    def _init_deep_research(self) -> None:
        """初始化深度调研组件"""
        # 研究规划器
        research_planner = ResearchPlanner(
            config=self._config.agents.search_planner,
            llm_client=self._llm,
        )

        # 迭代搜索器
        iterative_searcher = IterativeSearcher(
            config=self._config.agents.search_planner,
            llm_client=self._llm,
            search_tool=self._search_tool,
            web_fetcher=self._web_fetcher,
        )

        # 知识综合器
        knowledge_synthesizer = KnowledgeSynthesizer(
            config=self._config.agents.report_generator,
            llm_client=self._llm,
        )

        # 报告生成器
        deep_report_builder = ReportBuilder(
            config=self._config.agents.report_generator,
            llm_client=self._llm,
        )

        # 编排器
        self.deep_research_orchestrator = DeepResearchOrchestrator(
            memory_system=self.memory,
            planner=research_planner,
            searcher=iterative_searcher,
            synthesizer=knowledge_synthesizer,
            report_builder=deep_report_builder,
            config=self._config.deep_research,
            progress_callback=self._on_research_progress,
        )

    def _on_research_progress(self, task, event: str, data: dict) -> None:
        """
        调研进度回调。

        可以在这里推送进度到 WebSocket。
        """
        print(f"[DeepResearch] Task {task.id}: {event} - {data}")

    def get_user_profile_text(self) -> str:
        """
        获取用户画像的文本表示。

        Returns:
            用户画像文本
        """
        profile = self.memory.get_user_profile()
        return profile.to_prompt_text()

    def get_recent_topics(self) -> list:
        """
        获取用户最近的话题。

        Returns:
            最近话题列表
        """
        return self.memory.get_recent_topics()

    def get_all_topics(self) -> list:
        """
        获取用户所有话题。

        Returns:
            所有话题列表
        """
        return self.memory.get_all_topics()

    # ==================== 会话状态跟踪 ====================

    def record_interaction(self) -> None:
        """
        记录用户交互。

        在用户发送消息时调用此方法更新交互状态。
        """
        now = datetime.now()
        self.last_interaction = now
        self._message_timestamps.append(now)
        self._trim_timestamps()

    def _trim_timestamps(self) -> None:
        """修剪时间戳列表，只保留最近的记录"""
        if len(self._message_timestamps) > self._max_timestamps:
            self._message_timestamps = self._message_timestamps[-self._max_timestamps:]

    def get_seconds_since_last_interaction(self) -> int:
        """
        获取距离上次交互的秒数。

        Returns:
            距离上次交互的秒数
        """
        delta = datetime.now() - self.last_interaction
        return int(delta.total_seconds())

    def get_recent_interaction_count(self, window_seconds: int = 300) -> int:
        """
        获取指定时间窗口内的交互次数。

        Args:
            window_seconds: 时间窗口（秒），默认 5 分钟

        Returns:
            时间窗口内的交互次数
        """
        now = datetime.now()
        count = sum(
            1 for ts in self._message_timestamps
            if (now - ts).total_seconds() < window_seconds
        )
        return count

    def get_interaction_stats(self) -> dict:
        """
        获取交互统计信息。

        Returns:
            包含 seconds_since_last_interaction 和 recent_interaction_count 的字典
        """
        return {
            "seconds_since_last_interaction": self.get_seconds_since_last_interaction(),
            "recent_interaction_count": self.get_recent_interaction_count(),
        }


# 用户依赖缓存
_user_dependencies: Dict[str, UserDependencies] = {}


def get_user_dependencies(user_id: str) -> UserDependencies:
    """
    获取用户依赖。

    如果用户依赖不存在，则创建新的实例。

    Args:
        user_id: 用户 ID

    Returns:
        用户依赖实例
    """
    if user_id not in _user_dependencies:
        _user_dependencies[user_id] = UserDependencies(user_id)
    return _user_dependencies[user_id]


def clear_user_dependencies(user_id: str) -> None:
    """
    清除用户依赖。

    Args:
        user_id: 用户 ID
    """
    if user_id in _user_dependencies:
        del _user_dependencies[user_id]


def get_all_user_ids() -> list:
    """
    获取所有已加载的用户 ID。

    Returns:
        用户 ID 列表
    """
    return list(_user_dependencies.keys())


def clear_all_dependencies() -> None:
    """清除所有用户依赖"""
    _user_dependencies.clear()
