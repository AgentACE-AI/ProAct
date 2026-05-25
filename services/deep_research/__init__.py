"""
Deep Research 模块。

提供深度调研功能，包括：
- 主题分析和规划
- 迭代搜索
- 知识综合
- 个性化报告生成
- 增量搜索支持
"""

from services.deep_research.models import (
    DeepResearchReport,
    ExtractedFact,
    FactConflict,
    KnowledgeGraph,
    ResearchConfig,
    ResearchDepth,
    ResearchOutline,
    ResearchPlan,
    ResearchState,
    ResearchTask,
    SearchIteration,
    SourceQuality,
    SourceResult,
    SubTopic,
)
from services.deep_research.orchestrator import DeepResearchOrchestrator
from services.deep_research.planner import ResearchPlanner
from services.deep_research.searcher import IterativeSearcher
from services.deep_research.synthesizer import KnowledgeSynthesizer
from services.deep_research.report_builder import ReportBuilder


__all__ = [
    # 核心编排器
    "DeepResearchOrchestrator",
    # 组件
    "ResearchPlanner",
    "IterativeSearcher",
    "KnowledgeSynthesizer",
    "ReportBuilder",
    # 数据模型
    "ResearchTask",
    "ResearchState",
    "ResearchDepth",
    "ResearchConfig",
    "ResearchPlan",
    "ResearchOutline",
    "SubTopic",
    "SearchIteration",
    "SourceResult",
    "SourceQuality",
    "ExtractedFact",
    "KnowledgeGraph",
    "FactConflict",
    "DeepResearchReport",
]
