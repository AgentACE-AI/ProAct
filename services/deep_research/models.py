"""
Deep Research 数据模型定义。

包含研究任务、搜索迭代、知识图谱等核心数据结构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ResearchState(Enum):
    """研究任务状态"""
    CREATED = "created"
    PLANNING = "planning"
    SEARCHING = "searching"
    SYNTHESIZING = "synthesizing"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchDepth(Enum):
    """研究深度"""
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class ResearchConfig:
    """研究配置"""
    depth: str = "standard"  # quick | standard | deep
    max_iterations: int = 10
    max_sources_per_iteration: int = 5
    max_total_sources: int = 50
    max_duration_minutes: int = 15
    min_source_quality: float = 0.6
    dedup_similarity_threshold: float = 0.85
    max_subtopics: int = 6
    subtopic_depth: int = 2

    @classmethod
    def quick(cls) -> "ResearchConfig":
        """快速模式配置"""
        return cls(
            depth="quick",
            max_iterations=1,
            max_sources_per_iteration=3,
            max_total_sources=20,
            max_duration_minutes=5,
            max_subtopics=4,
            subtopic_depth=1,
        )

    @classmethod
    def standard(cls) -> "ResearchConfig":
        """标准模式配置"""
        return cls(
            depth="standard",
            max_iterations=10,
            max_sources_per_iteration=5,
            max_total_sources=50,
            max_duration_minutes=15,
            max_subtopics=6,
            subtopic_depth=2,
        )

    @classmethod
    def deep(cls) -> "ResearchConfig":
        """深度模式配置"""
        return cls(
            depth="deep",
            max_iterations=20,
            max_sources_per_iteration=5,
            max_total_sources=100,
            max_duration_minutes=30,
            max_subtopics=8,
            subtopic_depth=3,
        )


@dataclass
class SubTopic:
    """子主题"""
    name: str
    priority: int = 5  # 1-10
    queries: List[str] = field(default_factory=list)
    covered: bool = False
    fact_count: int = 0


@dataclass
class ResearchOutline:
    """研究大纲"""
    sections: List[str] = field(default_factory=list)
    key_questions: List[str] = field(default_factory=list)


@dataclass
class ResearchPlan:
    """研究计划"""
    original_topic: str
    refined_topic: str
    research_goals: List[str] = field(default_factory=list)
    subtopics: List[SubTopic] = field(default_factory=list)
    initial_queries: List[Dict[str, Any]] = field(default_factory=list)
    outline: ResearchOutline = field(default_factory=ResearchOutline)
    suggested_title: str = ""
    complexity: str = "medium"  # simple | medium | complex
    domain: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "original_topic": self.original_topic,
            "refined_topic": self.refined_topic,
            "research_goals": self.research_goals,
            "subtopics": [
                {"name": st.name, "priority": st.priority, "queries": st.queries}
                for st in self.subtopics
            ],
            "initial_queries": self.initial_queries,
            "outline": {
                "sections": self.outline.sections,
                "key_questions": self.outline.key_questions,
            },
            "suggested_title": self.suggested_title,
            "complexity": self.complexity,
            "domain": self.domain,
        }


@dataclass
class SourceQuality:
    """来源质量评估"""
    relevance: float = 0.0  # 0-1, 与主题相关性
    credibility: float = 0.0  # 0-1, 可信度
    freshness: float = 0.0  # 0-1, 时效性
    overall: float = 0.0  # 综合得分


@dataclass
class SourceResult:
    """搜索来源结果"""
    url: str
    title: str
    snippet: str
    full_content: str = ""
    quality: SourceQuality = field(default_factory=SourceQuality)
    extracted_facts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "full_content": self.full_content,
            "quality": {
                "relevance": self.quality.relevance,
                "credibility": self.quality.credibility,
                "freshness": self.quality.freshness,
                "overall": self.quality.overall,
            },
            "extracted_facts": self.extracted_facts,
        }


@dataclass
class ExecutedQuery:
    """已执行的查询"""
    query: str
    purpose: str
    priority: int
    results_count: int = 0
    success: bool = True
    error: str = ""


@dataclass
class ExtractedFact:
    """提取的事实"""
    content: str
    source_url: str
    source_title: str
    subtopic: str = ""
    confidence: float = 0.8
    summary: str = ""
    publish_date: str = ""  # 网页发布日期（如果能提取到）
    access_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))  # 访问日期

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "subtopic": self.subtopic,
            "confidence": self.confidence,
            "summary": self.summary,
            "publish_date": self.publish_date,
            "access_date": self.access_date,
        }

    def to_citation(self, index: int) -> str:
        """生成引用格式"""
        date_str = self.publish_date if self.publish_date else f"Accessed on {self.access_date}"
        return f"[{index}] {self.source_title}. {date_str}. {self.source_url}"


@dataclass
class SearchIteration:
    """搜索迭代"""
    iteration_number: int
    queries_executed: List[ExecutedQuery] = field(default_factory=list)
    sources_found: List[SourceResult] = field(default_factory=list)
    sources_added: int = 0
    new_facts: List[ExtractedFact] = field(default_factory=list)
    new_questions: List[str] = field(default_factory=list)  # 发现的新问题
    discovered_subtopics: List[str] = field(default_factory=list)
    coverage_improvement: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "iteration_number": self.iteration_number,
            "queries_executed": [
                {"query": q.query, "purpose": q.purpose, "results": q.results_count}
                for q in self.queries_executed
            ],
            "sources_found": len(self.sources_found),
            "sources_added": self.sources_added,
            "new_facts_count": len(self.new_facts),
            "new_questions": self.new_questions,
            "discovered_subtopics": self.discovered_subtopics,
            "coverage_improvement": self.coverage_improvement,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class FactConflict:
    """事实冲突"""
    fact1: str
    fact2: str
    source1: str
    source2: str
    resolution: str = ""  # 解决建议


@dataclass
class KnowledgeGraph:
    """知识图谱"""
    facts: List[ExtractedFact] = field(default_factory=list)
    facts_by_subtopic: Dict[str, List[ExtractedFact]] = field(default_factory=dict)
    conflicts: List[FactConflict] = field(default_factory=list)
    coverage_by_subtopic: Dict[str, float] = field(default_factory=dict)
    total_coverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_facts": len(self.facts),
            "facts_by_subtopic": {
                st: len(facts) for st, facts in self.facts_by_subtopic.items()
            },
            "conflicts_count": len(self.conflicts),
            "coverage_by_subtopic": self.coverage_by_subtopic,
            "total_coverage": self.total_coverage,
        }


@dataclass
class DeepResearchReport:
    """深度调研报告"""
    id: str
    topic: str
    title: str
    summary: str
    content: str
    sources: List[SourceResult] = field(default_factory=list)
    methodology: str = ""
    limitations: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "topic": self.topic,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "methodology": self.methodology,
            "limitations": self.limitations,
            "created_at": self.created_at,
        }


@dataclass
class ResearchTask:
    """研究任务"""
    id: str
    user_id: str
    original_topic: str
    refined_topic: str = ""
    state: ResearchState = ResearchState.CREATED
    progress: int = 0  # 0-100
    config: ResearchConfig = field(default_factory=ResearchConfig)
    plan: Optional[ResearchPlan] = None
    iterations: List[SearchIteration] = field(default_factory=list)
    knowledge_graph: Optional[KnowledgeGraph] = None
    final_report: Optional[DeepResearchReport] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    source: str = "proactive"  # proactive | user_request

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "original_topic": self.original_topic,
            "refined_topic": self.refined_topic,
            "state": self.state.value,
            "progress": self.progress,
            "config": {
                "depth": self.config.depth,
                "max_iterations": self.config.max_iterations,
            },
            "plan": self.plan.to_dict() if self.plan else None,
            "iterations_count": len(self.iterations),
            "knowledge_graph": self.knowledge_graph.to_dict() if self.knowledge_graph else None,
            "has_report": self.final_report is not None,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "source": self.source,
        }

    def update_progress(self, state: ResearchState, progress: int) -> None:
        """更新任务进度"""
        self.state = state
        self.progress = min(100, max(0, progress))

    def mark_completed(self, report: DeepResearchReport) -> None:
        """标记任务完成"""
        self.state = ResearchState.COMPLETED
        self.progress = 100
        self.final_report = report
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """标记任务失败"""
        self.state = ResearchState.FAILED
        self.error = error
        self.completed_at = datetime.now()
