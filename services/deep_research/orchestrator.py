"""
深度调研编排器。

负责研究任务的生命周期管理，协调各组件完成调研。
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from core.config import DeepResearchConfig
from core.models import UserProfile
from services.deep_research.models import (
    DeepResearchReport,
    ExtractedFact,
    ResearchConfig,
    ResearchState,
    ResearchTask,
)
from services.deep_research.planner import ResearchPlanner
from services.deep_research.searcher import IterativeSearcher
from services.deep_research.synthesizer import KnowledgeSynthesizer
from services.deep_research.report_builder import ReportBuilder

if TYPE_CHECKING:
    from memory.memory_system import MemorySystem


class DeepResearchOrchestrator:
    """
    深度调研编排器。

    职责:
    1. 研究任务的生命周期管理
    2. 协调 Planner, Searcher, Synthesizer, ReportBuilder
    3. 支持增量搜索
    4. 进度推送（通过回调）
    """

    def __init__(
        self,
        memory_system: "MemorySystem",
        planner: ResearchPlanner,
        searcher: IterativeSearcher,
        synthesizer: KnowledgeSynthesizer,
        report_builder: ReportBuilder,
        config: DeepResearchConfig,
        progress_callback: Optional[Callable[[ResearchTask, str, Dict], None]] = None,
    ):
        """
        初始化编排器。

        Args:
            memory_system: 记忆系统
            planner: 研究规划器
            searcher: 迭代搜索器
            synthesizer: 知识综合器
            report_builder: 报告生成器
            config: 深度调研配置
            progress_callback: 进度回调函数 (task, event, data)
        """
        self.memory = memory_system
        self.planner = planner
        self.searcher = searcher
        self.synthesizer = synthesizer
        self.report_builder = report_builder
        self.config = config
        self.progress_callback = progress_callback

        # 活跃任务
        self._active_tasks: Dict[str, ResearchTask] = {}

    async def start_research(
        self,
        topic: str,
        config: Optional[ResearchConfig] = None,
        source: str = "proactive",
        incremental: bool = True,
    ) -> ResearchTask:
        """
        启动深度研究任务。

        Args:
            topic: 调研主题
            config: 研究配置（可选）
            source: 来源 ("proactive" | "user_request")
            incremental: 是否启用增量搜索

        Returns:
            研究任务
        """
        # 创建研究配置
        if config is None:
            depth_config = self.config.get_config_for_depth()
            config = ResearchConfig(
                depth=self.config.default_depth,
                max_iterations=depth_config["max_iterations"],
                max_total_sources=depth_config["max_sources"],
                max_duration_minutes=depth_config["max_duration_minutes"],
            )

        # 创建任务
        task = ResearchTask(
            id=str(uuid.uuid4()),
            user_id=self.memory.user_id,
            original_topic=topic,
            config=config,
            source=source,
        )

        self._active_tasks[task.id] = task
        self._push_progress(task, "created", {"topic": topic})

        # 检查增量搜索
        if incremental and self.config.incremental_search_enabled:
            existing = self.memory.get_existing_research(
                topic,
                similarity_threshold=self.config.topic_similarity_threshold,
            )

            if existing:
                coverage = existing["coverage_score"]

                if coverage >= self.config.reuse_threshold * 100:
                    # 直接复用已有调研生成报告
                    print(f"[DeepResearchOrchestrator] 复用已有调研: {existing['related_topic']} (覆盖度: {coverage}%)")
                    return await self._generate_from_existing(task, existing)

                elif coverage >= self.config.incremental_threshold * 100:
                    # 增量搜索
                    print(f"[DeepResearchOrchestrator] 增量搜索已有调研: {existing['related_topic']} (覆盖度: {coverage}%)")
                    return await self._execute_incremental(task, existing)

        # 完整搜索
        return await self._execute_full_research(task)

    async def _execute_full_research(self, task: ResearchTask) -> ResearchTask:
        """执行完整的调研流程"""
        try:
            # 1. 规划阶段
            task.update_progress(ResearchState.PLANNING, 5)
            self._push_progress(task, "planning_started", {})

            user_profile = self.memory.get_user_profile()
            plan = await self.planner.create_plan(
                topic=task.original_topic,
                user_profile=user_profile,
                config=task.config,
            )
            task.plan = plan
            task.refined_topic = plan.refined_topic

            self._push_progress(task, "planning_completed", {
                "subtopics": [st.name for st in plan.subtopics],
            })
            print(f"[DeepResearchOrchestrator] 规划完成，子主题: {[st.name for st in plan.subtopics]}")

            # 2. 搜索阶段
            task.update_progress(ResearchState.SEARCHING, 10)
            self._push_progress(task, "searching_started", {})

            all_sources = []
            queries_to_execute = list(plan.initial_queries)

            iteration_num = 0
            while not self.searcher.check_convergence(task.iterations, task.config):
                iteration_num += 1

                if not queries_to_execute:
                    # 从子主题获取更多查询
                    for st in plan.subtopics:
                        if not st.covered and st.queries:
                            queries_to_execute.extend([
                                {"query": q, "purpose": st.name, "priority": st.priority}
                                for q in st.queries
                            ])
                            st.covered = True
                            break

                if not queries_to_execute:
                    break

                # 执行本轮搜索
                iteration = await self.searcher.execute_iteration(
                    iteration_number=iteration_num,
                    queries=queries_to_execute[:3],  # 每轮最多3个查询
                    existing_sources=all_sources,
                    research_config=task.config,
                    topic=task.refined_topic,
                )

                task.iterations.append(iteration)
                all_sources.extend(iteration.sources_found)

                # 使用发现的新问题作为下一轮查询
                queries_to_execute = queries_to_execute[3:]  # 移除已执行的
                if iteration.new_questions:
                    queries_to_execute.extend([
                        {"query": q, "purpose": "深入探索", "priority": 6}
                        for q in iteration.new_questions[:2]
                    ])

                # 更新进度
                progress = 10 + int(60 * iteration_num / task.config.max_iterations)
                task.update_progress(ResearchState.SEARCHING, min(progress, 70))
                self._push_progress(task, "iteration_completed", {
                    "iteration": iteration_num,
                    "sources_added": iteration.sources_added,
                    "facts_found": len(iteration.new_facts),
                })

            # 3. 综合阶段
            task.update_progress(ResearchState.SYNTHESIZING, 75)
            self._push_progress(task, "synthesizing_started", {})

            knowledge_graph = await self.synthesizer.synthesize(
                iterations=task.iterations,
                plan=plan,
            )
            task.knowledge_graph = knowledge_graph

            self._push_progress(task, "synthesizing_completed", {
                "total_facts": len(knowledge_graph.facts),
                "conflicts": len(knowledge_graph.conflicts),
            })

            # 4. 报告生成阶段
            task.update_progress(ResearchState.BUILDING, 85)
            self._push_progress(task, "building_started", {})

            report = await self.report_builder.build(
                plan=plan,
                knowledge_graph=knowledge_graph,
                iterations=task.iterations,
                user_profile=user_profile,
            )

            # 5. 存储调研知识
            self._save_research_knowledge(task, knowledge_graph)

            # 6. 更新用户画像
            self.memory.profile_store.add_research_topic(task.refined_topic)

            # 7. 完成
            task.mark_completed(report)
            self._push_progress(task, "completed", {
                "report_id": report.id,
                "title": report.title,
            })

            return task

        except Exception as e:
            import traceback
            traceback.print_exc()
            task.mark_failed(str(e))
            self._push_progress(task, "failed", {"error": str(e)})
            return task

        finally:
            # 清理活跃任务
            if task.id in self._active_tasks:
                del self._active_tasks[task.id]

    async def _execute_incremental(
        self,
        task: ResearchTask,
        existing: Dict[str, Any],
    ) -> ResearchTask:
        """执行增量搜索"""
        try:
            task.update_progress(ResearchState.PLANNING, 5)
            self._push_progress(task, "incremental_started", {
                "existing_topic": existing["related_topic"],
                "coverage": existing["coverage_score"],
            })

            # 1. 规划
            user_profile = self.memory.get_user_profile()

            # 格式化已有事实
            existing_facts_text = "\n".join([
                f.get("summary_preview", f.get("content", "")[:100])
                for f in existing["facts"][:10]
            ])

            plan = await self.planner.create_plan(
                topic=task.original_topic,
                user_profile=user_profile,
                config=task.config,
                existing_knowledge=existing_facts_text,
            )
            task.plan = plan
            task.refined_topic = plan.refined_topic

            # 2. 分析缺口
            gaps = await self.planner.analyze_incremental_gaps(
                new_topic=task.original_topic,
                existing_topic=existing["related_topic"],
                similarity=existing["similarity"],
                existing_facts=existing["facts"],
                target_subtopics=[st.name for st in plan.subtopics],
            )

            # 3. 只搜索缺失的部分
            task.update_progress(ResearchState.SEARCHING, 20)

            supplementary_queries = gaps.get("supplementary_queries", [])
            if supplementary_queries:
                all_sources = []
                for i, query_data in enumerate(supplementary_queries[:5]):
                    iteration = await self.searcher.execute_iteration(
                        iteration_number=i + 1,
                        queries=[query_data],
                        existing_sources=all_sources,
                        research_config=task.config,
                        topic=task.refined_topic,
                    )
                    task.iterations.append(iteration)
                    all_sources.extend(iteration.sources_found)

                    progress = 20 + int(40 * (i + 1) / len(supplementary_queries))
                    task.update_progress(ResearchState.SEARCHING, min(progress, 60))

            # 4. 合并已有事实和新事实
            task.update_progress(ResearchState.SYNTHESIZING, 65)

            # 转换已有事实
            all_facts = existing["facts"] + [
                f.to_dict() for it in task.iterations for f in it.new_facts
            ]

            knowledge_graph = await self.synthesizer.synthesize_from_facts(
                facts=all_facts,
                plan=plan,
            )
            task.knowledge_graph = knowledge_graph

            # 5. 生成报告
            task.update_progress(ResearchState.BUILDING, 80)

            report = await self.report_builder.build(
                plan=plan,
                knowledge_graph=knowledge_graph,
                iterations=task.iterations,
                user_profile=user_profile,
            )

            # 6. 存储新发现的知识
            self._save_research_knowledge(task, knowledge_graph, new_only=True)

            # 7. 更新用户画像
            self.memory.profile_store.add_research_topic(task.refined_topic)

            # 8. 完成
            task.mark_completed(report)
            self._push_progress(task, "completed", {
                "report_id": report.id,
                "title": report.title,
                "incremental": True,
            })

            return task

        except Exception as e:
            import traceback
            traceback.print_exc()
            task.mark_failed(str(e))
            self._push_progress(task, "failed", {"error": str(e)})
            return task

        finally:
            if task.id in self._active_tasks:
                del self._active_tasks[task.id]

    async def _generate_from_existing(
        self,
        task: ResearchTask,
        existing: Dict[str, Any],
    ) -> ResearchTask:
        """直接从已有调研生成报告"""
        try:
            task.update_progress(ResearchState.PLANNING, 10)
            self._push_progress(task, "reusing_existing", {
                "existing_topic": existing["related_topic"],
                "coverage": existing["coverage_score"],
            })

            # 1. 简化规划
            user_profile = self.memory.get_user_profile()

            existing_facts_text = "\n".join([
                f.get("summary_preview", f.get("content", "")[:100])
                for f in existing["facts"][:10]
            ])

            plan = await self.planner.create_plan(
                topic=task.original_topic,
                user_profile=user_profile,
                config=task.config,
                existing_knowledge=existing_facts_text,
            )
            task.plan = plan
            task.refined_topic = plan.refined_topic

            # 2. 直接综合已有知识
            task.update_progress(ResearchState.SYNTHESIZING, 40)

            knowledge_graph = await self.synthesizer.synthesize_from_facts(
                facts=existing["facts"],
                plan=plan,
            )
            task.knowledge_graph = knowledge_graph

            # 3. 生成报告
            task.update_progress(ResearchState.BUILDING, 70)

            report = await self.report_builder.build(
                plan=plan,
                knowledge_graph=knowledge_graph,
                iterations=[],  # 无新搜索
                user_profile=user_profile,
            )

            # 4. 更新用户画像
            self.memory.profile_store.add_research_topic(task.refined_topic)

            # 5. 完成
            task.mark_completed(report)
            self._push_progress(task, "completed", {
                "report_id": report.id,
                "title": report.title,
                "reused": True,
            })

            return task

        except Exception as e:
            import traceback
            traceback.print_exc()
            task.mark_failed(str(e))
            self._push_progress(task, "failed", {"error": str(e)})
            return task

        finally:
            if task.id in self._active_tasks:
                del self._active_tasks[task.id]

    def _save_research_knowledge(
        self,
        task: ResearchTask,
        knowledge_graph,
        new_only: bool = False,
    ) -> None:
        """保存调研知识到记忆系统"""
        facts_to_save = []

        if new_only:
            # 只保存新迭代中的事实
            for iteration in task.iterations:
                for fact in iteration.new_facts:
                    facts_to_save.append(fact.to_dict())
        else:
            # 保存所有事实
            for fact in knowledge_graph.facts:
                facts_to_save.append(fact.to_dict())

        if facts_to_save:
            self.memory.add_research_knowledge(
                topic=task.refined_topic,
                task_id=task.id,
                facts=facts_to_save,
                sources=[],
            )

    def _push_progress(
        self,
        task: ResearchTask,
        event: str,
        data: Dict[str, Any],
    ) -> None:
        """推送进度更新"""
        if self.progress_callback:
            try:
                self.progress_callback(task, event, data)
            except Exception as e:
                print(f"[DeepResearchOrchestrator] 进度回调失败: {e}")

    def get_active_task(self, task_id: str) -> Optional[ResearchTask]:
        """获取活跃任务"""
        return self._active_tasks.get(task_id)

    def get_all_active_tasks(self) -> List[ResearchTask]:
        """获取所有活跃任务"""
        return list(self._active_tasks.values())
