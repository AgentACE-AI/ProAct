"""
主动任务实现。

包含主动搜索、记忆验证、主动对话等后台任务的具体实现。
"""

import asyncio
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.models import (
    MemoryUpdateRequest,
    PushAction,
    Report,
    UpdateType,
)
from services.deep_research import ResearchConfig
from services.proactive import (
    DecisionContext,
    EligibilityContext,
    ProactiveCandidate,
    ProactiveItemState,
)

if TYPE_CHECKING:
    from server.dependencies import UserDependencies


async def run_proactive_search(deps: "UserDependencies", is_online: bool = False) -> List[Dict[str, Any]]:
    """
    执行主动搜索任务（使用 Deep Research）。

    流程:
    1. 收集研究主题（知识缺口 + 场景/关联预测）
    2. 价值评估过滤低价值主题
    3. 使用 DeepResearchOrchestrator 执行深度调研
    4. 生成报告并添加到待推送队列

    Args:
        deps: 用户依赖
        is_online: 用户是否在线
            - True: 侧重当前聊天内容和最近话题
            - False: 基于记忆系统的历史信息

    Returns:
        proactive delivery payloads
    """
    deliveries: List[Dict[str, Any]] = []
    print(f"Running proactive search... (online={is_online})")

    try:
        # 获取用户上下文
        memory = deps.memory
        recent_topics = memory.get_recent_topics()
        all_topics = memory.get_all_topics()

        # 冷启动检查：至少需要1个话题才进行搜索
        if not recent_topics and not all_topics:
            return deliveries

        if not deps.proactive_policy.allow_source("research_predictor"):
            return deliveries

        # 收集研究主题（根据在线/离线使用不同策略）
        research_topics = await _collect_research_topics(deps, is_online=is_online)

        if not research_topics:
            return deliveries

        # 取最高优先级的主题
        top_topic = research_topics[0]
        topic_name = top_topic.get("topic", "")

        if not topic_name:
            return deliveries

        # 检查是否启用深度调研
        if deps._config.deep_research.enabled:
            # 使用 Deep Research Orchestrator
            task = await deps.deep_research_orchestrator.start_research(
                topic=topic_name,
                config=ResearchConfig.quick(),
                source="proactive",
                incremental=True,  # 启用增量搜索
            )

            # 如果调研成功，创建报告
            if task.final_report:
                report = Report(
                    id=task.final_report.id,
                    topic=task.refined_topic,
                    title=task.final_report.title,
                    summary=task.final_report.summary,
                    content=task.final_report.content,
                    sources=[],  # Deep Research 的 sources 格式不同
                    relevance=top_topic.get("priority", 50),
                    urgency=50,
                )
                deliveries.extend(
                    _finalize_background_report(
                        deps=deps,
                        report=report,
                        is_online=is_online,
                        top_topic=top_topic,
                    )
                )
        else:
            # 使用原有的简单搜索
            result = deps.search_service.search_for_report(
                topic=topic_name,
                purpose=top_topic.get("reason", "主动搜索"),
            )

            if result.success and result.report:
                report = Report(
                    id=str(uuid.uuid4()),
                    topic=topic_name,
                    title=topic_name,
                    summary=result.report[:200] if result.report else "",
                    content=result.report,
                    sources=result.sources,
                    relevance=top_topic.get("priority", 50),
                    urgency=50,
                )
                deliveries.extend(
                    _finalize_background_report(
                        deps=deps,
                        report=report,
                        is_online=is_online,
                        top_topic=top_topic,
                    )
                )

    except Exception as e:
        print(f"[proactive_search] 错误: {e}")
        import traceback
        traceback.print_exc()

    return deliveries


def _finalize_background_report(
    deps: "UserDependencies",
    report: Report,
    is_online: bool,
    top_topic: Dict[str, Any],
) -> List[Dict[str, Any]]:
    deliveries: List[Dict[str, Any]] = []

    if not deps.proactive_policy.allow_source("report_completion"):
        return deliveries

    active_items = deps.proactive_item_service.list_active()
    profile = deps.memory.get_user_profile()
    topic = report.topic or top_topic.get("topic", "")
    dedupe_key = re.sub(r"\s+", "-", f"report:{topic}".strip().lower())[:200]
    has_active_dedupe = (
        deps.proactive_item_service.find_active_by_dedupe_key(dedupe_key)
        is not None
    )
    active_items_per_topic = sum(
        1 for item in active_items
        if item.candidate.topic == topic
    )

    allowed = deps.proactive_eligibility_gate.allow(
        source="report_completion",
        mode=deps._config.proactive.mode,
        context=EligibilityContext(
            topic=topic,
            reason=top_topic.get("reason", report.summary or report.title),
            evidence=[report.summary or report.title],
            dedupe_key=dedupe_key,
            message=report.summary or report.title,
            turn_count=0,
            current_turn_index=0,
            has_new_information=True,
            has_decision_pressure=False,
            is_smalltalk=False,
            active_dedupe_keys=[item.candidate.dedupe_key for item in active_items],
            last_proactive_turn_by_topic={},
            active_items_per_user=len(active_items),
            active_items_per_topic=active_items_per_topic,
            is_briefable=bool(report.summary or report.title),
            signal_fresh=True,
            user_deliverable=True,
        ),
    )
    if not allowed:
        return deliveries

    candidate = ProactiveCandidate(
        candidate_id=f"pc_{uuid.uuid4().hex[:8]}",
        source="report_completion",
        topic=topic,
        candidate_confidence=min(1.0, float(top_topic.get("priority", 50)) / 100.0),
        channel_hint="push",
        reason=top_topic.get("reason", report.summary or report.title),
        evidence=[report.summary or report.title],
        artifact_ref={"type": "report", "id": report.id},
        dedupe_key=dedupe_key,
    )
    item = deps.proactive_item_service.add_from_candidate(candidate)
    brief = deps.proactive_brief_service.build_for_candidate(candidate, report=report)
    deps.proactive_item_service.mark_brief_ready(item.item_id, brief)

    interaction_stats = deps.get_interaction_stats()
    decision = deps.proactive_decision_service.decide(
        candidate=candidate,
        brief=brief,
        context=DecisionContext(
            user_interests=getattr(profile, "interests", []),
            current_context="",
            seconds_since_last_interaction=interaction_stats.get(
                "seconds_since_last_interaction", 0
            ),
            recent_interaction_count=interaction_stats.get(
                "recent_interaction_count", 0
            ),
            is_user_online=is_online,
            has_active_dedupe=has_active_dedupe,
            topic_on_cooldown=False,
            active_items_per_topic=active_items_per_topic,
        ),
    )

    if decision.channel == "push" and decision.should_trigger:
        deps.proactive_item_service.mark_delivered(item.item_id, decision)
        deliveries.append(
            {
                "channel": "push",
                "content": brief.summary,
                "item_id": item.item_id,
                "message_type": "report",
                "title": brief.title,
            }
        )
    elif decision.channel == "queue" and decision.should_trigger:
        deps.proactive_item_service.mark_queued(item.item_id, decision)
    else:
        deps.proactive_item_service.close(item.item_id, decision.reason or "dropped")

    return deliveries


async def run_memory_validation(deps: "UserDependencies") -> Dict[str, Any]:
    """
    执行记忆验证任务。

    验证记忆的完整性和一致性，发现知识缺口时触发补全搜索。

    注意：此函数已被 run_memory_maintenance() 取代，保留用于向后兼容。

    Args:
        deps: 用户依赖

    Returns:
        验证结果，包含发现的问题和执行的搜索
    """
    # 直接调用新的合并函数
    return await run_memory_maintenance(deps)


async def run_memory_maintenance(deps: "UserDependencies") -> Dict[str, Any]:
    """
    执行记忆维护任务。

    合并了两个子任务：
    1. 记忆验证（原 memory_validation）：检测知识缺口、逻辑问题
    2. 过期检查（原 stale_check）：检测可能过期的信息

    Args:
        deps: 用户依赖

    Returns:
        维护结果，包含发现的问题、过期项和执行的搜索
    """
    issues = []
    stale_items = []
    searches = []

    try:
        memory = deps.memory
        recent_topics = memory.get_recent_topics(limit=5)

        for topic in recent_topics:
            summary = memory.get_topic_summary(topic)
            if not summary:
                continue

            # === 1. 记忆验证 ===
            try:
                validation_result = await asyncio.to_thread(
                    deps.memory_critic.run,
                    topic=topic,
                    memory_content=summary,
                )

                status = validation_result.get("status", "PASS")
                if status != "PASS":
                    issue = {
                        "topic": topic,
                        "status": status,
                        "confidence_score": validation_result.get("confidence_score", 50),
                        "knowledge_gaps": validation_result.get("knowledge_gaps", []),
                        "logical_issues": validation_result.get("logical_issues", []),
                        "summary": validation_result.get("summary", ""),
                    }
                    issues.append(issue)

                    # 收集建议的搜索（知识缺口）
                    for gap in validation_result.get("knowledge_gaps", [])[:2]:
                        if gap.get("search_query"):
                            searches.append({
                                "type": "knowledge_gap",
                                "topic": topic,
                                "query": gap.get("search_query"),
                                "reason": gap.get("description", "填补知识缺口"),
                            })

                    # 收集建议的搜索（逻辑问题）
                    for logical_issue in validation_result.get("logical_issues", [])[:1]:
                        if logical_issue.get("search_query"):
                            searches.append({
                                "type": "logical_issue",
                                "topic": topic,
                                "query": logical_issue.get("search_query"),
                                "reason": logical_issue.get("description", "验证信息准确性"),
                            })

            except Exception as e:
                print(f"[memory_maintenance] 验证话题 {topic} 失败: {e}")

            # === 2. 过期检查（基于关键词和时间敏感信息）===
            try:
                # 检查内容中是否有时间敏感的关键词
                time_sensitive_keywords = [
                    "最新", "当前", "目前", "现在", "今年", "本月",
                    "latest", "current", "now", "today", "this year",
                    "版本", "更新", "发布", "release", "version"
                ]

                summary_lower = summary.lower()
                has_time_sensitive = any(kw in summary_lower for kw in time_sensitive_keywords)

                if has_time_sensitive:
                    stale_items.append({
                        "topic": topic,
                        "reason": "内容包含时间敏感信息，可能需要更新",
                        "suggestion": f"更新关于 {topic} 的最新信息",
                    })

                    # 添加更新搜索建议
                    searches.append({
                        "type": "stale_info",
                        "topic": topic,
                        "query": f"{topic} 最新动态 2024",
                        "reason": "更新可能过期的信息",
                    })

            except Exception as e:
                print(f"[memory_maintenance] 检查话题 {topic} 过期信息失败: {e}")

        # === 3. 执行补全搜索（如果有）===
        # 按优先级排序：知识缺口 > 逻辑问题 > 过期信息
        priority_order = {"knowledge_gap": 0, "logical_issue": 1, "stale_info": 2}
        searches.sort(key=lambda x: priority_order.get(x.get("type", ""), 3))

        for search in searches[:3]:  # 最多执行3个搜索
            try:
                deps.search_service.search_for_query(
                    query=search["query"],
                    purpose=search["reason"],
                )
            except Exception as e:
                print(f"[memory_maintenance] 补全搜索失败: {e}")

    except Exception as e:
        print(f"[memory_maintenance] 错误: {e}")
        import traceback
        traceback.print_exc()

    return {
        "issues": issues,
        "stale_items": stale_items,
        "searches": searches,
        "topics_checked": len(deps.memory.get_recent_topics(limit=5)),
    }


async def run_initiative_check(deps: "UserDependencies") -> Optional[Dict[str, Any]]:
    """
    检查是否应该主动推送。

    检查项目：
    1. 高优先级报告
    2. 用户空闲时的相关信息
    3. 时间敏感的提醒
    4. 主动建议

    Args:
        deps: 用户依赖

    Returns:
        主动消息配置，或 None（如果不应该主动推送）
    """
    try:
        queued_items = [
            item for item in deps.proactive_item_service.list_all()
            if item.state == ProactiveItemState.QUEUED and item.brief is not None
        ]
        if queued_items:
            item = queued_items[0]
            deps.proactive_item_service.mark_delivered(item.item_id, item.decision)
            return {
                "should_initiate": True,
                "type": "report",
                "priority": "medium",
                "content": item.brief.summary,
                "push_id": item.item_id,
                "options": [],
            }

        # 检查 1: 高优先级待推送报告
        if hasattr(deps, "report_service"):
            initiative = await _check_high_priority_reports(deps)
            if initiative:
                return initiative

        # 检查 2: 用户空闲时的相关信息
        if hasattr(deps, "report_service"):
            initiative = await _check_idle_with_info(deps)
            if initiative:
                return initiative

        # 检查 3: 主动建议
        if hasattr(deps, "memory") and hasattr(deps, "_llm"):
            initiative = await _check_proactive_suggestions(deps)
            if initiative:
                return initiative

    except Exception as e:
        print(f"[initiative_check] 错误: {e}")

    return None


async def _collect_research_topics(deps: "UserDependencies", is_online: bool = False) -> List[Dict[str, Any]]:
    """
    收集值得研究的主题。

    流程:
    1. 收集候选主题（知识缺口 + 场景/关联预测）
    2. 价值评估（过滤低价值主题）
    3. 返回通过评估的主题（如果没有则返回空列表）

    策略区分:
    - 用户在线时：侧重当前聊天内容（包括未保存的新话题）和最近更新的话题
    - 用户离线时：基于记忆系统的历史信息（所有话题、画像）

    Args:
        deps: 用户依赖
        is_online: 用户是否在线

    Returns:
        按优先级排序的研究主题列表（可能为空）
    """
    candidates = []
    memory = deps.memory
    recent_topics = memory.get_recent_topics()
    profile = memory.get_user_profile()
    research_history = profile.research_history or []

    # ========== 1. 收集候选主题 ==========

    if is_online:
        # ===== 在线模式：侧重当前对话 =====

        # 1.1 来源A：当前正在进行的对话（可能是新话题，尚未保存到记忆）
        current_messages = memory.current_messages
        current_topic = memory.current_topic

        if current_messages:
            # 从当前对话中提取研究主题
            current_conversation = "\n".join([
                f"{m.role}: {m.content}" for m in current_messages[-10:]
            ])

            try:
                # 使用 memory_critic 分析当前对话内容
                validation_result = await asyncio.to_thread(
                    deps.memory_critic.run,
                    topic=current_topic or "当前对话",
                    memory_content=current_conversation,
                )

                if validation_result.get("knowledge_gaps"):
                    for gap in validation_result.get("knowledge_gaps", [])[:2]:
                        search_query = gap.get("search_query", "")
                        if search_query:
                            candidates.append({
                                "topic": search_query,
                                "source": "current_conversation",
                                "reason": gap.get("description", "当前对话中的知识缺口"),
                                "confidence": 0.8,  # 当前对话优先级较高
                                "existing_knowledge": current_conversation[:500],
                                "related_topic": current_topic or "当前对话",
                            })

            except Exception as e:
                print(f"[_collect_research_topics] 分析当前对话失败: {e}")

        # 1.2 来源B：最近更新的1个话题（补充）
        for topic in recent_topics[:1]:
            if topic == current_topic:
                continue  # 跳过当前话题

            topic_summary = memory.get_topic_summary(topic)
            if not topic_summary:
                continue

            try:
                validation_result = await asyncio.to_thread(
                    deps.memory_critic.run,
                    topic=topic,
                    memory_content=topic_summary,
                )

                if validation_result.get("knowledge_gaps"):
                    for gap in validation_result.get("knowledge_gaps", [])[:1]:
                        search_query = gap.get("search_query", "")
                        if search_query and not any(c["topic"] == search_query for c in candidates):
                            candidates.append({
                                "topic": search_query,
                                "source": "knowledge_gap",
                                "reason": gap.get("description", "填补知识缺口"),
                                "confidence": 0.6,
                                "existing_knowledge": topic_summary,
                                "related_topic": topic,
                            })

            except Exception as e:
                print(f"[_collect_research_topics] 分析话题 {topic} 失败: {e}")

    else:
        # ===== 离线模式：基于历史记忆 =====

        # 1.1 来源A：知识缺口（检查最近3个话题）
        for topic in recent_topics[:3]:
            topic_summary = memory.get_topic_summary(topic)
            if not topic_summary:
                continue

            try:
                validation_result = await asyncio.to_thread(
                    deps.memory_critic.run,
                    topic=topic,
                    memory_content=topic_summary,
                )

                if validation_result.get("knowledge_gaps") or validation_result.get("logical_issues"):
                    confidence = validation_result.get("confidence_score", 50)

                    for gap in validation_result.get("knowledge_gaps", [])[:2]:
                        search_query = gap.get("search_query", "")
                        if search_query:
                            candidates.append({
                                "topic": search_query,
                                "source": "knowledge_gap",
                                "reason": gap.get("description", "Fill a knowledge gap"),
                                "confidence": (100 - confidence) / 100,
                                "existing_knowledge": topic_summary,
                                "related_topic": topic,
                            })

            except Exception as e:
                print(f"[_collect_research_topics] 分析话题 {topic} 失败: {e}")

    # 1.3 来源C：场景/关联预测（在线和离线都使用，但构建方式不同）
    try:
        # 根据在线/离线构建不同的交互记录
        if is_online:
            recent_interactions = _build_online_interactions(deps)
        else:
            recent_interactions = _build_offline_interactions(deps)

        predictions = await asyncio.to_thread(
            deps.research_predictor.run,
            user_profile_text=profile.to_prompt_text(),
            recent_interactions=recent_interactions,
            recent_topics=recent_topics,
            research_history=research_history,
        )

        for pred in predictions.get("predictions", []):
            topic = pred.get("topic", "")
            if topic and not any(c["topic"] == topic for c in candidates):
                candidates.append({
                    "topic": topic,
                    "source": pred.get("type", "predicted"),
                    "reason": pred.get("reason", "Predicted user need"),
                    "confidence": pred.get("confidence", 0.5),
                    "existing_knowledge": "",
                    "trigger": pred.get("trigger", ""),
                })

    except Exception as e:
        print(f"[_collect_research_topics] 研究主题预测失败: {e}")

    # 如果没有候选主题，直接返回空列表（不再兜底）
    if not candidates:
        print("[_collect_research_topics] 没有发现候选研究主题")
        return []

    # ========== 2. 价值评估 ==========
    threshold = deps._config.tasks.research_value_threshold
    evaluated_topics = []

    def get_existing_knowledge(topic: str) -> str:
        """获取主题的现有知识"""
        # 先检查是否是已有话题
        summary = memory.get_topic_summary(topic)
        if summary:
            return summary
        # 否则搜索相关知识
        knowledge_items = memory.search_knowledge(topic, n_results=3)
        if knowledge_items:
            return "\n".join([k.summary or k.content[:200] for k in knowledge_items])
        return ""

    for candidate in candidates:
        try:
            existing_knowledge = candidate.get("existing_knowledge") or get_existing_knowledge(candidate["topic"])

            result = await asyncio.to_thread(
                deps.research_evaluator.evaluate,
                candidate_topic=candidate,
                user_profile_text=profile.to_prompt_text(),
                existing_knowledge=existing_knowledge,
                research_history=research_history,
            )

            if result["should_research"] and result["score"] >= threshold:
                evaluated_topics.append({
                    "topic": candidate["topic"],
                    "source": candidate["source"],
                    "reason": candidate["reason"],
                    "priority": result["score"],
                    "existing_knowledge": existing_knowledge,
                    "evaluation": result,
                })
                print(f"[_collect_research_topics] 主题通过评估: {candidate['topic']} (score={result['score']})")
            else:
                print(f"[_collect_research_topics] 主题未通过评估: {candidate['topic']} (score={result['score']}, threshold={threshold})")

        except Exception as e:
            print(f"[_collect_research_topics] 评估主题 {candidate['topic']} 失败: {e}")

    # 按优先级排序
    evaluated_topics.sort(key=lambda x: x.get("priority", 0), reverse=True)

    return evaluated_topics


def _build_recent_interactions(deps: "UserDependencies") -> str:
    """
    构建近期交互记录文本。

    包含：
    1. 当前对话的消息（current_messages）
    2. 最近2-3个话题的对话摘要

    Args:
        deps: 用户依赖

    Returns:
        格式化的近期交互记录文本
    """
    memory = deps.memory
    sections = []

    # 1. 当前对话消息
    if memory.current_messages:
        current_topic = memory.current_topic or "Current conversation"
        messages_text = "\n".join([
            f"  - {m.role}: {m.content[:150]}..."
            if len(m.content) > 150 else f"  - {m.role}: {m.content}"
            for m in memory.current_messages[-10:]  # 最多取最近10条
        ])
        sections.append(f"[{current_topic}]\n{messages_text}")

    # 2. 最近话题的摘要
    recent_topics = memory.get_recent_topics(limit=3)
    for topic in recent_topics:
        if topic == memory.current_topic:
            continue  # 跳过当前话题（已包含在上面）

        summary = memory.get_topic_summary(topic)
        if summary:
            # 截取摘要
            summary_short = summary[:300] + "..." if len(summary) > 300 else summary
            sections.append(f"[{topic}]\n  Summary: {summary_short}")

    if not sections:
        return "No recent interaction records"

    return "\n\n".join(sections)


def _build_online_interactions(deps: "UserDependencies") -> str:
    """
    构建在线用户的交互记录（侧重当前对话）。

    侧重：
    1. 当前正在进行的对话（完整内容）
    2. 最近1个话题的摘要（补充上下文）

    Args:
        deps: 用户依赖

    Returns:
        格式化的交互记录文本
    """
    memory = deps.memory
    sections = []

    # 1. 当前对话消息（更详细）
    if memory.current_messages:
        current_topic = memory.current_topic or "Current conversation"
        messages_text = "\n".join([
            f"  - {m.role}: {m.content[:300]}..."
            if len(m.content) > 300 else f"  - {m.role}: {m.content}"
            for m in memory.current_messages[-15:]  # 取更多消息
        ])
        sections.append(f"[Current conversation: {current_topic}] (in progress)\n{messages_text}")

    # 2. 最近1个话题的摘要（补充上下文）
    recent_topics = memory.get_recent_topics(limit=2)
    for topic in recent_topics[:1]:
        if topic == memory.current_topic:
            continue

        summary = memory.get_topic_summary(topic)
        if summary:
            summary_short = summary[:200] + "..." if len(summary) > 200 else summary
            sections.append(f"[Recent topic: {topic}]\n  Summary: {summary_short}")

    if not sections:
        return "The user has just started the conversation; no history is available"

    return "\n\n".join(sections)


def _build_offline_interactions(deps: "UserDependencies") -> str:
    """
    构建离线用户的交互记录（基于历史记忆）。

    侧重：
    1. 最近3-5个话题的摘要
    2. 用户画像中的兴趣和目标

    Args:
        deps: 用户依赖

    Returns:
        格式化的交互记录文本
    """
    memory = deps.memory
    sections = []

    # 1. 最近话题的摘要
    recent_topics = memory.get_recent_topics(limit=5)
    for topic in recent_topics:
        summary = memory.get_topic_summary(topic)
        if summary:
            summary_short = summary[:300] + "..." if len(summary) > 300 else summary
            sections.append(f"[Historical topic: {topic}]\n  Summary: {summary_short}")

    # 2. 用户画像关键信息
    profile = memory.get_user_profile()
    profile_sections = []

    if profile.interests:
        profile_sections.append(f"Interests: {', '.join(profile.interests[:5])}")
    if profile.goals:
        profile_sections.append(f"Goals: {', '.join(profile.goals[:3])}")
    if profile.challenges:
        profile_sections.append(f"Challenges: {', '.join(profile.challenges[:3])}")

    if profile_sections:
        sections.append(f"[User profile]\n  " + "\n  ".join(profile_sections))

    if not sections:
        return "No historical interaction records"

    return "\n\n".join(sections)


async def _check_high_priority_reports(deps: "UserDependencies") -> Optional[Dict[str, Any]]:
    """
    检查高优先级待推送报告。

    Args:
        deps: 用户依赖

    Returns:
        主动消息配置，或 None
    """
    pending = deps.report_service.get_pending()
    if not pending:
        return None

    # 获取用户上下文
    profile = deps.memory.get_user_profile()
    user_interests = profile.interests

    # 获取当前对话上下文
    current_context = ""
    if deps.memory.current_messages:
        recent = deps.memory.current_messages[-3:]
        current_context = "\n".join([m.content[:100] for m in recent])

    # 评估每个报告
    for report in pending:
        try:
            # 获取实际的交互统计
            interaction_stats = deps.get_interaction_stats()

            decision = deps.push_service.evaluate(
                report=report,
                user_interests=user_interests,
                current_context=current_context,
                seconds_since_last_interaction=interaction_stats["seconds_since_last_interaction"],
                recent_interaction_count=interaction_stats["recent_interaction_count"],
            )

            # 基于 action 和 score 决定推送类型
            if decision.action == PushAction.NOTIFY:
                # 高分推送为高优先级，中等分数为普通通知
                if decision.score >= 70:
                    return {
                        "should_initiate": True,
                        "type": "alert",
                        "priority": "high",
                        "content": f"紧急提醒：{report.title}\n\n{report.summary}",
                        "report_id": report.id,
                        "push_id": str(uuid.uuid4()),
                        "score": decision.score,
                        "options": ["查看详情", "稍后提醒", "忽略"],
                    }
                else:
                    return {
                        "should_initiate": True,
                        "type": "notification",
                        "priority": "medium",
                        "content": f"{report.title}: {report.summary[:100]}...",
                        "report_id": report.id,
                        "push_id": str(uuid.uuid4()),
                        "score": decision.score,
                        "options": ["查看", "忽略"],
                    }
            # decision.action == PushAction.SILENT: 继续检查下一个报告

        except Exception as e:
            print(f"[_check_high_priority_reports] 评估报告失败: {e}")

    return None


async def _check_idle_with_info(deps: "UserDependencies") -> Optional[Dict[str, Any]]:
    """
    检查用户空闲时是否有相关待推送信息。

    Args:
        deps: 用户依赖

    Returns:
        主动消息配置，或 None
    """
    pending = deps.report_service.get_pending()
    if not pending:
        return None

    # 获取最相关的报告
    best_report = pending[0]

    if best_report.relevance > 60:
        return {
            "should_initiate": True,
            "type": "suggestion",
            "priority": "low",
            "content": f"嗨！我发现了一些你可能感兴趣的内容：{best_report.title}",
            "report_id": best_report.id,
            "push_id": str(uuid.uuid4()),
            "options": ["感兴趣", "不需要"],
        }

    return None


async def _check_proactive_suggestions(deps: "UserDependencies") -> Optional[Dict[str, Any]]:
    """
    检查是否应该发起主动建议。

    Args:
        deps: 用户依赖

    Returns:
        主动消息配置，或 None
    """
    # 获取用户上下文
    memory = deps.memory
    current_topic = memory.current_topic or "None"
    recent_messages = memory.current_messages[-5:] if memory.current_messages else []

    if not recent_messages:
        return None

    # 获取用户兴趣
    profile = memory.get_user_profile()
    interests = profile.interests

    # 获取待推送报告
    pending = deps.report_service.get_pending()[:3]
    pending_titles = [r.title for r in pending]

    # 使用 LLM 判断是否应该主动发起
    try:
        from core.prompts import Prompts

        prompt = Prompts.INITIATIVE_CHECK.format(
            current_topic=current_topic,
            recent_messages="\n".join([f"- {m.content[:100]}" for m in recent_messages]),
            user_interests=", ".join(interests) if interests else "Unknown",
            pending_info=", ".join(pending_titles) if pending_titles else "None",
            idle_minutes=5,  # 假设值
        )

        response = await asyncio.to_thread(
            deps._llm.chat_json,
            messages=[
                {"role": "system", "content": "You are an assistant that decides whether the system should proactively start a conversation."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",  # 使用快速模型
            temperature=0.3,
        )

        if response.get("should_initiate"):
            return {
                "should_initiate": True,
                "type": response.get("type", "suggestion"),
                "priority": response.get("priority", "low"),
                "content": response.get("content", ""),
                "push_id": str(uuid.uuid4()),
                "options": response.get("options", ["好的", "不需要"]),
            }

    except Exception as e:
        print(f"[_check_proactive_suggestions] LLM 检查失败: {e}")

    return None
