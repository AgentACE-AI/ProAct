from dataclasses import dataclass, field
from typing import List

from core.config import ProactiveConfig
from services.proactive.models import (
    ProactiveBrief,
    ProactiveCandidate,
    ProactiveChannel,
    ProactiveDecision,
)


@dataclass
class DecisionContext:
    user_interests: List[str] = field(default_factory=list)
    current_context: str = ""
    seconds_since_last_interaction: int = 0
    recent_interaction_count: int = 0
    is_user_online: bool = True
    has_active_dedupe: bool = False
    topic_on_cooldown: bool = False
    active_items_per_topic: int = 0


class ProactiveDecisionService:
    def __init__(self, config: ProactiveConfig, push_score_agent):
        self.config = config
        self.push_score_agent = push_score_agent

    def decide(
        self,
        candidate: ProactiveCandidate,
        brief: ProactiveBrief,
        context: DecisionContext,
    ) -> ProactiveDecision:
        if (
            context.has_active_dedupe
            or context.topic_on_cooldown
            or context.active_items_per_topic >= self.config.max_active_items_per_topic
        ):
            return ProactiveDecision(
                should_trigger=False,
                channel=ProactiveChannel.DROP.value,
                score=0.0,
                reason="duplicate_or_cooldown",
            )

        if candidate.channel_hint == ProactiveChannel.INLINE.value:
            if candidate.candidate_confidence >= self.config.inline_trigger_threshold:
                return ProactiveDecision(
                    should_trigger=True,
                    channel=ProactiveChannel.INLINE.value,
                    score=candidate.candidate_confidence,
                    reason="inline_threshold_met",
                )
            return ProactiveDecision(
                should_trigger=False,
                channel=ProactiveChannel.DROP.value,
                score=candidate.candidate_confidence,
                reason="inline_threshold_not_met",
            )

        eval_result = self.push_score_agent.run(
            report_topic=candidate.topic,
            report_summary=brief.summary,
            user_interests=context.user_interests,
            current_context=context.current_context,
            seconds_since_last_interaction=context.seconds_since_last_interaction,
            recent_interaction_count=context.recent_interaction_count,
        )
        value = eval_result.get("value", 50)
        cost = eval_result.get("cost", 50)
        score = max(0, min(100, value - cost + 50))
        reason = eval_result.get("reason", "")

        if score >= self.config.push_trigger_threshold:
            if context.is_user_online:
                return ProactiveDecision(
                    should_trigger=True,
                    channel=ProactiveChannel.PUSH.value,
                    score=score,
                    reason=reason or "push_threshold_met",
                )
            return ProactiveDecision(
                should_trigger=True,
                channel=ProactiveChannel.QUEUE.value,
                score=score,
                reason=reason or "queued_until_user_online",
            )

        if score >= self.config.queue_threshold:
            return ProactiveDecision(
                should_trigger=True,
                channel=ProactiveChannel.QUEUE.value,
                score=score,
                reason=reason or "queue_threshold_met",
            )

        return ProactiveDecision(
            should_trigger=False,
            channel=ProactiveChannel.DROP.value,
            score=score,
            reason=reason or "below_threshold",
        )
