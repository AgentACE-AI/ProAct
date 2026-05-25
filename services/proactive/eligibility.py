from dataclasses import dataclass, field
from typing import Dict, List

from core.config import ProactiveConfig
from services.proactive.policy import HYBRID_SOURCES, ProactivePolicy


@dataclass
class EligibilityContext:
    topic: str
    reason: str
    evidence: List[str] = field(default_factory=list)
    dedupe_key: str = ""
    message: str = ""
    turn_count: int = 0
    current_turn_index: int = 0
    has_new_information: bool = False
    has_decision_pressure: bool = False
    is_smalltalk: bool = False
    same_turn_candidate_emitted: bool = False
    active_dedupe_keys: List[str] = field(default_factory=list)
    last_proactive_turn_by_topic: Dict[str, int] = field(default_factory=dict)
    active_items_per_user: int = 0
    active_items_per_topic: int = 0
    is_briefable: bool = True
    signal_fresh: bool = True
    user_deliverable: bool = True


class ProactiveEligibilityGate:
    def __init__(self, config: ProactiveConfig):
        self.config = config

    def allow(self, source: str, mode: str, context: EligibilityContext) -> bool:
        if source not in HYBRID_SOURCES:
            return False

        if not self._passes_global_gate(context):
            return False

        if not ProactivePolicy(mode).allow_source(source):
            return False

        if source == "need_predictor":
            return self._allow_need_predictor(context)

        return True

    def _passes_global_gate(self, context: EligibilityContext) -> bool:
        if not context.topic or not context.reason or not context.evidence:
            return False

        if not context.is_briefable or not context.signal_fresh or not context.user_deliverable:
            return False

        if context.dedupe_key and context.dedupe_key in set(context.active_dedupe_keys):
            return False

        if context.active_items_per_user >= self.config.max_active_items_per_user:
            return False

        if context.active_items_per_topic >= self.config.max_active_items_per_topic:
            return False

        if self._topic_is_on_cooldown(context):
            return False

        return True

    def _allow_need_predictor(self, context: EligibilityContext) -> bool:
        if context.turn_count < self.config.inline_min_turns:
            return False

        if len(context.message.strip()) < self.config.min_message_chars_for_need_prediction:
            return False

        if context.is_smalltalk or context.same_turn_candidate_emitted:
            return False

        if not (context.has_new_information or context.has_decision_pressure):
            return False

        return True

    def _topic_is_on_cooldown(self, context: EligibilityContext) -> bool:
        last_turn = context.last_proactive_turn_by_topic.get(context.topic)
        if last_turn is None:
            return False

        return (
            context.current_turn_index - last_turn
        ) < self.config.inline_same_topic_cooldown_turns
