import re
from dataclasses import dataclass
from typing import Protocol

from core.config import ProactiveConfig


@dataclass(eq=True)
class TurnSignals:
    is_smalltalk: bool
    has_decision_pressure: bool
    has_new_information: bool
    rationale: str = ""


class TurnSignalExtractor(Protocol):
    def extract(
        self,
        *,
        message: str,
        conversation_history: str,
        current_topic: str,
        memory_context: str,
    ) -> TurnSignals:
        ...


class RuleTurnSignalExtractor:
    def extract(
        self,
        *,
        message: str,
        conversation_history: str,
        current_topic: str,
        memory_context: str,
    ) -> TurnSignals:
        lowered = message.strip().lower()
        return TurnSignals(
            is_smalltalk=self._is_smalltalk(lowered),
            has_decision_pressure=self._has_decision_pressure(lowered),
            has_new_information=self._has_new_information(lowered),
            rationale="rule_based_fallback",
        )

    @staticmethod
    def _is_smalltalk(message: str) -> bool:
        normalized = re.sub(r"[!?.~。！，、\s]+", "", message)
        smalltalk_tokens = {
            "hi", "hello", "thanks", "thankyou", "ok", "okay", "gotit",
            "收到", "谢谢", "好的", "嗯", "好",
        }
        return normalized in smalltalk_tokens

    @staticmethod
    def _has_decision_pressure(message: str) -> bool:
        markers = [
            "which", "choose", "decide", "should i", "next step",
            "compare", "better", "or", "怎么选", "下一步", "应该", "选择", "比较",
        ]
        return any(marker in message for marker in markers) or "?" in message or "？" in message

    @staticmethod
    def _has_new_information(message: str) -> bool:
        patterns = [
            r"\d",
            r"\bcompare\b",
            r"\bversus\b",
            r"\bneed to\b",
            r"\bbudget\b",
            r"预算",
            r"孩子",
            r"日期",
            r"时间",
            r"地点",
        ]
        return any(re.search(pattern, message) for pattern in patterns)


class LLMTurnSignalExtractor:
    def __init__(self, llm_client, model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model

    def extract(
        self,
        *,
        message: str,
        conversation_history: str,
        current_topic: str,
        memory_context: str,
    ) -> TurnSignals:
        prompt = f"""
Classify the latest user turn for proactive eligibility gating.

Return JSON with exactly these fields:
{{
  "is_smalltalk": boolean,
  "has_decision_pressure": boolean,
  "has_new_information": boolean,
  "rationale": "short explanation"
}}

Definitions:
- is_smalltalk: greeting, thanks, acknowledgement, filler, or social reply that adds no task-relevant constraint or question.
- has_decision_pressure: the user is choosing among options, confirming a plan, resolving uncertainty, asking for executable next steps, or needs information in the next 1-2 turns.
- has_new_information: the user turn adds new entities, numeric constraints, time/location constraints, comparisons, narrowing constraints, blockers, or new next-step requirements.

Current topic:
{current_topic or "None"}

Conversation history:
{conversation_history or "None"}

Memory context:
{memory_context or "None"}

Latest user message:
{message}
""".strip()
        result = self.llm_client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic turn classifier for proactive gating. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.0,
            max_tokens=200,
        )
        return TurnSignals(
            is_smalltalk=bool(result.get("is_smalltalk", False)),
            has_decision_pressure=bool(result.get("has_decision_pressure", False)),
            has_new_information=bool(result.get("has_new_information", False)),
            rationale=str(result.get("rationale", "")),
        )


def build_turn_signal_extractor(
    config: ProactiveConfig,
    llm_client,
) -> TurnSignalExtractor:
    use_llm = config.turn_signal_backend == "llm"
    if config.mode == "need_only" and not config.enable_llm_turn_signals_in_need_only:
        use_llm = False

    if use_llm:
        return LLMTurnSignalExtractor(llm_client=llm_client, model=config.turn_signal_model)

    return RuleTurnSignalExtractor()
