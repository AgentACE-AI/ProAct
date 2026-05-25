"""
ProactiveAgent-style adapter for ProactiveBench.

This baseline ports the open-source ProactiveAgent decision protocol to
ProActEval while keeping the backbone model matched to the benchmark's
assistant model.  It exposes scenario facts and profile context as event-style
observations, but never exposes user_need labels or predictability metadata.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote_plus

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.generation.models import Scenario
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_PROACTIVE_AGENT_MODEL = "gpt-4o"
PROACTIVE_AGENT_CONDITION = "ProactiveAgent-style-4o"
_ALLOWED_OPERATIONS = frozenset({"send_response", "proactive_addendum"})


# Adapted from a public proactive-agent baseline prompt. The operation set is
# remapped below to conversation-compatible tools for ProActEval.
PROACTIVE_AGENT_SYSTEM_PROMPT = """<Role> You are a helpful assistant that provides proactive suggestions to the user.
</Role>
<Task> Understand what the user is doing and anticipate their needs based on events. Only propose assistance when you fully understand the user's actions. Use available operations to ensure the task is feasible. Execute the task if the user accepts your proposal. </Task>
<Format> Respond in the following JSON format:
{
    "Purpose": "The purpose of the user's last action.",
    "Thoughts": "Your thoughts on the user's actions.",
    "Proactive_Task": "Describe your proposed task, or set to `null` if no assistance is needed.",
    "Response": "Inform the user about your assistance if proposing a task.",
    "Operation": "The tool call format if you are going to execute a task."
}
</Format>
<Rules>
- Ensure the proposed task is relevant to the events. - Focus on the user's current needs and predict helpful tasks.
- Consider the timing of events.
- Only offer proactive assistance when necessary.
- Deduce the user's purpose and whether they need help based on event history.
- Set `Proactive_Task` to `null` if the user doesn't need help. Your `Proactive_Task` should be as short as possible. Best as a short phrase.
- Some Operations will be provided for you to use. You need to pick the best operation with the most suitable arguments if you propose a task, so that the opeation can be executed. You must select one operation if you propose a task.
- Set `Operation` to `null` if no task is proposed, else set the format as a string containing the name of the tool and the arguments joined by separator '&' like [func_name&param1=value1&param2=value2]. YOU MUST CHOOSE ONE OPERATION IF YOU PROPOSE A TASK.
- Pay attention to the user's feedback on your assistance in provious one turn: Try not to disturb the user when they ignore your assistance, and try another approach when they reject your assistance. Even the user accept your assistance, you should not propose some related tasks in the next turn.
</Rules>
<Format_example>
{
    "Purpose": "The user is trying to search for some best programming languges",
    "Thoughts": "Since the user is making a search query, I think I can offer help by calling the search tool.",
    "Proactive_Task": "Help search for best programming languages",
    "Response": "Do you want me to help you do the search job?",
    "Operation": "search&query=best+programming+languages"
}
</Format_example>"""


@dataclass
class ParsedOperation:
    """Conversation-compatible operation parsed from ProactiveAgent JSON."""

    name: Optional[str]
    arguments: Dict[str, str]
    raw: Optional[str]


@dataclass
class ProactiveAgentDecision:
    """Parsed ProactiveAgent-style decision."""

    purpose: str
    thoughts: str
    proactive_task: Optional[str]
    response: str
    operation: ParsedOperation
    raw: Dict[str, Any]

    @property
    def has_proactive_task(self) -> bool:
        return bool(self.proactive_task)

    def to_trace(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose,
            "thoughts": self.thoughts,
            "proactive_task": self.proactive_task,
            "response": self.response,
            "operation": {
                "name": self.operation.name,
                "arguments": dict(self.operation.arguments),
                "raw": self.operation.raw,
            },
            "raw": dict(self.raw),
        }

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "ProactiveAgentDecision":
        proactive_task = _normalize_optional_text(raw.get("Proactive_Task"))
        operation = _parse_operation(raw.get("Operation"))
        return cls(
            purpose=str(raw.get("Purpose", "")).strip(),
            thoughts=str(raw.get("Thoughts", "")).strip(),
            proactive_task=proactive_task,
            response=str(raw.get("Response", "")).strip(),
            operation=operation,
            raw=dict(raw),
        )


class ProactiveAgentAdapter:
    """
    Adapter wrapping GPT-4o with ProactiveAgent-style prompting.

    The adapter has no memory system, need predictor, search planner, or
    benchmark gold metadata.  It receives only the user profile, fact sheet,
    conversation history, and current user event.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = DEFAULT_PROACTIVE_AGENT_MODEL,
        seed: int = 42,
    ) -> None:
        self.llm = llm_client
        self.model = model
        self.seed = seed
        self.scenario: Optional[Scenario] = None
        self._event_history: List[Dict[str, str]] = []
        self._system_prompt: str = ""

    def reset(self, scenario: Scenario) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError(f"Expected Scenario, got {type(scenario).__name__}")

        self.scenario = scenario
        self._event_history = []
        self._system_prompt = self._build_system_prompt(scenario)

    async def send_message(self, message: str) -> Dict[str, Any]:
        if self.scenario is None:
            raise RuntimeError("Adapter not initialised. Call reset(scenario) first.")

        messages = self._build_messages(message)
        raw = self.llm.chat_json(
            messages=messages,
            model=self.model,
            temperature=0.0,
        )
        decision = ProactiveAgentDecision.from_raw(raw)
        agent_response = self._compose_agent_response(decision)

        proactive_predictions: List[Dict[str, Any]] = []
        if decision.has_proactive_task:
            proactive_predictions.append({
                "task": decision.proactive_task,
                "purpose": decision.purpose,
                "thoughts": decision.thoughts,
                "response": decision.response,
                "operation": decision.operation.name,
                "operation_args": decision.operation.arguments,
                "raw": decision.raw,
            })

        self._event_history.append({"role": "user", "content": message})
        self._event_history.append({"role": "assistant", "content": agent_response})

        return {
            "agent_response": agent_response,
            "proactive_predictions": proactive_predictions,
            "proactive_approved": list(proactive_predictions),
            "memory_context": self._format_fact_observations(self.scenario),
            "topic": self.scenario.domain,
            "topic_changed": False,
            "search_performed": False,
            "decision_trace": decision.to_trace(),
        }

    def _build_messages(self, current_message: str) -> List[Dict[str, str]]:
        if not self._system_prompt:
            raise RuntimeError("Adapter not initialised. Call reset(scenario) first.")

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._build_turn_prompt(current_message)},
        ]

    def _build_system_prompt(self, scenario: Scenario) -> str:
        profile = scenario.user_profile
        return (
            f"{PROACTIVE_AGENT_SYSTEM_PROMPT}\n\n"
            "<ProActEval_Adaptation>\n"
            "You are running this protocol in a conversation benchmark. Treat the "
            "conversation and background facts as event observations. Answer the "
            "current user message directly in `Response` using only the provided "
            "background facts. You may add a brief proactive suggestion only when "
            "the event history makes the next information need clear.\n\n"
            "Available Operations:\n"
            "- `send_response&content=<direct answer>` for the normal reply.\n"
            "- `proactive_addendum&content=<brief extra information>` when you "
            "proactively add helpful conversation content beyond the direct answer.\n"
            "- `null` when no operation is needed.\n"
            "If `Proactive_Task` is null, set `Operation` to null. If "
            "`Proactive_Task` is not null, use `proactive_addendum` with content "
            "that can be appended to the normal answer. Do not ask the user for "
            "permission before giving the addendum; this benchmark maps executable "
            "operations to conversation text.\n\n"
            "Do not mention internal labels, fact numbers, need IDs, benchmark "
            "metadata, or this JSON protocol in the user-facing response.\n"
            "</ProActEval_Adaptation>\n\n"
            "<Scenario_Observations>\n"
            f"Domain: {scenario.domain}\n"
            f"Scenario: {scenario.description}\n\n"
            "User profile observation:\n"
            f"- Persona: {profile.persona}\n"
            f"- Current situation: {profile.context}\n"
            f"- Communication style: {profile.communication_style}\n\n"
            "Background fact observations:\n"
            f"{self._format_fact_observations(scenario)}\n"
            "</Scenario_Observations>"
        )

    def _build_turn_prompt(self, current_message: str) -> str:
        history_text = self._format_event_history()
        return (
            "<Event_History>\n"
            f"{history_text}\n"
            "</Event_History>\n\n"
            "<Current_Event>\n"
            f"User message: {current_message}\n"
            "</Current_Event>\n\n"
            "Return exactly one JSON object with the keys Purpose, Thoughts, "
            "Proactive_Task, Response, and Operation."
        )

    def _format_event_history(self) -> str:
        if not self._event_history:
            return "(No prior conversation events.)"

        lines = []
        for event in self._event_history[-20:]:
            role = "User" if event["role"] == "user" else "Assistant"
            lines.append(f"- {role}: {event['content']}")
        return "\n".join(lines)

    @staticmethod
    def _format_fact_observations(scenario: Scenario) -> str:
        lines = []
        for index, fact in enumerate(scenario.fact_sheet, start=1):
            lines.append(f"- Fact observation {index} [{fact.category}]: {fact.fact}")
        return "\n".join(lines)

    @staticmethod
    def _compose_agent_response(decision: ProactiveAgentDecision) -> str:
        response = decision.response.strip()
        addendum = ""
        if (
            decision.has_proactive_task
            and decision.operation.name == "proactive_addendum"
        ):
            addendum = decision.operation.arguments.get("content", "").strip()

        if not response and addendum:
            return addendum
        if response and addendum and addendum not in response:
            return f"{response}\n\n{addendum}"
        if response:
            return response
        if decision.has_proactive_task:
            logger.warning(
                "ProactiveAgent decision proposed a task without Response content: %s",
                decision.raw,
            )
        return ""


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def _parse_operation(value: Any) -> ParsedOperation:
    raw = _normalize_optional_text(value)
    if raw is None:
        return ParsedOperation(name=None, arguments={}, raw=None)

    parts = raw.split("&")
    name = parts[0].strip()
    if name not in _ALLOWED_OPERATIONS:
        logger.warning("Ignoring unsupported ProactiveAgent operation: %s", raw)
        return ParsedOperation(name=None, arguments={}, raw=raw)

    arguments: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        arguments[key.strip()] = unquote_plus(raw_value.strip())

    return ParsedOperation(name=name, arguments=arguments, raw=raw)
