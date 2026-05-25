import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.ProactiveBench.eval.proactive_agent_adapter import (
    DEFAULT_PROACTIVE_AGENT_MODEL,
    ProactiveAgentAdapter,
)
from experiments.ProactiveBench.generation.models import (
    Fact,
    Scenario,
    ScenarioMetadata,
    SimulatorConfig,
    UserNeed,
    UserProfile,
)


def _scenario_with_gold_metadata() -> Scenario:
    return Scenario(
        scenario_id="metadata_leak_check",
        domain="employee_onboarding",
        description="A new employee needs first-day logistics.",
        user_profile=UserProfile(
            persona="New backend engineer",
            context="First day at HelioWorks",
            communication_style="Direct and detail-oriented",
        ),
        fact_sheet=[
            Fact(
                id="F_LEAK_FACT_ID",
                category="office",
                fact="The temporary desk is in Building C, Floor 4, Room 418.",
            ),
            Fact(
                id="F02",
                category="IT",
                fact="The Wi-Fi network is HelioWorks-Staff and the password is orbit-418.",
            ),
        ],
        user_needs=[
            UserNeed(
                id="N_LEAK_NEED_ID",
                description="LEAK_NEED_DESCRIPTION: where the desk is located",
                level="must-have",
                key_fact_ids=["F_LEAK_FACT_ID"],
                predictable_after=None,
                prediction_reason=None,
                turn_order=1,
            ),
            UserNeed(
                id="N2",
                description="How to connect to Wi-Fi",
                level="nice-to-have",
                key_fact_ids=["F02"],
                predictable_after="N_LEAK_NEED_ID",
                prediction_reason="LEAK_PREDICTION_REASON",
                turn_order=2,
            ),
        ],
        simulator_config=SimulatorConfig(max_turns=3),
        metadata=ScenarioMetadata(seed_id="seed"),
    )


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat_json(self, messages, model, temperature=0.0, max_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.payload


class ProactiveAgentAdapterTests(unittest.TestCase):
    def test_prompt_includes_profile_and_facts_but_not_gold_metadata(self) -> None:
        adapter = ProactiveAgentAdapter(_FakeLLM({}))
        adapter.reset(_scenario_with_gold_metadata())

        messages = adapter._build_messages("Where should I sit today?")
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertIn("New backend engineer", prompt_text)
        self.assertIn("Room 418", prompt_text)
        self.assertIn("Purpose", prompt_text)
        self.assertIn("Proactive_Task", prompt_text)
        self.assertIn("send_response", prompt_text)
        self.assertIn("proactive_addendum", prompt_text)

        self.assertNotIn("LEAK_NEED_DESCRIPTION", prompt_text)
        self.assertNotIn("N_LEAK_NEED_ID", prompt_text)
        self.assertNotIn("F_LEAK_FACT_ID", prompt_text)
        self.assertNotIn("key_fact_ids", prompt_text)
        self.assertNotIn("predictable_after", prompt_text)
        self.assertNotIn("LEAK_PREDICTION_REASON", prompt_text)
        self.assertNotIn("turn_order", prompt_text)
        self.assertNotIn("reveal_group", prompt_text)

    def test_null_proactive_task_returns_direct_response_without_predictions(self) -> None:
        llm = _FakeLLM(
            {
                "Purpose": "The user is asking where to sit.",
                "Thoughts": "The desk fact directly answers the request.",
                "Proactive_Task": None,
                "Response": "Your temporary desk is in Building C, Floor 4, Room 418.",
                "Operation": None,
            }
        )
        adapter = ProactiveAgentAdapter(llm)
        adapter.reset(_scenario_with_gold_metadata())

        result = asyncio.run(adapter.send_message("Where should I sit today?"))

        self.assertEqual(
            "Your temporary desk is in Building C, Floor 4, Room 418.",
            result["agent_response"],
        )
        self.assertEqual([], result["proactive_predictions"])
        self.assertEqual([], result["proactive_approved"])
        self.assertEqual(
            {
                "purpose": "The user is asking where to sit.",
                "thoughts": "The desk fact directly answers the request.",
                "proactive_task": None,
                "response": "Your temporary desk is in Building C, Floor 4, Room 418.",
                "operation": {
                    "name": None,
                    "arguments": {},
                    "raw": None,
                },
                "raw": llm.payload,
            },
            result["decision_trace"],
        )
        self.assertEqual(DEFAULT_PROACTIVE_AGENT_MODEL, llm.calls[0]["model"])

    def test_non_empty_proactive_task_appends_conversation_addendum(self) -> None:
        llm = _FakeLLM(
            {
                "Purpose": "The user is asking where to sit.",
                "Thoughts": "After desk location, Wi-Fi setup is a likely next step.",
                "Proactive_Task": "Help with Wi-Fi setup",
                "Response": "Your temporary desk is in Building C, Floor 4, Room 418.",
                "Operation": (
                    "proactive_addendum&content=You may also want the Wi-Fi details: "
                    "network HelioWorks-Staff, password orbit-418."
                ),
            }
        )
        adapter = ProactiveAgentAdapter(llm)
        adapter.reset(_scenario_with_gold_metadata())

        result = asyncio.run(adapter.send_message("Where should I sit today?"))

        self.assertIn("Room 418", result["agent_response"])
        self.assertIn("HelioWorks-Staff", result["agent_response"])
        self.assertEqual("Help with Wi-Fi setup", result["proactive_predictions"][0]["task"])
        self.assertEqual("proactive_addendum", result["proactive_predictions"][0]["operation"])
        self.assertEqual(result["proactive_predictions"], result["proactive_approved"])
        self.assertEqual("Help with Wi-Fi setup", result["decision_trace"]["proactive_task"])
        self.assertEqual("proactive_addendum", result["decision_trace"]["operation"]["name"])


if __name__ == "__main__":
    unittest.main()
