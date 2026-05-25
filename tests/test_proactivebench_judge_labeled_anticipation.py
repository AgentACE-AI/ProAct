import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.ProactiveBench.eval.judge_labeled_anticipation import (
    compute_judge_labeled_anticipation,
)
from experiments.ProactiveBench.eval.models import (
    ConversationResult,
    JudgmentResult,
    NeedCoverage,
    TurnTrace,
)


def _trace(turn: int, need_id: str, mode: str) -> TurnTrace:
    return TurnTrace(
        turn_number=turn,
        user_message=f"user turn {turn}",
        active_need_id=f"N{turn}",
        needs_skipped=[],
        agent_response=f"assistant turn {turn}",
        proactive_predictions=[],
        proactive_approved=[],
        memory_context="",
        judgment=JudgmentResult(
            turn_number=turn,
            facts_conveyed=[],
            facts_distorted=[],
            hallucinated_claims=[],
            needs_addressed=[NeedCoverage(need_id=need_id, mode=mode)],
        ),
        cumulative_covered_needs={need_id},
    )


class JudgeLabeledAnticipationTests(unittest.TestCase):
    def test_compute_judge_labeled_anticipation_counts_predictable_proactive_needs(self):
        result = ConversationResult(
            scenario_id="scenario_a",
            condition="ProactiveAgent-style-4o",
            domain="demo",
            seed=42,
            total_turns=3,
            turn_traces=[
                _trace(1, "N2", "proactive"),
                _trace(2, "N3", "reactive"),
                _trace(3, "N4", "proactive"),
            ],
            final_covered_needs={"N2", "N3", "N4"},
            all_need_ids={"N1", "N2", "N3", "N4"},
            must_have_ids={"N1", "N2", "N3"},
            predictable_need_ids={"N2", "N3", "N5"},
        )

        metrics = compute_judge_labeled_anticipation(result)

        self.assertEqual("scenario_a", metrics.scenario_id)
        self.assertEqual("ProactiveAgent-style-4o", metrics.condition)
        self.assertEqual(3, metrics.n_predictable_needs)
        self.assertEqual(["N2"], metrics.anticipated_predictable_need_ids)
        self.assertEqual(["N4"], metrics.non_predictable_proactive_need_ids)
        self.assertAlmostEqual(1 / 3, metrics.judge_labeled_anticipation_recall)


if __name__ == "__main__":
    unittest.main()
