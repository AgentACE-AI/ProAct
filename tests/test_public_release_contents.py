import importlib
import unittest
from pathlib import Path


class PublicReleaseContentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_release_has_200_scenarios(self) -> None:
        scenarios = list(
            (self.root / "experiments" / "ProactiveBench" / "data" / "scenarios").glob("*.json")
        )
        self.assertEqual(200, len(scenarios))

    def test_required_docs_exist(self) -> None:
        for rel_path in [
            "README.md",
            "RUN_COMMANDS.md",
            "LICENSES.md",
            "LICENSE",
            "CITATION.cff",
            "ARTIFACT_MANIFEST.json",
        ]:
            with self.subTest(rel_path=rel_path):
                self.assertTrue((self.root / rel_path).exists())

    def test_excluded_runtime_paths_are_absent(self) -> None:
        absent = [
            ".env",
            ".trellis",
            ".codexpotter",
            "CLAUDE.md",
            "experiments/ProactiveBench/runtime",
            "experiments/ProactiveBench/results",
            "experiments/MemBench/results",
        ]
        for rel_path in absent:
            with self.subTest(rel_path=rel_path):
                self.assertFalse((self.root / rel_path).exists())

    def test_reproduction_cli_modules_import(self) -> None:
        modules = [
            "experiments.ProactiveBench.generation.validate_scenario",
            "experiments.ProactiveBench.eval.runner",
            "experiments.ProactiveBench.eval.run_proactive_agent",
            "experiments.ProactiveBench.eval.judge_labeled_anticipation",
            "experiments.MemBench.benchmark.comparison_experiment",
        ]
        for module_name in modules:
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
