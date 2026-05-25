import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ProactiveBenchGenerationModuleImportTests(unittest.TestCase):
    def test_generation_tools_import_as_modules(self) -> None:
        module_names = [
            "experiments.ProactiveBench.generation.validate_scenario",
            "experiments.ProactiveBench.generation.generate_scenarios",
            "experiments.ProactiveBench.generation.expand_variants",
            "experiments.ProactiveBench.generation.review_tool",
        ]

        for module_name in module_names:
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
