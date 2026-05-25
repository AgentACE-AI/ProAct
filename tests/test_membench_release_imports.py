import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MemBenchReleaseImportTests(unittest.TestCase):
    def test_comparison_experiment_imports_without_upstream_membench_checkout(self) -> None:
        importlib.import_module("experiments.MemBench.benchmark.comparison_experiment")


if __name__ == "__main__":
    unittest.main()
