import re
import unittest
from pathlib import Path


class AnonymityScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_no_packaging_metadata_or_workspace_paths(self) -> None:
        forbidden_path_parts = {
            ".env",
            ".trellis",
            ".codexpotter",
            ".pytest_cache",
        }
        for path in self.root.rglob("*"):
            rel_parts_tuple = path.relative_to(self.root).parts
            if rel_parts_tuple and rel_parts_tuple[0] == ".git":
                continue
            rel_parts = set(path.relative_to(self.root).parts)
            self.assertFalse(rel_parts & forbidden_path_parts, path)

    def test_no_obvious_secret_or_author_path_leaks(self) -> None:
        mac_user_path = "/" + "Users/"
        secret_key = "sk-" + r"(?:proj-)?[A-Za-z0-9_-]{16,}"
        generic_email = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        linux_home = "/" + "home/" + r"[A-Za-z0-9._-]+"
        patterns = [
            ("macOS user path", re.compile(mac_user_path)),
            ("local worktree path", re.compile(r"(?:^|[\"'/])\.worktrees/")),
            ("local release branch name", re.compile(r"proactive-refactor-sync")),
            ("API secret", re.compile(secret_key)),
            ("non-placeholder OpenAI key", re.compile(r"OPENAI_API_KEY\s*=\s*(?!$|your-api-key-here\b)\S+")),
        ]
        text_suffixes = {"", ".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
        for path in self.root.rglob("*"):
            rel_parts_tuple = path.relative_to(self.root).parts
            if rel_parts_tuple and rel_parts_tuple[0] == ".git":
                continue
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            if path.name == "test_anonymity_scan.py":
                continue
            text = path.read_text(encoding="utf-8")
            rel_path = path.relative_to(self.root).as_posix()
            active_patterns = list(patterns)
            if "/data/scenarios/" not in rel_path:
                active_patterns.append(("email address", re.compile(generic_email)))
                active_patterns.append(("linux home path", re.compile(linux_home)))
            for label, pattern in active_patterns:
                self.assertIsNone(pattern.search(text), f"{label}: {rel_path}")


if __name__ == "__main__":
    unittest.main()
