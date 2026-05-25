from dataclasses import dataclass


NEED_ONLY_SOURCES = {"need_predictor"}
RESEARCH_SOURCES = {
    "research_predictor",
    "report_completion",
    "deep_research",
    "memory_critic",
    "stale_check",
}
HYBRID_SOURCES = NEED_ONLY_SOURCES | RESEARCH_SOURCES


@dataclass(frozen=True)
class ProactivePolicy:
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in {"need_only", "research", "hybrid"}:
            raise ValueError(
                f"Unsupported proactive mode: {self.mode!r}"
            )

    def allow_source(self, source: str) -> bool:
        if self.mode == "need_only":
            return source in NEED_ONLY_SOURCES

        if self.mode == "research":
            return source in RESEARCH_SOURCES

        return source in HYBRID_SOURCES

    def default_delivery_preference(self) -> str:
        if self.mode == "research":
            return "queue"

        return "inline"
