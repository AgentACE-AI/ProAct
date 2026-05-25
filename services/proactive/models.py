from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class ProactiveChannelHint(Enum):
    INLINE = "inline"
    PUSH = "push"
    EITHER = "either"


class ProactiveChannel(Enum):
    INLINE = "inline"
    PUSH = "push"
    QUEUE = "queue"
    DROP = "drop"


class ProactiveItemState(Enum):
    CREATED = "created"
    BRIEF_READY = "brief_ready"
    QUEUED = "queued"
    DELIVERED = "delivered"
    CLOSED = "closed"


class ProactiveCloseReason(Enum):
    CONSUMED = "consumed"
    DUPLICATE = "duplicate"
    STALE = "stale"
    DROPPED = "dropped"
    EXPIRED = "expired"


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class ProactiveCandidate:
    candidate_id: str
    source: str
    topic: str
    candidate_confidence: float
    channel_hint: str
    reason: str
    evidence: List[str] = field(default_factory=list)
    artifact_ref: Dict[str, Any] = field(default_factory=dict)
    dedupe_key: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "topic": self.topic,
            "candidate_confidence": self.candidate_confidence,
            "channel_hint": self.channel_hint,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "artifact_ref": dict(self.artifact_ref),
            "dedupe_key": self.dedupe_key,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProactiveCandidate":
        return cls(
            candidate_id=data["candidate_id"],
            source=data["source"],
            topic=data["topic"],
            candidate_confidence=data["candidate_confidence"],
            channel_hint=data["channel_hint"],
            reason=data["reason"],
            evidence=list(data.get("evidence", [])),
            artifact_ref=dict(data.get("artifact_ref", {})),
            dedupe_key=data.get("dedupe_key", ""),
            created_at=data.get("created_at", _now_iso()),
        )


@dataclass
class ProactiveBrief:
    brief_id: str
    candidate_id: str
    title: str
    summary: str
    action_hint: str = ""
    source_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brief_id": self.brief_id,
            "candidate_id": self.candidate_id,
            "title": self.title,
            "summary": self.summary,
            "action_hint": self.action_hint,
            "source_refs": list(self.source_refs),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProactiveBrief":
        return cls(
            brief_id=data["brief_id"],
            candidate_id=data["candidate_id"],
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            action_hint=data.get("action_hint", ""),
            source_refs=list(data.get("source_refs", [])),
        )


@dataclass
class ProactiveDecision:
    should_trigger: bool
    channel: str
    score: float
    reason: str
    expires_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_trigger": self.should_trigger,
            "channel": self.channel,
            "score": self.score,
            "reason": self.reason,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProactiveDecision":
        return cls(
            should_trigger=data.get("should_trigger", False),
            channel=data.get("channel", ProactiveChannel.DROP.value),
            score=data.get("score", 0.0),
            reason=data.get("reason", ""),
            expires_at=data.get("expires_at"),
        )


@dataclass
class ProactiveItem:
    item_id: str
    candidate: ProactiveCandidate
    brief: Optional[ProactiveBrief] = None
    decision: Optional[ProactiveDecision] = None
    state: ProactiveItemState = ProactiveItemState.CREATED
    close_reason: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_candidate(
        cls,
        candidate: ProactiveCandidate,
        item_id: Optional[str] = None,
    ) -> "ProactiveItem":
        return cls(
            item_id=item_id or f"pi_{uuid4().hex[:8]}",
            candidate=candidate,
            state=ProactiveItemState.CREATED,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "candidate": self.candidate.to_dict(),
            "brief": self.brief.to_dict() if self.brief else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "state": self.state.value,
            "close_reason": self.close_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProactiveItem":
        state = data.get("state", ProactiveItemState.CREATED.value)
        return cls(
            item_id=data["item_id"],
            candidate=ProactiveCandidate.from_dict(data["candidate"]),
            brief=(
                ProactiveBrief.from_dict(data["brief"])
                if data.get("brief")
                else None
            ),
            decision=(
                ProactiveDecision.from_dict(data["decision"])
                if data.get("decision")
                else None
            ),
            state=ProactiveItemState(state),
            close_reason=data.get("close_reason"),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
        )
