import re
from typing import List, Optional
from uuid import uuid4

from core.models import Report
from services.proactive.models import ProactiveBrief, ProactiveCandidate


class ProactiveBriefService:
    def build_for_candidate(
        self,
        candidate: ProactiveCandidate,
        report: Optional[Report] = None,
        memory_snippets: Optional[List[str]] = None,
    ) -> ProactiveBrief:
        if candidate.source == "report_completion" and report is not None:
            return self._build_from_report(candidate, report)

        return self._build_from_snippets(candidate, memory_snippets or [])

    def _build_from_snippets(
        self,
        candidate: ProactiveCandidate,
        memory_snippets: List[str],
    ) -> ProactiveBrief:
        snippets = [self._clean_sentence(s) for s in memory_snippets if s and s.strip()]
        sentences = snippets[:2]

        if not sentences:
            sentences.append(self._clean_sentence(candidate.reason))

        if candidate.reason and candidate.reason.strip():
            reason_sentence = self._clean_sentence(candidate.reason)
            if reason_sentence not in sentences:
                sentences.append(reason_sentence)

        summary = " ".join(sentences[:3]).strip()
        return ProactiveBrief(
            brief_id=f"pb_{uuid4().hex[:8]}",
            candidate_id=candidate.candidate_id,
            title=self._title_for(candidate),
            summary=summary,
            action_hint="I can compare the next decision point if you want.",
            source_refs=[candidate.artifact_ref.get("id", "")] if candidate.artifact_ref else [],
        )

    def _build_from_report(
        self,
        candidate: ProactiveCandidate,
        report: Report,
    ) -> ProactiveBrief:
        report_sentences = self._split_sentences(report.summary)
        cleaned = [
            self._clean_sentence(sentence)
            for sentence in report_sentences
            if sentence.strip()
        ]

        if not cleaned:
            cleaned = [self._clean_sentence(candidate.reason)]

        summary = " ".join(cleaned[:4]).strip()
        summary = re.sub(r"\breport ready\b", "update", summary, flags=re.IGNORECASE)
        summary = re.sub(
            r"\breport has been generated\b",
            "useful findings are available",
            summary,
            flags=re.IGNORECASE,
        )

        return ProactiveBrief(
            brief_id=f"pb_{uuid4().hex[:8]}",
            candidate_id=candidate.candidate_id,
            title=report.title or self._title_for(candidate),
            summary=summary,
            action_hint="I can break this down into a short shortlist next.",
            source_refs=[report.id],
        )

    @staticmethod
    def _title_for(candidate: ProactiveCandidate) -> str:
        topic = candidate.topic.strip()
        return topic[:1].upper() + topic[1:] if topic else "Proactive update"

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]

    @staticmethod
    def _clean_sentence(text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return ""
        if text[-1] not in ".!?":
            text += "."
        return text
