"""
CoverageJudge: LLM-as-Judge evaluation module.

Evaluates each agent response turn against the scenario's fact_sheet and
user_needs, producing a structured JudgmentResult with:
  - facts_conveyed:      Fact IDs accurately communicated
  - facts_distorted:     Fact IDs mentioned with errors
  - hallucinated_claims: Claims not grounded in fact_sheet
  - needs_addressed:     User needs covered this turn (reactive vs proactive)
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.eval.models import JudgmentResult, NeedCoverage
from experiments.ProactiveBench.generation.models import Scenario
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CoverageJudge:
    """LLM-based judge that evaluates a single agent response against
    the ground-truth fact sheet and user needs list."""

    def __init__(
        self,
        scenario: Scenario,
        llm_client: LLMClient,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.scenario = scenario
        self.llm = llm_client
        self.model = model

        # Pre-build lookup sets for post-validation
        self._valid_fact_ids = {f.id for f in scenario.fact_sheet}
        self._valid_need_ids = {n.id for n in scenario.user_needs}
        # Map each need to its key_fact_ids for fact-based coverage enforcement
        self._need_key_facts = {
            n.id: set(n.key_fact_ids) for n in scenario.user_needs
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge_turn(
        self,
        turn_number: int,
        user_message: str,
        agent_response: str,
        active_need_id: Optional[str],
        conversation_history: List[Dict[str, str]],
        satellite_need_ids: Optional[List[str]] = None,
    ) -> JudgmentResult:
        """
        Judge a single agent response for factual accuracy and need coverage.

        Args:
            turn_number: Current turn (1-based).
            user_message: The user's message this turn.
            agent_response: The agent's response this turn.
            active_need_id: The need ID the user explicitly raised (None if none).
            conversation_history: Full dialogue history prior to this turn.
            satellite_need_ids: Related need IDs mentioned in the same turn as context.

        Returns:
            A validated JudgmentResult.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            turn_number, user_message, agent_response,
            active_need_id, conversation_history, satellite_need_ids,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.llm.chat_json(messages, model=self.model, temperature=0.0)
        return self._parse_and_validate(turn_number, raw)

    def judge_push_turn(
        self,
        turn_number: int,
        push_content: str,
        conversation_history: List[Dict[str, str]],
        allowed_fact_ids: Optional[List[str]] = None,
    ) -> JudgmentResult:
        """
        Judge a proactive push message (no user request triggered it).

        All needs addressed by a push are classified as "proactive".
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_push_prompt(
            turn_number, push_content, conversation_history,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.llm.chat_json(messages, model=self.model, temperature=0.0)
        result = self._parse_and_validate(
            turn_number,
            raw,
            response_text=push_content,
            allowed_fact_ids=allowed_fact_ids,
        )

        for nc in result.needs_addressed:
            nc.mode = "proactive"

        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt containing fact sheet, needs list, and
        judging instructions."""
        fact_lines = self._format_fact_sheet()
        need_lines = self._format_user_needs()

        return (
            "You are a strict evaluation judge for an AI assistant benchmark. "
            "Given the [Fact Sheet] and [User Needs List] below, analyze the "
            "AI assistant's response and determine:\n"
            "1. facts_conveyed   - Fact IDs whose information is accurately communicated\n"
            "2. facts_distorted  - Fact IDs that are mentioned but contain errors\n"
            "3. hallucinated_claims - Specific claims NOT grounded in the Fact Sheet\n"
            "4. needs_addressed  - Which user needs are substantively covered this turn\n\n"
            "Coverage mode definitions:\n"
            '"reactive":  The user explicitly asked about this need in the current turn, '
            "or mentioned it in passing / mentioned in passing as related context in the current turn, "
            "and the assistant provided substantive factual information from the Fact Sheet.\n"
            '"proactive": The user did NOT ask about this need in the current turn, '
            "but the assistant volunteered substantive factual information that addresses it.\n\n"
            "CRITICAL judging criteria:\n"
            "- A fact counts as conveyed ONLY when the assistant's statement is "
            "semantically consistent with the fact sheet entry.\n"
            "- If the assistant mentions a fact but gets details wrong (numbers, names, "
            "dates, etc.), it counts as distorted.\n"
            "- Concrete claims in the response that are entirely absent from the fact "
            "sheet count as hallucinated. Generic pleasantries or transition phrases "
            "do NOT count as hallucinations.\n"
            "- **IMPORTANT**: A need is addressed ONLY when the response conveys "
            "SUBSTANTIVE information that directly answers the need using facts "
            "from the Fact Sheet. Specifically:\n"
            "  - The response MUST convey at least one fact from the need's "
            "key_fact_ids to count as addressed.\n"
            "  - Generic advice like 'contact HR', 'check the company website', "
            "'ask your manager' does NOT count as addressing a need.\n"
            "  - Saying 'I don't have this information' does NOT address the need.\n"
            "  - A need with zero facts_conveyed from its key_fact_ids is NOT addressed.\n\n"
            f"[Fact Sheet]\n{fact_lines}\n\n"
            f"[User Needs List]\n{need_lines}\n\n"
            "You MUST respond in JSON with exactly this structure:\n"
            "{\n"
            '  "facts_conveyed": ["F01", "F02"],\n'
            '  "facts_distorted": ["F05"],\n'
            '  "hallucinated_claims": ["Description of the hallucinated claim"],\n'
            '  "needs_addressed": [\n'
            '    {"need_id": "N1", "mode": "reactive"},\n'
            '    {"need_id": "N3", "mode": "proactive"}\n'
            "  ]\n"
            "}\n"
        )

    def _build_user_prompt(
        self,
        turn_number: int,
        user_message: str,
        agent_response: str,
        active_need_id: Optional[str],
        conversation_history: List[Dict[str, str]],
        satellite_need_ids: Optional[List[str]] = None,
    ) -> str:
        """Build the user prompt with conversation context and the current
        turn to be judged."""
        # Include recent history (last 10 exchanges = 20 messages max)
        history_text = ""
        recent_history = conversation_history[-20:]
        if recent_history:
            history_lines = []
            for msg in recent_history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"[{role_label}]: {msg['content']}")
            history_text = "\n".join(history_lines)

        if active_need_id:
            active_need_hint = (
                f"The user explicitly raised primary need ID: {active_need_id} this turn."
            )
        else:
            active_need_hint = (
                "The user did NOT explicitly raise a specific need this turn "
                "(may be casual chat or a follow-up)."
            )

        satellite_need_ids = satellite_need_ids or []
        satellite_hint = ""
        if satellite_need_ids:
            joined = ", ".join(satellite_need_ids)
            satellite_hint = (
                f"The user also mentioned related need IDs: {joined}. "
                "Treat those related needs as user-mentioned context, so if the assistant "
                "addresses them this turn they should be judged as reactive, not proactive.\n\n"
            )

        return (
            f"[Turn {turn_number}]\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Current user message: {user_message}\n\n"
            f"Current assistant response: {agent_response}\n\n"
            f"{active_need_hint}\n\n"
            f"{satellite_hint}"
            "Based on the Fact Sheet and User Needs List, judge this turn's "
            "assistant response for factual accuracy and need coverage."
        )

    def _build_push_prompt(
        self,
        turn_number: int,
        push_content: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """Build the user prompt for judging a proactive push message."""
        history_text = ""
        recent_history = conversation_history[-20:]
        if recent_history:
            history_lines = []
            for msg in recent_history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"[{role_label}]: {msg['content']}")
            history_text = "\n".join(history_lines)

        return (
            f"[Turn {turn_number} — PROACTIVE PUSH]\n\n"
            f"Conversation history:\n{history_text}\n\n"
            "The user did NOT send a message this turn. The assistant "
            "proactively pushed the following information:\n\n"
            f"Push content: {push_content}\n\n"
            "All needs addressed by this push should be marked as 'proactive'. "
            "Judge this push for factual accuracy and need coverage."
        )

    def _format_fact_sheet(self) -> str:
        """Format the fact sheet for inclusion in the prompt."""
        lines = []
        for fact in self.scenario.fact_sheet:
            lines.append(f"{fact.id} [{fact.category}]: {fact.fact}")
        return "\n".join(lines)

    def _format_user_needs(self) -> str:
        """Format the user needs list for inclusion in the prompt."""
        lines = []
        for need in self.scenario.user_needs:
            fact_refs = ", ".join(need.key_fact_ids)
            lines.append(
                f"{need.id} ({need.level}): {need.description} "
                f"[key facts: {fact_refs}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing & validation
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        turn_number: int,
        raw: Dict[str, Any],
        response_text: Optional[str] = None,
        allowed_fact_ids: Optional[List[str]] = None,
    ) -> JudgmentResult:
        """Parse LLM JSON response and post-validate all IDs, discarding
        any references that do not exist in the scenario."""
        allowed_fact_id_set = self._allowed_fact_id_set(
            allowed_fact_ids,
            response_text,
        )

        # --- facts_conveyed: keep only valid fact IDs ---
        raw_conveyed = raw.get("facts_conveyed", [])
        valid_facts_conveyed = [
            fid for fid in raw_conveyed if fid in self._valid_fact_ids
        ]
        if len(valid_facts_conveyed) != len(raw_conveyed):
            discarded = set(raw_conveyed) - set(valid_facts_conveyed)
            logger.warning(
                "Turn %d: discarded invalid fact IDs from facts_conveyed: %s",
                turn_number, discarded,
            )
        facts_conveyed = self._filter_allowed_fact_ids(
            turn_number,
            "facts_conveyed",
            valid_facts_conveyed,
            allowed_fact_id_set,
        )

        # --- facts_distorted: keep only valid fact IDs ---
        raw_distorted = raw.get("facts_distorted", [])
        valid_facts_distorted = [
            fid for fid in raw_distorted if fid in self._valid_fact_ids
        ]
        if len(valid_facts_distorted) != len(raw_distorted):
            discarded = set(raw_distorted) - set(valid_facts_distorted)
            logger.warning(
                "Turn %d: discarded invalid fact IDs from facts_distorted: %s",
                turn_number, discarded,
            )
        facts_distorted = self._filter_allowed_fact_ids(
            turn_number,
            "facts_distorted",
            valid_facts_distorted,
            allowed_fact_id_set,
        )

        # --- hallucinated_claims: free-text, sanitize no-op placeholders ---
        hallucinated_claims = self._sanitize_hallucinated_claims(
            raw.get("hallucinated_claims", []),
        )

        # --- needs_addressed: keep only valid need IDs with valid modes ---
        valid_modes = {"reactive", "proactive"}
        needs_addressed: List[NeedCoverage] = []
        for entry in raw.get("needs_addressed", []):
            need_id = entry.get("need_id", "")
            mode = entry.get("mode", "reactive")
            if need_id not in self._valid_need_ids:
                logger.warning(
                    "Turn %d: discarded invalid need ID '%s' from needs_addressed",
                    turn_number, need_id,
                )
                continue
            if mode not in valid_modes:
                logger.warning(
                    "Turn %d: invalid mode '%s' for need '%s', defaulting to 'reactive'",
                    turn_number, mode, need_id,
                )
                mode = "reactive"
            needs_addressed.append(NeedCoverage(need_id=need_id, mode=mode))

        # --- Post-validation: enforce fact-based coverage rule ---
        # A need is only truly addressed if at least one of its key_fact_ids
        # appears in facts_conveyed (or facts_distorted, since distorted still
        # means the topic was engaged with factual content).
        conveyed_or_distorted = set(facts_conveyed) | set(facts_distorted)
        validated_needs: List[NeedCoverage] = []
        for nc in needs_addressed:
            key_facts = self._need_key_facts.get(nc.need_id, set())
            if key_facts & conveyed_or_distorted:
                validated_needs.append(nc)
            else:
                logger.info(
                    "Turn %d: need %s marked as addressed but no key facts "
                    "(%s) were conveyed — removing from needs_addressed",
                    turn_number, nc.need_id, key_facts,
                )

        return JudgmentResult(
            turn_number=turn_number,
            facts_conveyed=facts_conveyed,
            facts_distorted=facts_distorted,
            hallucinated_claims=hallucinated_claims,
            needs_addressed=validated_needs,
        )

    def _allowed_fact_id_set(
        self,
        allowed_fact_ids: Optional[List[str]],
        response_text: Optional[str],
    ) -> Optional[Set[str]]:
        if allowed_fact_ids is not None:
            return {
                fid for fid in allowed_fact_ids
                if fid in self._valid_fact_ids
            }
        if response_text:
            extracted = self._extract_fact_ids(response_text)
            if extracted:
                return extracted & self._valid_fact_ids
        return None

    @staticmethod
    def _extract_fact_ids(text: str) -> Set[str]:
        return {match.group(0).upper() for match in re.finditer(r"\bF\d+\b", text)}

    @staticmethod
    def _filter_allowed_fact_ids(
        turn_number: int,
        field_name: str,
        fact_ids: List[str],
        allowed_fact_ids: Optional[Set[str]],
    ) -> List[str]:
        if allowed_fact_ids is None:
            return fact_ids
        filtered = [fid for fid in fact_ids if fid in allowed_fact_ids]
        discarded = set(fact_ids) - set(filtered)
        if discarded:
            logger.warning(
                "Turn %d: discarded %s not present in push content/allowed facts: %s",
                turn_number,
                field_name,
                discarded,
            )
        return filtered

    @staticmethod
    def _sanitize_hallucinated_claims(raw_claims: Any) -> List[str]:
        if isinstance(raw_claims, str):
            raw_claims = [raw_claims]
        if not isinstance(raw_claims, list):
            return []

        sanitized: List[str] = []
        noop_markers = (
            "no hallucinated",
            "no hallucination",
            "not provide any hallucinated",
            "did not provide any hallucinated",
            "no unsupported",
        )
        for claim in raw_claims:
            if not isinstance(claim, str):
                continue
            text = claim.strip()
            if not text:
                continue
            normalized = text.lower().strip(" .。")
            if normalized in {"none", "n/a", "null"}:
                continue
            if any(marker in normalized for marker in noop_markers):
                continue
            sanitized.append(text)
        return sanitized
