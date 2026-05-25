"""
BenchmarkUserSimulator: scenario-driven user simulator for ProactiveBench.

Walks through the scenario's need sequence in turn_order, generating natural
user messages via LLM. Tracks which needs have been revealed to avoid
re-asking, and skips needs already covered by proactive agent behaviour.
"""

from difflib import SequenceMatcher
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ProactiveBench.generation.models import Scenario, UserNeed
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class SimulatorOutput:
    """Output of a single simulator step."""

    message: str                  # The generated user message
    active_need_id: Optional[str] # The need this message is asking about
    needs_skipped: List[str]      # Need IDs skipped because already covered
    is_final: bool                # Whether this is the last message
    satellite_need_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "active_need_id": self.active_need_id,
            "primary_need_id": self.active_need_id,
            "satellite_need_ids": list(self.satellite_need_ids or []),
            "needs_skipped": self.needs_skipped,
            "is_final": self.is_final,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulatorOutput":
        return cls(
            message=data["message"],
            active_need_id=data.get("active_need_id", data.get("primary_need_id")),
            needs_skipped=data.get("needs_skipped", []),
            is_final=data.get("is_final", False),
            satellite_need_ids=data.get("satellite_need_ids", []),
        )

    @property
    def primary_need_id(self) -> Optional[str]:
        return self.active_need_id


class BenchmarkUserSimulator:
    """Deterministic-order user simulator that follows the scenario's need
    sequence and generates natural questions via LLM."""

    _DUPLICATE_SIMILARITY_THRESHOLD = 0.85

    def __init__(
        self,
        scenario: Scenario,
        llm_client: LLMClient,
        model: str = "gpt-4o",
    ) -> None:
        self.scenario = scenario
        self.llm = llm_client
        self.model = model

        # Build needs queue sorted by turn_order
        self._needs_queue: List[UserNeed] = sorted(
            scenario.user_needs, key=lambda n: n.turn_order,
        )

        # Lookup sets
        self._must_have_ids: Set[str] = {
            n.id for n in scenario.user_needs if n.level == "must-have"
        }

        # Patience affects message generation temperature:
        # low = more direct/impatient, high = more patient/elaborate
        patience = getattr(scenario.simulator_config, "patience", "medium")
        self._temperature: float = {
            "low": 0.3, "medium": 0.5, "high": 0.7
        }.get(patience, 0.5)

        # Mutable state
        self._revealed_needs: Set[str] = set()   # Needs already asked about
        self._current_turn: int = 0
        self._max_turns: int = scenario.simulator_config.max_turns
        self._done: bool = False
        self._reveal_strategy: str = getattr(
            scenario.simulator_config,
            "need_reveal_strategy",
            "sequential",
        )
        self._needs_by_id: Dict[str, UserNeed] = {
            need.id: need for need in self._needs_queue
        }
        self._group_members: Dict[str, List[UserNeed]] = self._build_group_members()
        self._group_order: List[str] = self._build_group_order()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_message(
        self,
        conversation_history: List[Dict[str, str]],
        covered_need_ids: Set[str],
    ) -> Optional[SimulatorOutput]:
        """
        Generate the next user message following the need sequence.

        Args:
            conversation_history: The dialogue so far (list of role/content dicts).
            covered_need_ids: Need IDs already covered (by reactive or proactive means).

        Returns:
            SimulatorOutput if there is a message to send, None if the
            conversation should end.
        """
        if self._done:
            return None

        if self._use_grouped_reveal():
            active_need, satellite_needs, needs_skipped = self._select_grouped_turn(
                covered_need_ids
            )
        else:
            active_need, satellite_needs, needs_skipped = self._select_sequential_turn(
                covered_need_ids
            )

        # Step 2: If no active need found, check if there are uncovered must-haves
        if active_need is None:
            uncovered_must_haves = [
                n for n in self._needs_queue
                if n.id in self._must_have_ids
                and n.id not in covered_need_ids
            ]
            if uncovered_must_haves:
                # Pick the highest-priority (lowest turn_order) uncovered must-have
                active_need = uncovered_must_haves[0]
                logger.info(
                    "No sequential need available; falling back to uncovered "
                    "must-have: %s", active_need.id,
                )
            else:
                # All must-haves covered and no more needs in queue
                self._done = True
                return None

        # Step 3: Check turn limit
        if self._current_turn >= self._max_turns:
            self._done = True
            return None

        # Step 4: Generate natural user message via LLM
        message = self._generate_message(
            active_need,
            satellite_needs,
            conversation_history,
        )

        # Step 5: Update state
        self._revealed_needs.add(active_need.id)
        for need in satellite_needs:
            self._revealed_needs.add(need.id)
        self._current_turn += 1

        # Determine if this is the final turn
        surfaced_ids = {active_need.id, *(need.id for need in satellite_needs)}
        if self._use_grouped_reveal():
            remaining_unrevealed = [
                n for n in self._needs_queue
                if n.id not in covered_need_ids and n.id not in surfaced_ids
            ]
        else:
            remaining_unrevealed = [
                n for n in self._needs_queue
                if n.id not in self._revealed_needs and n.id not in covered_need_ids
            ]
        remaining_uncovered_must_haves = [
            n for n in self._needs_queue
            if n.id in self._must_have_ids and n.id not in covered_need_ids
            # Note: active_need may have just been revealed but not yet covered;
            # the agent response hasn't been evaluated yet, so we include it
        ]
        is_final = (
            len(remaining_unrevealed) == 0
            or self._current_turn >= self._max_turns
        )

        if is_final:
            self._done = True

        return SimulatorOutput(
            message=message,
            active_need_id=active_need.id,
            needs_skipped=needs_skipped,
            is_final=is_final,
            satellite_need_ids=[need.id for need in satellite_needs],
        )

    def is_done(self) -> bool:
        """Return True if the simulator has no more messages to generate."""
        return self._done

    # ------------------------------------------------------------------
    # LLM message generation
    # ------------------------------------------------------------------

    def _generate_message(
        self,
        active_need: UserNeed,
        satellite_needs: List[UserNeed],
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """Use LLM to generate a natural, in-character user question for
        the given need."""
        previous_user_message = self._last_user_message(conversation_history)
        message = self._chat_for_message(
            active_need,
            satellite_needs,
            conversation_history,
        )
        if (
            previous_user_message
            and self._is_duplicate_message(message, previous_user_message)
        ):
            logger.info(
                "Simulator generated duplicate-like user turn for need %s; retrying with guard.",
                active_need.id,
            )
            retried = self._chat_for_message(
                active_need,
                satellite_needs,
                conversation_history,
                previous_user_message=previous_user_message,
            )
            if not self._is_duplicate_message(retried, previous_user_message):
                return retried
            logger.info(
                "Simulator retry still duplicated prior user turn for need %s; using fallback rephrase.",
                active_need.id,
            )
            return self._fallback_message(active_need, satellite_needs)

        return message

    def _chat_for_message(
        self,
        active_need: UserNeed,
        satellite_needs: List[UserNeed],
        conversation_history: List[Dict[str, str]],
        previous_user_message: Optional[str] = None,
    ) -> str:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            active_need,
            satellite_needs,
            conversation_history,
            previous_user_message=previous_user_message,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm.chat(
            messages, model=self.model, temperature=self._temperature,
        )
        return response.strip()

    def _build_system_prompt(self) -> str:
        """Build the system prompt with user persona information.
        Deliberately excludes the fact sheet to prevent information leakage."""
        profile = self.scenario.user_profile
        return (
            "You are role-playing as a user in a conversation with an AI assistant. "
            "Stay in character and generate natural, realistic messages.\n\n"
            f"Your persona: {profile.persona}\n"
            f"Your current situation: {profile.context}\n"
            f"Your communication style: {profile.communication_style}\n\n"
            "Rules:\n"
            "- Generate ONLY the user's message, nothing else.\n"
            "- Do NOT include any metadata, labels, or quotation marks around the message.\n"
            "- Keep the message natural and conversational.\n"
            "- Match the communication style described above.\n"
            "- The message should be in the same language as the scenario "
            "(Chinese if the scenario is in Chinese, English if in English).\n"
            "- Do NOT mention fact IDs, need IDs, or any evaluation metadata.\n"
            "- Your message should naturally include some context about your "
            "situation — mention why you need this information, what you were "
            "doing, or how it relates to your current task. Do NOT just ask a "
            "bare question.\n"
        )

    def _build_user_prompt(
        self,
        active_need: UserNeed,
        satellite_needs: List[UserNeed],
        conversation_history: List[Dict[str, str]],
        previous_user_message: Optional[str] = None,
    ) -> str:
        """Build the user prompt containing the need description and
        recent conversation context."""
        # Format recent history (last 10 exchanges)
        history_text = "(No prior conversation)"
        recent = conversation_history[-20:]
        if recent:
            lines = []
            for msg in recent:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"[{role_label}]: {msg['content']}")
            history_text = "\n".join(lines)

        turn_label = f"Turn {self._current_turn + 1}"

        satellite_block = ""
        satellite_requirements = ""
        if satellite_needs:
            satellite_lines = "\n".join(
                f"- {need.description}" for need in satellite_needs
            )
            satellite_block = (
                "Related concerns to mention naturally in the same turn:\n"
                f"{satellite_lines}\n\n"
            )
            satellite_requirements = (
                "- Ask directly about the primary need.\n"
                "- Also weave in the related concerns as brief, natural side mentions.\n"
            )

        duplicate_guard_block = ""
        if previous_user_message:
            duplicate_guard_block = (
                "Important anti-duplication constraint:\n"
                f"- Your previous user message was: {previous_user_message}\n"
                "- Do NOT repeat or closely paraphrase that previous message.\n"
                "- This turn must ask about a different information need than the previous turn.\n\n"
            )

        return (
            f"[{turn_label}]\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"You now want to ask about the following primary need:\n"
            f"{active_need.description}\n\n"
            f"{satellite_block}"
            f"{duplicate_guard_block}"
            "Generate a natural message that expresses this information need "
            "in a way consistent with your persona. Requirements:\n"
            "- Do NOT copy the need description verbatim; rephrase it naturally.\n"
            f"{satellite_requirements}"
            "- Include some situational context (e.g., why you need this now, "
            "what prompted the question, or what you are currently doing).\n"
            "- If this is a follow-up, naturally reference what was discussed before."
        )

    @staticmethod
    def _last_user_message(
        conversation_history: List[Dict[str, str]],
    ) -> Optional[str]:
        for message in reversed(conversation_history):
            if message.get("role") == "user":
                content = str(message.get("content", "")).strip()
                if content:
                    return content
        return None

    @classmethod
    def _is_duplicate_message(cls, candidate: str, previous: str) -> bool:
        candidate_norm = cls._normalize_message(candidate)
        previous_norm = cls._normalize_message(previous)
        if not candidate_norm or not previous_norm:
            return False
        similarity = SequenceMatcher(None, candidate_norm, previous_norm).ratio()
        return similarity >= cls._DUPLICATE_SIMILARITY_THRESHOLD

    @staticmethod
    def _normalize_message(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _fallback_message(
        active_need: UserNeed,
        satellite_needs: List[UserNeed],
    ) -> str:
        primary = active_need.description.strip().rstrip("?.!")
        if primary:
            primary = primary[0].lower() + primary[1:]
        message = f"I'm moving to the next step and need clarity on {primary}."
        if satellite_needs:
            satellite_text = "; ".join(
                need.description.strip().rstrip("?.!") for need in satellite_needs
            )
            if satellite_text:
                message += f" I also want to touch on {satellite_text.lower()}."
        return message

    def _use_grouped_reveal(self) -> bool:
        return self._reveal_strategy == "grouped" and bool(self._group_order)

    def _select_sequential_turn(
        self,
        covered_need_ids: Set[str],
    ) -> tuple[Optional[UserNeed], List[UserNeed], List[str]]:
        active_need: Optional[UserNeed] = None
        needs_skipped: List[str] = []

        for need in self._needs_queue:
            if need.id in self._revealed_needs:
                continue
            if need.id in covered_need_ids:
                needs_skipped.append(need.id)
                self._revealed_needs.add(need.id)
                continue
            active_need = need
            break

        return active_need, [], needs_skipped

    def _select_grouped_turn(
        self,
        covered_need_ids: Set[str],
    ) -> tuple[Optional[UserNeed], List[UserNeed], List[str]]:
        prioritized_fallback: Optional[
            tuple[UserNeed, List[UserNeed], List[str]]
        ] = None

        for group_id in self._group_order:
            if not self._is_group_trigger_ready(group_id, covered_need_ids):
                continue

            remaining_members, needs_skipped = self._partition_group_members(
                group_id,
                covered_need_ids,
            )
            if not remaining_members:
                continue

            gating_members = [
                member
                for member in remaining_members
                if member.id in {
                    gating.id for gating in self._group_gating_members(group_id)
                }
            ]
            if gating_members:
                return self._build_group_turn(remaining_members, needs_skipped)

            if prioritized_fallback is None:
                prioritized_fallback = self._build_group_turn(
                    remaining_members,
                    needs_skipped,
                )

        if prioritized_fallback is not None:
            return prioritized_fallback

        return None, [], []

    def _partition_group_members(
        self,
        group_id: str,
        covered_need_ids: Set[str],
    ) -> tuple[List[UserNeed], List[str]]:
        remaining_members: List[UserNeed] = []
        needs_skipped: List[str] = []
        for need in self._group_members[group_id]:
            if need.id in covered_need_ids:
                needs_skipped.append(need.id)
                self._revealed_needs.add(need.id)
                continue
            remaining_members.append(need)
        return remaining_members, needs_skipped

    def _build_group_turn(
        self,
        remaining_members: List[UserNeed],
        needs_skipped: List[str],
    ) -> tuple[UserNeed, List[UserNeed], List[str]]:
        primary = next(
            (need for need in remaining_members if need.reveal_priority == 1),
            remaining_members[0],
        )
        satellites = [
            need for need in remaining_members if need.id != primary.id
        ]
        return primary, satellites, needs_skipped

    def _build_group_members(self) -> Dict[str, List[UserNeed]]:
        group_members: Dict[str, List[UserNeed]] = {}
        if self._reveal_strategy != "grouped":
            return group_members

        for group in self.scenario.reveal_groups:
            members = [
                self._needs_by_id[need_id]
                for need_id in group.member_need_ids
                if need_id in self._needs_by_id
            ]
            group_members[group.group_id] = sorted(
                members,
                key=lambda need: (
                    need.reveal_priority if need.reveal_priority is not None else 999,
                    need.turn_order,
                ),
            )

        return group_members

    def _build_group_order(self) -> List[str]:
        if self._reveal_strategy != "grouped":
            return []

        min_turn_by_group: Dict[str, int] = {}
        for group in self.scenario.reveal_groups:
            member_turns = [
                self._needs_by_id[need_id].turn_order
                for need_id in group.member_need_ids
                if need_id in self._needs_by_id
            ]
            min_turn_by_group[group.group_id] = min(member_turns) if member_turns else 10**9

        return [
            group.group_id
            for group in sorted(
                self.scenario.reveal_groups,
                key=lambda group: (
                    min_turn_by_group.get(group.group_id, 10**9),
                    group.group_id,
                ),
            )
        ]

    def _is_group_trigger_ready(
        self,
        group_id: str,
        covered_need_ids: Set[str],
    ) -> bool:
        group_meta = next(
            (group for group in self.scenario.reveal_groups if group.group_id == group_id),
            None,
        )
        if group_meta is None or group_meta.trigger_after is None:
            return True

        gating_members = self._group_gating_members(group_meta.trigger_after)
        return all(member.id in covered_need_ids for member in gating_members)

    def _group_gating_members(self, group_id: str) -> List[UserNeed]:
        members = self._group_members.get(group_id, [])
        gating_members = [
            member for member in members if member.level == "must-have"
        ]
        if not gating_members:
            gating_members = [
                member for member in members if member.reveal_priority == 1
            ]
        if not gating_members:
            gating_members = members
        return gating_members
