import json
from datetime import datetime
from typing import Dict, List, Optional

from core.config import StorageConfig
from services.proactive.models import ProactiveBrief, ProactiveCandidate, ProactiveDecision, ProactiveItem, ProactiveItemState


class ProactiveItemService:
    def __init__(
        self,
        user_id: str,
        storage_config: StorageConfig,
        *,
        include_delivered_in_active: bool = True,
    ):
        self.user_id = user_id
        self.storage_config = storage_config
        self.include_delivered_in_active = include_delivered_in_active
        self._path = storage_config.get_proactive_items_path(user_id)
        self._items: Dict[str, ProactiveItem] = {}
        self._load()

    def add(self, item: ProactiveItem) -> ProactiveItem:
        self._items[item.item_id] = item
        self._save()
        return item

    def add_from_candidate(self, candidate: ProactiveCandidate) -> ProactiveItem:
        return self.add(ProactiveItem.from_candidate(candidate))

    def get(self, item_id: str) -> Optional[ProactiveItem]:
        return self._items.get(item_id)

    def list_all(self) -> List[ProactiveItem]:
        return list(self._items.values())

    def list_active(self) -> List[ProactiveItem]:
        active_states = {
            ProactiveItemState.CREATED,
            ProactiveItemState.BRIEF_READY,
            ProactiveItemState.QUEUED,
        }
        if self.include_delivered_in_active:
            active_states.add(ProactiveItemState.DELIVERED)
        return [item for item in self._items.values() if item.state in active_states]

    def mark_brief_ready(self, item_id: str, brief: ProactiveBrief) -> Optional[ProactiveItem]:
        item = self.get(item_id)
        if not item:
            return None
        item.brief = brief
        item.state = ProactiveItemState.BRIEF_READY
        item.updated_at = datetime.now().isoformat()
        self._save()
        return item

    def mark_queued(
        self,
        item_id: str,
        decision: Optional[ProactiveDecision] = None,
    ) -> Optional[ProactiveItem]:
        item = self.get(item_id)
        if not item:
            return None
        if decision:
            item.decision = decision
        item.state = ProactiveItemState.QUEUED
        item.updated_at = datetime.now().isoformat()
        self._save()
        return item

    def mark_delivered(
        self,
        item_id: str,
        decision: Optional[ProactiveDecision] = None,
    ) -> Optional[ProactiveItem]:
        item = self.get(item_id)
        if not item:
            return None
        if decision:
            item.decision = decision
        item.state = ProactiveItemState.DELIVERED
        item.updated_at = datetime.now().isoformat()
        self._save()
        return item

    def close(self, item_id: str, reason: str) -> Optional[ProactiveItem]:
        item = self.get(item_id)
        if not item:
            return None
        item.close_reason = reason
        item.state = ProactiveItemState.CLOSED
        item.updated_at = datetime.now().isoformat()
        self._save()
        return item

    def find_active_by_dedupe_key(self, dedupe_key: str) -> Optional[ProactiveItem]:
        for item in self.list_active():
            if item.candidate.dedupe_key == dedupe_key:
                return item
        return None

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return

        items = data.get("items", [])
        self._items = {
            item_data["item_id"]: ProactiveItem.from_dict(item_data)
            for item_data in items
        }

    def _save(self) -> None:
        payload = {
            "items": [item.to_dict() for item in self._items.values()],
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
