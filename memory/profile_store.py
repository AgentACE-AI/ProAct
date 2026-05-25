"""
用户画像存储。

基于 JSON 文件存储用户画像信息。
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config import StorageConfig
from core.models import UserProfile


class ProfileStore:
    """
    用户画像存储。

    使用 JSON 文件存储用户画像，支持：
    - 创建/获取画像
    - 增量更新画像字段
    - 格式化为 Prompt 文本
    """

    def __init__(self, user_id: str, storage_config: StorageConfig):
        """
        初始化画像存储。

        Args:
            user_id: 用户 ID
            storage_config: 存储配置
        """
        self.user_id = user_id
        self.storage_config = storage_config
        self.file_path = storage_config.get_profile_path(user_id)
        self._profile: Optional[UserProfile] = None
        self._load()

    def _load(self) -> None:
        """从文件加载画像"""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._profile = UserProfile.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[ProfileStore] 加载画像失败: {e}")
                self._profile = None

    def _save(self) -> None:
        """保存画像到文件"""
        if self._profile:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._profile.to_dict(), f, ensure_ascii=False, indent=2)

    def get(self) -> UserProfile:
        """
        获取用户画像。

        如果画像不存在，创建一个空画像。

        Returns:
            用户画像
        """
        if not self._profile:
            self._profile = UserProfile(user_id=self.user_id)
            self._save()
        return self._profile

    def save(self, profile: UserProfile) -> None:
        """
        保存用户画像。

        Args:
            profile: 要保存的画像
        """
        self._profile = profile
        self._profile.updated_at = datetime.now().isoformat()
        self._save()

    def update(self, updates: Dict[str, Any]) -> UserProfile:
        """
        增量更新用户画像。

        对于列表类型字段（interests, goals, etc.），会追加新值（去重）。
        对于字符串类型字段（role, age, location），会替换。

        Args:
            updates: 要更新的字段
                - role: str (替换)
                - age: str (替换)
                - location: str (替换)
                - interests: List[str] (追加)
                - goals: List[str] (追加)
                - challenges: List[str] (追加)
                - traits: List[str] (追加)
                - communication_style: List[str] (追加)

        Returns:
            更新后的画像
        """
        profile = self.get()

        # 字符串字段：替换
        if "role" in updates and updates["role"]:
            profile.role = updates["role"]

        if "age" in updates and updates["age"]:
            profile.age = updates["age"]

        if "location" in updates and updates["location"]:
            profile.location = updates["location"]

        # 列表字段：追加（去重）
        list_fields = [
            "interests",
            "goals",
            "challenges",
            "traits",
            "communication_style",
            "research_history",
        ]

        for field in list_fields:
            if field in updates and updates[field]:
                current_list = getattr(profile, field, [])
                new_items = updates[field]
                if isinstance(new_items, list):
                    for item in new_items:
                        if item and item not in current_list:
                            current_list.append(item)
                elif isinstance(new_items, str):
                    if new_items not in current_list:
                        current_list.append(new_items)

        profile.updated_at = datetime.now().isoformat()
        self._profile = profile
        self._save()

        return profile

    def add_interest(self, interest: str) -> None:
        """
        添加兴趣。

        Args:
            interest: 兴趣描述
        """
        if interest:
            self.update({"interests": [interest]})

    def add_goal(self, goal: str) -> None:
        """
        添加目标。

        Args:
            goal: 目标描述
        """
        if goal:
            self.update({"goals": [goal]})

    def add_trait(self, trait: str) -> None:
        """
        添加特点。

        Args:
            trait: 特点描述
        """
        if trait:
            self.update({"traits": [trait]})

    def add_challenge(self, challenge: str) -> None:
        """
        添加挑战。

        Args:
            challenge: 挑战描述
        """
        if challenge:
            self.update({"challenges": [challenge]})

    def add_research_topic(self, topic: str) -> None:
        """
        添加调研主题到历史记录。

        Args:
            topic: 调研主题
        """
        if topic:
            self.update({"research_history": [topic]})

    def get_research_history(self) -> List[str]:
        """
        获取调研历史主题列表。

        Returns:
            调研历史主题列表
        """
        profile = self.get()
        return profile.research_history

    def exists(self) -> bool:
        """
        检查画像是否存在且有内容。

        Returns:
            画像是否有实际内容
        """
        if not self._profile:
            return False

        return bool(
            self._profile.role
            or self._profile.interests
            or self._profile.goals
            or self._profile.traits
            or self._profile.challenges
        )

    def format_for_prompt(self) -> str:
        """
        格式化画像为 Prompt 文本。

        Returns:
            格式化的画像文本
        """
        profile = self.get()
        return profile.to_prompt_text()

    def clear(self) -> None:
        """清空画像"""
        self._profile = UserProfile(user_id=self.user_id)
        self._save()

    def get_interests(self) -> List[str]:
        """
        获取用户兴趣列表。

        Returns:
            兴趣列表
        """
        profile = self.get()
        return profile.interests

    def get_goals(self) -> List[str]:
        """
        获取用户目标列表。

        Returns:
            目标列表
        """
        profile = self.get()
        return profile.goals
