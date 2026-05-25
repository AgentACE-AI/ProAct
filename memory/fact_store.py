"""
用户事实存储。

基于 JSON 文件存储用户提到的事实/实体信息。
"""

import json
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from core.config import StorageConfig
from core.models import UserFact


class FactStore:
    """
    用户事实存储。

    使用 JSON 文件存储用户提到的事实信息，支持：
    - 添加/获取事实
    - 按实体名称查找和更新
    - 格式化为 Prompt 文本
    """

    def __init__(self, user_id: str, storage_config: StorageConfig):
        """
        初始化事实存储。

        Args:
            user_id: 用户 ID
            storage_config: 存储配置
        """
        self.user_id = user_id
        self.storage_config = storage_config
        self.file_path = storage_config.get_facts_path(user_id)
        self._facts: Dict[str, UserFact] = {}  # key: entity name (normalized)
        self._load()

    def _normalize_entity_name(self, name: str) -> str:
        """标准化实体名称（用于去重和查找）"""
        return name.strip().lower()

    # ==================== 实体去重（只读合并，用于 format_for_prompt）====================

    _RELATION_WORDS: Set[str] = {
        "brother", "sister", "mother", "father", "mom", "dad",
        "uncle", "aunt", "cousin", "nephew", "niece",
        "boss", "subordinate", "coworker", "colleague",
        "friend", "mentor", "roommate", "partner",
        "wife", "husband", "son", "daughter",
        "grandma", "grandpa", "grandmother", "grandfather",
        "girlfriend", "boyfriend", "spouse", "mentee",
        "male cousin", "female cousin",
    }

    @staticmethod
    def _base_name(entity: str) -> str:
        """提取基础名字（去掉括号内容，统一小写去多余空格）"""
        name = re.sub(r"\s*\(.*?\)\s*", "", entity or "").strip().lower()
        return re.sub(r"\s+", " ", name)

    def _is_relation_word(self, name: str) -> bool:
        """判断名字是否为纯关系词（如 subordinate、male cousin）"""
        base = self._base_name(name)
        tokens = [t for t in base.split() if t not in {"my", "the", "a", "an"}]
        return bool(tokens) and " ".join(tokens) in self._RELATION_WORDS

    def _deduplicate_facts(self, facts: List[UserFact]) -> List[UserFact]:
        """
        对 person 类型实体做只读去重合并，返回合并后的事实列表。

        不修改底层 _facts 存储，仅影响 format_for_prompt 输出。

        合并规则：
        1. 名字包含：短名是长名的子串 → 合并到长名
        2. 关系匹配：同 relationship 且一方为纯关系词 → 合并到具名实体
        """
        persons = [f for f in facts if f.entity_type == "person"]
        others = [f for f in facts if f.entity_type != "person"]

        if len(persons) < 2:
            return facts

        n = len(persons)
        base_names = [self._base_name(p.entity) for p in persons]
        rels = [(p.relationship or "").strip().lower() for p in persons]
        is_rel = [self._is_relation_word(p.entity) for p in persons]

        # 纯关系词实体如果 relationship 为空，用其 base_name 作为关系
        for i in range(n):
            if is_rel[i] and not rels[i]:
                rels[i] = base_names[i]

        # Union-Find: parent[i] 指向合并目标
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(src: int, dst: int) -> bool:
            rs, rd = find(src), find(dst)
            if rs == rd:
                return False
            parent[rs] = rd
            return True

        # 关系词实体合并后标记为已消费，防止传递性误合并
        consumed_rel: Set[int] = set()

        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue

                ni, nj = base_names[i], base_names[j]

                # 规则 1：名字词级包含（短名的所有词都出现在长名中）
                # 使用词集合而非子串匹配，避免 "ann" ⊂ "joanna" 的误合并
                if ni and nj and ni != nj:
                    words_i = set(ni.split())
                    words_j = set(nj.split())
                    if words_i < words_j:  # i 的词是 j 的真子集
                        union(i, j)
                        continue
                    if words_j < words_i:  # j 的词是 i 的真子集
                        union(j, i)
                        continue

                # 规则 2：同 relationship，纯关系词合并到具名实体
                # 关系词一旦被合并就标记为已消费，防止传递性误合并
                # （如 "subordinate" 桥接 "John Doe" 和 "Jane Roe"）
                if rels[i] and rels[i] == rels[j]:
                    if is_rel[i] and not is_rel[j] and i not in consumed_rel:
                        union(i, j)
                        consumed_rel.add(i)
                    elif is_rel[j] and not is_rel[i] and j not in consumed_rel:
                        union(j, i)
                        consumed_rel.add(j)

        # 按组聚合
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        merged_persons: List[UserFact] = []
        for root, indices in sorted(groups.items(), key=lambda kv: min(kv[1])):
            if len(indices) == 1:
                merged_persons.append(deepcopy(persons[indices[0]]))
                continue

            # 以代表元素（root）为基准，合并所有属性
            canonical = deepcopy(persons[root])
            for idx in indices:
                if idx == root:
                    continue
                canonical.merge_attributes(persons[idx].attributes)

            # 确保 relationship 不为空
            if not canonical.relationship:
                for idx in indices:
                    if persons[idx].relationship:
                        canonical.relationship = persons[idx].relationship
                        break

            merged_persons.append(canonical)

        return merged_persons + others

    def _load(self) -> None:
        """从文件加载事实"""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    facts_data = data.get("facts", {})
                    for key, fact_dict in facts_data.items():
                        self._facts[key] = UserFact.from_dict(fact_dict)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[FactStore] 加载事实失败: {e}")
                self._facts = {}

    def _save(self) -> None:
        """保存事实到文件"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "user_id": self.user_id,
            "facts": {
                key: fact.to_dict()
                for key, fact in self._facts.items()
            },
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_fact(self, fact: UserFact) -> None:
        """
        添加或更新事实。

        如果实体已存在，合并属性；否则添加新实体。

        Args:
            fact: 要添加的事实
        """
        key = self._normalize_entity_name(fact.entity)

        if key in self._facts:
            # 已存在：合并属性
            existing = self._facts[key]
            existing.merge_attributes(fact.attributes)
            # 更新其他可能变化的字段
            if fact.relationship and not existing.relationship:
                existing.relationship = fact.relationship
            if fact.entity_type != "other" and existing.entity_type == "other":
                existing.entity_type = fact.entity_type
        else:
            # 新实体
            self._facts[key] = fact

        self._save()

    def add_facts_from_extraction(
        self,
        facts_data: List[Dict[str, Any]],
        source_topic: str = "",
    ) -> int:
        """
        从 LLM 提取结果批量添加事实。

        Args:
            facts_data: LLM 提取的事实列表，每项包含:
                - entity: str
                - entity_type: str (可选)
                - attributes: Dict[str, str] (可选)
                - relationship: str (可选)
            source_topic: 来源话题

        Returns:
            添加的事实数量
        """
        count = 0
        for fact_dict in facts_data:
            entity = fact_dict.get("entity", "").strip()
            if not entity:
                continue

            fact = UserFact(
                entity=entity,
                entity_type=fact_dict.get("entity_type", "other"),
                attributes=fact_dict.get("attributes", {}),
                relationship=fact_dict.get("relationship", ""),
                source_topic=source_topic,
            )
            self.add_fact(fact)
            count += 1

        return count

    def get_fact(self, entity: str) -> Optional[UserFact]:
        """
        获取指定实体的事实。

        Args:
            entity: 实体名称

        Returns:
            事实对象，或 None
        """
        key = self._normalize_entity_name(entity)
        return self._facts.get(key)

    def get_all_facts(self) -> List[UserFact]:
        """
        获取所有事实。

        Returns:
            所有事实列表
        """
        return list(self._facts.values())

    def get_facts_by_type(self, entity_type: str) -> List[UserFact]:
        """
        按类型获取事实。

        Args:
            entity_type: 实体类型 (person/place/organization/event/other)

        Returns:
            指定类型的事实列表
        """
        return [
            fact for fact in self._facts.values()
            if fact.entity_type == entity_type
        ]

    def format_for_prompt(self) -> str:
        """
        格式化所有事实为 Prompt 文本。

        在输出前对 person 类型实体做去重合并，将碎片化的同一人信息合并展示。

        Returns:
            格式化的事实文本
        """
        if not self._facts:
            return ""

        # 去重合并后再格式化
        facts_for_prompt = self._deduplicate_facts(list(self._facts.values()))
        if not facts_for_prompt:
            return ""

        lines = []

        # 按类型分组
        by_type: Dict[str, List[UserFact]] = {}
        for fact in facts_for_prompt:
            t = fact.entity_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(fact)

        # 格式化输出
        type_labels = {
            "person": "People",
            "place": "Places",
            "organization": "Organizations",
            "event": "Events",
            "other": "Other",
        }

        for entity_type, facts in by_type.items():
            label = type_labels.get(entity_type, entity_type)
            lines.append(f"[{label}]")
            for fact in facts:
                lines.append(f"  - {fact.to_prompt_text()}")

        return "\n".join(lines)

    def exists(self) -> bool:
        """
        检查是否有存储的事实。

        Returns:
            是否有事实
        """
        return bool(self._facts)

    def clear(self) -> None:
        """清空所有事实"""
        self._facts = {}
        self._save()

    def remove_fact(self, entity: str) -> bool:
        """
        删除指定实体的事实。

        Args:
            entity: 实体名称

        Returns:
            是否成功删除
        """
        key = self._normalize_entity_name(entity)
        if key in self._facts:
            del self._facts[key]
            self._save()
            return True
        return False
