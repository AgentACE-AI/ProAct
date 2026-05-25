"""
Proactive Bench 数据模型。

定义 Fact-Grounded Scenario 的数据结构和校验逻辑。
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Fact:
    """事实表中的一条原子事实。"""
    id: str              # e.g., "F01"
    category: str        # e.g., "office", "IT", "HR"
    fact: str            # 原子事实文本

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        return cls(id=data["id"], category=data["category"], fact=data["fact"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserNeed:
    """用户的一个信息需求。"""
    id: str                                  # e.g., "N1"
    description: str                         # 需求描述
    level: str                               # "must-have" | "nice-to-have"
    key_fact_ids: List[str]                  # 满足该需求所需的 fact IDs
    predictable_after: Optional[str]         # 该需求在哪个前序需求之后变得可预判 (need ID or null)
    prediction_reason: Optional[str]         # 为什么可以预判
    turn_order: int                          # 用户暴露该需求的顺序
    reveal_group: Optional[str] = None       # benchmark-only simulator metadata
    reveal_group_label: Optional[str] = None
    reveal_priority: Optional[int] = None    # 1 = primary, 2+ = satellite

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserNeed":
        return cls(
            id=data["id"],
            description=data["description"],
            level=data.get("level", "must-have"),
            key_fact_ids=data.get("key_fact_ids", []),
            predictable_after=data.get("predictable_after"),
            prediction_reason=data.get("prediction_reason"),
            turn_order=data.get("turn_order", 0),
            reveal_group=data.get("reveal_group"),
            reveal_group_label=data.get("reveal_group_label"),
            reveal_priority=data.get("reveal_priority"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserProfile:
    """用户画像。"""
    persona: str                # 用户身份描述
    context: str                # 用户当前情境
    communication_style: str    # 沟通风格

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            persona=data["persona"],
            context=data["context"],
            communication_style=data.get("communication_style", "直接简洁"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RevealGroupMeta:
    """分组 reveal 元数据，仅供 benchmark simulator / judge 使用。"""

    group_id: str
    label: str
    member_need_ids: List[str]
    trigger_after: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RevealGroupMeta":
        return cls(
            group_id=data["group_id"],
            label=data.get("label", ""),
            member_need_ids=data.get("member_need_ids", []),
            trigger_after=data.get("trigger_after"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulatorConfig:
    """用户模拟器配置。"""
    need_reveal_strategy: str = "sequential"   # "sequential" | "grouped"
    patience: str = "medium"                   # "low" | "medium" | "high"
    max_turns: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulatorConfig":
        return cls(
            need_reveal_strategy=data.get("need_reveal_strategy", "sequential"),
            patience=data.get("patience", "medium"),
            max_turns=data.get("max_turns", 10),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioMetadata:
    """场景元数据。"""
    seed_id: str                           # 种子场景 ID
    variant: Optional[str] = None          # 变体标识 (e.g., "v1"), null 表示种子本身
    generated_by: str = "gpt-4o"           # 生成使用的模型
    reviewed: bool = False                 # 是否经过人工审核
    review_status: Optional[str] = None    # "approved" | "needs-revision" | None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioMetadata":
        return cls(
            seed_id=data.get("seed_id", ""),
            variant=data.get("variant"),
            generated_by=data.get("generated_by", "gpt-4o"),
            reviewed=data.get("reviewed", False),
            review_status=data.get("review_status"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Scenario:
    """完整的 Fact-Grounded Scenario。"""
    scenario_id: str
    domain: str
    description: str
    user_profile: UserProfile
    fact_sheet: List[Fact]
    user_needs: List[UserNeed]
    reveal_groups: List[RevealGroupMeta] = field(default_factory=list)  # benchmark-only
    simulator_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    metadata: ScenarioMetadata = field(default_factory=lambda: ScenarioMetadata(seed_id=""))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=data["scenario_id"],
            domain=data["domain"],
            description=data["description"],
            user_profile=UserProfile.from_dict(data["user_profile"]),
            fact_sheet=[Fact.from_dict(f) for f in data["fact_sheet"]],
            user_needs=[UserNeed.from_dict(n) for n in data["user_needs"]],
            reveal_groups=[
                RevealGroupMeta.from_dict(group)
                for group in data.get("reveal_groups", [])
            ],
            simulator_config=SimulatorConfig.from_dict(data.get("simulator_config", {})),
            metadata=ScenarioMetadata.from_dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "domain": self.domain,
            "description": self.description,
            "user_profile": self.user_profile.to_dict(),
            "fact_sheet": [f.to_dict() for f in self.fact_sheet],
            "user_needs": [n.to_dict() for n in self.user_needs],
            "reveal_groups": [group.to_dict() for group in self.reveal_groups],
            "simulator_config": self.simulator_config.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


class ScenarioValidator:
    """场景结构与一致性校验器。"""

    def validate(self, scenario: Scenario) -> List[str]:
        """
        校验场景的结构一致性。

        Returns:
            issues: 问题列表，空列表表示通过
        """
        issues: List[str] = []

        fact_ids = {f.id for f in scenario.fact_sheet}
        need_ids = {n.id for n in scenario.user_needs}

        # 基数检查
        if len(scenario.fact_sheet) < 10:
            issues.append(f"fact_sheet 过少: {len(scenario.fact_sheet)} < 10")
        if len(scenario.user_needs) < 4:
            issues.append(f"user_needs 过少: {len(scenario.user_needs)} < 4")

        must_have_count = sum(1 for n in scenario.user_needs if n.level == "must-have")
        if must_have_count < 3:
            issues.append(f"must-have 需求过少: {must_have_count} < 3")

        # Fact ID 唯一性
        if len(fact_ids) != len(scenario.fact_sheet):
            issues.append("fact_sheet 中存在重复 ID")

        # Need ID 唯一性
        if len(need_ids) != len(scenario.user_needs):
            issues.append("user_needs 中存在重复 ID")

        # key_fact_ids 引用检查
        for need in scenario.user_needs:
            for fid in need.key_fact_ids:
                if fid not in fact_ids:
                    issues.append(f"Need {need.id}: key_fact_ids 引用了不存在的 Fact '{fid}'")

            if not need.key_fact_ids:
                issues.append(f"Need {need.id}: key_fact_ids 为空")

        # predictable_after 引用检查
        for need in scenario.user_needs:
            if need.predictable_after is not None:
                if need.predictable_after not in need_ids:
                    issues.append(
                        f"Need {need.id}: predictable_after 引用了不存在的 Need '{need.predictable_after}'"
                    )
                if need.predictable_after == need.id:
                    issues.append(f"Need {need.id}: predictable_after 指向自身")
                if not need.prediction_reason:
                    issues.append(f"Need {need.id}: 有 predictable_after 但缺少 prediction_reason")

        # 循环依赖检查
        cycle_issues = self._check_cycles(scenario.user_needs)
        issues.extend(cycle_issues)

        # grouped reveal 结构检查
        issues.extend(self._validate_reveal_groups(scenario))

        # turn_order 检查
        orders = [n.turn_order for n in scenario.user_needs]
        if sorted(orders) != list(range(1, len(orders) + 1)):
            issues.append(f"turn_order 应为连续递增序列 1..N，实际: {sorted(orders)}")

        # level 值检查
        valid_levels = {"must-have", "nice-to-have"}
        for need in scenario.user_needs:
            if need.level not in valid_levels:
                issues.append(f"Need {need.id}: level '{need.level}' 不在 {valid_levels} 中")

        # 必填字段检查
        if not scenario.scenario_id:
            issues.append("缺少 scenario_id")
        if not scenario.domain:
            issues.append("缺少 domain")
        if not scenario.description:
            issues.append("缺少 description")

        return issues

    def _check_cycles(self, needs: List[UserNeed]) -> List[str]:
        """检查 predictable_after 链中是否存在循环依赖。"""
        issues = []
        graph = {}
        for need in needs:
            if need.predictable_after is not None:
                graph[need.id] = need.predictable_after

        for start_id in graph:
            visited = set()
            current = start_id
            while current in graph:
                if current in visited:
                    issues.append(f"predictable_after 存在循环依赖，涉及: {visited}")
                    break
                visited.add(current)
                current = graph[current]

        return issues

    def _validate_reveal_groups(self, scenario: Scenario) -> List[str]:
        """检查 grouped reveal 元数据的一致性。"""
        needs_by_id = {need.id: need for need in scenario.user_needs}
        has_group_metadata = (
            scenario.simulator_config.need_reveal_strategy == "grouped"
            or bool(scenario.reveal_groups)
            or any(need.reveal_group for need in scenario.user_needs)
        )
        if not has_group_metadata:
            return []

        issues: List[str] = []
        group_members_from_needs: Dict[str, List[UserNeed]] = defaultdict(list)
        for need in scenario.user_needs:
            if not need.reveal_group:
                issues.append(f"Need {need.id}: grouped reveal 缺少 reveal_group")
                continue
            group_members_from_needs[need.reveal_group].append(need)
            if not need.reveal_group_label:
                issues.append(f"Need {need.id}: grouped reveal 缺少 reveal_group_label")
            if need.reveal_priority is None or need.reveal_priority < 1:
                issues.append(
                    f"Need {need.id}: grouped reveal 需要 reveal_priority >= 1"
                )

        group_meta_by_id = {group.group_id: group for group in scenario.reveal_groups}

        for group_id, members in group_members_from_needs.items():
            if group_id not in group_meta_by_id:
                issues.append(f"Reveal group {group_id}: 缺少 Scenario.reveal_groups 元数据")
            if len(members) > 4:
                issues.append(
                    f"Reveal group {group_id}: group size {len(members)} 超过 4"
                )
            if not any(member.reveal_priority == 1 for member in members):
                issues.append(
                    f"Reveal group {group_id}: 必须至少有一个 reveal_priority=1 的 primary need"
                )

        for group in scenario.reveal_groups:
            if not group.member_need_ids:
                issues.append(f"Reveal group {group.group_id}: member_need_ids 不能为空")
                continue
            for need_id in group.member_need_ids:
                if need_id not in needs_by_id:
                    issues.append(
                        f"Reveal group {group.group_id}: 引用了不存在的 Need '{need_id}'"
                    )
                    continue
                if needs_by_id[need_id].reveal_group != group.group_id:
                    issues.append(
                        f"Reveal group {group.group_id}: Need {need_id} 的 reveal_group 不匹配"
                    )
            if group.trigger_after is not None and group.trigger_after not in group_meta_by_id:
                issues.append(
                    f"Reveal group {group.group_id}: trigger_after 引用了不存在的 group '{group.trigger_after}'"
                )

        issues.extend(self._validate_cross_group_prediction_quality(scenario))
        issues.extend(self._check_reveal_group_cycles(scenario.reveal_groups))
        return issues

    def _validate_cross_group_prediction_quality(
        self,
        scenario: Scenario,
    ) -> List[str]:
        """Validate whether grouped-reveal data leaves enough auditable
        proactive headroom for Route C."""
        issues: List[str] = []
        needs_by_id = {need.id: need for need in scenario.user_needs}
        group_members: Dict[str, List[UserNeed]] = defaultdict(list)
        for need in scenario.user_needs:
            if need.reveal_group:
                group_members[need.reveal_group].append(need)

        intra_count = 0
        cross_count = 0
        for need in scenario.user_needs:
            if need.predictable_after is None:
                continue
            pred = needs_by_id.get(need.predictable_after)
            if pred is None:
                continue  # referential integrity checked elsewhere
            if need.reveal_group and pred.reveal_group:
                if need.reveal_group == pred.reveal_group:
                    intra_count += 1
                else:
                    cross_count += 1

        total_links = intra_count + cross_count
        if total_links > 0 and cross_count * 2 < total_links:
            issues.append(
                "At least half of predictable_after links must be cross-group "
                f"for meaningful proactive credit ({cross_count} cross, {intra_count} intra)."
            )

        auditable_targets: List[UserNeed] = []
        for need in scenario.user_needs:
            if need.predictable_after is None or not need.reveal_group:
                continue
            pred = needs_by_id.get(need.predictable_after)
            if pred is None or not pred.reveal_group:
                continue
            if need.reveal_group == pred.reveal_group:
                continue

            members = group_members.get(need.reveal_group, [])
            if len(members) == 1:
                auditable_targets.append(need)
                continue

            if need.reveal_priority != 1:
                continue

            other_members = [
                member for member in members if member.id != need.id
            ]
            if any(member.level == "must-have" for member in other_members):
                continue
            auditable_targets.append(need)

        if len(auditable_targets) < 2:
            issues.append(
                "Grouped reveal scenarios must include at least 2 auditable proactive targets."
            )

        if auditable_targets and not any(
            need.level == "nice-to-have" for need in auditable_targets
        ):
            issues.append(
                "Grouped reveal scenarios must include at least 1 nice-to-have auditable proactive target."
            )

        # Validate trigger ordering for cross-group links
        issues.extend(
            self._check_cross_group_trigger_order(scenario, needs_by_id)
        )
        return issues

    def _check_cross_group_trigger_order(
        self,
        scenario: Scenario,
        needs_by_id: Dict[str, UserNeed],
    ) -> List[str]:
        """Verify cross-group predictable_after respects trigger_after ordering.

        For each cross-group link (need in group_target predicted after
        predecessor in group_pred), group_pred should be reachable before
        group_target in the trigger chain.  If group_target is an ancestor of
        group_pred (i.e. target fires first), the link is backwards.
        """
        issues: List[str] = []
        group_meta = {g.group_id: g for g in scenario.reveal_groups}
        if not group_meta:
            return issues

        def ancestors(gid: str) -> set:
            """Return set of all ancestor group IDs via trigger_after chain."""
            visited: set = set()
            current = group_meta.get(gid)
            while current and current.trigger_after:
                if current.trigger_after in visited:
                    break  # cycle; handled by _check_reveal_group_cycles
                visited.add(current.trigger_after)
                current = group_meta.get(current.trigger_after)
            return visited

        for need in scenario.user_needs:
            if need.predictable_after is None or not need.reveal_group:
                continue
            pred = needs_by_id.get(need.predictable_after)
            if pred is None or not pred.reveal_group:
                continue
            if need.reveal_group == pred.reveal_group:
                continue  # intra-group, skip

            target_group = need.reveal_group
            pred_group = pred.reveal_group

            # Check if target_group is an ancestor of pred_group
            # (meaning target fires before pred — backwards link)
            pred_ancestors = ancestors(pred_group)
            if target_group in pred_ancestors:
                issues.append(
                    f"[WARNING] Need {need.id} ({target_group}) has "
                    f"predictable_after={need.predictable_after} ({pred_group}), "
                    f"but {target_group} triggers before {pred_group} in the "
                    f"trigger chain — the prediction link is backwards."
                )

        return issues

    def _check_reveal_group_cycles(
        self,
        reveal_groups: List[RevealGroupMeta],
    ) -> List[str]:
        """检查 reveal_group trigger graph 中是否存在循环依赖。"""
        issues: List[str] = []
        graph = {
            group.group_id: group.trigger_after
            for group in reveal_groups
            if group.trigger_after is not None
        }

        for start_id in graph:
            visited = set()
            current = start_id
            while current in graph:
                if current in visited:
                    issues.append(
                        f"reveal_group trigger_after 存在循环依赖，涉及: {visited}"
                    )
                    break
                visited.add(current)
                current = graph[current]

        return issues
