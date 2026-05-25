"""
MemBench Benchmark 数据加载器。

负责加载、转换和预处理 MemBench 数据集。
注意：只支持 FirstAgent (Participation) 数据，因为 Memory System V2 是对话记忆系统。

支持两种数据格式：
- raw: 原始 data/ 目录下的数据
- data2test: 论文中采样的 data2test/0-10k/ 和 data2test/100k/ 数据
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    BenchmarkConfig,
    MemBenchMessage,
    Trajectory,
)

logger = logging.getLogger(__name__)


# ============== 问题类型映射 (data2test -> 标准类型) ==============

# LowLevel 映射
LOWLEVEL_TYPE_MAP = {
    "simple": "Single-hop",
    "conditional": "Multi-hop",
    "comparative": "Comparative",
    "aggregative": "Aggregative",
    "post_processing": "Post_processing",
    "knowledge_update": "Knowledge_updating",
    "lowlevel_rec": "Single-session-assistant",
    "RecMultiSession": "Multi-session-assistant",
    "noisy": "Noisy",
}

# HighLevel 映射
# highlevel/emotion -> Emotion
# highlevel/{movie,food,book} -> Preference
# highlevel_rec/* -> Preference
HIGHLEVEL_EMOTION_SCENARIOS = {"emotion"}
HIGHLEVEL_PREFERENCE_SCENARIOS = {"movie", "food", "book"}


class MemBenchDataLoader:
    """MemBench 数据加载与转换（仅支持 FirstAgent）"""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        初始化数据加载器。

        Args:
            config: Benchmark 配置
        """
        self.config = config or BenchmarkConfig()
        self._cache: Dict[str, Any] = {}

    def _detect_data_format(self, raw_data: Dict) -> str:
        """
        检测数据格式类型。

        Args:
            raw_data: 加载的原始 JSON 数据

        Returns:
            "raw" 或 "data2test"
        """
        keys = set(raw_data.keys())

        # data2test HighLevel 格式: highlevel, highlevel_rec
        if "highlevel" in keys or "highlevel_rec" in keys:
            return "data2test"

        # data2test LowLevel 格式: simple, conditional, etc.
        if "simple" in keys or "conditional" in keys:
            return "data2test"

        # raw 格式: Emotion, Preference, Single-hop, Multi-hop, etc.
        return "raw"

    def load_trajectories(
        self,
        level: str = "LowLevel",
        question_types: Optional[List[str]] = None,
        max_per_type: Optional[int] = None,
        capacity_test: bool = False,
        min_tokens: int = 8000,
    ) -> List[Trajectory]:
        """
        加载轨迹数据（仅 FirstAgent）。

        Args:
            level: "LowLevel" 或 "HighLevel"
            question_types: 要加载的问题类型列表，None 表示全部
            max_per_type: 每种类型最大加载数量，None 表示全部
            capacity_test: 是否为 100k 容量测试（选择更长的轨迹）
            min_tokens: 容量测试时，轨迹的最小 token 数

        Returns:
            轨迹列表
        """
        data_path = self.config.get_data_path("FirstAgent", level)
        cache_key = str(data_path)

        # 尝试从缓存加载
        if cache_key not in self._cache:
            logger.info(f"Loading data from {data_path}...")
            with open(data_path, "r", encoding="utf-8") as f:
                self._cache[cache_key] = json.load(f)

        raw_data = self._cache[cache_key]

        # 检测数据格式
        data_format = self._detect_data_format(raw_data)
        logger.info(f"Detected data format: {data_format}")

        # 根据格式和级别加载数据
        if data_format == "data2test":
            if level == "HighLevel":
                trajectories = self._load_data2test_highlevel(
                    raw_data, question_types, max_per_type
                )
            else:
                trajectories = self._load_data2test_lowlevel(
                    raw_data, question_types, max_per_type
                )
        else:
            trajectories = self._load_raw_format(
                raw_data, question_types, max_per_type, capacity_test, min_tokens
            )

        logger.info(f"Loaded {len(trajectories)} trajectories (capacity_test={capacity_test})")
        return trajectories

    def _load_data2test_highlevel(
        self,
        raw_data: Dict,
        question_types: Optional[List[str]],
        max_per_type: Optional[int],
    ) -> List[Trajectory]:
        """
        加载 data2test 格式的 HighLevel 数据。

        数据结构:
        {
            "highlevel": {"movie": [...], "food": [...], "book": [...], "emotion": [...]},
            "highlevel_rec": {"movie": [...], "food": [...], "book": [...]}
        }

        映射:
        - highlevel/emotion -> Emotion
        - highlevel/{movie,food,book} -> Preference
        - highlevel_rec/* -> Preference
        """
        trajectories_by_type: Dict[str, List[Trajectory]] = {}

        for top_key, scenarios in raw_data.items():
            if not isinstance(scenarios, dict):
                continue

            for scenario, items in scenarios.items():
                if not isinstance(items, list):
                    continue

                # 确定问题类型
                if scenario.lower() in HIGHLEVEL_EMOTION_SCENARIOS:
                    qtype = "Emotion"
                else:
                    qtype = "Preference"

                # 过滤问题类型
                if question_types and qtype not in question_types:
                    continue

                if qtype not in trajectories_by_type:
                    trajectories_by_type[qtype] = []

                for item in items:
                    try:
                        # 转换 message_list 格式
                        converted_item = self._convert_data2test_item(item)
                        traj = Trajectory.from_dict(converted_item, qtype, scenario)
                        trajectories_by_type[qtype].append(traj)
                    except Exception as e:
                        logger.warning(f"Failed to parse trajectory: {e}")
                        continue

        # 合并并限制数量
        all_trajectories = []
        for qtype, trajs in trajectories_by_type.items():
            if max_per_type:
                trajs = trajs[:max_per_type]
            all_trajectories.extend(trajs)
            logger.info(f"  {qtype}: {len(trajs)} trajectories")

        return all_trajectories

    def _load_data2test_lowlevel(
        self,
        raw_data: Dict,
        question_types: Optional[List[str]],
        max_per_type: Optional[int],
    ) -> List[Trajectory]:
        """
        加载 data2test 格式的 LowLevel 数据。

        数据结构:
        {
            "simple": {"roles": [...], "events": [...]},
            "conditional": {"roles": [...], "events": [...]},
            ...
        }
        """
        trajectories_by_type: Dict[str, List[Trajectory]] = {}

        for raw_type, scenarios in raw_data.items():
            if not isinstance(scenarios, dict):
                continue

            # 映射到标准问题类型
            qtype = LOWLEVEL_TYPE_MAP.get(raw_type, raw_type)

            # 过滤问题类型
            if question_types and qtype not in question_types:
                continue

            if qtype not in trajectories_by_type:
                trajectories_by_type[qtype] = []

            for scenario, items in scenarios.items():
                if not isinstance(items, list):
                    continue

                for item in items:
                    try:
                        converted_item = self._convert_data2test_item(item)
                        traj = Trajectory.from_dict(converted_item, qtype, scenario)
                        trajectories_by_type[qtype].append(traj)
                    except Exception as e:
                        logger.warning(f"Failed to parse trajectory: {e}")
                        continue

        # 合并并限制数量
        all_trajectories = []
        for qtype, trajs in trajectories_by_type.items():
            if max_per_type:
                trajs = trajs[:max_per_type]
            all_trajectories.extend(trajs)
            logger.info(f"  {qtype}: {len(trajs)} trajectories")

        return all_trajectories

    def _convert_data2test_item(self, item: Dict) -> Dict:
        """
        转换 data2test 格式的单条数据为标准格式。

        data2test 格式:
        {
            "tid": 0,
            "message_list": [{"user": "...", "agent": "..."}, ...],
            "QA": {...}
        }

        标准格式:
        {
            "tid": 0,
            "message_list": [[{"user_message": "...", "assistant_message": "...", "time": "...", "place": "..."}, ...]],
            "QA": {...}
        }
        """
        converted_messages = []

        for msg in item.get("message_list", []):
            user_text = msg.get("user", "")
            agent_text = msg.get("agent", "")

            # 从文本中提取时间和地点
            time, place = self._extract_time_place(user_text)

            # 清理文本中的时间地点信息
            user_clean = self._clean_message_text(user_text)
            agent_clean = self._clean_message_text(agent_text)

            converted_messages.append({
                "user_message": user_clean,
                "assistant_message": agent_clean,
                "time": time,
                "place": place,
            })

        return {
            "tid": item.get("tid", 0),
            "message_list": [converted_messages],  # 包装成嵌套列表
            "QA": item.get("QA", {}),
        }

    def _extract_time_place(self, text: str) -> Tuple[str, str]:
        """
        从消息文本中提取时间和地点。

        格式: "... (place: Boston, MA; time: '2024-10-01 08:00' Tuesday)"
        """
        time = ""
        place = ""

        # 匹配 (place: ...; time: ...)
        pattern = r"\(place:\s*([^;]+);\s*time:\s*([^)]+)\)"
        match = re.search(pattern, text)
        if match:
            place = match.group(1).strip()
            time = match.group(2).strip()

        return time, place

    def _clean_message_text(self, text: str) -> str:
        """
        清理消息文本，移除时间地点信息。
        """
        # 移除 (place: ...; time: ...)
        pattern = r"\s*\(place:\s*[^;]+;\s*time:\s*[^)]+\)"
        return re.sub(pattern, "", text).strip()

    def _load_raw_format(
        self,
        raw_data: Dict,
        question_types: Optional[List[str]],
        max_per_type: Optional[int],
        capacity_test: bool,
        min_tokens: int,
    ) -> List[Trajectory]:
        """
        加载原始 data/ 目录格式的数据。
        """
        trajectories = []

        for qtype, type_data in raw_data.items():
            if question_types and qtype not in question_types:
                continue

            type_trajectories = []

            if isinstance(type_data, dict):
                for scenario, items in type_data.items():
                    if not isinstance(items, list):
                        continue

                    for item in items:
                        try:
                            traj = Trajectory.from_dict(item, qtype, scenario)
                            type_trajectories.append(traj)
                        except Exception as e:
                            logger.warning(f"Failed to parse trajectory: {e}")
                            continue

            elif isinstance(type_data, list):
                for item in type_data:
                    try:
                        traj = Trajectory.from_dict(item, qtype, "default")
                        type_trajectories.append(traj)
                    except Exception as e:
                        logger.warning(f"Failed to parse trajectory: {e}")
                        continue

            # 容量测试处理
            if capacity_test:
                type_trajectories = self._filter_by_capacity(
                    type_trajectories, qtype, min_tokens
                )

            if max_per_type:
                type_trajectories = type_trajectories[:max_per_type]

            trajectories.extend(type_trajectories)

        return trajectories

    def _filter_by_capacity(
        self,
        trajectories: List[Trajectory],
        qtype: str,
        min_tokens: int,
    ) -> List[Trajectory]:
        """
        按 token 数筛选轨迹（用于容量测试）。
        """
        traj_with_tokens = []
        for traj in trajectories:
            tokens = self._estimate_tokens(traj)
            traj_with_tokens.append((traj, tokens))

        traj_with_tokens.sort(key=lambda x: -x[1])

        filtered = [(t, tok) for t, tok in traj_with_tokens if tok >= min_tokens]
        if not filtered:
            filtered = traj_with_tokens

        logger.info(
            f"  {qtype}: {len(filtered)} trajectories with >={min_tokens} tokens "
            f"(max: {filtered[0][1] if filtered else 0} tokens)"
        )

        return [t for t, _ in filtered]

    def _estimate_tokens(self, trajectory: Trajectory) -> int:
        """
        估计轨迹的 token 数。
        """
        total_chars = 0

        for session in trajectory.message_list:
            if isinstance(session, list):
                for msg in session:
                    if isinstance(msg, dict):
                        user = msg.get("user_message") or msg.get("user", "")
                        assistant = msg.get("assistant_message") or msg.get("assistant", "") or msg.get("agent", "")
                        total_chars += len(user) + len(assistant)
            elif isinstance(session, dict):
                # data2test 格式：message_list 是扁平列表
                user = session.get("user_message") or session.get("user", "")
                assistant = session.get("assistant_message") or session.get("assistant", "") or session.get("agent", "")
                total_chars += len(user) + len(assistant)

        return total_chars // 4

    def convert_to_messages(self, trajectory: Trajectory) -> List[MemBenchMessage]:
        """
        将轨迹转换为统一消息格式。

        支持两种格式：
        - 原始格式: message_list = [[{user_message, assistant_message, time, place}, ...], ...]
        - data2test 格式（转换后）: message_list = [[{user_message, assistant_message, time, place}, ...]]
        """
        messages = []
        global_step_id = 0

        for session_idx, session in enumerate(trajectory.message_list):
            if isinstance(session, list):
                for msg in session:
                    if not isinstance(msg, dict):
                        continue

                    messages.append(
                        MemBenchMessage(
                            step_id=global_step_id,
                            session_id=f"session_{session_idx}",
                            user_content=msg.get("user_message") or msg.get("user", ""),
                            assistant_content=msg.get("assistant_message") or msg.get("assistant", "") or msg.get("agent", ""),
                            time=msg.get("time", ""),
                            place=msg.get("place", ""),
                        )
                    )
                    global_step_id += 1
            elif isinstance(session, dict):
                # 处理扁平列表格式
                messages.append(
                    MemBenchMessage(
                        step_id=global_step_id,
                        session_id="session_0",
                        user_content=session.get("user_message") or session.get("user", ""),
                        assistant_content=session.get("assistant_message") or session.get("assistant", "") or session.get("agent", ""),
                        time=session.get("time", ""),
                        place=session.get("place", ""),
                    )
                )
                global_step_id += 1

        return messages

    def get_target_step_ids(self, trajectory: Trajectory) -> List[int]:
        """
        获取目标步骤的全局索引。

        MemBench 的 target_step_id 格式:
        - LowLevel: [[global_step_id, attr_index], ...]
        - HighLevel: [[step_id, ...], ...] 或扁平 [step_id, ...]
        """
        target_ids = trajectory.qa.target_step_id
        global_indices = []

        for target in target_ids:
            if isinstance(target, list):
                if target:
                    global_indices.append(int(target[0]))
            elif isinstance(target, int):
                global_indices.append(target)

        return list(set(global_indices))

    def get_trajectory_stats(self, trajectory: Trajectory) -> Dict[str, Any]:
        """
        获取轨迹统计信息。
        """
        messages = self.convert_to_messages(trajectory)

        total_chars = sum(
            len(m.user_content) + len(m.assistant_content) for m in messages
        )
        total_tokens = total_chars // 4

        return {
            "tid": trajectory.tid,
            "question_type": trajectory.question_type,
            "scenario": trajectory.scenario,
            "num_sessions": len(trajectory.message_list),
            "num_messages": len(messages),
            "total_chars": total_chars,
            "estimated_tokens": total_tokens,
            "num_target_steps": len(trajectory.qa.target_step_id),
        }

    def get_dataset_summary(self, level: str = "LowLevel") -> Dict[str, Any]:
        """
        获取数据集摘要信息。
        """
        trajectories = self.load_trajectories(level=level)

        by_type: Dict[str, Dict[str, Any]] = {}
        for traj in trajectories:
            qtype = traj.question_type
            if qtype not in by_type:
                by_type[qtype] = {"count": 0, "scenarios": set()}
            by_type[qtype]["count"] += 1
            by_type[qtype]["scenarios"].add(traj.scenario)

        for qtype in by_type:
            by_type[qtype]["scenarios"] = list(by_type[qtype]["scenarios"])

        return {
            "level": level,
            "total_trajectories": len(trajectories),
            "by_question_type": by_type,
        }


if __name__ == "__main__":
    # 测试数据加载器
    from .models import BenchmarkConfig

    # 测试 10k 数据
    config = BenchmarkConfig(dataset_variant="10k")
    loader = MemBenchDataLoader(config)

    print("=== Testing 10k HighLevel ===")
    trajs = loader.load_trajectories(
        level="HighLevel",
        question_types=["Emotion", "Preference"],
        max_per_type=2,
    )
    for traj in trajs:
        stats = loader.get_trajectory_stats(traj)
        print(stats)
