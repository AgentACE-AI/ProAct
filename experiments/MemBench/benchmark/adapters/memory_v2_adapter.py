"""
Memory System v2 适配器。

实现 MemBench 的 BaseMemory 接口，将 Memory System v2 适配到 MemBench 评估框架。

增强版: 完整利用 Memory System v2 的能力：
- 用户画像提取和更新
- 从对话中提取关键信息和偏好
- 组合知识+对话+偏好作为上下文
- 时序情感查询 (针对 Emotion 类型问题)
- 用户事实存储和检索 (人物、地点、组织等)
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# 导入 MemBench 的 BaseMemory
import sys

sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "Membench" / "benchmark" / "memory")
)
try:
    from BaseMemory import BaseMemory
except ModuleNotFoundError:
    class BaseMemory:  # type: ignore[no-redef]
        """Minimal MemBench BaseMemory fallback for the release subset."""

        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config

# 导入 Memory System v2 组件
from core.config import Config
from core.models import Message, MemoryUpdateRequest, UpdateType
from memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)


class MemorySystemV2Adapter(BaseMemory):
    """
    Memory System v2 的 MemBench 适配器（增强版）。

    实现 BaseMemory 接口:
    - reset(): 清空记忆
    - store(observation): 存储观察，提取用户偏好、情感和事实
    - recall(observation): 检索相关记忆，包含用户画像和事实
    - retri(observation): 返回检索到的索引
    - manage(): 记忆管理，更新用户画像
    - train(): 训练（不适用）

    新增:
    - 时序情感查询 (get_sentiment_at_time)
    - 用户事实存储 (FactStore)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化适配器。

        Args:
            config: 配置字典，格式:
                {
                    "user_id": "membench_user",
                    "args": {
                        "max_words": 2000,
                        "enable_profile_extraction": True,  # 是否启用画像提取
                        "profile_update_interval": 10,      # 每 N 条消息更新一次画像
                        "enable_sentiment_extraction": True, # 是否启用情感提取
                        "enable_fact_extraction": True,      # 是否启用事实提取
                    },
                    "llm_client": <LLM client instance>
                }
        """
        super().__init__(config)

        # 解析配置
        self.user_id = config.get("user_id", f"membench_{uuid.uuid4().hex[:8]}")
        self.args = config.get("args", {})
        self.max_words = self.args.get("max_words", 2000)
        self._llm_client = config.get("llm_client")
        self._llm_model = config.get("llm_model", "gpt-4o-mini")

        # 画像提取配置
        self.enable_profile_extraction = self.args.get("enable_profile_extraction", True)
        self.profile_update_interval = self.args.get("profile_update_interval", 5)

        # 情感提取配置
        self.enable_sentiment_extraction = self.args.get("enable_sentiment_extraction", True)

        # 事实提取配置
        self.enable_fact_extraction = self.args.get("enable_fact_extraction", True)

        # 初始化配置
        self._config = Config()

        # Memory System v2 实例
        self._memory_system: Optional[MemorySystem] = None

        # step_id 追踪（用于 Recall 评估）
        self._step_id_to_doc: Dict[int, str] = {}
        self._doc_to_step_id: Dict[str, int] = {}
        self._current_step = 0

        # 消息缓存（用于批量画像/事实提取）
        self._message_buffer: List[Dict[str, str]] = []
        self._message_count = 0

        # step_id 到时间戳的映射 (用于情感查询)
        self._step_id_to_timestamp: Dict[int, str] = {}

        # 轨迹上下文（用于确定性 user_id 生成）
        self._pending_trajectory_context: Optional[Dict[str, Any]] = None
        self.run_id: str = ""

        # 检索缓存（recall/retri 共享，避免重复向量查询）
        self._last_search_query: Optional[str] = None
        self._last_search_results: Optional[List[Dict[str, Any]]] = None

        # 初始化
        self._init_memory_system()

    def set_trajectory_context(
        self,
        question_type: str,
        scenario: str,
        tid: int,
        run_id: str = "",
    ) -> None:
        """
        设置当前轨迹上下文，reset() 将据此生成确定性 user_id。

        Args:
            question_type: 问题类型
            scenario: 场景名称
            tid: 轨迹 ID
            run_id: 本次运行标识
        """
        self._pending_trajectory_context = {
            "question_type": question_type,
            "scenario": scenario,
            "tid": tid,
            "run_id": run_id,
        }

    def _init_memory_system(self) -> None:
        """初始化或重新初始化 Memory System"""
        self._memory_system = MemorySystem(
            user_id=self.user_id,
            config=self._config,
            llm_client=self._llm_client,
        )
        self._memory_system.set_current_topic("MemBench")

        if self._llm_client:
            self._memory_system.set_llm_client(self._llm_client)

    def reset(self) -> None:
        """
        清空所有记忆内容。

        实现: 使用新的用户 ID 创建全新的 MemorySystem 实例。
        这样可以避免 ChromaDB 的数据库锁问题。
        """
        # 清空追踪状态
        self._step_id_to_doc.clear()
        self._doc_to_step_id.clear()
        self._step_id_to_timestamp.clear()
        self._current_step = 0
        self._message_buffer.clear()
        self._message_count = 0
        self._last_search_query = None
        self._last_search_results = None

        # 根据轨迹上下文生成确定性 user_id，便于追溯实验数据
        ctx = self._pending_trajectory_context
        if ctx:
            self.run_id = str(ctx.get("run_id", ""))
            raw_id = (
                f"mb_{ctx.get('run_id', '')}"
                f"_{ctx.get('question_type', '')}"
                f"_{ctx.get('scenario', '')}"
                f"_{ctx.get('tid', '')}"
            )
            # 合规化: 小写、非法字符替换为下划线、去连续下划线、截断到 63 字符
            sanitized = re.sub(r"[^a-z0-9_]", "_", raw_id.lower())
            sanitized = re.sub(r"_+", "_", sanitized).strip("_")
            self.user_id = sanitized[:63] if sanitized else f"membench_{uuid.uuid4().hex[:8]}"
        else:
            # 无上下文时降级为随机 user_id
            self.run_id = ""
            self.user_id = f"membench_{uuid.uuid4().hex[:8]}"

        # 重新初始化
        self._init_memory_system()

    def store(self, observation: str, time: Optional[int] = None) -> None:
        """
        存储观察（对话消息）。

        增强版:
        1. 存储到向量库
        2. 提取并存储情感信息
        3. 缓存消息用于画像提取
        4. 定期提取用户偏好并更新画像

        MemBench 观察格式:
        - "{step_id}[|]'user': {user_msg}; 'agent': {agent_msg}"
        - "{step_id}[|]{content}"

        Args:
            observation: 观察字符串
            time: 可选的时间步
        """
        # 解析观察
        step_id, content = self._parse_observation(observation)
        # print(f"step_id: {step_id}, content: {content}")
        # input("press enter to continue...")

        # 解析用户/助手内容
        user_content, agent_content = self._parse_user_agent(content)
        # print(f"user_content: {user_content}, agent_content: {agent_content}")
        # input("press enter to continue...")

        # 解析时间戳 (如果有)
        timestamp = self._extract_timestamp_from_content(content)
        # print(f"timestamp: {timestamp}")
        # input("press enter to continue...")

        # 生成文档 ID
        doc_id = str(uuid.uuid4())
        # print(f"doc_id: {doc_id}")
        # input("press enter to continue...")

        # 提取情感 (如果启用)
        # sentiment = None
        # sentiment_score = None
        # if self.enable_sentiment_extraction and user_content and self._llm_client:
        #     sentiment, sentiment_score = self._extract_sentiment(user_content)

        # 创建消息对象 (带情感)
        user_message = Message(
            role="user",
            content=user_content,
            timestamp=timestamp or f"step_{step_id}",
            # sentiment=sentiment,
            # sentiment_score=sentiment_score,
        )
        assistant_message = Message(
            role="assistant",
            content=agent_content or "Acknowledged.",
            timestamp=timestamp or f"step_{step_id}",
        )

        # 存储到 Memory System v2
        topic = self._memory_system.current_topic or "MemBench"
        # print(f"Storing to topic: {topic}")
        # input("press enter to continue...")

        # 使用 append_interaction_to_topic 存储
        self._memory_system.append_interaction_to_topic(
            topic=topic,
            user_message=user_message,
            assistant_message=assistant_message,
            running_summary=f"Conversation at step {step_id}",
            key_info=[f"step_id:{step_id}"],
        )

        # 同时添加到向量库（带 step_id metadata）
        self._memory_system.vector_store.add_conversation(
            doc_id=doc_id,
            content=f"{user_content} {agent_content}".strip(),
            metadata={
                "topic": topic,
                "step_id": step_id,
                "user_content": user_content,
                "agent_content": agent_content,
                "timestamp": timestamp or "",
                # "sentiment": sentiment or "",
                # "sentiment_score": sentiment_score if sentiment_score is not None else "",
            },
        )

        # 记录映射
        self._step_id_to_doc[step_id] = doc_id
        self._doc_to_step_id[doc_id] = step_id
        if timestamp:
            self._step_id_to_timestamp[step_id] = timestamp
        self._current_step = max(self._current_step, step_id + 1)

        # 缓存消息用于画像提取
        if self.enable_profile_extraction and user_content:
            self._message_buffer.append({
                "user": user_content,
                "agent": agent_content or "",
            })
            self._message_count += 1

            # 定期提取画像
            if self._message_count % self.profile_update_interval == 0:
                self._extract_and_update_profile()

    def _extract_timestamp_from_content(self, content: str) -> Optional[str]:
        """
        从内容中提取时间戳。

        支持的格式:
        1. [time: '2024-10-01 08:00' Tuesday] - 新的 observation 格式
        2. "time": "'2024-10-01 08:00' Tuesday" - 原始 MemBench 格式

        Args:
            content: 消息内容

        Returns:
            ISO 格式时间戳或 None
        """
        from datetime import datetime as dt

        # 格式1: [time: '2024-10-01 08:00' Tuesday] - 新格式
        time_tag_match = re.search(r"\[time:\s*'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)'\s*\w*\]", content)
        if time_tag_match:
            time_str = time_tag_match.group(1)
            try:
                parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M")
                return parsed.isoformat()
            except ValueError:
                pass

        # 格式2: 带引号的时间 '2024-10-01 08:00' - 兼容原格式
        quoted_match = re.search(r"'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)'", content)
        if quoted_match:
            time_str = quoted_match.group(1)
            try:
                parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M")
                return parsed.isoformat()
            except ValueError:
                pass

        return None

    def _extract_sentiment(self, user_content: str) -> tuple:
        """
        使用 LLM 从用户消息中提取情感。

        Args:
            user_content: 用户消息内容

        Returns:
            (sentiment, sentiment_score) 元组
        """
        try:
            prompt = f"""Analyze the sentiment of this message and respond with a JSON object.

Message: "{user_content[:500]}"

Respond with ONLY a JSON object (no other text):
{{"sentiment": "<one of: surprise/anger/sadness/joy/fear/neutral/disgust>", "score": <float from -1.0 to 1.0>}}"""

            response = self._llm_client.chat(
                messages=[
                    {"role": "system", "content": "You analyze sentiment. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self._llm_model,
                temperature=0.0,
            )

            import json
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                sentiment = result.get("sentiment", "neutral")
                score = float(result.get("score", 0.0))
                return sentiment, score

        except Exception as e:
            logger.debug(f"Failed to extract sentiment: {e}")

        return None, None

    def _extract_core_query(self, query: str) -> str:
        """从含噪声的查询中提取核心问题。

        MemBench Noisy 类型会在问题前加入干扰文本，如
        "Hold on, what I truly wanted to clarify is ..."，
        本方法提取 marker 之后的核心查询部分。
        """
        if not query:
            return query

        noise_markers = re.compile(
            r"(?:"
            r"what I (?:truly|actually|really) wanted (?:to (?:clarify|ask|understand|know)|was)"
            r"|hold on,?\s*(?:actually )?what I wanted to ask (?:is|was)"
            r"|wait a minute,?\s*(?:actually )?what I wanted to ask (?:is|was)"
            r"|oops,?\s*(?:actually )?what I (?:wanted to ask|meant) (?:is|was)"
            r")\s*[:,\-]*\s*",
            re.IGNORECASE,
        )
        match = noise_markers.search(query)
        if match:
            core = query[match.end():].strip()
            if core:
                return core

        return query.strip()

    def recall(self, observation: str) -> str:
        """
        检索相关记忆上下文。

        增强版:
        - 使用 format_context_for_query() 获取完整上下文
        - 自动检测时间查询并补充情感信息
        - 包含用户事实、画像、对话历史

        Args:
            observation: 查询字符串（通常是问题）

        Returns:
            检索到的记忆上下文，截断到 max_words
        """
        # 去除 Noisy 类型查询中的噪声前缀
        query = self._extract_core_query(observation)

        context_parts = []
        is_sentiment_query = bool(re.search(r"sentiment", query, re.IGNORECASE))

        # 使用 Memory System v2 的 format_context_for_query()
        # 情感查询时它只返回时序上下文（已在内部做了精简）
        system_context = self._memory_system.format_context_for_query(query)
        if system_context and system_context != "暂无相关记忆":
            context_parts.append(f"[Memory System Context]\n{system_context}")

        # 补充：搜索相关对话（带 step_id，用于 MemBench 评估）
        conv_results = self._memory_system.vector_store.search_conversations(
            query=query, n_results=10
        )
        # 缓存检索结果，供 retri() 复用，避免重复向量查询
        self._last_search_query = query
        self._last_search_results = conv_results

        # 情感查询：仅使用时序上下文，跳过 Related Conversations（避免混入无关情感消息）
        if not is_sentiment_query:
            conversation_parts = []
            for r in conv_results:
                metadata = r.get("metadata", {})
                user_content = metadata.get("user_content", "")
                agent_content = metadata.get("agent_content", "")
                step_id = metadata.get("step_id", "")
                sentiment = metadata.get("sentiment", "")

                if user_content or agent_content:
                    entry = f"{step_id}[|]'user': {user_content}; 'agent': {agent_content}"
                    if sentiment:
                        entry += f" [sentiment: {sentiment}]"
                    conversation_parts.append(entry)
                elif r.get("content"):
                    conversation_parts.append(f"{step_id}[|]{r['content']}")

            if conversation_parts:
                context_parts.append("[Related Conversations with step_id]\n" + "\n".join(conversation_parts))

        # 合并上下文
        context = "\n\n".join(context_parts)

        # 截断到 max_words
        words = context.split()
        if len(words) > self.max_words:
            context = " ".join(words[: self.max_words])

        return context

    def retri(self, observation: str) -> List[int]:
        """
        返回检索到的 step_id 列表。

        用于 Recall@K 评估。

        Args:
            observation: 查询字符串

        Returns:
            检索到的 step_id 列表
        """
        # 去除噪声前缀，与 recall() 保持一致
        query = self._extract_core_query(observation)

        retrieved_ids = []

        # 优先复用 recall() 阶段缓存的检索结果
        if query == self._last_search_query and self._last_search_results is not None:
            conv_results = self._last_search_results
        else:
            conv_results = self._memory_system.vector_store.search_conversations(
                query=query, n_results=10
            )

        for r in conv_results:
            metadata = r.get("metadata", {})
            step_id = metadata.get("step_id")

            if step_id is not None:
                try:
                    retrieved_ids.append(int(step_id))
                except (ValueError, TypeError):
                    pass

        # 去重
        return list(set(retrieved_ids))

    def manage(self) -> None:
        """
        记忆管理。

        增强版: 处理缓存的消息，提取并更新用户画像。
        """
        if self.enable_profile_extraction and self._message_buffer:
            self._extract_and_update_profile()

    def train(self, **kwargs) -> None:
        """训练（不适用于 Memory System v2）"""
        pass

    # ==================== 画像和事实提取方法 ====================

    def _extract_and_update_profile(self) -> None:
        """
        从缓存的消息中提取用户偏好、画像和事实信息。
        """
        if not self._message_buffer or not self._llm_client:
            return

        try:
            # 构建对话文本
            conversation_text = "\n".join([
                f"User: {m['user']}\nAssistant: {m['agent']}"
                for m in self._message_buffer[-self.profile_update_interval:]
            ])

            # 使用 LLM 提取用户信息和事实
            extraction_prompt = f"""Analyze the following conversation and extract user information and facts.

Conversation:
{conversation_text}

IMPORTANT RULES:
1. Entity names: Always use the person's FULL NAME when mentioned (e.g., "Amelia Brooks", NOT "sister", "she", or "subordinate"). Put the relationship in the "relationship" field instead.
2. Fact UPDATES: If a fact was corrected or changed (e.g., "hobby changed from hiking to camping"), record ONLY the NEW/CURRENT value.
3. Exact values: Record phone numbers, email addresses, ages, heights, and birthdays with their EXACT values — never paraphrase or summarize numbers.

Extract the following (if present, otherwise leave empty):
1. User interests (topics they're curious about)
2. User preferences (likes, dislikes, communication style)
3. User goals (what they're trying to achieve)
4. Key traits about the user
5. Facts mentioned by the user (people, places, organizations, events they mentioned)

Respond in JSON format:
{{
    "interests": ["interest1", "interest2"],
    "preferences": ["preference1", "preference2"],
    "goals": ["goal1", "goal2"],
    "traits": ["trait1", "trait2"],
    "extracted_facts": [
        {{
            "entity": "Full Name (e.g., Amelia Brooks)",
            "entity_type": "person/place/organization/event/other",
            "attributes": {{"attribute_name": "attribute_value"}},
            "relationship": "relationship to user (e.g., sister, brother, boss, coworker)"
        }}
    ]
}}

Only include items that are clearly stated or strongly implied. Be concise."""

            # Fix B: 注入已有实体列表，让 LLM 复用已有名字，减少碎片化
            existing_entities = []
            try:
                all_facts = self._memory_system.get_all_facts()
                if all_facts:
                    for f in all_facts:
                        if f.entity_type == "person":
                            existing_entities.append(f"{f.entity} ({f.relationship})")
            except Exception:
                pass

            if existing_entities:
                extraction_prompt += f"""

KNOWN ENTITIES (reuse these names if the same person is mentioned):
{chr(10).join(f'- {e}' for e in existing_entities[:20])}
"""

            response = self._llm_client.chat(
                messages=[
                    {"role": "system", "content": "You extract user information and facts from conversations. Respond only in valid JSON."},
                    {"role": "user", "content": extraction_prompt},
                ],
                model=self._llm_model,
                temperature=0.0,
            )

            # 解析响应
            import json
            # 尝试提取 JSON (支持嵌套)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())

                # 更新用户画像
                profile_updates = {}
                if extracted.get("interests"):
                    profile_updates["interests"] = extracted["interests"]
                if extracted.get("preferences"):
                    profile_updates["communication_style"] = extracted["preferences"]
                if extracted.get("goals"):
                    profile_updates["goals"] = extracted["goals"]
                if extracted.get("traits"):
                    profile_updates["traits"] = extracted["traits"]

                if profile_updates:
                    self._memory_system.profile_store.update(profile_updates)
                    logger.debug(f"Updated user profile: {profile_updates}")

                # 存储偏好到向量库
                for pref in extracted.get("preferences", []):
                    if pref:
                        self._memory_system.vector_store.add_preference(
                            doc_id=str(uuid.uuid4()),
                            content=pref,
                            metadata={"source": "conversation_extraction"},
                        )

                # 存储事实到 FactStore
                if self.enable_fact_extraction and extracted.get("extracted_facts"):
                    try:
                        topic = self._memory_system.current_topic or "MemBench"
                        added = self._memory_system.add_facts(
                            facts_data=extracted["extracted_facts"],
                            source_topic=topic,
                        )
                        if added > 0:
                            logger.debug(f"Added {added} new facts")
                    except AttributeError:
                        # FactStore 可能不存在（旧版本）
                        pass

        except Exception as e:
            logger.warning(f"Failed to extract profile/facts: {e}")

        # 清空缓存（保留最近几条用于上下文）
        self._message_buffer = self._message_buffer[-3:]

    def _format_profile_for_context(self, profile) -> str:
        """
        将用户画像格式化为上下文文本。

        Args:
            profile: UserProfile 对象

        Returns:
            格式化的画像文本
        """
        parts = []

        if profile.interests:
            parts.append(f"Interests: {', '.join(profile.interests[:5])}")
        if profile.goals:
            parts.append(f"Goals: {', '.join(profile.goals[:3])}")
        if profile.traits:
            parts.append(f"Traits: {', '.join(profile.traits[:3])}")
        if profile.communication_style:
            parts.append(f"Preferences: {', '.join(profile.communication_style[:3])}")

        return "; ".join(parts) if parts else ""

    # ==================== 辅助方法 ====================

    def _parse_observation(self, observation: str) -> tuple:
        """
        解析观察字符串，提取 step_id 和内容。

        Args:
            observation: 观察字符串

        Returns:
            (step_id, content) 元组
        """
        if "[|]" in observation:
            parts = observation.split("[|]", 1)
            try:
                step_id = int(parts[0])
                content = parts[1] if len(parts) > 1 else ""
            except ValueError:
                step_id = self._current_step
                content = observation
        else:
            step_id = self._current_step
            content = observation

        return step_id, content

    def _parse_user_agent(self, content: str) -> tuple:
        """
        解析用户和助手内容。

        Args:
            content: 内容字符串，格式 "'user': xxx; 'agent': yyy"

        Returns:
            (user_content, agent_content) 元组
        """
        user_content = ""
        agent_content = ""

        # 尝试以 "; 'agent':" 为完整分隔符解析（避免用户消息中的分号被截断）
        full_match = re.search(r"'user':\s*(.+?);\s*'agent':\s*(.*)", content, re.DOTALL)
        if full_match:
            user_content = full_match.group(1).strip()
            agent_content = full_match.group(2).strip()
        else:
            # fallback: 仅有 user 部分
            user_match = re.search(r"'user':\s*(.+)", content, re.DOTALL)
            if user_match:
                user_content = user_match.group(1).strip()

        # 如果没有匹配，整个内容作为用户内容
        if not user_content and not agent_content:
            user_content = content

        return user_content, agent_content

    def set_llm_client(self, llm_client: Any) -> None:
        """
        设置 LLM 客户端。

        Args:
            llm_client: LLM 客户端实例
        """
        self._llm_client = llm_client
        if self._memory_system:
            self._memory_system.set_llm_client(llm_client)

    def get_stats(self) -> Dict[str, Any]:
        """获取适配器统计信息"""
        profile = self._memory_system.get_user_profile() if self._memory_system else None

        # 获取事实数量
        facts_count = 0
        try:
            facts = self._memory_system.get_all_facts()
            facts_count = len(facts) if facts else 0
        except AttributeError:
            pass

        return {
            "user_id": self.user_id,
            "total_stored": len(self._step_id_to_doc),
            "current_step": self._current_step,
            "message_buffer_size": len(self._message_buffer),
            "profile_extraction_enabled": self.enable_profile_extraction,
            "sentiment_extraction_enabled": self.enable_sentiment_extraction,
            "fact_extraction_enabled": self.enable_fact_extraction,
            "profile_interests_count": len(profile.interests) if profile else 0,
            "profile_goals_count": len(profile.goals) if profile else 0,
            "facts_count": facts_count,
            "memory_stats": self._memory_system.get_stats()
            if self._memory_system
            else {},
        }
