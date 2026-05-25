"""
WebSocket 连接管理。

处理实时通信、消息推送和连接管理。
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect


@dataclass
class WebSocketConnection:
    """
    WebSocket 连接。

    表示单个 WebSocket 连接的状态。
    """
    websocket: WebSocket
    user_id: str
    connected_at: datetime
    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # 推送状态管理
    push_state: str = "idle"  # "idle" | "waiting_response"
    pending_push_id: Optional[str] = None
    pending_push_sent_at: Optional[datetime] = None

    def set_push_waiting(self, push_id: str) -> None:
        """设置为等待推送回复状态"""
        self.push_state = "waiting_response"
        self.pending_push_id = push_id
        self.pending_push_sent_at = datetime.now()

    def clear_push_waiting(self) -> None:
        """清除推送等待状态"""
        self.push_state = "idle"
        self.pending_push_id = None
        self.pending_push_sent_at = None

    def is_waiting_push_response(self) -> bool:
        """检查是否正在等待推送回复"""
        return self.push_state == "waiting_response"


class WebSocketManager:
    """
    WebSocket 连接管理器。

    管理所有 WebSocket 连接，支持：
    - 连接管理（连接/断开）
    - 消息广播
    - 用户定向消息
    - 推送通知
    """

    def __init__(self):
        """初始化 WebSocket 管理器"""
        # user_id -> list of connections (支持多客户端)
        self.connections: Dict[str, List[WebSocketConnection]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """
        接受并注册新的 WebSocket 连接。

        Args:
            websocket: WebSocket 实例
            user_id: 用户 ID

        Returns:
            连接 ID
        """
        await websocket.accept()

        connection = WebSocketConnection(
            websocket=websocket,
            user_id=user_id,
            connected_at=datetime.now(),
        )

        async with self._lock:
            if user_id not in self.connections:
                self.connections[user_id] = []
            self.connections[user_id].append(connection)

        print(f"[WebSocket] 用户 {user_id} 已连接 (id: {connection.connection_id[:8]})")
        return connection.connection_id

    async def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """
        移除 WebSocket 连接。

        在移除连接后，会尝试保存用户的当前对话（如果有）。
        这是安全保存逻辑，用于处理非正常断开的情况。

        Args:
            websocket: WebSocket 实例
            user_id: 用户 ID
        """
        async with self._lock:
            if user_id in self.connections:
                # 找到并移除特定连接
                self.connections[user_id] = [
                    conn for conn in self.connections[user_id]
                    if conn.websocket != websocket
                ]

                # 清理空列表
                if not self.connections[user_id]:
                    del self.connections[user_id]

        print(f"[WebSocket] 用户 {user_id} 已断开连接")

        # 安全保存：保存用户的当前对话（如果有）
        await self._safe_save_conversation(user_id)

    async def send_to_user(
        self,
        user_id: str,
        message: Dict[str, Any],
    ) -> bool:
        """
        发送消息给特定用户的所有连接。

        Args:
            user_id: 用户 ID
            message: 消息内容

        Returns:
            是否至少发送到一个连接
        """
        if user_id not in self.connections:
            return False

        message_json = json.dumps(message, ensure_ascii=False)
        disconnected = []
        sent_count = 0

        for conn in self.connections.get(user_id, []):
            try:
                await conn.websocket.send_text(message_json)
                sent_count += 1
            except Exception as e:
                print(f"[WebSocket] 发送给 {user_id} 失败: {e}")
                disconnected.append(conn)

        # 清理断开的连接
        if disconnected and user_id in self.connections:
            self.connections[user_id] = [
                conn for conn in self.connections[user_id]
                if conn not in disconnected
            ]

        return sent_count > 0

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        广播消息给所有连接的用户。

        Args:
            message: 消息内容
        """
        message_json = json.dumps(message, ensure_ascii=False)
        disconnected = []

        for user_id, connections in list(self.connections.items()):
            for conn in connections:
                try:
                    await conn.websocket.send_text(message_json)
                except Exception:
                    disconnected.append((user_id, conn))

        # 清理断开的连接
        for user_id, conn in disconnected:
            if user_id in self.connections:
                self.connections[user_id] = [
                    c for c in self.connections[user_id] if c != conn
                ]

    def is_user_connected(self, user_id: str) -> bool:
        """
        检查用户是否有活跃连接。

        Args:
            user_id: 用户 ID

        Returns:
            是否有活跃连接
        """
        return user_id in self.connections and len(self.connections[user_id]) > 0

    def get_connected_users(self) -> List[str]:
        """
        获取所有已连接用户的 ID 列表。

        Returns:
            用户 ID 列表
        """
        return list(self.connections.keys())

    def get_connection(self, user_id: str) -> Optional[WebSocketConnection]:
        """
        获取用户的第一个连接。

        Args:
            user_id: 用户 ID

        Returns:
            WebSocket 连接或 None
        """
        connections = self.connections.get(user_id, [])
        return connections[0] if connections else None

    async def send_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        content: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        发送通知给用户。

        Args:
            user_id: 用户 ID
            notification_type: 通知类型 (alert, info, report, initiative)
            title: 通知标题
            content: 通知内容
            data: 附加数据

        Returns:
            是否发送成功
        """
        message = {
            "type": "notification",
            "notification_type": notification_type,
            "title": title,
            "content": content,
            "data": data or {},
            "timestamp": datetime.now().isoformat(),
        }
        return await self.send_to_user(user_id, message)

    async def send_initiative_message(
        self,
        user_id: str,
        message_type: str,
        content: str,
        options: Optional[List[str]] = None,
        push_id: Optional[str] = None,
        requires_response: bool = False,
    ) -> bool:
        """
        发送主动消息（Agent 主动发起对话）。

        Args:
            user_id: 用户 ID
            message_type: 消息类型 (recommendation, reminder, question, report)
            content: 消息内容
            options: 用户可选的回复选项
            push_id: 推送 ID（用于跟踪回复）
            requires_response: 是否需要用户回复

        Returns:
            是否发送成功
        """
        push_id = push_id or str(uuid.uuid4())

        message = {
            "type": "initiative",
            "message_type": message_type,
            "content": content,
            "options": options or [],
            "push_id": push_id,
            "requires_response": requires_response,
            "timestamp": datetime.now().isoformat(),
        }

        sent = await self.send_to_user(user_id, message)

        # 如果需要回复，更新连接状态
        if sent and requires_response:
            conn = self.get_connection(user_id)
            if conn:
                conn.set_push_waiting(push_id)

        return sent

    async def send_report_ready(
        self,
        user_id: str,
        report_id: str,
        topic: str,
        title: str,
        summary: str,
    ) -> bool:
        """
        发送报告就绪通知。

        Args:
            user_id: 用户 ID
            report_id: 报告 ID
            topic: 报告主题
            title: 报告标题
            summary: 报告摘要

        Returns:
            是否发送成功
        """
        return await self.send_notification(
            user_id=user_id,
            notification_type="report",
            title=f"报告已生成: {title}",
            content=summary[:200],
            data={
                "report_id": report_id,
                "topic": topic,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        获取 WebSocket 统计信息。

        Returns:
            统计信息
        """
        total_connections = sum(
            len(conns) for conns in self.connections.values()
        )
        return {
            "connected_users": len(self.connections),
            "total_connections": total_connections,
            "user_ids": list(self.connections.keys()),
        }

    async def _safe_save_conversation(self, user_id: str) -> None:
        """
        安全保存用户的当前对话。

        当用户断开连接时调用此方法，保存未保存的对话。
        如果对话已经被保存（如通过 /quit 命令），则不会重复保存。

        Args:
            user_id: 用户 ID
        """
        try:
            from server.dependencies import get_user_dependencies, _user_dependencies
            from core.models import MemoryUpdateRequest, UpdateType

            # 检查用户是否已加载到内存中
            if user_id not in _user_dependencies:
                # 用户未加载，无需保存
                return

            deps = get_user_dependencies(user_id)
            memory = deps.memory

            # 检查是否有需要保存的对话
            # 如果 current_messages 不为空，说明用户非正常断开（没有执行 /quit 或 /reset）
            if memory.current_messages and memory.current_topic:
                memory.submit_update(MemoryUpdateRequest(
                    update_type=UpdateType.ADD_CONVERSATION,
                    source="disconnect",
                    user_id=user_id,
                    data={
                        "topic": memory.current_topic,
                        "messages": [m.to_dict() for m in memory.current_messages],
                        "summary": f"关于 {memory.current_topic} 的对话（连接断开时保存）",
                        "key_info": [],
                        "user_preferences": [],
                    },
                ))
                memory.clear_current_session()
                print(f"[WebSocket] 用户 {user_id}: 断开时保存了对话")

        except Exception as e:
            print(f"[WebSocket] 用户 {user_id}: 断开保存失败: {e}")


async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    manager: WebSocketManager,
    on_push_response: Optional[Callable] = None,
) -> None:
    """
    WebSocket 端点处理函数。

    处理消息类型：
    - ping: 心跳
    - ack: 确认收到通知
    - push_response: 推送消息的回复

    Args:
        websocket: WebSocket 实例
        user_id: 用户 ID
        manager: WebSocketManager 实例
        on_push_response: 推送回复回调函数
    """
    await manager.connect(websocket, user_id)

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                # 处理不同类型的消息
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                elif message.get("type") == "ack":
                    # 客户端确认收到通知
                    print(f"[WebSocket] 用户 {user_id} 确认: {message.get('id')}")

                elif message.get("type") == "push_response":
                    # 客户端响应推送消息
                    push_id = message.get("push_id")
                    option = message.get("option")
                    print(f"[WebSocket] 用户 {user_id} 响应推送 {push_id}: 选项 {option}")

                    # 清除推送等待状态
                    conn = manager.get_connection(user_id)
                    if conn and conn.pending_push_id == push_id:
                        conn.clear_push_waiting()
                        print(f"[WebSocket] 已清除用户 {user_id} 的推送等待状态")

                    # 调用回调（如果提供）
                    if on_push_response:
                        await on_push_response(user_id, push_id, option)

                    # 发送确认
                    await websocket.send_text(json.dumps({
                        "type": "push_response_ack",
                        "push_id": push_id,
                        "status": "received",
                    }))

            except json.JSONDecodeError:
                print(f"[WebSocket] 无效 JSON 来自 {user_id}: {data}")

    except WebSocketDisconnect:
        await manager.disconnect(websocket, user_id)
    except Exception as e:
        print(f"[WebSocket] 用户 {user_id} 错误: {e}")
        await manager.disconnect(websocket, user_id)
