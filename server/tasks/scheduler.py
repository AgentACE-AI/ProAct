"""
任务调度器。

管理后台任务的调度和执行：
- 主动搜索任务 (全局 - 对所有用户执行)
- 记忆维护任务 (全局 - 合并 memory_validation + stale_check)
- 主动对话任务 (在线 - 仅对在线用户执行)

注：idle_check 功能已移至 WebSocket 断开事件处理
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from core.config import Config

if TYPE_CHECKING:
    from server.websocket.ws_handler import WebSocketManager


@dataclass
class TaskStats:
    """任务统计信息"""
    runs: int = 0
    last_run: Optional[datetime] = None
    errors: int = 0
    last_error: Optional[str] = None
    last_duration: float = 0.0
    users_processed: int = 0


class TaskScheduler:
    """
    后台任务调度器。

    任务分为两类：
    1. 全局任务（对所有用户执行）：proactive_search, memory_maintenance
    2. 在线任务（仅对在线用户执行）：initiative_check
    """

    def __init__(
        self,
        config: Config,
        ws_manager: "WebSocketManager",
    ):
        """
        初始化任务调度器。

        Args:
            config: 全局配置
            ws_manager: WebSocket 管理器
        """
        self.config = config
        self.ws_manager = ws_manager
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()

        # 任务统计
        self._task_stats: Dict[str, TaskStats] = {
            "proactive_search": TaskStats(),
            "memory_maintenance": TaskStats(),
            "initiative_check": TaskStats(),
        }

    async def start(self) -> None:
        """启动所有调度任务。"""
        if self.is_running:
            return

        self.is_running = True
        self._stop_event.clear()

        # 获取任务间隔
        intervals = self.config.tasks.get_intervals(self.config.debug_mode)
        mode = "开发模式" if self.config.debug_mode else "生产模式"

        print(f"[TaskScheduler] 启动 - {mode}")
        print(f"[TaskScheduler] 任务间隔配置:")
        print(f"  • 主动搜索 (全局): {intervals['proactive_search']}秒")
        print(f"  • 记忆维护 (全局): {intervals['memory_maintenance']}秒")
        print(f"  • 主动对话 (在线): {intervals['initiative_check']}秒")

        # 创建任务
        self._tasks = [
            asyncio.create_task(self._proactive_search_loop(intervals["proactive_search"])),
            asyncio.create_task(self._memory_maintenance_loop(intervals["memory_maintenance"])),
            asyncio.create_task(self._initiative_check_loop(intervals["initiative_check"])),
        ]

        print("[TaskScheduler] 所有任务已启动")

    async def stop(self) -> None:
        """停止所有调度任务。"""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        print("[TaskScheduler] 正在停止所有任务...")

        # 取消所有任务
        for task in self._tasks:
            task.cancel()

        # 等待任务完成
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        print("[TaskScheduler] 所有任务已停止")

    # ==================== 辅助方法：获取用户列表 ====================

    def _get_all_existing_users(self) -> List[str]:
        """
        获取所有存在的用户（包括未加载到内存的）。

        用于全局任务（proactive_search, memory_maintenance）。
        """
        return self.config.storage.get_all_existing_user_ids()

    def _get_online_users(self) -> List[str]:
        """
        获取当前在线（已连接 WebSocket）的用户。

        用于需要实时推送的任务（initiative_check）。
        """
        return self.ws_manager.get_connected_users()

    # ==================== 任务循环 ====================

    async def _proactive_search_loop(self, interval: int) -> None:
        """
        主动搜索任务循环（全局任务）。

        Args:
            interval: 执行间隔（秒）
        """
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(interval)

                if self._stop_event.is_set():
                    break

                await self._run_task("proactive_search", self._execute_proactive_search)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._record_error("proactive_search", str(e))
                print(f"[TaskScheduler] proactive_search 循环异常: {e}")

    async def _memory_maintenance_loop(self, interval: int) -> None:
        """
        记忆维护任务循环（全局任务）。

        合并了原来的 memory_validation 和 stale_check。

        Args:
            interval: 执行间隔（秒）
        """
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(interval)

                if self._stop_event.is_set():
                    break

                await self._run_task("memory_maintenance", self._execute_memory_maintenance)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._record_error("memory_maintenance", str(e))
                print(f"[TaskScheduler] memory_maintenance 循环异常: {e}")

    async def _initiative_check_loop(self, interval: int) -> None:
        """
        主动对话检查任务循环（仅在线用户）。

        Args:
            interval: 执行间隔（秒）
        """
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(interval)

                if self._stop_event.is_set():
                    break

                await self._run_task("initiative_check", self._execute_initiative_check)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._record_error("initiative_check", str(e))
                print(f"[TaskScheduler] initiative_check 循环异常: {e}")

    # ==================== 任务执行 ====================

    async def _run_task(
        self,
        task_name: str,
        task_func: Callable,
    ) -> None:
        """
        运行单个任务并记录统计。

        Args:
            task_name: 任务名称
            task_func: 任务函数
        """
        start_time = time.time()
        stats = self._task_stats[task_name]
        stats.runs += 1
        stats.last_run = datetime.now()

        try:
            users_count = await task_func()
            stats.users_processed = users_count
        except Exception as e:
            self._record_error(task_name, str(e))
            raise
        finally:
            stats.last_duration = time.time() - start_time

    async def _execute_proactive_search(self) -> int:
        """
        执行主动搜索任务。

        对所有用户执行主动搜索（包括未加载到内存的）。
        根据用户在线/离线状态使用不同的搜索策略。

        Returns:
            处理的用户数量
        """
        from server.tasks.proactive_tasks import run_proactive_search
        from server.dependencies import get_user_dependencies

        # 获取所有存在的用户
        user_ids = self._get_all_existing_users()
        if not user_ids:
            print("[proactive_search] 无用户可执行搜索")
            return 0

        print(f"[proactive_search] 开始 | 用户数: {len(user_ids)}")

        for user_id in user_ids:
            try:
                deps = get_user_dependencies(user_id)

                # 检查用户是否在线
                is_online = self.ws_manager.is_user_connected(user_id)

                # 根据在线状态使用不同策略
                deliveries = await run_proactive_search(deps, is_online=is_online)
                print(f"[proactive_search] 用户 {user_id}: proactive delivery 数 {len(deliveries)} (online={is_online})")
                await self._deliver_proactive_deliveries(user_id, deliveries)

            except Exception as e:
                print(f"[proactive_search] 用户 {user_id} 错误: {e}")

        return len(user_ids)

    async def _execute_memory_maintenance(self) -> int:
        """
        执行记忆维护任务。

        合并了 memory_validation 和 stale_check：
        1. 记忆验证：检测知识缺口、逻辑问题
        2. 过期检查：检测过期信息

        对所有用户执行。

        Returns:
            处理的用户数量
        """
        from server.tasks.proactive_tasks import run_memory_maintenance
        from server.dependencies import get_user_dependencies

        user_ids = self._get_all_existing_users()
        if not user_ids:
            return 0

        print(f"[memory_maintenance] 开始 | 用户数: {len(user_ids)}")

        for user_id in user_ids:
            try:
                deps = get_user_dependencies(user_id)
                result = await run_memory_maintenance(deps)

                if result.get("issues") or result.get("stale_items"):
                    issues_count = len(result.get("issues", []))
                    stale_count = len(result.get("stale_items", []))
                    print(f"[memory_maintenance] 用户 {user_id}: {issues_count} 个问题, {stale_count} 个过期项")

            except Exception as e:
                print(f"[memory_maintenance] 用户 {user_id} 错误: {e}")

        return len(user_ids)

    async def _execute_initiative_check(self) -> int:
        """
        执行主动对话检查任务。

        仅对在线用户执行（需要 WebSocket 连接来推送消息）。

        Returns:
            处理的用户数量
        """
        from server.tasks.proactive_tasks import run_initiative_check
        from server.dependencies import get_user_dependencies

        # 只处理在线用户
        connected_users = self._get_online_users()
        if not connected_users:
            return 0

        for user_id in connected_users:
            try:
                deps = get_user_dependencies(user_id)
                initiative = await run_initiative_check(deps)

                if initiative and initiative.get("should_initiate"):
                    push_id = initiative.get("push_id")
                    has_options = bool(initiative.get("options"))

                    sent = await self.ws_manager.send_initiative_message(
                        user_id=user_id,
                        message_type=initiative.get("type", "info"),
                        content=initiative.get("content", ""),
                        options=initiative.get("options", []),
                        push_id=push_id,
                        requires_response=has_options,
                    )

                    if sent:
                        print(f"[initiative_check] 用户 {user_id}: 已推送 [{initiative.get('type')}]")

            except Exception as e:
                print(f"[initiative_check] 用户 {user_id} 错误: {e}")

        return len(connected_users)

    # ==================== 辅助方法 ====================

    def _record_error(self, task_name: str, error: str) -> None:
        """
        记录任务错误。

        Args:
            task_name: 任务名称
            error: 错误信息
        """
        if task_name in self._task_stats:
            self._task_stats[task_name].errors += 1
            self._task_stats[task_name].last_error = error

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取任务统计信息。

        Returns:
            统计信息字典
        """
        result = {}
        for name, stats in self._task_stats.items():
            result[name] = {
                "runs": stats.runs,
                "last_run": stats.last_run.isoformat() if stats.last_run else None,
                "errors": stats.errors,
                "last_error": stats.last_error,
                "last_duration": stats.last_duration,
                "users_processed": stats.users_processed,
            }
        return result

    # ==================== 事件触发方法 ====================

    async def trigger_proactive_search(self, user_id: str) -> None:
        """
        为特定用户触发主动搜索（如话题切换时）。

        Args:
            user_id: 用户 ID
        """
        from server.tasks.proactive_tasks import run_proactive_search
        from server.dependencies import get_user_dependencies

        try:
            deps = get_user_dependencies(user_id)
            deliveries = await run_proactive_search(deps)
            await self._deliver_proactive_deliveries(user_id, deliveries)

            print(f"[proactive_search] 触发搜索完成 | 用户: {user_id} | proactive delivery: {len(deliveries)}")

        except Exception as e:
            print(f"[proactive_search] 触发搜索失败 | 用户: {user_id} | 错误: {e}")

    async def _deliver_proactive_deliveries(
        self,
        user_id: str,
        deliveries: List[Dict[str, Any]],
    ) -> None:
        for delivery in deliveries:
            if delivery.get("channel") != "push":
                continue
            if not self.ws_manager.is_user_connected(user_id):
                continue

            await self.ws_manager.send_initiative_message(
                user_id=user_id,
                message_type=delivery.get("message_type", "report"),
                content=delivery.get("content", ""),
                options=[],
                push_id=delivery.get("item_id"),
                requires_response=False,
            )
