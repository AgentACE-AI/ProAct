"""
FastAPI 应用入口。

提供 HTTP API 和 WebSocket 端点。
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import config
from server.dependencies import (
    get_config,
    get_user_dependencies,
    clear_user_dependencies,
    get_all_user_ids,
    UserDependencies,
)
from server.handlers.chat_handler import ChatHandler
from server.handlers.report_handler import ReportHandler
from server.websocket.ws_handler import WebSocketManager, websocket_endpoint
from server.tasks.scheduler import TaskScheduler


logger = logging.getLogger(__name__)

# 全局管理器实例
ws_manager = WebSocketManager()
task_scheduler: Optional[TaskScheduler] = None

INTERNAL_ERROR_DETAIL = "内部服务错误，请稍后重试"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。

    启动时初始化任务调度器，关闭时停止调度器。
    """
    global task_scheduler

    # 启动
    print("[Server] 启动中...")

    # 根据配置决定是否启用任务调度器
    if config.tasks.enabled:
        task_scheduler = TaskScheduler(
            config=config,
            ws_manager=ws_manager,
        )
        await task_scheduler.start()
        print("[Server] 任务调度器已启动")
    else:
        print("[Server] 主动任务已禁用")
    
    yield

    # 关闭
    print("[Server] 关闭中...")
    if task_scheduler:
        await task_scheduler.stop()
    print("[Server] 任务调度器已停止")


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例。

    Returns:
        FastAPI 应用
    """
    application = FastAPI(
        title="Proactive Personalized Agent System",
        description="具有记忆增强和主动能力的对话 AI 系统",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS 来源可通过 CORS_ALLOW_ORIGINS 环境变量配置（逗号分隔）
    cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] or ["*"]

    # 添加 CORS 中间件
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


# 创建应用实例
app = create_app()


# ValueError（如非法 user_id）统一返回 422 而非 500
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    """对话请求"""
    user_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, max_length=100
    )


class ChatResponse(BaseModel):
    """对话响应"""
    reply: str
    topic: Optional[str] = None
    topic_changed: bool = False
    search_performed: bool = False
    alerts: List[Dict[str, Any]] = []
    notifications: List[Dict[str, Any]] = []


class GenerateReportRequest(BaseModel):
    """生成报告请求"""
    topic: str
    purpose: str = "Research requested by the user"


class ReportResponse(BaseModel):
    """报告响应"""
    success: bool
    report_id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    """统计响应"""
    user_id: str
    memory_stats: Dict[str, Any]
    report_stats: Dict[str, Any]
    current_topic: Optional[str] = None


# ==================== 根端点 ====================

@app.get("/")
async def root():
    """根端点"""
    return {
        "name": "Proactive Personalized Agent System",
        "version": "2.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


# ==================== 对话 API ====================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    处理对话消息。

    流程：
    1. 获取或创建用户依赖
    2. 如果提供了对话历史，预加载到 memory
    3. 通过 ChatHandler 处理消息
    4. 返回响应
    """
    try:
        deps = get_user_dependencies(request.user_id)
        handler = ChatHandler(deps)

        # 如果提供了对话历史，预加载到 memory（用于模拟模式等场景）
        if request.conversation_history:
            handler.preload_conversation_history(request.conversation_history)

        result = await handler.handle(request.message)

        return ChatResponse(
            reply=result.reply,
            topic=result.topic,
            topic_changed=result.topic_changed,
            search_performed=result.search_performed,
            alerts=result.alerts,
            notifications=result.notifications,
        )

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.get("/api/chat/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """获取对话历史"""
    try:
        deps = get_user_dependencies(user_id)
        handler = ChatHandler(deps)

        history = handler.get_conversation_history(limit)

        return {
            "user_id": user_id,
            "messages": history,
            "current_topic": deps.memory.current_topic,
        }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.post("/api/chat/reset/{user_id}")
async def reset_chat(user_id: str):
    """重置对话会话"""
    try:
        deps = get_user_dependencies(user_id)
        handler = ChatHandler(deps)

        handler.reset_session()

        return {"status": "ok", "message": f"已重置用户 {user_id} 的会话"}

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


# ==================== 报告 API ====================

@app.post("/api/reports/{user_id}/generate", response_model=ReportResponse)
async def generate_report(user_id: str, request: GenerateReportRequest):
    """生成调研报告"""
    try:
        deps = get_user_dependencies(user_id)
        handler = ReportHandler(deps)

        result = await handler.generate_report(
            topic=request.topic,
            purpose=request.purpose,
        )

        if result.success and result.report:
            return ReportResponse(
                success=True,
                report_id=result.report.id,
                title=result.report.title,
                summary=result.report.summary,
                content=result.report.content,
            )
        else:
            return ReportResponse(
                success=False,
                error=result.error,
            )

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.get("/api/reports/{user_id}")
async def list_reports(user_id: str):
    """获取报告列表"""
    try:
        deps = get_user_dependencies(user_id)
        handler = ReportHandler(deps)

        reports = handler.list_reports()

        return {
            "user_id": user_id,
            "reports": reports,
        }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.get("/api/reports/{user_id}/{report_id}")
async def get_report(user_id: str, report_id: str):
    """获取报告详情"""
    try:
        deps = get_user_dependencies(user_id)
        handler = ReportHandler(deps)

        report = handler.get_report(report_id)

        if report:
            return {
                "success": True,
                "report": report.to_dict(),
            }
        else:
            return {
                "success": False,
                "error": "报告不存在",
            }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.get("/api/reports/{user_id}/pending")
async def get_pending_reports(user_id: str):
    """获取待推送报告列表"""
    try:
        deps = get_user_dependencies(user_id)
        handler = ReportHandler(deps)

        pending = handler.get_pending_reports()

        return {
            "user_id": user_id,
            "pending_reports": [r.to_dict() for r in pending],
        }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


# ==================== 系统 API ====================

@app.get("/api/system/stats")
async def get_system_stats():
    """获取系统统计"""
    try:
        user_ids = get_all_user_ids()
        ws_stats = ws_manager.get_stats()
        task_stats = task_scheduler.get_stats() if task_scheduler else {}

        return {
            "active_users": len(user_ids),
            "user_ids": user_ids,
            "websocket": ws_stats,
            "tasks": task_stats,
        }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.get("/api/system/users")
async def list_users():
    """获取所有用户列表"""
    try:
        user_ids = get_all_user_ids()
        users = []

        for user_id in user_ids:
            deps = get_user_dependencies(user_id)
            profile = deps.memory.get_user_profile()

            users.append({
                "user_id": user_id,
                "interests_count": len(profile.interests),
                "current_topic": deps.memory.current_topic,
                "is_connected": ws_manager.is_user_connected(user_id),
            })

        return {"users": users}

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.get("/api/system/users/{user_id}/stats", response_model=StatsResponse)
async def get_user_stats(user_id: str):
    """获取用户统计"""
    try:
        deps = get_user_dependencies(user_id)
        handler = ReportHandler(deps)

        memory_stats = deps.memory.get_stats()
        report_stats = handler.get_report_stats()

        return StatsResponse(
            user_id=user_id,
            memory_stats=memory_stats,
            report_stats=report_stats,
            current_topic=deps.memory.current_topic,
        )

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.get("/api/system/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """获取用户画像"""
    try:
        deps = get_user_dependencies(user_id)
        profile = deps.memory.get_user_profile()

        return {
            "user_id": user_id,
            "profile": profile.to_dict(),
        }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.get("/api/system/users/{user_id}/exists")
async def check_user_exists(user_id: str):
    """检查用户是否存在"""
    try:
        exists = config.storage.user_exists(user_id)
        return {"exists": exists}

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


class CreateUserRequest(BaseModel):
    """创建用户请求"""
    user_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    mode: str = "real"  # "real" 或 "simulation"
    use_default_profile: bool = True


@app.post("/api/system/users/create")
async def create_user(request: CreateUserRequest):
    """
    创建新用户。

    对于模拟用户，如果 use_default_profile=True，将从预设的模拟画像路径加载画像。
    """
    try:
        import json

        # 检查用户是否已存在
        if config.storage.user_exists(request.user_id):
            return {
                "success": False,
                "error": f"用户 {request.user_id} 已存在",
            }

        # 获取用户依赖（会自动创建用户数据）
        deps = get_user_dependencies(request.user_id)
        profile = deps.memory.get_user_profile()

        # 设置交互模式
        profile.interaction_mode = request.mode

        # 如果是模拟模式且使用默认画像，从预设路径加载模拟画像
        if request.mode == "simulation" and request.use_default_profile:
            # 尝试从预设路径加载模拟画像
            simulator_profile_path = config.storage.get_user_dir(request.user_id) / "simulator_profile.json"

            # 如果没有预设画像，创建一个基础的模拟画像
            if not simulator_profile_path.exists():
                default_simulator_profile = {
                    "user_id": request.user_id,
                    "core_profile": {
                        "role": "模拟用户",
                        "core_tags": ["测试用户"]
                    },
                    "personality_and_traits": ["好奇心强", "乐于学习"],
                    "interests_and_focus": ["技术", "人工智能"],
                    "goals_and_motivations": ["学习新知识", "提高效率"],
                    "pain_points_and_challenges": [],
                    "communication_style": ["简洁明了"]
                }
                with open(simulator_profile_path, "w", encoding="utf-8") as f:
                    json.dump(default_simulator_profile, f, ensure_ascii=False, indent=2)

        # 保存画像
        deps.memory._profile_store.save(profile)

        return {
            "success": True,
            "user_id": request.user_id,
            "mode": request.mode,
        }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


class SimulateRequest(BaseModel):
    """模拟对话请求"""
    user_id: str
    turns: int = 5
    profile: Optional[str] = None


@app.post("/api/system/simulate")
async def simulate_conversation(request: SimulateRequest):
    """
    运行模拟对话。

    使用 UserSimulatorAgent 根据用户画像生成模拟消息。
    """
    import json
    from agents.user_simulator import UserSimulatorAgent

    try:
        deps = get_user_dependencies(request.user_id)
        handler = ChatHandler(deps)

        # 创建模拟器 Agent
        simulator = UserSimulatorAgent(
            config=config.agents.user_simulator,
            llm_client=deps._llm,
        )

        # 加载模拟画像
        simulator_profile_path = config.storage.get_user_dir(request.user_id) / "simulator_profile.json"
        if simulator_profile_path.exists():
            with open(simulator_profile_path, "r", encoding="utf-8") as f:
                simulator_profile_data = json.load(f)
            user_profile_text = UserSimulatorAgent.format_simulator_profile(simulator_profile_data)
        else:
            # 使用普通画像
            profile = deps.memory.get_user_profile()
            user_profile_text = profile.to_prompt_text()

        conversations = []
        conversation_history: List[Dict[str, str]] = []

        for i in range(request.turns):
            # 使用 UserSimulatorAgent 生成模拟消息
            simulated_msg = simulator.run(
                user_profile=user_profile_text,
                conversation_history=conversation_history,
            )

            # 处理模拟消息
            result = await handler.handle(simulated_msg)

            # 记录对话
            conversations.append({
                "turn": i + 1,
                "user": simulated_msg,
                "assistant": result.reply,
            })

            # 更新对话历史
            conversation_history.append({"role": "user", "content": simulated_msg})
            conversation_history.append({"role": "assistant", "content": result.reply})

        return {
            "success": True,
            "user_id": request.user_id,
            "turns": len(conversations),
            "conversations": conversations,
        }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


class ImportSimulatorProfileRequest(BaseModel):
    """导入模拟器画像请求"""
    profile: Dict[str, Any]


@app.post("/api/system/users/{user_id}/profile/simulator")
async def import_simulator_profile(user_id: str, request: ImportSimulatorProfileRequest):
    """导入模拟器用户画像"""
    import json

    try:
        # 确保用户存在
        deps = get_user_dependencies(user_id)

        # 设置为模拟模式
        profile = deps.memory.get_user_profile()
        profile.interaction_mode = "simulation"
        deps.memory._profile_store.save(profile)

        # 保存模拟器画像
        simulator_profile_path = config.storage.get_user_dir(user_id) / "simulator_profile.json"
        profile_data = request.profile
        profile_data["user_id"] = user_id

        with open(simulator_profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "user_id": user_id,
            "message": "模拟器画像导入成功",
        }

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


@app.post("/api/reports/mark-read/{user_id}/{report_id}")
async def mark_report_read(user_id: str, report_id: str):
    """标记报告为已读"""
    try:
        deps = get_user_dependencies(user_id)
        handler = ReportHandler(deps)

        success = handler.mark_report_pushed(report_id)

        if success:
            return {"success": True, "message": f"报告 {report_id} 已标记为已读"}
        else:
            return {"success": False, "error": "标记失败"}

    except Exception as e:
        logger.exception("API error: %s", e)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)


# ==================== WebSocket 端点 ====================

@app.websocket("/ws/{user_id}")
async def websocket_route(websocket: WebSocket, user_id: str):
    """
    WebSocket 端点。

    用于实时推送通知和消息。
    """
    await websocket_endpoint(
        websocket=websocket,
        user_id=user_id,
        manager=ws_manager,
    )


# ==================== 辅助函数 ====================

def get_ws_manager() -> WebSocketManager:
    """获取 WebSocket 管理器实例"""
    return ws_manager


def get_task_scheduler() -> Optional[TaskScheduler]:
    """获取任务调度器实例"""
    return task_scheduler
