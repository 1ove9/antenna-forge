# ============================================================
# REFERENCE
#   仿造来源：Tidy3D Web API + FastAPI 标准模板
#             @ https://github.com/flexcompute/tidy3d
#   对标文件：tidy3d/web/, FastAPI 官方 lifespan 示例
#   对标类/函数：FastAPI(lifespan=...), CORSMiddleware, include_router
#   关键设计点：
#     - FastAPI 应用工厂 + asynccontextmanager lifespan
#     - CORS 中间件（localhost:5173 + 3000）
#     - 模块化路由挂载（designs/simulations/optimizations/websocket）
#     - /health + /api/v1/health 双健康检查端点
#   YAF 的差异化改造：
#     - 内置 CORS 配置（无需 nginx 反向代理）
#     - 直接挂载 YAF 专属 routers
#     - 无数据库初始化（in-memory store 模式）
# ============================================================

"""
YAF API Server — FastAPI application entry point.

Provides REST and WebSocket endpoints for the antenna design platform.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from yaf_api.routers import designs, simulations, optimizations, websocket


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup/shutdown."""
    # Startup
    print("YAF API starting...")
    yield
    # Shutdown
    print("YAF API shutting down...")


app = FastAPI(
    title="Source Sequence Antenna Forge API",
    description="AI-driven antenna invention platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(designs.router)
app.include_router(simulations.router)
app.include_router(optimizations.router)
app.include_router(websocket.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/v1/health")
async def health_v1() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}
