# YAF 系统架构文档

## 1. 概述

源序天线锻造平台 (Source Sequence Antenna Forge, YAF) 是一个以"发明新天线"为目标的智能设计平台。它采用**微内核 + 适配器 + 插件 SDK**三层架构，将 AI 生成、电磁仿真、拓扑优化和主动学习整合为闭环流水线。

## 2. 架构层级

### 2.1 表示层 (Presentation)

- **Web UI**: React 18 + TypeScript + Three.js + WebGPU
- 组件: 3D 编辑器、设计空间浏览器 (t-SNE/UMAP)、仿真监控、插件管理器
- 通信: REST (CRUD) + WebSocket (实时进度)

### 2.2 应用层 (Application)

- **API 网关**: FastAPI + Pydantic v2
- 路由: /api/v1/designs, /simulations, /optimizations
- WebSocket: /ws/simulation/{job_id} 实时进度推送
- 中间件: CORS, 结构化日志 (structlog)

### 2.3 编排层 (Orchestration)

- **任务调度**: Celery + Redis
- **设计版本控制**: 每个 Design 拥有不可变版本历史
- **事件总线**: Redis Streams 用于跨服务事件

### 2.4 领域核心层 (Domain Core)

```
yaf_core/
├── domain/         # 领域模型 (Design, Geometry, Simulation, Optimization)
├── ports/          # 抽象协议 (SolverAdapter, AIBackend, CADBackend)
├── geometry/       # 几何内核 (OpenCASCADE, SIREN, 拓扑优化)
└── physics/        # 物理模型 (超表面, RIS, OAM, 石墨烯, 时空调制)
```

### 2.5 插件层 (Plugin)

所有外部能力通过 Protocol 接入：

| 插件类型 | 协议 | 示例实现 |
|----------|------|----------|
| 求解器 | SolverAdapter | openEMS, NEC2, HFSS, CST, FEKO |
| AI 模型 | AIBackend | VAE, Diffusion, PINN, FNO |
| CAD 引擎 | CADBackend | FreeCAD, Blender, Rhino |

## 3. 数据流

```
DesignSpec → [1] AI 生成 → [2] FNO 筛选 → [3] 可微 FDTD 精修
           → [4] 拓扑优化 → [5] 高保真验证 → [6] 结果入库 → 闭环
```

## 4. 设计状态机

```
DRAFT → GENERATING → MESHING → SOLVING → SOLVED
  ↓                                        ↑
  └──────────→ FAILED ←────────────────────┘
```

## 5. 技术决策

| 决策 | 理由 |
|------|------|
| Pydantic v2 数据模型 | 类型安全 + JSON Schema 自动生成 |
| async/await 全线 IO | 高并发求解器编排 |
| JAX 可微分 FDTD | 端到端梯度反传, GPU 加速 |
| Protocol 类型（非 ABC）| 插件热加载, 无继承耦合 |
| pyproject.toml + uv | 极速包管理 |
