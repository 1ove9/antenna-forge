# ============================================================
# REFERENCE
#   仿造来源：无（首创实现）
#   理由：YAF 专属的多求解器仿真调度 API，
#        Tidy3D Web API 和 Ansys AEDT 均为单一求解器 API。
#   关键设计点：
#     - POST /api/v1/simulations 提交作业（design_id + solver + 频率范围）
#     - 内联求解器路由（nec2 → NEC2Adapter, openems → OpenEMSAdapter）
#     - in-memory job store（演示模式）/ Celery dispatch（生产模式）
#     - 同步降级执行（demo 用）/ 异步 Celery 调度
#     - GET /{job_id} 状态查询 + GET /{job_id}/result 结果获取
# ============================================================

"""
Simulations router — trigger and monitor EM simulations.

POST   /api/v1/simulations          Submit a simulation job
GET    /api/v1/simulations/{job_id} Get simulation status/result
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from yaf_core.domain.simulation import (
    SimulationJob,
    SimulationResult,
    SimulationSpec,
)
from yaf_core.domain.geometry import Geometry

router = APIRouter(prefix="/api/v1/simulations", tags=["simulations"])

# In-memory job store
_jobs: dict[uuid.UUID, SimulationJob] = {}
_results: dict[uuid.UUID, SimulationResult] = {}


@router.post("", response_model=SimulationJob)
async def submit_simulation(
    design_id: uuid.UUID,
    solver: str = "nec2",
    frequency_min: float = 1e9,
    frequency_max: float = 2e9,
) -> SimulationJob:
    """Submit an EM simulation job.

    The job is dispatched to a Celery worker and progress is
    reported via WebSocket.
    """
    spec = SimulationSpec(
        name="simulation",
        frequency_range=(frequency_min, frequency_max),
        frequency_points=51,
    )

    job = SimulationJob(
        design_id=design_id,
        design_version=1,
        spec=spec,
        solver_name=solver,
        state="pending",
    )

    _jobs[job.id] = job

    # In a real deployment, this dispatches to Celery:
    # submit_simulation_task.delay(str(job.id), solver, spec.dict())

    # For demo, run synchronously
    try:
        job.state = "running"
        job.started_at = datetime.now(timezone.utc)

        if solver == "nec2":
            from yaf_solvers.nec2_adapter.adapter import NEC2Adapter
            adapter = NEC2Adapter()
            geom = Geometry()
            mesh = await adapter.mesh(geom, spec)
            result = await adapter.solve(mesh, spec)
        elif solver == "openems":
            from yaf_solvers.openems_adapter.adapter import OpenEMSAdapter
            adapter = OpenEMSAdapter()
            geom = Geometry()
            mesh = await adapter.mesh(geom, spec)
            result = await adapter.solve(mesh, spec)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        result.job_id = job.id
        _results[job.id] = result
        job.state = "completed"
        job.completed_at = datetime.now(timezone.utc)

    except Exception as e:
        job.state = "failed"
        job.error_message = str(e)

    _jobs[job.id] = job
    return job


@router.get("/{job_id}", response_model=SimulationJob)
async def get_simulation(job_id: uuid.UUID) -> SimulationJob:
    """Get simulation job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.get("/{job_id}/result", response_model=SimulationResult)
async def get_simulation_result(job_id: uuid.UUID) -> SimulationResult:
    """Get simulation result."""
    if job_id not in _results:
        raise HTTPException(status_code=404, detail="Result not found")
    return _results[job_id]
