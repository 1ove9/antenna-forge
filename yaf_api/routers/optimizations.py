"""
Optimizations router — trigger and monitor optimization runs.

POST /api/v1/optimizations          Start an optimization run
GET  /api/v1/optimizations/{run_id} Get optimization status
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from yaf_core.domain.optimization import (
    OptimizationMethod,
    OptimizationRun,
    OptimizationState,
    Trial,
)

router = APIRouter(prefix="/api/v1/optimizations", tags=["optimizations"])

_runs: dict[uuid.UUID, OptimizationRun] = {}


@router.post("", response_model=OptimizationRun)
async def start_optimization(
    design_id: uuid.UUID,
    method: OptimizationMethod = OptimizationMethod.BAYESIAN,
    max_trials: int = 50,
) -> OptimizationRun:
    """Start an optimization run."""
    run = OptimizationRun(
        design_id=design_id,
        method=method,
        max_trials=max_trials,
        state=OptimizationState.RUNNING,
    )
    _runs[run.id] = run
    return run


@router.get("/{run_id}", response_model=OptimizationRun)
async def get_optimization(run_id: uuid.UUID) -> OptimizationRun:
    """Get optimization run status."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return _runs[run_id]
