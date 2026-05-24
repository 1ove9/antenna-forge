"""
Optimization domain model — runs, trials, and optimization state.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class OptimizationMethod(str, enum.Enum):
    """Optimization algorithm type."""

    BAYESIAN = "bayesian"
    NSGA2 = "nsga2"
    TOPOLOGY_SIMP = "topology_simp"
    TOPOLOGY_LEVELSET = "topology_levelset"
    GRADIENT_DESCENT = "gradient_descent"
    RANDOM_SEARCH = "random_search"
    CUSTOM = "custom"


class OptimizationState(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Trial(BaseModel):
    """A single evaluation within an optimization run."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    run_id: uuid.UUID
    trial_number: int
    parameters: dict[str, Any]  # e.g. {"length": 0.03, "width": 0.01}
    metrics: dict[str, float] = Field(default_factory=dict)  # {"gain_dbi": 5.2, "vswr": 1.3}
    status: str = "pending"
    simulation_job_id: uuid.UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OptimizationRun(BaseModel):
    """A complete multi-trial optimization run."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    design_id: uuid.UUID
    method: OptimizationMethod
    state: OptimizationState = OptimizationState.PENDING

    # Parameter space definition
    parameter_space: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="param_name -> (min, max)"
    )

    # Objective function (composite)
    objectives: list[str] = Field(
        default_factory=list, description="Metric names to optimize"
    )
    objective_weights: list[float] = Field(default_factory=list)

    # Constraints
    constraints: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="metric -> (min, max)"
    )

    # Budget
    max_trials: int = 100
    max_time_sec: float | None = None

    # Results
    trials: list[Trial] = Field(default_factory=list)
    best_trial: Trial | None = None
    pareto_front: list[Trial] = Field(default_factory=list)

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
