"""
Solver adapter base class — provides common infrastructure for solver adapters.

All concrete solver adapters inherit from BaseSolverAdapter, which
implements the SolverAdapter Protocol with shared logic for:
- Error handling
- Progress reporting
- Logging
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable

from yaf_core.domain.geometry import Geometry, Mesh
from yaf_core.domain.simulation import SimulationResult, SimulationSpec


class YAFError(Exception):
    """Base exception for all YAF errors."""
    pass


class SolverError(YAFError):
    """Solver execution failure."""
    def __init__(self, solver: str, job_id: str, message: str) -> None:
        super().__init__(f"[{solver}] Job {job_id}: {message}")
        self.solver = solver
        self.job_id = job_id


class SolverUnavailable(YAFError):
    """Required solver backend (binary, Python binding, license) is missing.

    Adapters MUST raise this instead of silently returning fabricated results
    when their underlying solver cannot be invoked.
    """
    def __init__(self, solver: str, reason: str) -> None:
        super().__init__(f"[{solver}] solver unavailable: {reason}")
        self.solver = solver
        self.reason = reason


class MeshError(YAFError):
    """Mesh generation failure."""
    pass


class GeometryError(YAFError):
    """Geometry validation or conversion failure."""
    pass


class BaseSolverAdapter(ABC):
    """Abstract base for all solver adapters.

    Subclasses must implement:
    - mesh()
    - solve()
    - to_native_format()
    - from_native_result()
    """

    name: str = "base"
    version: str = "0.1.0"
    supports: set[str] = set()

    def __init__(self) -> None:
        self._running_jobs: dict[str, asyncio.Task[Any]] = {}

    async def capabilities(self) -> dict[str, Any]:
        """Return solver capabilities metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "methods": sorted(self.supports),
            "frequency_range": [1e6, 100e9],
            "max_cells": 1e7,
            "gpu_support": False,
        }

    @abstractmethod
    async def mesh(self, geometry: Geometry, spec: SimulationSpec) -> Mesh:
        ...

    @abstractmethod
    async def solve(
        self,
        mesh: Mesh,
        spec: SimulationSpec,
        progress_callback: Callable[[float], Any] | None = None,
    ) -> SimulationResult:
        ...

    @abstractmethod
    def to_native_format(self, geometry: Geometry) -> bytes:
        ...

    @abstractmethod
    async def from_native_result(self, raw_output: bytes) -> SimulationResult:
        ...

    async def cancel(self, job_id: str) -> None:
        """Cancel a running job."""
        if job_id in self._running_jobs:
            self._running_jobs[job_id].cancel()
            del self._running_jobs[job_id]

    async def health_check(self) -> bool:
        """Check solver executable availability."""
        return True

    async def close(self) -> None:
        """Release resources."""
        for job_id in list(self._running_jobs):
            await self.cancel(job_id)
