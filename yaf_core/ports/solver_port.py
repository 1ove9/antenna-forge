# ============================================================
# REFERENCE
#   仿造来源：无（首创实现）
#   理由：多求解器统一 Protocol 接口为 YAF 核心架构创新，
#        openEMS/NEC2/HFSS/CST/FEKO/COMSOL 各自 API 差异巨大，
#        无现成统一协议可对标。设计借鉴了 Tidy3D 的声明式理念
#        和 PyAEDT 的面向对象建模。
#   关键设计点：
#     - typing.Protocol + @runtime_checkable 实现鸭子类型插件化
#     - 异步优先接口（async mesh/solve/cancel/health_check）
#     - 能力自省（capabilities() 返回方法/频率/GPU/材料支持）
#     - to_native_format/from_native_result 双向格式转换
#     - 插件发现：plugin.toml → entrypoint → 热加载
#   YAF 的差异化改造：
#     - 统一 Geometry/Mesh → solver-native 的翻译层
#     - SimulationResult 规范返回格式（S 参数 + 远场 + 近场）
#     - progress_callback 支持 WebSocket 实时推送
# ============================================================

"""
SolverAdapter Protocol — every external EM solver must satisfy this interface.

Defines the contract that bridges the YAF domain model with
third-party solvers (openEMS, NEC2, HFSS, CST, FEKO, COMSOL, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from yaf_core.domain.geometry import Geometry, Mesh
from yaf_core.domain.simulation import SimulationResult, SimulationSpec


@runtime_checkable
class SolverAdapter(Protocol):
    """Protocol that all external EM solvers must implement.

    Each solver is a plugin: discoverable via plugin.toml, instantiated
    by the orchestrator, and called through this async interface.
    """

    # -- Class-level metadata (required) --

    name: str
    version: str
    supports: set[str]  # e.g. {"fdtd", "mom", "fem", "po"}

    # -- Capability introspection --

    async def capabilities(self) -> dict[str, Any]:
        """Return solver capabilities: methods, freq range, max cells, etc.

        Returns:
            dict with keys like:
                methods: list[str]
                frequency_range: [min_hz, max_hz]
                max_cells: int
                gpu_support: bool
                supported_materials: list[str]
        """
        ...

    # -- Meshing --

    async def mesh(self, geometry: Geometry, spec: SimulationSpec) -> Mesh:
        """Generate a solver-native mesh from canonical geometry.

        Args:
            geometry: Canonical YAF geometry (mesh or BREP).
            spec: Simulation spec (frequency, ports, BCs).

        Returns:
            Solver-ready Mesh with nodes, elements, and material assignments.

        Raises:
            MeshError: If mesh generation fails.
        """
        ...

    # -- Solving --

    async def solve(
        self,
        mesh: Mesh,
        spec: SimulationSpec,
        progress_callback: Any = None,
    ) -> SimulationResult:
        """Run the solver on the given mesh and return results.

        Args:
            mesh: Solver-ready mesh.
            spec: Simulation specification.
            progress_callback: Optional async callable(progress_pct: float).

        Returns:
            SimulationResult with S-params, far-field, and derived metrics.

        Raises:
            SolverError: If the solver fails.
        """
        ...

    # -- Cancellation --

    async def cancel(self, job_id: str) -> None:
        """Cancel a running simulation job.

        Args:
            job_id: The job identifier.

        Raises:
            SolverError: If cancellation fails or job not found.
        """
        ...

    # -- I/O --

    def to_native_format(self, geometry: Geometry) -> bytes:
        """Convert canonical geometry to the solver's native input format.

        Args:
            geometry: Canonical YAF geometry.

        Returns:
            Solver-native input file as bytes (e.g. .xml for openEMS,
            .nec for NEC2, .aedt for HFSS).
        """
        ...

    async def from_native_result(self, raw_output: bytes) -> SimulationResult:
        """Parse solver output into canonical SimulationResult.

        Args:
            raw_output: Raw solver output (text, binary, etc.).

        Returns:
            Canonical SimulationResult.
        """
        ...

    # -- Lifecycle --

    async def health_check(self) -> bool:
        """Verify solver executable is available and responsive.

        Returns:
            True if solver is healthy.
        """
        ...

    async def close(self) -> None:
        """Release solver resources (processes, temp files, licenses)."""
        ...
