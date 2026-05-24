"""HFSS FEM solver adapter (skeleton). Requires PyAEDT + ANSYS HFSS license."""
from __future__ import annotations
from typing import Any, Callable
from yaf_core.domain.geometry import Geometry, Mesh
from yaf_core.domain.simulation import SimulationResult, SimulationSpec
from yaf_solvers.base import BaseSolverAdapter
class HFSSAdapter(BaseSolverAdapter):
    name="hfss";version="2024.1";supports={"fem"}
    async def capabilities(self)->dict[str, Any]:
        c=await super().capabilities();c.update({"requires_license":True});return c
    async def mesh(self,g:Geometry,s:SimulationSpec)->Mesh:
        return Mesh(geometry_id=g.id,solver_name=self.name,element_type="tet10",metadata={"status":"skeleton"})
    async def solve(self,m:Mesh,s:SimulationSpec,cb:Callable[[float], None] | None=None)->SimulationResult:
        import uuid;return SimulationResult(job_id=uuid.uuid4(),solver_name=self.name,solver_version=self.version,status="skeleton_not_implemented")
    def to_native_format(self,g:Geometry)->bytes:return b""
    async def from_native_result(self,r:bytes)->SimulationResult:
        import uuid;return SimulationResult(job_id=uuid.uuid4(),solver_name=self.name,solver_version=self.version,status="success")
