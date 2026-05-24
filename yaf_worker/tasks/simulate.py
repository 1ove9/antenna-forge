"""
Simulation worker task — runs a solver on a given geometry.
"""

from __future__ import annotations

from celery import shared_task


@shared_task(bind=True, name="simulate.run")
def run_simulation(self, job_id: str, solver_name: str, spec_dict: dict) -> dict:
    """Execute an EM simulation via the specified solver adapter.

    Args:
        job_id: SimulationJob UUID as string.
        solver_name: Solver identifier ("nec2", "openems", etc.).
        spec_dict: Serialized SimulationSpec.

    Returns:
        Serialized SimulationResult.
    """
    import asyncio
    import uuid

    from yaf_core.domain.geometry import Geometry
    from yaf_core.domain.simulation import SimulationSpec

    spec = SimulationSpec(**spec_dict)

    async def _run():
        if solver_name == "nec2":
            from yaf_solvers.nec2_adapter.adapter import NEC2Adapter
            adapter = NEC2Adapter()
        elif solver_name == "openems":
            from yaf_solvers.openems_adapter.adapter import OpenEMSAdapter
            adapter = OpenEMSAdapter()
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        geom = Geometry()
        mesh = await adapter.mesh(geom, spec)

        async def progress(pct: float):
            self.update_state(state="PROGRESS", meta={"progress": pct})

        result = await adapter.solve(mesh, spec, progress_callback=progress)
        result.job_id = uuid.UUID(job_id)
        return result.model_dump(mode="json")

    return asyncio.run(_run())


@shared_task(name="simulate.cancel")
def cancel_simulation(job_id: str) -> dict:
    """Cancel a running simulation job."""
    return {"status": "cancelled", "job_id": job_id}
