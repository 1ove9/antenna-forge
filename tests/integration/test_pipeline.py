"""Integration test: PIPELINE — runs the full inverse design pipeline demo."""

import asyncio
import sys

def test_pipeline_demo():
    """Run the end-to-end inverse design pipeline and verify it completes."""
    from yaf_ai.inverse_design.pipeline import (
        InverseDesignPipeline, PipelineConfig, demo_pipeline,
    )
    from yaf_core.domain.design import (
        BoundingBox, DesignSpec, Polarization,
    )
    config = PipelineConfig(
        n_candidates=4, top_k=2, max_pipeline_loops=1,
        use_surrogate=True, use_diff_fdtd=False, use_topo=False, use_high_fidelity=False,
    )
    spec = DesignSpec(
        name="test_pipeline", frequency_range=(2.4e9, 2.5e9),
        size_constraint=BoundingBox(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, z_min=-0.1, z_max=0.1),
        target_gain_dbi=2.0, material_palette=["copper"],
    )
    pipeline = InverseDesignPipeline(config)
    result = asyncio.run(pipeline.run(spec))
    assert result.loop_count >= 1
    assert len(result.all_candidates) > 0


def test_solver_nec2_integration():
    """Integration: run NEC2 (necpp) on a real half-wave dipole at 2.45 GHz."""
    import pytest
    try:
        import necpp  # noqa: F401
    except Exception:
        pytest.skip("necpp not installed")
    from yaf_core.domain.geometry import Geometry
    from yaf_core.domain.simulation import SimulationSpec
    from yaf_solvers.nec2_adapter.adapter import NEC2Adapter

    # λ/2 at 2.45 GHz ≈ 61.2 mm; centered along z
    half = 0.0306
    geom = Geometry(vertices=[[0, 0, -half], [0, 0, half]], faces=[[0, 1]])
    spec = SimulationSpec(
        frequency_range=(2.4e9, 2.5e9),
        frequency_points=11,
        solver_settings={"wire_radius": 0.0001},
    )
    adapter = NEC2Adapter()
    mesh = asyncio.run(adapter.mesh(geom, spec))
    result = asyncio.run(adapter.solve(mesh, spec))
    assert result.status == "success"
    assert result.gain_dbi is not None
    assert result.vswr is not None
    assert result.s_params is not None
    assert len(result.s_params.frequency) == 11


def test_solver_openems_integration():
    """Integration: run openEMS solver and verify result structure."""
    import uuid
    from yaf_core.domain.geometry import Geometry
    from yaf_core.domain.simulation import SimulationSpec
    from yaf_solvers.openems_adapter.adapter import OpenEMSAdapter

    adapter = OpenEMSAdapter()
    geom = Geometry()
    spec = SimulationSpec(frequency_range=(2.4e9, 2.5e9), frequency_points=21)
    mesh = asyncio.run(adapter.mesh(geom, spec))
    result = asyncio.run(adapter.solve(mesh, spec))
    assert result.status == "success"
    if result.s_params:
        assert len(result.s_params.frequency) == 21
