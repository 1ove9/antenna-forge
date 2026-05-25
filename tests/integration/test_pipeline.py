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
    """Integration: run openEMS FDTD on a small microstrip patch.

    Real full-wave run (no analytical fallback); skipped when the openEMS
    Python bindings are not importable.
    """
    import math

    import numpy as np
    import pytest

    try:
        from openEMS import openEMS  # noqa: F401
        from CSXCAD import ContinuousStructure  # noqa: F401
    except Exception:
        pytest.skip("openEMS Python bindings not installed")

    from yaf_core.domain.geometry import Geometry
    from yaf_core.domain.simulation import SimulationSpec
    from yaf_solvers.openems_adapter.adapter import OpenEMSAdapter

    eps0 = 8.8541878128e-12
    sub_t = 1.524
    kappa = 1e-3 * 2 * math.pi * 2.45e9 * eps0 * 3.38
    structures = [
        {"kind": "metal", "name": "patch", "start": [-16, -20, sub_t],
         "stop": [16, 20, sub_t], "priority": 10, "add_edges": "xy",
         "metal_edge_res": True},
        {"kind": "material", "name": "substrate", "epsilon": 3.38, "kappa": kappa,
         "start": [-30, -30, 0], "stop": [30, 30, sub_t], "priority": 0},
        {"kind": "metal", "name": "gnd", "start": [-30, -30, 0],
         "stop": [30, 30, 0], "priority": 10, "add_edges": "xy"},
    ]
    spec = SimulationSpec(
        frequency_range=(1e9, 3e9),
        frequency_points=21,
        solver_settings={
            "unit": 1e-3,
            "resolution": 15,
            "air_box": {"x": [-100, 100], "y": [-100, 100], "z": [-50, 100]},
            "extra_mesh_lines": {"z": list(np.linspace(0, sub_t, 5))},
            "structures": structures,
            "ports": [{"nr": 1, "R": 50.0, "start": [-6, 0, 0],
                       "stop": [-6, 0, sub_t], "dir": "z", "excite": 1.0,
                       "priority": 5, "edges2grid": "xy"}],
            "nf2ff_center": [0, 0, 1e-3],
        },
    )
    adapter = OpenEMSAdapter()
    geom = Geometry()
    mesh = asyncio.run(adapter.mesh(geom, spec))
    result = asyncio.run(adapter.solve(mesh, spec))
    assert result.status == "success"
    assert result.s_params is not None
    assert len(result.s_params.frequency) == 21
    assert result.gain_dbi is not None
    assert result.solver_metadata["backend"] == "openEMS-python"
