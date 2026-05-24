#!/usr/bin/env python3
"""
Demo: Half-wave dipole antenna simulation and S11 analysis.

Runs a simple dipole design through the YAF pipeline:
1. Create a DesignSpec for a 2.4 GHz dipole
2. Simulate with NEC2 solver adapter
3. Report S-parameter results

Usage:
    python scripts/demo_dipole.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from yaf_core.domain.design import BoundingBox, DesignSpec, PatternTarget, Polarization
from yaf_core.domain.geometry import Geometry
from yaf_core.domain.simulation import SimulationSpec


def create_dipole_spec() -> DesignSpec:
    """Create a 2.4 GHz half-wave dipole design specification."""
    return DesignSpec(
        name="2.4GHz Dipole Demo",
        frequency_range=(2.4e9, 2.5e9),
        target_gain_dbi=2.15,
        target_pattern=PatternTarget(
            theta_range=(0.0, 180.0),
            phi_range=(0.0, 360.0),
            main_lobe_direction=(90.0, 0.0),
        ),
        polarization=Polarization.LINEAR,
        size_constraint=BoundingBox(
            x_min=-0.1, x_max=0.1,
            y_min=-0.1, y_max=0.1,
            z_min=-0.1, z_max=0.1,
        ),
        material_palette=["copper"],
        target_vswr=2.0,
    )


async def run_demo() -> None:
    """Run the dipole demo end-to-end."""
    print("=" * 60)
    print("  YAF Dipole Demo — 2.4 GHz Half-Wave Dipole")
    print("=" * 60)

    # 1. Create design spec
    spec = create_dipole_spec()
    f_center = (spec.frequency_range[0] + spec.frequency_range[1]) / 2
    wavelength = 3e8 / f_center

    print(f"\n  Design: {spec.name}")
    print(f"  Center frequency: {f_center / 1e9:.2f} GHz")
    print(f"  Wavelength: {wavelength:.3f} m")
    print(f"  Expected dipole length: {wavelength / 2:.3f} m")

    # 2. Create geometry — a single thin wire is the canonical NEC2 input
    # for a dipole. The old triangle-mesh form was a relic of an analytical
    # fallback path and produces overlapping NEC wires that real NEC2 rejects.
    length = wavelength / 2
    geom = Geometry(
        name="Half-wave dipole",
        representation="mesh",
        vertices=[
            [0.0, 0.0, -length / 2],
            [0.0, 0.0, length / 2],
        ],
        faces=[[0, 1]],
    )

    print(f"  Geometry: {geom.num_vertices} vertices, {geom.num_faces} faces")

    # 3. Create simulation spec
    sim_spec = SimulationSpec(
        name="Dipole S11 sweep",
        frequency_range=spec.frequency_range,
        frequency_points=51,
        far_field_request={"n_theta": 91, "n_phi": 1},
    )

    # 4. Run NEC2 simulation
    print("\n  Running NEC2 simulation...")

    from yaf_solvers.nec2_adapter.adapter import NEC2Adapter

    adapter = NEC2Adapter()
    mesh = await adapter.mesh(geom, sim_spec)
    result = await adapter.solve(mesh, sim_spec)

    print(f"\n  Solver: {result.solver_name} v{result.solver_version}")
    print(f"  Status: {result.status}")
    print(f"  Simulation time: {result.simulation_time_sec:.2f} s")

    # 5. Report S-parameters
    if result.s_params:
        s_params = result.s_params
        s11_db = 20 * np.log10(np.abs(s_params.s_matrix[0][0][0]) + 1e-12)

        print(f"\n  --- S-Parameter Results ---")
        print(f"  Frequency points: {len(s_params.frequency)}")
        print(f"  S11 at center ({f_center / 1e9:.2f} GHz): {s11_db:.2f} dB")

        # Find best S11
        s11_vals = [
            np.abs(s_params.s_matrix[i][0][0])
            for i in range(len(s_params.frequency))
        ]
        s11_db_vals = 20 * np.log10(np.array(s11_vals) + 1e-12)
        best_idx = np.argmin(s11_db_vals)

        print(
            f"  Best S11: {s11_db_vals[best_idx]:.2f} dB "
            f"@{s_params.frequency[best_idx] / 1e9:.3f} GHz"
        )

        # Bandwidth estimation
        threshold = -10.0
        bandwidth_mask = s11_db_vals < threshold
        if np.any(bandwidth_mask):
            bw_freqs = np.array(s_params.frequency)[bandwidth_mask]
            bw_hz = bw_freqs[-1] - bw_freqs[0]
            bw_pct = bw_hz / f_center * 100
            print(f"  -10 dB bandwidth: {bw_hz / 1e6:.1f} MHz ({bw_pct:.1f}%)")
        else:
            print("  -10 dB bandwidth: not met")

    # 6. Report far-field
    print(f"\n  --- Far-Field Results ---")
    print(f"  Gain: {result.gain_dbi:.2f} dBi")
    if result.efficiency is not None:
        print(f"  Efficiency: {result.efficiency:.2%}")
    if result.vswr is not None:
        print(f"  VSWR: {result.vswr:.2f}")
    if result.bandwidth_hz is not None:
        print(f"  Bandwidth: {result.bandwidth_hz / 1e6:.1f} MHz")

    if result.far_field:
        ff = result.far_field
        print(f"  Far-field grid: {len(ff.theta)} theta × {len(ff.phi)} phi")
        gain_pattern = ff.gain_dbi()
        print(f"  Peak gain: {np.max(gain_pattern):.2f} dBi")

    print(f"\n  {'='*60}")
    print(f"  Demo complete.")
    print(f"  {'='*60}")


def main() -> None:
    """Entry point."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
