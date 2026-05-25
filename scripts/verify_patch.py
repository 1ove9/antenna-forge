#!/usr/bin/env python3
"""openEMS truth check: rectangular microstrip patch resonant frequency.

Drives `yaf_solvers.openems_adapter.adapter.OpenEMSAdapter` (real openEMS FDTD,
no analytical fallback) on a standard inset-fed rectangular patch and compares
the simulated resonance to the classical cavity-model prediction.

Reference design — Rogers RO4003C laminate, a common analyzable case:

    substrate  : eps_r = 3.38, h = 1.524 mm, loss tangent 1e-3
    patch      : L = 32 mm (resonant, x) x W = 40 mm (radiating edge, y)
    feed       : lumped 50 ohm probe at x = -6 mm

Cavity-model resonance for the dominant TM010 mode (Balanis, "Antenna Theory",
microstrip-patch chapter):

    eps_eff = (er+1)/2 + (er-1)/2 * (1 + 12 h / W) ** -0.5
    dL      = 0.412 h * (eps_eff+0.3)(W/h+0.264) / ((eps_eff-0.258)(W/h+0.8))
    f_res   = c0 / (2 (L + 2 dL) sqrt(eps_eff))

Assertion:
    * simulated resonance (deepest S11 dip) within +/- 10 % of f_res above.

This is the openEMS counterpart of verify_dipole.py (the half-wave dipole /
73 ohm NEC2 truth check).
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

# repo root on sys.path so this runs without `pip install -e .`
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from yaf_core.domain.geometry import Geometry  # noqa: E402
from yaf_core.domain.simulation import SimulationSpec  # noqa: E402

# --- reference patch (Rogers RO4003C) ---
EPS_R = 3.38
SUB_H_MM = 1.524
PATCH_L_MM = 32.0   # resonant length (x)
PATCH_W_MM = 40.0   # radiating-edge width (y)
FEED_X_MM = -6.0
FEED_R = 50.0
LOSS_TAN = 1e-3
DESIGN_F = 2.45e9   # frequency at which the substrate loss is specified

# sweep / FDTD band
F_MIN = 1e9
F_MAX = 3e9
F_POINTS = 401
RESOLUTION = 20     # cells per wavelength
TOL_FRAC = 0.10

C0 = 299792458.0
EPS0 = 8.8541878128e-12


def analytical_resonance() -> tuple[float, float, float]:
    """Return (f_res_hz, eps_eff, delta_L_mm) from the cavity model."""
    er = EPS_R
    h = SUB_H_MM * 1e-3
    L = PATCH_L_MM * 1e-3
    W = PATCH_W_MM * 1e-3
    eps_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / W) ** -0.5
    dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264)) / (
        (eps_eff - 0.258) * (W / h + 0.8)
    )
    f_res = C0 / (2 * (L + 2 * dL) * math.sqrt(eps_eff))
    return f_res, eps_eff, dL * 1e3


def patch_simulation_spec() -> SimulationSpec:
    kappa = LOSS_TAN * 2 * math.pi * DESIGN_F * EPS0 * EPS_R
    sub_t = SUB_H_MM
    sub_half = 30.0  # substrate / ground half-extent (mm)
    structures = [
        {"kind": "metal", "name": "patch",
         "start": [-PATCH_L_MM / 2, -PATCH_W_MM / 2, sub_t],
         "stop": [PATCH_L_MM / 2, PATCH_W_MM / 2, sub_t],
         "priority": 10, "add_edges": "xy", "metal_edge_res": True},
        {"kind": "material", "name": "substrate", "epsilon": EPS_R, "kappa": kappa,
         "start": [-sub_half, -sub_half, 0], "stop": [sub_half, sub_half, sub_t],
         "priority": 0},
        {"kind": "metal", "name": "gnd",
         "start": [-sub_half, -sub_half, 0], "stop": [sub_half, sub_half, 0],
         "priority": 10, "add_edges": "xy"},
    ]
    return SimulationSpec(
        name="ro4003_patch",
        frequency_range=(F_MIN, F_MAX),
        frequency_points=F_POINTS,
        solver_settings={
            "unit": 1e-3,
            "resolution": RESOLUTION,
            "boundary": ["MUR"] * 6,
            "air_box": {"x": [-100, 100], "y": [-100, 100], "z": [-50, 100]},
            "extra_mesh_lines": {"z": list(np.linspace(0, sub_t, 5))},
            "structures": structures,
            "ports": [{"nr": 1, "R": FEED_R, "start": [FEED_X_MM, 0, 0],
                       "stop": [FEED_X_MM, 0, sub_t], "dir": "z", "excite": 1.0,
                       "priority": 5, "edges2grid": "xy"}],
            "nf2ff_center": [0, 0, 1e-3],
            "nr_timesteps": 60000,
            "end_criteria": 1e-4,
        },
    )


def main() -> int:
    from yaf_solvers.base import SolverUnavailable
    from yaf_solvers.openems_adapter.adapter import OpenEMSAdapter

    f_theory, eps_eff, dL_mm = analytical_resonance()

    print("=== Truth check: rectangular microstrip patch (Rogers RO4003C) ===")
    print(f"    eps_r={EPS_R}, h={SUB_H_MM} mm, "
          f"L={PATCH_L_MM} mm (resonant), W={PATCH_W_MM} mm")
    print(f"    cavity model: eps_eff={eps_eff:.4f}, dL={dL_mm:.4f} mm")
    print(f"    analytical f_res = {f_theory/1e9:.4f} GHz")
    print(f"    FDTD band {F_MIN/1e9:.1f}-{F_MAX/1e9:.1f} GHz, "
          f"resolution {RESOLUTION} cells/wavelength")
    print()

    adapter = OpenEMSAdapter()
    spec = patch_simulation_spec()
    geom = Geometry(name="ro4003_patch")
    try:
        mesh = asyncio.run(adapter.mesh(geom, spec))
        print("running openEMS FDTD (this takes tens of seconds) ...")
        result = asyncio.run(adapter.solve(mesh, spec))
    except SolverUnavailable as e:
        print(f"FAIL: openEMS unavailable: {e}")
        return 2

    meta = result.solver_metadata
    f_sim = float(meta["resonant_freq_hz"])
    s11_db = float(meta["resonant_s11_db"])
    zr, zi = meta["zin_at_resonance"]
    err_frac = abs(f_sim - f_theory) / f_theory

    print()
    print("=== Result ===")
    print(f"  measured f_res   = {f_sim/1e9:7.4f} GHz   "
          f"(S11 dip {s11_db:6.2f} dB, Zin {zr:5.1f}{zi:+.1f}j ohm)")
    print(f"  analytical f_res = {f_theory/1e9:7.4f} GHz")
    print(f"  relative error   = {err_frac*100:6.2f} %   (tol +/- {TOL_FRAC*100:.0f} %)")
    print(f"  directivity      = {result.gain_dbi:6.2f} dBi")
    print(f"  FDTD cells       = {meta.get('num_cells', 'n/a')}, "
          f"sim time {result.simulation_time_sec:.1f} s")
    print()

    if err_frac <= TOL_FRAC:
        print("PASS: real openEMS FDTD matches cavity-model patch resonance "
              f"within +/- {TOL_FRAC*100:.0f} %.")
        return 0

    print(f"FAIL: simulated resonance off by {err_frac*100:.1f} % "
          f"(> {TOL_FRAC*100:.0f} %).")
    print("  Diagnostics to check, in order: mesh resolution / edge refinement,")
    print("  feed position, substrate eps_r/kappa, FDTD convergence (energy decay).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
