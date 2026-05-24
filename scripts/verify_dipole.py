#!/usr/bin/env python3
"""NEC2 truth check: half-wave dipole impedance and gain via real NEC2 (necpp).

Drives `yaf_solvers.nec2_adapter.adapter.NEC2Adapter` (which now calls necpp,
no analytical fallback) on a 0.47 m wire dipole at 300 MHz and compares to
classical textbook values for a thin half-wave dipole:

    R_in  ≈ 73 ohm
    X_in  ≈ +42 ohm   (for an ideal 0.5 λ; for ~0.47 λ resonance crosses zero)
    G_max ≈ 2.15 dBi

Assertions:
    * R within ±15 % of 73 ohm at 300 MHz
    * X passes through zero across a small length sweep (proves resonance is
      where it should be; 0.5 λ thin-wire X is positive, slightly shorter is
      negative, the sweep must cross zero in between)
    * Gain in [1.9, 2.4] dBi
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# repo root on sys.path so this runs without `pip install -e .`
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from yaf_core.domain.geometry import Geometry  # noqa: E402
from yaf_core.domain.simulation import SimulationSpec  # noqa: E402
from yaf_solvers.nec2_adapter.adapter import NEC2Adapter  # noqa: E402

FREQ_HZ = 300e6
SEGMENTS = 21
WIRE_RADIUS_M = 0.0005

# Theory targets
THEORY_R = 73.0
THEORY_X = 42.0
THEORY_G_DBI = 2.15
R_TOL_FRAC = 0.15
G_BAND = (1.9, 2.4)


def _run_one(length_m: float) -> tuple[float, float, float]:
    """Returns (R_ohm, X_ohm, gain_dBi) for a center-fed dipole of total length L."""
    half = length_m / 2.0
    geom = Geometry(
        vertices=[[0.0, 0.0, -half], [0.0, 0.0, half]],
        faces=[[0, 1]],
    )
    spec = SimulationSpec(
        frequency_range=(FREQ_HZ, FREQ_HZ),
        frequency_points=1,
        solver_settings={
            "segments": SEGMENTS,
            "wire_radius": WIRE_RADIUS_M,
        },
    )
    adapter = NEC2Adapter()
    mesh = asyncio.run(adapter.mesh(geom, spec))
    result = asyncio.run(adapter.solve(mesh, spec))
    R, X = result.solver_metadata["impedance_per_freq"][0]
    return float(R), float(X), float(result.gain_dbi or 0.0)


def main() -> int:
    print(f"=== Truth check: half-wave dipole @ {FREQ_HZ/1e6:.0f} MHz ===")
    print(f"    segments={SEGMENTS}, wire radius={WIRE_RADIUS_M*1000:.2f} mm")
    print()

    # Length sweep around the textbook half-wave (0.5 λ = 0.5 m at 300 MHz),
    # spanning the canonical "0.47–0.50 λ" range so X crosses zero (resonance).
    sweep_lengths = [0.46, 0.47, 0.48, 0.49, 0.50]
    print(f"{'L [m]':>8} {'L/λ':>7} {'R [Ω]':>10} {'X [Ω]':>10} {'G [dBi]':>10}")
    print("-" * 50)
    sweep: list[tuple[float, float, float, float]] = []
    for L in sweep_lengths:
        R, X, G = _run_one(L)
        sweep.append((L, R, X, G))
        wavelength = 3e8 / FREQ_HZ
        print(f"{L:8.3f} {L/wavelength:7.3f} {R:10.3f} {X:+10.3f} {G:10.3f}")
    print()

    # Operating point: total length ≈ 0.47 m (slightly past resonance for a
    # thin wire at 300 MHz, so X(L=0.47 m) is just-capacitive).
    L_ref = 0.47
    R_ref, X_ref, G_ref = next(
        (R, X, G) for (L, R, X, G) in sweep if abs(L - L_ref) < 1e-9
    )

    # Resonance check: across the sweep X must cross zero (sign change)
    x_vals = [X for (_, _, X, _) in sweep]
    resonance_crossed = any(
        x_vals[i] * x_vals[i + 1] < 0 for i in range(len(x_vals) - 1)
    )

    R_err_frac = abs(R_ref - THEORY_R) / THEORY_R
    G_in_band = G_BAND[0] <= G_ref <= G_BAND[1]

    print("=== Comparison at L = 0.47 m (≈ 0.47 λ) ===")
    print(f"  R measured = {R_ref:7.3f} Ω    theory ≈ {THEORY_R:6.1f} Ω    "
          f"err = {R_err_frac*100:5.2f} %    tol ±{R_TOL_FRAC*100:.0f} %")
    print(f"  X measured = {X_ref:+7.3f} Ω    theory ≈ {THEORY_X:+6.1f} Ω (at 0.5 λ)")
    print(f"            -- resonance (X→0) crossed in sweep: {resonance_crossed}")
    print(f"  G measured = {G_ref:7.3f} dBi  theory ≈ {THEORY_G_DBI:6.2f} dBi  "
          f"band = [{G_BAND[0]}, {G_BAND[1]}]")
    print()

    failures: list[str] = []
    if R_err_frac > R_TOL_FRAC:
        failures.append(
            f"R={R_ref:.2f} Ω outside ±{R_TOL_FRAC*100:.0f} % of {THEORY_R} Ω "
            f"(err {R_err_frac*100:.1f} %)"
        )
    if not resonance_crossed:
        failures.append(
            "X did not cross zero in the 0.46–0.50 m sweep "
            "(expected resonance somewhere in 0.47–0.49 m)"
        )
    if not G_in_band:
        failures.append(
            f"G_max={G_ref:.2f} dBi outside [{G_BAND[0]}, {G_BAND[1]}] dBi"
        )

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS: real NEC2 (necpp) matches half-wave dipole theory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
