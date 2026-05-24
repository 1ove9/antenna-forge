#!/usr/bin/env python3
"""Inverse-design demo: real NEC2 in the optimization loop.

Goal: find the dipole length L that resonates at f_target = 300 MHz
(i.e. minimize |X_in(L, f_target)|) while staying near peak gain. Each
candidate L is evaluated by *actually running NEC2* via necpp — no mock,
no analytical fallback, no surrogate model.

Optimizer: golden-section search on objective J(L) = |X(L; 300 MHz)|.
This is the simplest 1-D minimizer that converges geometrically and is
honest about "each iteration costs one solver call".

Outputs:
  * Iteration table printed to stdout (L, R, X, G, cost, bracket width).
  * docs/assets/inverse_design_convergence.png:
        - left:  evaluated (L, |X|) samples with bracket shrinking.
        - right: |X| best-so-far vs iteration (log scale).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import necpp
except ImportError as e:
    raise SystemExit(
        "necpp not installed — cannot run real-NEC2 inverse design. "
        "Install with: pip install necpp --break-system-packages"
    ) from e


# --------------------------------------------------------------------------- #
# Problem
# --------------------------------------------------------------------------- #

F_TARGET_MHZ = 300.0
SEGMENTS = 21
WIRE_RADIUS_M = 0.0005
L_LOW_M, L_HIGH_M = 0.40, 0.55         # search bracket for total dipole length
TOL_M = 1e-4                            # 0.1 mm geometric tolerance
MAX_ITER = 14

OUT_PATH = _REPO / "docs" / "assets" / "inverse_design_convergence.png"


def evaluate(L_m: float, f_mhz: float = F_TARGET_MHZ) -> tuple[float, float, float]:
    """Run real NEC2 for a thin-wire dipole of total length L_m.

    Returns (R, X, G_max_dBi) at f_mhz. No fallback — any failure raises.
    """
    half = L_m / 2.0
    nec = necpp.nec_create()
    try:
        rc = necpp.nec_wire(
            nec, 1, SEGMENTS,
            0.0, 0.0, -half,
            0.0, 0.0, half,
            WIRE_RADIUS_M, 1.0, 1.0,
        )
        if rc != 0:
            raise RuntimeError(f"nec_wire rc={rc}: {necpp.nec_error_message()}")
        if necpp.nec_geometry_complete(nec, 0) != 0:
            raise RuntimeError("nec_geometry_complete failed")
        if necpp.nec_fr_card(nec, 0, 1, f_mhz, 0) != 0:
            raise RuntimeError("nec_fr_card failed")
        ex_seg = SEGMENTS // 2 + 1
        if necpp.nec_ex_card(nec, 0, 1, ex_seg, 0,
                             1.0, 0.0, 0.0, 0.0, 0.0, 0.0) != 0:
            raise RuntimeError("nec_ex_card failed")
        # n_theta=2, dtheta=90° → broadside sample so nec_gain_max is real
        if necpp.nec_rp_card(nec, 0, 2, 1, 0, 5, 0, 0,
                             0.0, 0.0, 90.0, 0.0, 0, 0) != 0:
            raise RuntimeError("nec_rp_card failed")
        return (
            float(necpp.nec_impedance_real(nec, 0)),
            float(necpp.nec_impedance_imag(nec, 0)),
            float(necpp.nec_gain_max(nec, 0)),
        )
    finally:
        necpp.nec_delete(nec)


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #

PHI = (math.sqrt(5.0) - 1.0) / 2.0       # ≈ 0.6180


def golden_section_search(
    f_obj,
    a: float,
    b: float,
    tol: float,
    max_iter: int,
    history: list[dict],
):
    """Standard golden-section search minimizing scalar f_obj over [a, b].

    Reuses the previous-iteration interior eval (one solver call per step
    after warm-up). `history` is appended to with one record per iteration.
    """
    L1 = b - PHI * (b - a)
    L2 = a + PHI * (b - a)
    R1, X1, G1 = f_obj(L1)
    R2, X2, G2 = f_obj(L2)
    f1, f2 = abs(X1), abs(X2)
    evals = 2

    for it in range(1, max_iter + 1):
        if f1 < f2:
            b = L2
            L2, X2_, R2_, G2_, f2 = L1, X1, R1, G1, f1
            L1 = b - PHI * (b - a)
            R1, X1, G1 = f_obj(L1)
            f1 = abs(X1)
            best_L, best_R, best_X, best_G = L1, R1, X1, G1
        else:
            a = L1
            L1, X1, R1, G1, f1 = L2, X2, R2, G2, f2
            L2 = a + PHI * (b - a)
            R2, X2, G2 = f_obj(L2)
            f2 = abs(X2)
            best_L, best_R, best_X, best_G = L2, R2, X2, G2
        evals += 1
        bracket = b - a
        history.append({
            "iter": it,
            "L": best_L,
            "R": best_R,
            "X": best_X,
            "G": best_G,
            "cost_absX": min(f1, f2),
            "bracket_m": bracket,
            "evals": evals,
        })
        if bracket < tol:
            break

    if f1 < f2:
        return (L1, R1, X1, G1), evals
    return (L2, R2, X2, G2), evals


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=== Inverse-design demo: dipole length for 300 MHz resonance ===")
    print(f"    Target  : f_res = {F_TARGET_MHZ} MHz  (minimize |X(L; f_target)|)")
    print(f"    Backend : real NEC2 via necpp — every candidate is a solver run")
    print(f"    Bracket : L ∈ [{L_LOW_M*1000:.0f}, {L_HIGH_M*1000:.0f}] mm")
    print(f"    Method  : golden-section search, tol = {TOL_M*1000:.2f} mm")
    print()
    print(f"{'iter':>4} {'L [mm]':>9} {'R [Ω]':>9} {'X [Ω]':>9} "
          f"{'G [dBi]':>9} {'|X|':>9} {'bracket [mm]':>13} {'evals':>6}")
    print("-" * 78)

    history: list[dict] = []
    t0 = time.perf_counter()
    (L_best, R_best, X_best, G_best), n_evals = golden_section_search(
        evaluate, L_LOW_M, L_HIGH_M, TOL_M, MAX_ITER, history,
    )
    elapsed = time.perf_counter() - t0

    for h in history:
        print(
            f"{h['iter']:>4} {h['L']*1000:>9.4f} {h['R']:>9.3f} "
            f"{h['X']:>+9.3f} {h['G']:>9.3f} {h['cost_absX']:>9.4f} "
            f"{h['bracket_m']*1000:>13.4f} {h['evals']:>6d}"
        )

    print()
    print("=== Best design found ===")
    print(f"  L     = {L_best*1000:.3f} mm  ({L_best:.5f} m)")
    print(f"  R     = {R_best:.3f} Ω")
    print(f"  X     = {X_best:+.3f} Ω    (target: 0 — resonance)")
    print(f"  G_max = {G_best:.3f} dBi")
    print(f"  total NEC2 calls = {n_evals}   wall time = {elapsed*1000:.1f} ms")

    # --- convergence plot ---
    iters = np.array([h["iter"] for h in history])
    L_vals = np.array([h["L"] for h in history]) * 1000.0  # mm
    cost = np.array([h["cost_absX"] for h in history])
    bracket = np.array([h["bracket_m"] for h in history]) * 1000.0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Left: candidate L per iteration with bracket shrinking
    axL.plot(iters, L_vals, "o-", color="#1f77b4", lw=1.6, ms=6,
             label="current best L")
    # Reconstruct bracket from history: each iter we replaced one endpoint,
    # so plot a±bracket/2 around L is a reasonable visualization.
    half_bracket = bracket / 2.0
    axL.fill_between(iters, L_vals - half_bracket, L_vals + half_bracket,
                     color="#1f77b4", alpha=0.12, label="±½ bracket")
    axL.axhline(L_best * 1000.0, color="green", ls="--", lw=1.0,
                label=f"converged L = {L_best*1000:.2f} mm")
    axL.set_xlabel("Iteration")
    axL.set_ylabel("Dipole length L  [mm]")
    axL.set_title("Geometry parameter convergence")
    axL.grid(True, alpha=0.3)
    axL.legend(loc="upper right", fontsize=9)

    # Right: objective convergence (log scale)
    axR.semilogy(iters, cost, "o-", color="#d62728", lw=1.6, ms=6,
                 label="|X|  [Ω]")
    axR.semilogy(iters, bracket, "s--", color="#7f7f7f", lw=1.2, ms=5,
                 label="bracket width  [mm]")
    axR.set_xlabel("Iteration")
    axR.set_ylabel("value  (log)")
    axR.set_title("Objective + bracket convergence")
    axR.grid(True, which="both", alpha=0.3)
    axR.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Inverse design: dipole L → f_res = {F_TARGET_MHZ:.0f} MHz "
        f"(real NEC2 in the loop, {n_evals} solver calls)",
        fontsize=14, fontweight="bold",
    )
    annot = (
        f"BEST DESIGN  (real NEC2)\n"
        f"  L     = {L_best*1000:.3f} mm\n"
        f"  R     = {R_best:.2f} Ω\n"
        f"  X     = {X_best:+.3f} Ω   ≈ 0 (resonant)\n"
        f"  G_max = {G_best:.2f} dBi"
    )
    fig.text(
        0.005, 0.005, annot,
        fontsize=9, family="monospace",
        bbox={"boxstyle": "round,pad=0.4",
              "facecolor": "#eef7ee", "edgecolor": "#4a8"},
        verticalalignment="bottom",
    )
    fig.text(
        0.995, 0.005,
        f"objective: |X(L; 300 MHz)|   bracket [{L_LOW_M*1000:.0f}, "
        f"{L_HIGH_M*1000:.0f}] mm   golden-section",
        fontsize=8, color="#666",
        horizontalalignment="right", verticalalignment="bottom",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"\nWrote {OUT_PATH.relative_to(_REPO)}  "
          f"({OUT_PATH.stat().st_size/1024:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
