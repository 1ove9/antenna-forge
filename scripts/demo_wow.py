#!/usr/bin/env python3
"""Wow-moment demo: one PNG that proves the NEC2 path is real.

Builds a thin-wire half-wave dipole (L = 0.47 m, 21 segments, r = 0.5 mm),
drives it through the real `necpp` Python binding (no mock, no analytical
fallback), sweeps 200–400 MHz, then renders three subplots into
`docs/assets/dipole_demo.png`:

  (1) Input impedance R(f), X(f) with the resonance frequency (X → 0) marked.
  (2) E-plane polar radiation pattern at the resonance frequency, with the
      true NEC2 peak gain labeled.
  (3) S11(f) in dB plus VSWR(f), with the −10 dB bandwidth marked.

Title is explicit: "Real NEC2 simulation, not mock". A corner annotation
shows measured-vs-theory for R(73 Ω), gain(2.15 dBi), and resonance.

Run:
    python3 scripts/demo_wow.py
"""

from __future__ import annotations

import sys
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
        "necpp not installed — cannot run real NEC2 demo. "
        "Install with: pip install necpp --break-system-packages"
    ) from e


# --------------------------------------------------------------------------- #
# Antenna + sweep parameters
# --------------------------------------------------------------------------- #

L_M = 0.47                     # total dipole length [m]
HALF = L_M / 2
SEGMENTS = 21
WIRE_RADIUS_M = 0.0005
EX_SEG = SEGMENTS // 2 + 1     # middle segment (feed point) — 11 of 21

FREQS_MHZ = np.linspace(200.0, 400.0, 81)   # 2.5 MHz resolution
Z0 = 50.0
F_TARGET_MHZ = 300.0
THETA_DEG_PATTERN = np.arange(0.0, 181.0, 5.0)   # 37 samples, 5° steps

OUT_PATH = _REPO / "docs" / "assets" / "dipole_demo.png"


# --------------------------------------------------------------------------- #
# NEC2 driver helpers — direct necpp calls so we KNOW it's not a fake path
# --------------------------------------------------------------------------- #

def _build_dipole(nec) -> None:
    rc = necpp.nec_wire(
        nec, 1, SEGMENTS,
        0.0, 0.0, -HALF,
        0.0, 0.0, HALF,
        WIRE_RADIUS_M, 1.0, 1.0,
    )
    if rc != 0:
        raise RuntimeError(f"nec_wire failed rc={rc}: {necpp.nec_error_message()}")
    rc = necpp.nec_geometry_complete(nec, 0)
    if rc != 0:
        raise RuntimeError(f"nec_geometry_complete failed rc={rc}")


def _excite_and_request_pattern(nec, n_theta: int, dtheta: float) -> None:
    rc = necpp.nec_ex_card(nec, 0, 1, EX_SEG, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if rc != 0:
        raise RuntimeError(f"nec_ex_card failed rc={rc}")
    rc = necpp.nec_rp_card(
        nec, 0, n_theta, 1, 0, 5, 0, 0,
        0.0, 0.0, dtheta, 0.0, 0, 0,
    )
    if rc != 0:
        raise RuntimeError(f"nec_rp_card failed rc={rc}")


def run_point(f_mhz: float, *, n_theta: int = 1, dtheta: float = 10.0):
    """One frequency point. Returns (R, X, gain_max, per_theta_gains)."""
    nec = necpp.nec_create()
    try:
        _build_dipole(nec)
        rc = necpp.nec_fr_card(nec, 0, 1, f_mhz, 0)
        if rc != 0:
            raise RuntimeError(f"nec_fr_card failed rc={rc}")
        _excite_and_request_pattern(nec, n_theta, dtheta)
        R = float(necpp.nec_impedance_real(nec, 0))
        X = float(necpp.nec_impedance_imag(nec, 0))
        G_max = float(necpp.nec_gain_max(nec, 0))
        per_theta = [float(necpp.nec_gain(nec, 0, i, 0)) for i in range(n_theta)]
        return R, X, G_max, per_theta
    finally:
        necpp.nec_delete(nec)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    print(f"=== Real NEC2 dipole demo (L={L_M} m, {SEGMENTS} seg, "
          f"r={WIRE_RADIUS_M*1000:.2f} mm) ===")
    print(f"Sweeping {len(FREQS_MHZ)} freqs from {FREQS_MHZ[0]:.0f} "
          f"to {FREQS_MHZ[-1]:.0f} MHz...")

    R_arr = np.zeros_like(FREQS_MHZ)
    X_arr = np.zeros_like(FREQS_MHZ)
    G_arr = np.zeros_like(FREQS_MHZ)
    # n_theta=2 at dtheta=90° → samples θ=0° (pole) and θ=90° (broadside).
    # broadside is where dipole peak gain lives, so nec_gain_max here is real.
    for i, f in enumerate(FREQS_MHZ):
        R, X, Gm, _ = run_point(f, n_theta=2, dtheta=90.0)
        R_arr[i] = R
        X_arr[i] = X
        G_arr[i] = Gm
        if i % 10 == 0 or i == len(FREQS_MHZ) - 1:
            print(f"  f={f:6.1f} MHz   R={R:7.2f}   X={X:+7.2f}   G={Gm:+.2f} dBi")

    # Find resonance: linearly interpolate X → 0
    sign = np.sign(X_arr)
    cross = np.where(np.diff(sign) != 0)[0]
    if len(cross) == 0:
        f_res_mhz = float(FREQS_MHZ[np.argmin(np.abs(X_arr))])
        print(f"  (no clean X-zero crossing; nearest point {f_res_mhz:.1f} MHz)")
    else:
        i0 = int(cross[0])
        f0, f1 = FREQS_MHZ[i0], FREQS_MHZ[i0 + 1]
        x0, x1 = X_arr[i0], X_arr[i0 + 1]
        f_res_mhz = float(f0 - x0 * (f1 - f0) / (x1 - x0))
    print(f"Resonance (X=0) ≈ {f_res_mhz:.2f} MHz")

    # R, gain at the closest swept frequency to resonance
    idx_res = int(np.argmin(np.abs(FREQS_MHZ - f_res_mhz)))
    R_at_res = float(R_arr[idx_res])
    G_at_res = float(G_arr[idx_res])

    # Full pattern at resonance frequency
    n_theta = len(THETA_DEG_PATTERN)
    dtheta = float(THETA_DEG_PATTERN[1] - THETA_DEG_PATTERN[0])
    _, _, G_peak, per_theta = run_point(
        f_res_mhz, n_theta=n_theta, dtheta=dtheta
    )
    pattern_db = np.array(per_theta)
    # NEC sentinel for unreachable directions (poles, etc.) is -999
    valid = pattern_db > -100
    G_floor = float(pattern_db[valid].min()) - 5.0
    pattern_plot = np.where(valid, pattern_db, G_floor)
    print(f"Peak gain at resonance: {G_peak:.3f} dBi")

    # S11 / VSWR
    Z = R_arr + 1j * X_arr
    gamma = (Z - Z0) / (Z + Z0)
    s11_db = 20.0 * np.log10(np.clip(np.abs(gamma), 1e-12, None))
    vswr = (1.0 + np.abs(gamma)) / np.maximum(1.0 - np.abs(gamma), 1e-9)

    # -10 dB bandwidth
    below_10 = FREQS_MHZ[s11_db < -10.0]
    bw_lo, bw_hi, bw_pct = None, None, None
    if len(below_10) > 0:
        bw_lo, bw_hi = float(below_10.min()), float(below_10.max())
        bw_pct = (bw_hi - bw_lo) / f_res_mhz * 100.0

    # --- plot --- #
    # Reserve explicit top band for the suptitle and a bottom band for the
    # "Measured vs Textbook" annotation so neither overlaps any axis label.
    fig = plt.figure(figsize=(17.0, 7.2))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 0.85, 1.0],
        wspace=0.32,
        left=0.05, right=0.97, top=0.82, bottom=0.20,
    )

    # subplot 1: Impedance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(FREQS_MHZ, R_arr, color="#1f77b4", lw=2.0, label="R (resistance)")
    ax1.plot(FREQS_MHZ, X_arr, color="#d62728", lw=2.0, label="X (reactance)")
    ax1.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax1.axvline(f_res_mhz, color="green", ls="--", lw=1.2,
                label=f"f_res ≈ {f_res_mhz:.1f} MHz")
    ax1.scatter([f_res_mhz], [0], color="green", zorder=5, s=40)
    ax1.set_xlabel("Frequency [MHz]")
    ax1.set_ylabel("Impedance [Ω]")
    ax1.set_title("(1) Input impedance Z = R + jX")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # subplot 2: polar pattern
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    theta_full = np.concatenate([THETA_DEG_PATTERN, 360.0 - THETA_DEG_PATTERN[::-1][1:]])
    pat_full = np.concatenate([pattern_plot, pattern_plot[::-1][1:]])
    ax2.plot(np.deg2rad(theta_full), pat_full, color="#1f77b4", lw=2.0)
    ax2.fill(np.deg2rad(theta_full), pat_full, color="#1f77b4", alpha=0.15)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    rmin = float(np.floor(pattern_plot[valid].min() / 5.0) * 5.0)
    rmax = float(np.ceil(G_peak / 2.0) * 2.0 + 1.0)
    ax2.set_ylim(rmin, rmax)
    ax2.set_rlabel_position(135)
    ax2.set_title(f"(2) E-plane pattern\n"
                  f"peak G = {G_peak:.2f} dBi  @ θ = 90°",
                  pad=10, fontsize=11)

    # subplot 3: S11 / VSWR
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(FREQS_MHZ, s11_db, color="#1f77b4", lw=2.0, label="S11 [dB]")
    ax3.axhline(-10.0, color="gray", ls=":", lw=1.0, label="−10 dB threshold")
    ax3.set_xlabel("Frequency [MHz]")
    ax3.set_ylabel("S11 [dB]", color="#1f77b4")
    ax3.tick_params(axis="y", labelcolor="#1f77b4")
    ax3.grid(True, alpha=0.3)

    ax3b = ax3.twinx()
    ax3b.plot(FREQS_MHZ, vswr, color="#d62728", lw=1.5, ls="--", label="VSWR")
    ax3b.set_ylabel("VSWR", color="#d62728")
    ax3b.tick_params(axis="y", labelcolor="#d62728")
    ax3b.set_ylim(1, min(10, float(np.nanmax(vswr)) * 1.1 + 0.5))

    if bw_lo is not None:
        ax3.axvspan(bw_lo, bw_hi, color="green", alpha=0.12)
        ax3.set_title(
            f"(3) Return loss & VSWR\n"
            f"−10 dB BW: {bw_lo:.1f}–{bw_hi:.1f} MHz "
            f"({bw_pct:.1f} % rel.)"
        )
    else:
        ax3.set_title("(3) Return loss & VSWR\n(no −10 dB band in scan)")

    # combined legend
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)

    # Top title sits in the top band we reserved above (top=0.82).
    fig.suptitle(
        "Real NEC2 simulation, not mock — thin-wire half-wave dipole",
        fontsize=15, fontweight="bold", y=0.94,
    )
    annot = (
        f"Measured (necpp MoM)        vs       Textbook\n"
        f"  R(f_res) = {R_at_res:6.2f} Ω         ≈  73 Ω\n"
        f"  G_max    = {G_peak:5.2f} dBi        ≈  2.15 dBi\n"
        f"  f_res    = {f_res_mhz:6.2f} MHz       ≈  300 MHz (L = λ/2)"
    )
    # Annotation lives in the bottom band (bottom=0.22 reserved). Anchored
    # below the subplot row so it does not clip "Frequency [MHz]" labels.
    fig.text(
        0.05, 0.02, annot,
        fontsize=9, family="monospace",
        bbox={"boxstyle": "round,pad=0.4",
              "facecolor": "#fafafa", "edgecolor": "#888"},
        verticalalignment="bottom",
    )
    fig.text(
        0.97, 0.02,
        f"L = {L_M} m    {SEGMENTS} segments    "
        f"r = {WIRE_RADIUS_M*1000:.2f} mm    backend: necpp",
        fontsize=9, color="#555",
        horizontalalignment="right", verticalalignment="bottom",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: do NOT pass bbox_inches="tight" — it overrides our explicit
    # subplots_adjust margins and the annotation re-clips the xlabels.
    fig.savefig(OUT_PATH, dpi=130)
    print(f"\nWrote {OUT_PATH.relative_to(_REPO)}  "
          f"({OUT_PATH.stat().st_size/1024:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
