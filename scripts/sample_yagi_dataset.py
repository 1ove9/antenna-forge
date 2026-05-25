#!/usr/bin/env python3
"""Uniform Latin-hypercube sampling of the Yagi design space (real NEC2).

The differential-evolution sweep in ``scripts/case_yagi.py`` concentrates its
evaluations in the high-performance corner of the 9-D design space — exactly
what an optimizer should do, but a poor training set for a surrogate that is
expected to be accurate *everywhere*. A surrogate fitted only on the DE history
extrapolates badly in the under-sampled regions (short elements, wide spacings,
detuned drivers).

This script fixes the sampling bias: it draws a Latin-hypercube design over a
widened-but-physical parameter box and evaluates every point with the *same*
real NEC2 (necpp MoM) solver used by the optimizer — no surrogate, no
analytical fallback, no mock. Each record stores the full 9-parameter input and
the five NEC2 outputs (G_fwd, G_back, F/B, R, X), so the resulting dataset can
be concatenated with the DE history to give the surrogate uniform coverage.

Output:
    results/yagi_uniform_sample.json

Run:
    python3 scripts/sample_yagi_dataset.py [N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.case_yagi import (  # noqa: E402
    BOUNDS,
    PARAM_NAMES,
    RESULTS_DIR,
    evaluate_yagi,
    params_to_geometry,
)

# --------------------------------------------------------------------------- #
# Sampling box
# --------------------------------------------------------------------------- #
# Reference: the DE bounds in case_yagi.py. We widen every dimension so the
# uniform sample reaches *beyond* the region the optimizer ever explored — the
# whole point is to cover the corners the DE neglected. Lengths span roughly
# 0.35..0.60 lambda (the original box was 0.38..0.55); spacings span
# 0.04..0.42 lambda (original 0.05..0.35). All values stay strictly positive and
# physically realizable as a wire antenna at 300 MHz.
SAMPLE_BOUNDS = np.array([
    [0.35, 0.60],   # L_ref
    [0.35, 0.60],   # L_drv
    [0.33, 0.55],   # L_d1
    [0.33, 0.55],   # L_d2
    [0.33, 0.55],   # L_d3
    [0.04, 0.42],   # s_ref
    [0.04, 0.40],   # s_d1
    [0.05, 0.45],   # s_d2
    [0.05, 0.45],   # s_d3
])

# A spacing below this would put two elements almost on top of each other on the
# boom — not a meaningful Yagi. With the box above no LHS point violates it, but
# we filter anyway so the validity rule is explicit and the dataset is clean.
MIN_SPACING_M = 0.02

OUTPUT_PATH = RESULTS_DIR / "yagi_uniform_sample.json"


def is_physical(p: np.ndarray) -> tuple[bool, str]:
    """Reject physically meaningless designs (negative length, overlap)."""
    lengths = p[:5]
    spacings = p[5:]
    if np.any(lengths <= 0.0):
        return False, "non-positive element length"
    if np.any(spacings <= 0.0):
        return False, "non-positive spacing"
    if np.any(spacings < MIN_SPACING_M):
        return False, f"spacing < {MIN_SPACING_M} m (elements overlap)"
    return True, ""


def latin_hypercube_points(n: int, seed: int) -> np.ndarray:
    """n LHS points scaled into SAMPLE_BOUNDS, keeping only physical designs.

    Oversamples and refills so exactly ``n`` valid points are returned even if
    the validity filter rejects some (it won't for the default box, but the
    contract holds for any box a caller might tighten).
    """
    sampler = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    lo, hi = SAMPLE_BOUNDS[:, 0], SAMPLE_BOUNDS[:, 1]
    kept: list[np.ndarray] = []
    draw = n
    while len(kept) < n:
        unit = sampler.random(draw)
        pts = scale(unit, lo, hi)
        for row in pts:
            ok, _ = is_physical(row)
            if ok:
                kept.append(row)
            if len(kept) >= n:
                break
        draw = max(16, (n - len(kept)))  # top up the remainder if any rejected
    return np.asarray(kept[:n], dtype=np.float64)


def run(n: int, seed: int) -> dict:
    print("=== Yagi uniform Latin-hypercube sampling (real NEC2 in the loop) ===")
    print(f"  requested points : {n}")
    print(f"  LHS seed         : {seed}")
    print("  sampling box (lambda units, widened vs DE bounds):")
    for name, (slo, shi), (blo, bhi) in zip(PARAM_NAMES, SAMPLE_BOUNDS, BOUNDS):
        print(f"    {name:6s} sample[{slo:.3f},{shi:.3f}]  "
              f"(DE bounds [{blo:.3f},{bhi:.3f}])")
    print()

    points = latin_hypercube_points(n, seed)

    samples: list[dict] = []
    failures: list[dict] = []
    t0 = time.perf_counter()
    for i, p in enumerate(points):
        try:
            perf = evaluate_yagi(params_to_geometry(p))
        except Exception as exc:  # necpp can raise on degenerate geometry
            failures.append({
                "params": dict(zip(PARAM_NAMES, [float(v) for v in p])),
                "error": f"exception: {exc}",
            })
            continue
        params_dict = dict(zip(PARAM_NAMES, [float(v) for v in p]))
        if not perf.converged:
            failures.append({"params": params_dict, "error": perf.error})
            continue
        samples.append({
            "params": params_dict,
            "G_fwd": perf.G_forward_dbi,
            "G_back": perf.G_backward_dbi,
            "FB": perf.FB_db,
            "R": perf.R_in_ohm,
            "X": perf.X_in_ohm,
        })
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{n} evaluated "
                  f"({len(samples)} ok, {len(failures)} failed)")
    elapsed = time.perf_counter() - t0

    print(f"\n  done: {len(samples)} succeeded, {len(failures)} failed "
          f"in {elapsed:.1f} s ({1000*elapsed/max(1,n):.2f} ms/eval)")

    return {
        "description": (
            "Uniform Latin-hypercube sample of the 9-D Yagi design space, every "
            "point evaluated with real NEC2 (necpp MoM). Complements the "
            "high-performance-biased DE history in yagi_optimized.json so a "
            "surrogate trained on the union is accurate across the whole space."
        ),
        "solver": "necpp (NEC2 Method of Moments) — real evaluation, no surrogate",
        "design_frequency_mhz": 300.0,
        "param_names": PARAM_NAMES,
        "sample_bounds": {n: [float(lo), float(hi)]
                          for n, (lo, hi) in zip(PARAM_NAMES, SAMPLE_BOUNDS)},
        "lhs_seed": seed,
        "min_spacing_m": MIN_SPACING_M,
        "n_requested": n,
        "n_success": len(samples),
        "n_failed": len(failures),
        "wall_time_sec": elapsed,
        "samples": samples,
        "failures": failures,
    }


def _range(values: list[float]) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.min()), float(a.max()), float(a.mean())


def report(payload: dict) -> None:
    samples = payload["samples"]
    keys = ["G_fwd", "G_back", "FB", "R", "X"]

    # Reference ranges from the DE-optimized history (converged evals only).
    opt_path = RESULTS_DIR / "yagi_optimized.json"
    opt_ranges: dict[str, tuple[float, float, float]] = {}
    if opt_path.exists():
        hist = json.loads(opt_path.read_text())["history"]
        conv = [h for h in hist if h.get("converged")]
        for k in keys:
            opt_ranges[k] = _range([h[k] for h in conv])
        n_opt = len(conv)
    else:
        n_opt = 0

    print("\n=== Output distributions: uniform sample vs DE-optimized history ===")
    print(f"  uniform sample : {len(samples)} converged NEC2 records")
    print(f"  DE history     : {n_opt} converged NEC2 records "
          f"(results/yagi_optimized.json)")
    print()
    hdr = (f"  {'metric':<7s} | {'uniform min':>11s} {'max':>9s} {'mean':>9s} "
           f"| {'optimized min':>13s} {'max':>9s} {'mean':>9s} | coverage")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in keys:
        umin, umax, umean = _range([s[k] for s in samples])
        if k in opt_ranges:
            omin, omax, omean = opt_ranges[k]
            wider = umin < omin and umax > omax
            note = "wider both ends" if wider else (
                "wider low" if umin < omin else
                ("wider high" if umax > omax else "within"))
            print(f"  {k:<7s} | {umin:11.3f} {umax:9.3f} {umean:9.3f} "
                  f"| {omin:13.3f} {omax:9.3f} {omean:9.3f} | {note}")
        else:
            print(f"  {k:<7s} | {umin:11.3f} {umax:9.3f} {umean:9.3f} "
                  f"| {'n/a':>13s} {'n/a':>9s} {'n/a':>9s} |")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", nargs="?", type=int, default=5000,
                    help="number of Latin-hypercube points (default 5000)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="LHS seed (default 12345; distinct from the DE seed 42)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = run(args.n, args.seed)
    with OUTPUT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> {OUTPUT_PATH.relative_to(_REPO)} "
          f"({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
    report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
