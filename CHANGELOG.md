# Changelog

All notable changes to Source Sequence Antenna Forge (YAF) are recorded in this
file. The format roughly follows [Keep a Changelog](https://keepachangelog.com/),
adapted for a research-oriented codebase: grouped summaries with the
underlying commits referenced by short hash.

This release tagging has **not** yet been applied to the repository.
This document is a release-candidate draft for `v0.1.0`.

## [Unreleased / v0.1.0-rc]

Status: release candidate. Tagging and any remote-publication decisions
are deferred to a maintainer review pass.

### Highlights

- **Real Method-of-Moments simulation end-to-end.** The NEC2 adapter
  now drives the in-process `necpp` Python binding directly; the
  half-wave dipole truth check (300 MHz) matches textbook values
  (R = 68.30 Ω vs 73 Ω, gain 2.12 dBi vs 2.15 dBi) to within published
  thin-wire tolerances.
- **Real full-wave FDTD simulation end-to-end.** The openEMS adapter
  now drives the `openEMS` / `CSXCAD` Python bindings directly (build CSX
  structure → mesh refinement → lumped-port excitation → time-domain
  run → S11/Zin from the port, gain pattern from NF2FF). The rectangular
  microstrip patch truth check (`scripts/verify_patch.py`) puts the
  simulated resonance within 3.1 % of the cavity-model prediction
  (2.435 GHz vs 2.513 GHz, S11 dip −27 dB).
- **AI × real solver inverse design.** A 9-parameter Yagi-Uda case
  study (`scripts/case_yagi.py`) drives `scipy.optimize.differential_evolution`
  with NEC2 in every objective evaluation (5858 calls in 12.7 s wall
  time) and Pareto-dominates the Viezbicke NBS TN 688 5-element
  reference design on both forward gain (+1.60 dB) and front-to-back
  ratio (+1.21 dB). See `docs/case_study_yagi.md`.
- **Three reproducible demo figures** in `docs/assets/` (dipole sweep,
  inverse-design convergence, Yagi case study) — all produced by
  single-command scripts that run real solver code only.
- **Honest status documentation.** `docs/HONEST_STATUS.md` and
  `DECISIONS.md` record exactly which AI modules are wired into a real
  physics oracle (🟢), demo-only (🟡), or placeholder / dead code (🔴),
  plus the pending legal-review items (GPL boundary with necpp).

### Added

- `yaf_solvers/nec2_adapter/adapter.py` — real `necpp` MoM backend
  driving `nec_create / nec_wire / nec_geometry_complete / nec_fr_card
  / nec_ex_card / nec_rp_card / nec_impedance_{real,imag} / nec_gain`.
  `FarFieldResult.e_theta` now carries the real per-direction NEC2
  pattern (not an analytic placeholder).
- `yaf_solvers/base.py::SolverUnavailable` — explicit exception for
  missing solver backends. Adapters MUST raise this rather than
  fabricate results.
- `scripts/verify_dipole.py` — NEC2 truth check: half-wave dipole
  at 300 MHz, sweeps length around resonance, asserts R within ±15 %
  of 73 Ω and gain in [1.9, 2.4] dBi.
- `scripts/demo_wow.py` — single-PNG showcase: impedance
  sweep + polar pattern + S11/VSWR with –10 dB bandwidth on one
  figure. Output at `docs/assets/dipole_demo.png`.
- `scripts/demo_inverse_design.py` — closed-loop demo:
  golden-section search converges a dipole length to 477.892 mm in
  16 real-NEC2 calls (~6 ms). Output at
  `docs/assets/inverse_design_convergence.png`.
- `scripts/case_yagi.py` + `scripts/plot_yagi.py` — flagship
  case study with the four published 5-element Yagi baselines
  (Viezbicke / ARRL / DL6WU / Lawson-Cebik) plus the AI 5-elem
  design. `results/yagi_optimized.json` contains the full 5858-record
  DE history (a free FNO surrogate training set).
- `docs/case_study_yagi.md` — engineer-credible 10-section write-up
  of the Yagi case study.
- `NOTICE` — third-party license-boundary document (GPL caveats with
  necpp / openems).
- `CONTRIBUTING.md`, `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`
  — standard open-source contributor onboarding scaffolding.

### Changed

- `yaf_solvers/nec2_adapter/adapter.py` no longer silently falls back
  to a hand-coded induced-EMF analytical model when the solver is
  unavailable. Missing `necpp` → `SolverUnavailable`; empty mesh →
  `SolverError`. The previous "always returns gain = 2.15 dBi" code
  path has been removed.
- `tests/unit/test_solvers.py` and `tests/integration/test_pipeline.py`
  updated to (a) assert that empty geometry raises `SolverError` and
  (b) verify a real 300 MHz / 2.45 GHz half-wave dipole against
  textbook impedance to ±15 %.
- `scripts/demo_dipole.py` geometry replaced from a 4-vertex /
  2-triangle mesh (which only "worked" via the deleted fallback) to a
  single clean wire; now reports a consistent 2.20 dBi from both
  `result.gain_dbi` and `result.far_field.gain_dbi()`.
- `docs/HONEST_STATUS.md` rewritten to use a three-tier AI module
  reality table (🟢 real / 🟡 demo only / 🔴 placeholder/dead). Adds
  a new §0a flagging the necpp GPL combined-work question as the
  pre-release legal item.
- `README.md` headline case study now reports the *clean 5-vs-5*
  comparison against Viezbicke (Pareto-dominant +1.60 dB / +1.21 dB)
  rather than the apples-to-oranges 3-vs-5 result. Quick Demo block
  is bilingual (English first, Chinese in a `<details>` block).
- Acceptance command block now includes the NEC2 truth checks and
  case-study demos alongside the original six commands.

### Removed

- The silent analytical-fallback path in `yaf_solvers/nec2_adapter/adapter.py`
  (commit `efe5cc0`). It returned hardcoded gain = 2.15 dBi regardless
  of input geometry — actively misleading.
- The analytical `cos(π/2 cos θ) / sin θ` placeholder pattern in
  `FarFieldResult.e_theta` (commit `d9a0b26`); replaced with real
  per-direction NEC2 gain values.
- The analytical RLC fallback in `yaf_solvers/openems_adapter/adapter.py`
  (the `s11 = detuning / (detuning + 1j·0.1)` placeholder and the dead
  `import openems` path that never imported). Replaced with a real
  openEMS full-wave FDTD path; missing `openEMS` / `CSXCAD` bindings now
  raise `SolverUnavailable`.

### Known limitations (carried into next release)

- The `yaf_solvers/openems_adapter/` adapter is now a real openEMS
  full-wave FDTD backend, but its truth coverage is a single known-answer
  case (patch resonance); reported gain is the NF2FF directivity
  (`10·log10(Dmax)`, not a mismatch-discounted realized gain), and only
  single-port lumped excitation has been validated.
- `yaf_ai/inverse_design/pipeline.py` is the six-stage
  generate/screen/refine/topo/verify/score framework, but its `verify`
  step expects 2D voxel geometries that the NEC2 adapter correctly
  refuses for wire antennas; the Yagi case study deliberately bypasses
  pipeline.py for that reason. Wiring pipeline.py to wire-antenna
  geometries is a follow-up task.
- `yaf_ai/surrogate/fno_solver.py` and `deeponet.py` remain untrained
  scaffolding. `DECISIONS.md` ADR-013 explains why this release did not
  attempt a half-baked integration; the 5858-record DE history from
  `case_yagi.py` is a ready-made training set for whoever picks this
  up next.
- mypy `--strict` passes but with documented per-module
  `ignore_missing_imports` overrides for stub-less third-party libs
  (OCC, openems, CSXCAD, necpp, …); see `docs/HONEST_STATUS.md` §4.
- Acceptance test count is "all pass", **not** "covers every code
  path". Tests are smoke + structure for most modules, with explicit
  physics-truth assertions only on the NEC2 dipole / Yagi paths.

### Compute / reproducibility

All numbers reported in `docs/case_study_yagi.md` were generated on a
stock laptop (Linux, single thread) running the exact scripts in
`scripts/`, which print their verbatim stdout for reproduction. Seeded
operations (`differential_evolution`, torch / jax training demos) use
deterministic seeds where applicable; exact reproducibility within
±1e-9 is **not** guaranteed across different BLAS / OpenMP / scipy
versions.

## [0.1.0] — not yet released

When tagged, this section will be promoted from `[Unreleased]`. Tagging,
remote publication, and repository-visibility changes are deliberately
left as manual maintainer steps and are *not* performed by any script in
this repository.
