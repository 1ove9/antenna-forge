# ⚡ Source Sequence Antenna Forge (YAF)

[English](README.md) | [中文](README.zh-CN.md)

**AI-driven antenna invention platform** — automatic exploration, generation,
optimization, and verification of antenna topologies that did not exist before.

## Quick Demo — one command, a real simulation figure

The PNG below is produced by `scripts/demo_wow.py`. **Every number on it comes
from a real NEC2 run** (Method of Moments via the `necpp` Python binding) —
no mock, no analytical fallback, and if `necpp` is missing the solver raises
`SolverUnavailable` rather than fabricating output. Reproduce it with one
command:

```bash
python3 scripts/demo_wow.py    # → docs/assets/dipole_demo.png
```

![Dipole real-NEC2 demo](docs/assets/dipole_demo.png)

The three panels are: (1) input impedance R(f) / X(f) with the resonance
point (X → 0) marked, (2) E-plane polar radiation pattern with the measured
peak gain of 2.13 dBi, and (3) S11(f) / VSWR(f) with the −10 dB bandwidth
shaded. The annotation box gives the field-by-field "measured vs. textbook"
comparison (textbook half-wave dipole reference: R ≈ 73 Ω, G ≈ 2.15 dBi).

Closed-loop inverse design — putting real NEC2 inside the optimizer:

```bash
python3 scripts/demo_inverse_design.py    # → docs/assets/inverse_design_convergence.png
```

![Inverse-design convergence](docs/assets/inverse_design_convergence.png)

Golden-section search over the dipole length, 14 iterations / **16 real NEC2
solver calls / ~6 ms** wall time, converges from a ±75 mm bracket down to
**L = 477.892 mm** (≈ 0.478 λ at 300 MHz), with R = 71.85 Ω, X = +0.03 Ω,
G = 2.13 dBi — i.e. the optimizer rediscovers the textbook thin-wire
resonant length to sub-millimeter precision with no antenna theory baked
into the objective.

### Headline case study — 9-parameter Yagi-Uda inverse design

The platform's flagship demonstration: a 5-element Yagi-Uda at 300 MHz with
**9 continuous design parameters** (5 element lengths + 4 inter-element
spacings), driven by `scipy.optimize.differential_evolution` and evaluated
by **real NEC2 in every single iteration** — 5858 solver calls, 12.7 s wall
time on a laptop. Full write-up:
[`docs/case_study_yagi.md`](docs/case_study_yagi.md).

```bash
python3 scripts/case_yagi.py     # baseline + optimization → JSON in results/
python3 scripts/plot_yagi.py     # → docs/assets/yagi_design.png
```

![Yagi-Uda inverse design](docs/assets/yagi_design.png)

**Clean 5-vs-5 contest (same element count, same NEC2 backend):**

| Quantity | Viezbicke 5-elem (NBS TN 688) | YAF AI 5-elem | Δ |
|---|---|---|---|
| Forward gain G_fwd | +11.03 dBi | **+12.63 dBi** | **+1.60 dB** |
| Front-to-back F/B | 13.79 dB | **15.00 dB** | **+1.21 dB** |
| Boom length | 1.00 λ | 1.17 λ | +0.17 λ |
| Element count | 5 | 5 | 0 (clean attribution) |

**The AI design Pareto-dominates the canonical Viezbicke 5-element
reference on both gain *and* F/B simultaneously**, with the same number
of elements. Across the broader published 5-element design space
(Viezbicke, ARRL Handbook, DL6WU, Lawson/Cebik) the AI strictly
dominates 3 of 4 on both axes (see `docs/case_study_yagi.md` §6).

The optimizer *independently* recovers Viezbicke-style director tapering
(L: 0.440 → 0.434 → 0.429 m, monotonically decreasing rear-to-front) and
a balanced 0.243 λ reflector spacing — both consistent with published
5-element Yagi recipes — without being told anything about antenna design.
The "AI" part of the loop is just differential evolution; the unique
platform contribution is *what it's optimizing against*: real
Method-of-Moments physics, not a surrogate or analytical model.

> *Bonus: the 5858-record DE history (`results/yagi_optimized.json`) is a
> free FNO-surrogate training set — every input geometry and output
> (R, X, G_fwd, G_back, F/B) is recorded, ready for whoever wants to try
> active-learning DE in a later phase.*

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Web UI (React + Three.js + WebGPU)                             │
│  3D editor │ Design browser │ Experiment tracking │ Live monitor│
└────────────────────────┬────────────────────────────────────────┘
                         │ REST / WebSocket
┌────────────────────────▼────────────────────────────────────────┐
│  API Gateway (FastAPI + Pydantic)                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Orchestration Core (Python asyncio + Celery)                   │
└─────┬────────────┬───────────┬─────────────┬──────────────┬─────┘
      │            │           │             │              │
┌──────────┐  ┌────────┐  ┌─────────┐  ┌───────────┐  ┌───────────┐
│ Geometry │  │ AI     │  │ Solver  │  │ Optimizer │  │ Post-proc │
│ Kernel   │  │ Engine │  │ Adapter │  │ Engine    │  │ Analyzer  │
└──────────┘  └────────┘  └─────────┘  └───────────┘  └───────────┘
```

## Quick start

```bash
# Clone the project
git clone https://github.com/1ove9/antenna-forge.git yaf && cd yaf

# Copy the env template
cp .env.example .env

# Start all services
docker compose up -d

# Health check
curl http://localhost:8000/health
# → {"status": "ok", "version": "0.1.0"}

# Open the frontend
open http://localhost:5173
```

## Core modules

| Module | Path | Description |
|--------|------|-------------|
| Domain models | `yaf_core/domain/` | Design, Geometry, Simulation, Optimization |
| Port protocols | `yaf_core/ports/` | SolverAdapter, AIBackend, CADBackend |
| Geometry kernel | `yaf_core/geometry/` | OpenCASCADE, parametric generators, SIREN, topology optimization |
| Physics models | `yaf_core/physics/` | Metasurfaces, RIS, OAM, graphene, space-time modulation |
| Solvers | `yaf_solvers/` | openEMS, NEC2, MEEP, HFSS, CST, FEKO |
| AI engine | `yaf_ai/` | Diffusion, VAE, GAN, FNO, PINN, differentiable FDTD, Bayesian optimization |
| API service | `yaf_api/` | FastAPI + WebSocket |
| Task queue | `yaf_worker/` | Celery + Redis |
| Database | `yaf_db/` | PostgreSQL + Qdrant |
| Frontend | `frontend/` | React 18 + Three.js + TypeScript |

## API quick start

```bash
# Create a design
curl -X POST http://localhost:8000/api/v1/designs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_dipole",
    "frequency_range": [2.4e9, 2.5e9],
    "size_constraint": {"x_min": -0.1, "x_max": 0.1, "y_min": -0.1, "y_max": 0.1, "z_min": -0.1, "z_max": 0.1},
    "polarization": "linear",
    "material_palette": ["copper"]
  }'

# Simulate with NEC2
curl -X POST http://localhost:8000/api/v1/simulations \
  -H "Content-Type: application/json" \
  -d '{"design_id": "<design-uuid>", "solver": "nec2", "frequency_min": 2400000000, "frequency_max": 2500000000}'
```

## AI demos

```bash
# Differentiable FDTD gradient optimization
python -m yaf_ai.differentiable.diff_fdtd_jax --demo

# VAE antenna-geometry generation (--epochs 2 already triggers a weight save; 20 is the default convergence run)
python -m yaf_ai.generative.vae_designer --train --epochs 20

# Bayesian optimization
python -m yaf_ai.optimization.bayesian --demo

# End-to-end inverse-design pipeline
python -m yaf_ai.inverse_design.pipeline --demo

# End-to-end half-wave dipole example (NEC2 → S11 + gain)
python scripts/demo_dipole.py
```

> See `docs/HONEST_STATUS.md`: **neither solver fabricates results.** NEC2
> (`necpp`) and openEMS both run real solvers and raise `SolverUnavailable`
> when their backend is missing — there is no silent analytical fallback on
> either path.

## Acceptance commands

The project's acceptance commands: 6 core commands (infrastructure, tests,
differentiable FDTD, generative models, demo) plus the real-NEC2 truth checks
and the Yagi case-study demo. The repository currently passes all of the below:

```bash
# 1. Infrastructure boot
docker compose up -d                                          # postgres / redis / minio / qdrant / api

# 2. API health check
curl -fsS http://localhost:8000/health                        # → 200 {"status":"ok","version":"0.1.0"}

# 3. Test suite (includes real-NEC2 half-wave dipole assertion)
pytest tests/ -x -q                                           # → all passed

# 4. Differentiable FDTD — gradient flow proof
python -m yaf_ai.differentiable.diff_fdtd_jax --demo          # → "✓ Gradient flow verified" + monotone loss

# 5. VAE training + weights checkpoint
python -m yaf_ai.generative.vae_designer --train --epochs 2   # → models/vae_designer.pt written

# 6. Dipole demo — S11 + gain (real NEC2)
python scripts/demo_dipole.py                                 # → S11/VSWR/Peak gain printed (2.20 dBi)

# 7. NEC2 truth check vs textbook
python3 scripts/verify_dipole.py                              # → PASS: R=68.30 Ω (err 6.4%), G=2.12 dBi

# 8. openEMS truth check — real full-wave FDTD vs cavity model
python3 scripts/verify_patch.py                               # → PASS: f_res 2.435 GHz vs 2.513 GHz (err 3.1%)

# 9. 3-panel showcase PNG (Z sweep / polar pattern / S11+BW)
python3 scripts/demo_wow.py                                   # → docs/assets/dipole_demo.png

# 10. Closed-loop inverse design (real NEC2 in the loop)
python3 scripts/demo_inverse_design.py                        # → 477.89 mm + inverse_design_convergence.png

# 11. Yagi-Uda case study — 9-param DE × real NEC2
python3 scripts/case_yagi.py                                  # → baselines + opt JSON, +1.60 dB Pareto-dominant vs Viezbicke
python3 scripts/plot_yagi.py                                  # → docs/assets/yagi_design.png

# Static type check
mypy yaf_core yaf_ai yaf_solvers --strict                     # → Success: no issues in 64 source files
```

A per-command credibility annotation lives in `docs/HONEST_STATUS.md`
(revised 2026-05-25); "what's still missing once everything is green" is in
`docs/next-steps.md`; the full Yagi case-study walkthrough is in
`docs/case_study_yagi.md`.

## Tech stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Pydantic v2 |
| Differentiable | JAX, Flax, Optax |
| Deep learning | PyTorch 2.x |
| Geometry | pythonocc-core, trimesh, gmsh |
| Task queue | Celery + Redis |
| Database | PostgreSQL 16, Qdrant (vector) |
| Object storage | MinIO (S3-compatible) |
| Frontend | React 18, TypeScript, Vite, Three.js |
| Deployment | Docker Compose (dev), Kubernetes (prod) |


## License

YAF source code is distributed under the **MIT License** — see
[`LICENSE`](LICENSE).

**Third-party dependencies carry their own licenses, some of them
copyleft.** In particular, the optional `necpp` Method-of-Moments
backend and the `openEMS` / `CSXCAD` FDTD backend are GPL-licensed.
YAF does not bundle or redistribute any of them; users install
them separately and assume the combined-work obligations that may
result. See [`NOTICE`](NOTICE) for the full license-boundary
discussion and mitigations for downstream redistributors. This is
provided in good faith and is not legal advice.


## Open-core model

YAF follows an **open-core** model. This repository is the **core engine**:
free, self-hostable, and MIT-licensed. It covers **wire antennas** (NEC2
Method-of-Moments) and **planar / patch antennas** (openEMS full-wave FDTD),
driven by **classical optimization** (differential evolution / golden-section
search), and it is complete and useful on its own for that scope.

**Available now in this open-source core**

- Wire-antenna simulation with real NEC2 Method-of-Moments via `necpp` — no
  analytical fallback (missing solver raises rather than fabricates).
- Full-wave FDTD simulation with real openEMS (`openEMS` / `CSXCAD` Python
  bindings): builds the CSX structure, runs the time-domain solve, and extracts
  S11 / input impedance from the port and the gain pattern via NF2FF. Same
  honesty rule — missing bindings raise `SolverUnavailable`, never fabricate.
- Classical optimization with the real solver inside every iteration:
  half-wave dipole resonance search and the 9-parameter Yagi-Uda inverse
  design.
- Known-answer truth checks and reproducible benchmarks: half-wave dipole
  (`scripts/verify_dipole.py`, NEC2) and a rectangular microstrip patch
  (`scripts/verify_patch.py`, openEMS — simulated resonance within 3.1 % of the
  cavity-model prediction). See also `docs/case_study_yagi.md`.
- FastAPI service, Pydantic domain models, and the solver / AI adapter
  interfaces.

Source Sequence maintains a separate **enhanced edition** — a hosted /
commercial product with an embedded web platform — for professional and
commercial users. To set expectations honestly, the capabilities below are
**planned / on the roadmap; they are *not yet shipped*, in either the
open-source core or the enhanced edition.**

**Planned / on the roadmap (not yet available)**

- Broader full-wave coverage — microstrip arrays, metasurfaces, and full 3-D
  structures, plus commercial solvers (HFSS / CST / FEKO / COMSOL). *(The
  openEMS FDTD backend is real and validated on a single patch-antenna
  truth check today; wider geometry coverage and the commercial-solver
  adapters are still on the roadmap — see `docs/HONEST_STATUS.md`.)*
- Generative AI geometry design (diffusion / VAE) connected to a real physics
  oracle. *(These generative models exist in the repo today only as
  **early / experimental** code: trained on synthetic geometry, not yet wired
  into a simulation loop. They are not production-ready.)*
- Multi-objective, multi-band joint optimization.
- RIS (reconfigurable intelligent surface) inverse design.
- In-browser visual design platform.
- Cloud compute — run designs without installing a solver locally.
- Team collaboration and design version management.

In short: **the open-source core lets you validate the method and reproduce
the benchmarks; the enhanced edition is aimed at taking that into real
engineering projects.** Nothing on the roadmap above is implied to work today.

- A commercial enhanced edition is in development; details will be announced.
- For commercial inquiries, please open a GitHub issue for now.


## Acknowledgements

This project was built by a single engineer, using AI coding assistants
to help with implementation. The architecture, the physics-validation
methodology (the real-NEC2 truth checks and known-answer regressions),
and the benchmark design (the 5-vs-5 Yagi comparison and the honesty
tiers in `docs/HONEST_STATUS.md`) are my own. Where a module's design is
informed by an open-source project, that project is cited at the top of
the file and in `NOTICE`.
