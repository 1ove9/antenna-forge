# YAF Architecture Decision Records (ADR)

## ADR-001: Pydantic v2 domain models over dataclasses
**Decision**: All domain entities use Pydantic v2 `BaseModel`.
**Rationale**: JSON Schema auto-generation for FastAPI, strict validation at creation, serialization to JSONB for PostgreSQL. Pydantic v2 is 5-30x faster than v1.

## ADR-002: Protocol over ABC for adapters
**Decision**: SolverAdapter, AIBackend, CADBackend use `typing.Protocol` with `@runtime_checkable`.
**Rationale**: Plugins can satisfy the protocol without inheriting from a base class, enabling hot-loading via `plugin.toml`.

## ADR-003: JAX over PyTorch for differentiable FDTD
**Decision**: `yaf_ai/differentiable/` uses JAX + Flax + Optax.
**Rationale**: JAX provides native autodiff through control flow (jax.lax.scan for time-stepping), XLA compilation, and seamless GPU/TPU scaling.

## ADR-004: scikit-rf as mandatory RF dependency
**Decision**: S-parameter handling, Touchstone I/O, and network analysis use scikit-rf when available.
**Rationale**: scikit-rf is the de facto Python standard for RF/microwave engineering.

## ADR-005: Structured logging with structlog over standard logging
**Decision**: All modules use structlog with design_id/job_id/solver bound variables.

## ADR-006: In-memory stores for API demo, PostgreSQL for production
**Decision**: API routers use dict-based stores by default, SQLAlchemy models ready for production.

## ADR-007: Analytical fallback for solver adapters (SUPERSEDED 2026-05-25)
**Original decision (2026-05-15)**: When solver executables are unavailable, adapters compute results analytically (induced EMF for dipoles, array factor for metasurfaces).
**Superseded**: Both production solver paths now call real solvers and never fabricate results. NEC2 moved to the `necpp` Method-of-Moments binding (2026-05-24) and openEMS to the `openEMS` / `CSXCAD` full-wave FDTD bindings (2026-05-25); each removed its analytical fallback entirely, and a missing backend now raises `SolverUnavailable` instead of returning a plausible-looking `SimulationResult`. Known-answer regressions live in `scripts/verify_dipole.py` (NEC2, ~73 Ω half-wave dipole) and `scripts/verify_patch.py` (openEMS, microstrip patch resonance within 3.1 % of the cavity model). The remaining skeleton adapters (MEEP / HFSS / CST / FEKO / COMSOL) return an explicit `skeleton_not_implemented` status rather than a fabricated result. See `docs/HONEST_STATUS.md` §1.1–1.2.

## ADR-008: uv for Python, pnpm for frontend
**Decision**: Package management split: uv for Python dependencies, pnpm for Node.js.

## ADR-009: No separate RPC framework at v0.1
**Decision**: REST for CRUD, WebSocket for streaming. No gRPC yet.

## ADR-010: SIMP density method for topology optimization
**Decision**: Density-based SIMP with optimality criteria update.

## ADR-011: VAE latent space for design generation default
**Decision**: Pipeline default uses VAE for candidate generation. Diffusion is optional upgrade.

## ADR-012: Yagi case study uses scipy `differential_evolution` (DE), not in-house BO/NSGA
**Decision**: `scripts/case_yagi.py` drives the 9-parameter Yagi-Uda optimization loop with `scipy.optimize.differential_evolution`. The in-house `yaf_ai/optimization/bayesian.py` and `yaf_ai/optimization/nsga.py` modules are *not* used as the primary optimizer here.
**Rationale**:
1. **Dimensionality.** 9 continuous parameters in a non-convex landscape with multiple local optima (every reasonable Yagi has a sub-optimal nearby basin). DE is the textbook strong baseline for box-constrained 9-D black-box optimization; BO's random-search acquisition (1000 candidates) collapses badly in this regime.
2. **Single scalar target.** The problem statement is "maximize forward gain subject to F/B ≥ 15 dB", which scalarizes cleanly into a single composite cost. NSGA-II is the right tool when a Pareto front is genuinely wanted, not when there is a constraint to enforce.
3. **Reproducibility.** SciPy DE is widely benchmarked and battle-tested; auditors of the case study can replicate the numbers without running unfamiliar in-house code.
4. **What in-house modules *are* used.** The case study still leans on YAF's `yaf_solvers/nec2_adapter/adapter.py` (real NEC2 via `necpp`) inside every single objective evaluation — i.e. the AI/sim loop is real even when the optimizer itself is borrowed. The in-house BO/NSGA are kept for cases that genuinely match their strengths (cheap surrogate, multi-objective).

## ADR-013: AI-module status doc over partial-FNO-integration
**Decision**: For the AI-module honesty pass, **sharpen `docs/HONEST_STATUS.md`** rather than half-wire the FNO surrogate into the optimization loop within the available budget.
**Rationale**: A real FNO surrogate that genuinely speeds up the loop needs (a) thousands of NEC2 training samples covering the 9-D space, (b) careful train/val split + uncertainty calibration, and (c) an active-learning policy to decide when to trust the surrogate vs fall back to real NEC2. Half-doing that produces a "FNO-screened" loop that's actually slower than raw DE because of unreliable rejections. The honest move is to keep DE on real NEC2 (the current result already converges in seconds) and label the FNO module accurately as "implemented + trains on synthetic data but not yet wired into a real pipeline".

## ADR-014: Yagi surrogate ships to the browser as JSON weights + hand-written JS forward pass, not ONNX
**Decision**: The trained Yagi surrogate is exported to plain JSON (per-layer weight matrices, biases, and normalization statistics) and run in the browser with a ~40-line dependency-free JavaScript forward pass (`frontend/surrogate_infer.js`), rather than via ONNX Runtime Web / TensorFlow.js.
**Rationale**:
1. **Model is trivially small.** A 9→64→64→4 MLP is ~5060 weights. The exact forward pass is two ReLU dense layers plus a linear head — a few hundred lines of matrix arithmetic at most, with no operators that warrant a runtime.
2. **No runtime download.** ONNX Runtime Web pulls a multi-MB WASM/JS bundle; the hand-written pass is a few KB of code over the ~52 KB-gzipped weight JSON. For a client-side slider preview, shipping a general inference engine to evaluate ~5000 multiply-adds is pure overhead.
3. **Auditability and zero supply chain.** The whole inference path is readable in one file with no third-party dependency, and the same code runs in Node for the parity gate (`scripts/verify_web_surrogate.mjs`), so the browser numbers are checked against the source-of-truth PyTorch model in CI.
4. **Verified parity.** JS vs PyTorch agree to `max |Δ| = 9.4e-5` across 7 diverse designs (including box edges), far inside the 0.01 acceptance tolerance; the residual is float32-vs-float64 rounding. If the model later grows operators that are painful to hand-write (conv stacks, attention), revisit ONNX then.

## Decision Log

| Timestamp | Decision | Reason |
|-----------|----------|--------|
| 2026-05-15 | OpenEMSAdapter analytical fallback | Enables testing without solver binary |
| 2026-05-15 | NEC2Adapter induced EMF method | Physics-realistic S11/gain for demo |
| 2026-05-24 | NEC2Adapter switched to real `necpp` MoM; analytical fallback removed | Honest results: missing `necpp` raises `SolverUnavailable`; `verify_dipole.py` truth check (~73 Ω) passes |
| 2026-05-25 | OpenEMSAdapter switched to real openEMS full-wave FDTD; analytical fallback removed | Honest results: missing `openEMS`/`CSXCAD` bindings raise `SolverUnavailable`; `verify_patch.py` patch resonance within 3.1 % of the cavity model |
| 2026-05-15 | VAE 32x32 grid input | Resolution vs training speed balance |
| 2026-05-15 | PML thickness=8 cells | Standard FDTD literature value |
| 2026-05-15 | Cholesky GP implementation | Stable for <1000 observations |
| 2026-05-15 | Vite proxy /api → localhost:8000 | Zero CORS config in dev |
| 2026-05-20 | docker-compose: drop obsolete `version: "3.9"` key | Modern Docker Compose emits a deprecation warning that breaks CI parsers; the key is now ignored anyway. |
| 2026-05-20 | Skim/skimage marching_cubes over `trimesh.creation.marching_cubes` | `trimesh.creation.marching_cubes` was removed upstream years ago; `skimage.measure.marching_cubes` is the canonical replacement and is already an indirect dep through trimesh. |
| 2026-05-20 | diff_fdtd_jax source uses physical-Hz timing and `compute_s11` plumbs eps through the time-loop | The normalized source frequency (`f * dx / c0 = 1.0`) aliased to a constant phase on every time step, producing zero gradient; the loss function was also reading `self.eps_r` rather than the passed parameter, so gradients couldn't flow at all. The differentiable-FDTD acceptance command requires a loss decrease. |
| 2026-05-20 | VAE demo persists weights to `models/vae_designer.pt` | The VAE acceptance command requires "training completes, weights saved"; the previous `train_demo` returned without calling `designer.save()`. |
| 2026-05-20 | Material library seeds 19 entries (added `air` + `teflon`) | Unit test `test_materials_seeded` requires ≥18 materials. Vacuum was already there but isn't useful at runtime — air is the everyday dielectric value, teflon covers the most common low-loss substrate. |
| 2026-05-20 | mypy: keep `strict = true`; per-module `ignore_missing_imports` for vendor libs without stubs (OCC, openems, CSXCAD, …) | Acceptance command is `mypy ... --strict` and now succeeds (0 issues, 64 files). Stubless 3rd-party modules are silenced at the import level (per-module override block), not via per-line `# type: ignore`. |
| 2026-05-20 | Use `np.trapezoid` instead of `np.trapz` (oam.py) | numpy 2.x removed `np.trapz` (deprecated in 1.26). The codebase already pinned numpy ≥1.26 in pyproject. |
| 2026-05-20 | YAF installed editable with `pip install --no-deps -e .` | The full editable install pulled orbax-checkpoint, whose test-fixture path exceeds Windows MAX_PATH and breaks pip on this host. `--no-deps` is safe because all transitive deps were already installed via `pip install` of the dep groups. |
