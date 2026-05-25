# Contributing to YAF

Thank you for considering contributing to Source Sequence Antenna Forge (YAF).
This document covers: setting up a local development environment,
running the acceptance commands, the style we aim for, and the
practical recipe for adding a new solver adapter or AI module.

Before diving in, please skim [`docs/HONEST_STATUS.md`](docs/HONEST_STATUS.md).
That document is unusually direct about what works, what is demo-only,
and what is placeholder code; it sets realistic expectations for where
contributions are most useful.

---

## 1. Development environment

### 1.1 Prerequisites

| Tool | Tested version | Notes |
|---|---|---|
| Python | 3.11 (CI), 3.12, 3.13 (used during development) | `pyproject.toml` declares `requires-python = ">=3.11"` |
| `git` | any modern version | |
| Docker + Compose | 24.x | required only for the infrastructure acceptance command |
| `make` | optional | the Makefile is a thin shortcut wrapper |
| C/C++ toolchain | distro default | needed if you build `necpp` from source |

### 1.2 Clone & install

```bash
git clone <your-fork-url> yaf
cd yaf
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .[dev]
```

If the editable install pulls a transitive that misbehaves on your OS
(seen on Windows with `orbax-checkpoint`'s test-fixture paths), the
fallback recipe used in the development environment is:

```bash
pip install -r <(python -c "import tomllib,sys; \
  d=tomllib.loads(open('pyproject.toml','rb').read().decode()); \
  print('\n'.join(d['project']['dependencies']))") \
  --break-system-packages 2>/dev/null || true
pip install --no-deps -e .
```

`DECISIONS.md` ADR records the reasoning behind the `--no-deps`
fallback. Use the normal `pip install -e .[dev]` path first.

### 1.3 Optional: real NEC2 backend

The Method-of-Moments code path uses the `necpp` Python binding.
**It is not a hard dependency** — you can develop most of YAF without
it. To enable real-MoM tests and the Yagi case study:

```bash
pip install necpp        # or: pip install necpp --break-system-packages
```

Note: `necpp` wraps the GPL-2 `nec2++` C++ library. The combined
process is governed by [`NOTICE`](NOTICE); please read it before
deciding whether to install in a setting where licensing is
commercially material.

### 1.4 Optional: openEMS backend

`yaf_solvers/openems_adapter/` is a real openEMS full-wave FDTD backend.
You do not need to install openEMS/CSXCAD to develop against the rest of
the codebase — the openEMS integration test and `scripts/verify_patch.py`
skip (or raise `SolverUnavailable`) when the bindings are absent, and they
are not required for the other acceptance commands. To run the openEMS
path, build openEMS with its Python interface and point
`CSXCAD_INSTALL_PATH` / `OPENEMS_INSTALL_PATH` / `LD_LIBRARY_PATH` at the
install so `from openEMS import openEMS` / `from CSXCAD import
ContinuousStructure` import. Because this loads the GPL-3 openEMS bindings
into the same process, the GPL-3 caveats in `NOTICE` apply once you do.

---

## 2. Running the acceptance commands

Every PR is expected to keep the acceptance commands in
[README → "Acceptance commands"](README.md) green. Below is the
minimum dev-time subset; consult the full block in the README for the
complete list including the v0.1.0 truth-check and case-study demos.

### 2.1 Test suite

```bash
pytest tests/ -x -q
```

Should report `all passed`. The dipole and Yagi tests require
`necpp`; they `pytest.skip` if it is not installed, but please
install it before opening PRs that touch `yaf_solvers/`.

### 2.2 Static type checking

```bash
mypy yaf_core yaf_ai yaf_solvers --strict
```

Must report `Success: no issues in N source files`. Per-module
`ignore_missing_imports` overrides for stub-less third-party libraries
are documented in `pyproject.toml`; do not add new ones without an
ADR.

### 2.3 Quick smoke tests for the headline features

```bash
python3 scripts/verify_dipole.py       # truth check: ~73 Ω, ~2.15 dBi
python3 scripts/demo_wow.py            # writes docs/assets/dipole_demo.png
python3 scripts/demo_inverse_design.py # 16 NEC2 calls → 477.892 mm
python3 scripts/case_yagi.py           # 5858 NEC2 calls, ~13 s
python3 scripts/plot_yagi.py           # writes docs/assets/yagi_design.png
```

PRs that change `yaf_solvers/nec2_adapter/` or `scripts/case_yagi.py`
should re-run the Yagi case study and update the JSONs in `results/`
+ the PNG in `docs/assets/`.

### 2.4 Infrastructure (only required if you touch `yaf_api/` or `yaf_worker/`)

```bash
docker compose up -d
curl -fsS http://localhost:8000/health    # 200 OK
```

---

## 3. Code style

* **Python**: PEP 8 with the small project-specific tightenings encoded
  in `pyproject.toml` (Ruff config). Run `ruff check .` and
  `ruff format .` before submitting.
* **Type hints required** on all public functions / classes;
  `mypy --strict` must pass.
* **Pydantic v2** for all domain models. No raw `dataclass` where a
  Pydantic model is appropriate — see ADR-001.
* **Docstrings**: every public symbol has at least a one-line summary;
  modules whose design is informed by an open-source project cite that
  project at the top of the file.
* **Logging**: structlog, not `logging`. See ADR-005.

### 3.1 What NOT to do

* Do not silently fall back to an analytical model when a solver is
  unavailable. Raise `SolverUnavailable` (defined in
  `yaf_solvers/base.py`) so the caller can decide. This is the lesson
  of the NEC2-adapter rewrite — see commit `efe5cc0` and the
  `DECISIONS.md` ADR around it.
* Do not add a new mypy `ignore_missing_imports` entry without
  documenting why in `DECISIONS.md`.
* Do not bundle GPL solver code into the repository. The license
  boundary in `NOTICE` is load-bearing.
* Do not commit local tooling state, build caches, or cloned reference
  repositories. They are gitignored for a reason.

---

## 4. Adding a new solver adapter

The solver-adapter pattern is the most common kind of contribution and
worth documenting in detail.

### 4.1 The contract

Every adapter implements the `SolverAdapter` protocol defined in
`yaf_core/ports/solver.py` (and inherits the helper class
`yaf_solvers.base.BaseSolverAdapter`). The minimum surface is:

```python
class MySolverAdapter(BaseSolverAdapter):
    name = "mysolver"
    version = "0.1"
    supports = {"fdtd"}       # one of: "mom", "fdtd", "fem", "moments"

    async def capabilities(self) -> dict[str, Any]: ...
    async def mesh(self, geometry: Geometry, spec: SimulationSpec) -> Mesh: ...
    async def solve(
        self, mesh: Mesh, spec: SimulationSpec,
        progress_callback: Callable[[float], Any] | None = None,
    ) -> SimulationResult: ...
    def to_native_format(self, geometry: Geometry) -> bytes: ...
    async def from_native_result(self, raw_output: bytes) -> SimulationResult: ...
    async def health_check(self) -> bool: ...
```

### 4.2 Required behaviour

1. **If the underlying solver binary / Python binding is missing**,
   raise `yaf_solvers.base.SolverUnavailable(name, reason)`. Never
   return a fabricated result. See the NEC2-adapter rewrite (commit
   `efe5cc0`).
2. **If the input geometry / mesh is empty or unsupported**, raise
   `yaf_solvers.base.SolverError(name, job_id, reason)`.
3. **`SimulationResult.far_field`** should carry per-direction
   amplitudes that reflect what the solver actually computed. The NEC2
   adapter is the canonical example: it stores an equivalent `|E_θ|`
   derived from `nec_gain(., ti, pj)` so that
   `FarFieldResult.gain_dbi()` recovers the same dBi the solver
   reported.
4. **Per-frequency raw values** belong in
   `SimulationResult.solver_metadata` so callers can audit the
   trajectory without rerunning the solver.

### 4.3 Tests every new adapter must have

* `test_capabilities()` — `await adapter.capabilities()` returns the
  expected method set.
* `test_to_native_format()` — round-trips a minimal geometry.
* `test_empty_geometry_raises()` — empty mesh → `SolverError`. This is
  the explicit anti-fallback assertion.
* `test_real_truth_case()` — a known-answer physics regression with a
  tight (e.g. ±15 %) tolerance against a textbook value. Skip with
  `pytest.skip` if the backend is not installed; do **not** silently
  pass.

Cribbed from
`tests/unit/test_solvers.py::test_dipole_solve_real_nec` and
`tests/integration/test_pipeline.py::test_solver_nec2_integration`.

### 4.4 Documentation

* A top-of-file note citing the upstream project the adapter is modeled
  on, if any.
* If the upstream is copyleft, update `NOTICE` to document the
  license-boundary implications.
* If the adapter is non-trivial, add a `docs/case_study_<your-solver>.md`
  following the structure of `docs/case_study_yagi.md`.

---

## 5. Adding an AI / optimizer module

* The bayes/NSGA/SIMP modules in `yaf_ai/optimization/` are the
  reference shape. New optimizers should either fit into that pattern
  or live in `yaf_ai/inverse_design/` if they are full-stack closed
  loops.
* Any module that is **not yet wired into a real physics oracle**
  should be marked `🟡 demo only` in `docs/HONEST_STATUS.md` §3 (the
  three-tier table). Honesty here is the project's selling point;
  please do not let modules drift from 🔴 to 🟡 to 🟢 by relabel rather
  than by integration.

---

## 6. Pull-request workflow

1. Fork the repo. Branch off `main` (the project's default branch).
2. Make commits small and focused. Acceptable commit messages follow
   the existing convention (look at `git log --oneline | head -30`
   for examples); the short summary line should be ≤ 72 characters.
3. Run **all** of §2 above before opening the PR.
4. Open the PR using the template in `.github/PULL_REQUEST_TEMPLATE.md`.
   The "What changed and why" + "Acceptance commands rerun" sections
   are required; "Backwards-incompatible changes" and "Known
   regressions" are optional but appreciated.
5. Expect at least one reviewer round. We prioritise correctness
   (especially physics correctness) and clear honest documentation
   over fast iteration.

If you discovered a bug rather than wanting to add a feature, open an
issue using the bug-report template in
`.github/ISSUE_TEMPLATE/bug_report.md` first — the maintainer can
help triage and may have context.

---

## 7. License

By submitting a contribution, you agree that your contribution will
be licensed under the same MIT License that covers the rest of the YAF
source code (see `LICENSE`). If your contribution depends on a
third-party project with a non-MIT-compatible license, please flag it
in the PR description so that `NOTICE` can be updated.

---

Thank you for taking the time to read this. If anything in the above
is unclear, please open a discussion or an issue rather than guessing —
the project's main risk is drift between what the code claims to do
and what it actually does, and clear contributor communication is the
best defence against that drift.
