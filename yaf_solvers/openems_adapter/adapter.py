# ============================================================
# REFERENCE
#   仿造来源：openEMS @ https://github.com/thliebig/openEMS
#   对标文件：openEMS/python/Tutorials/Simple_Patch_Antenna.py
#   对标类/函数：openEMS.openEMS, CSXCAD.ContinuousStructure,
#             FDTD.SetGaussExcite / SetBoundaryCond / AddLumpedPort /
#             AddEdges2Grid / CreateNF2FFBox / Run, LumpedPort.CalcPort,
#             nf2ff.CalcNF2FF
#   关键设计点：
#     - EC-FDTD（Equivalent Circuit FDTD），工程界标准
#     - CSXCAD 几何/材料分离 + 结构化 Yee 网格
#     - 高斯激励覆盖整个频带，单次时域迭代得全频段 S 参数
#     - AddEdges2Grid + SmoothMeshLines 两步网格细化
#     - NF2FF 近场转远场提取方向图与方向性
#   YAF 的差异化改造：
#     - 没有 openEMS Python 绑定 → SolverUnavailable，绝不静默返回假值
#       （与 nec2_adapter 的诚实原则一致，无解析降级占位）
#     - structures/ports 声明式描述（metal/material 盒 + lumped 端口）映射到 CSX
#     - SimulationSpec.frequency_range → 高斯激励 f0/fc 自动推导
#     - 端口 CalcPort → S11/Zin，NF2FF → 增益方向图，填充 SimulationResult
# ============================================================

"""openEMS FDTD solver adapter — real full-wave backend, no analytical fallback.

Drives the openEMS Python bindings (``openEMS`` + ``CSXCAD``) to run a true
time-domain FDTD simulation: build the CSX structure, refine the mesh, add a
lumped excitation port, run the solver, then extract S-parameters / input
impedance from the port and the gain pattern from a near-field-to-far-field
transform.

If the openEMS bindings cannot be imported the adapter raises
``SolverUnavailable`` — it never fabricates results.
"""

from __future__ import annotations

import math
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import numpy as np

from yaf_core.domain.geometry import Geometry, Mesh
from yaf_core.domain.simulation import (
    FarFieldResult,
    SimulationResult,
    SimulationSpec,
    SParamResult,
)
from yaf_solvers.base import (
    BaseSolverAdapter,
    MeshError,
    SolverError,
    SolverUnavailable,
)

try:
    from openEMS import openEMS as _OpenEMS  # type: ignore[import-not-found]
    from CSXCAD import ContinuousStructure as _ContinuousStructure  # type: ignore[import-not-found]
    from openEMS.physical_constants import C0 as _C0  # type: ignore[import-not-found]

    _OPENEMS_IMPORT_ERROR: str | None = None
except Exception as _e:  # pragma: no cover - exercised only without openEMS
    _OpenEMS = None  # type: ignore[assignment]
    _ContinuousStructure = None  # type: ignore[assignment]
    _C0 = 299792458.0
    _OPENEMS_IMPORT_ERROR = repr(_e)


def _require_openems() -> None:
    if _OpenEMS is None or _ContinuousStructure is None:
        raise SolverUnavailable(
            "openems",
            "Python bindings `openEMS`/`CSXCAD` not importable: "
            f"{_OPENEMS_IMPORT_ERROR}. Build openEMS with its Python interface "
            "and set CSXCAD_INSTALL_PATH / OPENEMS_INSTALL_PATH / "
            "LD_LIBRARY_PATH so the modules import.",
        )


class OpenEMSAdapter(BaseSolverAdapter):
    """FDTD solver adapter for openEMS via its Python bindings.

    The simulation is described declaratively through ``spec.solver_settings``:

    ``unit``
        Length unit in metres for all coordinates (default ``1e-3`` = mm).
    ``structures``
        List of box primitives. Each is a dict with ``kind`` ∈ {``metal``,
        ``material``}, ``start``/``stop`` (3-vectors, in ``unit``), ``priority``,
        and for ``material`` an ``epsilon`` (+ optional ``kappa``/``mue``).
        A ``metal`` box may carry ``add_edges`` (e.g. ``"xy"``) and
        ``metal_edge_res`` to refine the mesh at its edges.
    ``ports``
        List of lumped-port dicts: ``nr``, ``R``, ``start``, ``stop``, ``dir``
        (``"x"``/``"y"``/``"z"``), ``excite`` (1.0 for the driven port),
        ``priority``, optional ``edges2grid``.
    ``air_box``
        ``{"x": [lo, hi], "y": [...], "z": [...]}`` simulation-box extent in
        ``unit``. If omitted it is derived from the structure bounds + padding.
    ``resolution``
        Cells per wavelength used to derive the mesh step (default 20).
    ``boundary``
        Six boundary-condition strings (default ``["MUR"] * 6``).

    When no ``structures`` are supplied the adapter treats each triangle of the
    canonical mesh as an axis-aligned PEC box, so arbitrary geometry still runs
    a real FDTD pass (driven by ``spec.ports``).
    """

    name = "openems"
    version = "0.0.36"
    supports = {"fdtd"}

    def __init__(self, executable: str = "openEMS") -> None:
        super().__init__()
        # kept only for back-compat / health reporting; solve() uses the bindings
        self.executable = executable

    async def capabilities(self) -> dict[str, Any]:
        caps = await super().capabilities()
        caps.update(
            {
                "methods": ["fdtd"],
                "frequency_range": [0, 100e9],
                "max_cells": int(1e8),
                "gpu_support": False,
                "excitation_types": ["lumped"],
                "boundary_conditions": ["PML_8", "PEC", "PMC", "MUR"],
                "backend": "openEMS-python" if _OpenEMS is not None else "unavailable",
            }
        )
        return caps

    async def mesh(self, geometry: Geometry, spec: SimulationSpec) -> Mesh:
        """Carry the canonical geometry into a Mesh.

        openEMS performs its own structured-grid discretisation at solve time
        (``SmoothMeshLines`` + ``AddEdges2Grid``); this method only packages the
        geometry and resolution hint so ``solve`` has what it needs.
        """
        try:
            elements = [list(f) for f in geometry.faces]
            return Mesh(
                geometry_id=geometry.id,
                solver_name=self.name,
                nodes=[list(v) for v in geometry.vertices],
                elements=elements,
                element_type="tri3",
                metadata={
                    "resolution": (spec.solver_settings or {}).get("resolution", 20),
                },
            )
        except Exception as e:
            raise MeshError(f"openEMS meshing failed: {e}") from e

    async def solve(
        self,
        mesh: Mesh,
        spec: SimulationSpec,
        progress_callback: Callable[[float], Any] | None = None,
    ) -> SimulationResult:
        _require_openems()

        settings = spec.solver_settings or {}
        structures = self._resolve_structures(mesh, settings)
        ports_cfg = self._resolve_ports(mesh, spec, settings)
        if not structures:
            raise SolverError(
                self.name,
                str(mesh.id),
                "no structures to simulate (provide solver_settings['structures'] "
                "or a mesh with faces)",
            )
        if not ports_cfg:
            raise SolverError(
                self.name,
                str(mesh.id),
                "no excitation port (provide solver_settings['ports'] or spec.ports)",
            )

        with TemporaryDirectory(prefix="openems_") as tmpdir:
            try:
                return self._run_simulation(
                    Path(tmpdir), mesh, spec, settings, structures,
                    ports_cfg, progress_callback,
                )
            except (SolverUnavailable, SolverError):
                raise
            except Exception as e:  # surface the real failure, never fake success
                raise SolverError(self.name, str(mesh.id), str(e)) from e

    # ------------------------------------------------------------------
    # Simulation construction & post-processing
    # ------------------------------------------------------------------

    def _run_simulation(
        self,
        sim_path: Path,
        mesh: Mesh,
        spec: SimulationSpec,
        settings: dict[str, Any],
        structures: list[dict[str, Any]],
        ports_cfg: list[dict[str, Any]],
        progress_callback: Callable[[float], Any] | None,
    ) -> SimulationResult:
        unit = float(settings.get("unit", 1e-3))
        f_min, f_max = float(spec.frequency_range[0]), float(spec.frequency_range[1])
        f0 = 0.5 * (f_min + f_max)
        fc = 0.5 * (f_max - f_min)
        if fc <= 0:  # single-frequency request: open a band around it
            fc = 0.5 * f0 if f0 > 0 else 1e9
            f_min, f_max = max(f0 - fc, 0.0), f0 + fc

        resolution = float(settings.get("resolution", 20))
        if "mesh_res" in settings:
            mesh_res = float(settings["mesh_res"])
        else:
            mesh_res = _C0 / (f0 + fc) / unit / resolution

        boundary = list(settings.get("boundary", ["MUR"] * 6))
        nr_ts = int(settings.get("nr_timesteps", 30000))
        end_crit = float(settings.get("end_criteria", 1e-4))
        smooth_ratio = float(settings.get("smooth_ratio", 1.4))

        FDTD = _OpenEMS(NrTS=nr_ts, EndCriteria=end_crit)
        FDTD.SetGaussExcite(f0, fc)
        FDTD.SetBoundaryCond(boundary)

        CSX = _ContinuousStructure()
        FDTD.SetCSX(CSX)
        grid = CSX.GetGrid()
        grid.SetDeltaUnit(unit)

        # 1) seed the grid with the air-box extent
        air_box = self._air_box(settings, structures, mesh_res)
        for axis in ("x", "y", "z"):
            grid.AddLine(axis, list(air_box[axis]))

        # 2) build the structure primitives (and their edge refinement)
        for st in structures:
            self._add_structure(CSX, FDTD, st, mesh_res)

        # 3) extra explicit mesh lines (e.g. cells across a thin substrate)
        for axis, lines in (settings.get("extra_mesh_lines", {}) or {}).items():
            grid.AddLine(axis, list(lines))

        # 4) excitation / measurement ports
        ports = []
        for pc in ports_cfg:
            kw: dict[str, Any] = {"priority": int(pc.get("priority", 5))}
            if pc.get("edges2grid"):
                kw["edges2grid"] = pc["edges2grid"]
            port = FDTD.AddLumpedPort(
                int(pc["nr"]),
                float(pc["R"]),
                list(pc["start"]),
                list(pc["stop"]),
                pc["dir"],
                float(pc.get("excite", 0.0)),
                **kw,
            )
            ports.append((port, pc))

        # 5) smooth the whole grid to the target resolution
        grid.SmoothMeshLines("all", mesh_res, smooth_ratio)

        # 6) near-field-to-far-field recording box (for the gain pattern)
        want_nf2ff = bool(settings.get("nf2ff", True))
        nf2ff = FDTD.CreateNF2FFBox() if want_nf2ff else None

        n_cells = self._grid_cell_count(grid)

        if progress_callback is not None:
            progress_callback(0.05)

        t_start = time.perf_counter()
        FDTD.Run(str(sim_path), cleanup=True, verbose=0)
        elapsed = time.perf_counter() - t_start

        if progress_callback is not None:
            progress_callback(0.9)

        # --- post-processing -------------------------------------------------
        n_pts = max(int(spec.frequency_points), 2)
        freqs = np.linspace(f_min, f_max, n_pts)

        driven, driven_cfg = self._driven_port(ports)
        driven.CalcPort(str(sim_path), freqs)

        s11 = np.asarray(driven.uf_ref) / np.asarray(driven.uf_inc)
        zin = np.asarray(driven.uf_tot) / np.asarray(driven.if_tot)
        z0 = float(driven_cfg["R"])

        s_matrix = [[[complex(v)]] for v in s11]
        s_params = SParamResult(
            frequency=[float(f) for f in freqs],
            s_matrix=s_matrix,
            z0=z0,
        )

        # resonance = deepest S11 dip
        s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-30))
        res_idx = int(np.argmin(s11_db))
        f_res = float(freqs[res_idx])
        z_res = complex(zin[res_idx])
        vswr = self._vswr_from_s11(complex(s11[res_idx]))

        far_field: FarFieldResult | None = None
        gain_dbi: float | None = None
        if nf2ff is not None:
            far_field, gain_dbi = self._far_field(
                nf2ff, sim_path, f_res, settings
            )

        if progress_callback is not None:
            progress_callback(1.0)

        try:
            job_uuid = uuid.UUID(str(mesh.id))
        except (ValueError, AttributeError):
            job_uuid = uuid.uuid4()

        return SimulationResult(
            job_id=job_uuid,
            solver_name=self.name,
            solver_version=self.version,
            status="success",
            s_params=s_params,
            far_field=far_field,
            gain_dbi=gain_dbi,
            efficiency=None,
            vswr=vswr,
            simulation_time_sec=elapsed,
            mesh_stats={"num_cells": n_cells},
            solver_metadata={
                "backend": "openEMS-python",
                "unit": unit,
                "f0_hz": f0,
                "fc_hz": fc,
                "mesh_res_unit": mesh_res,
                "boundary": boundary,
                "resonant_freq_hz": f_res,
                "resonant_s11_db": float(s11_db[res_idx]),
                "zin_at_resonance": (z_res.real, z_res.imag),
                "zin_per_freq": [(complex(z).real, complex(z).imag) for z in zin],
                "s11_db_per_freq": [float(v) for v in s11_db],
                "num_cells": n_cells,
                "directivity_dbi": gain_dbi,
            },
        )

    def _far_field(
        self,
        nf2ff: Any,
        sim_path: Path,
        f_res: float,
        settings: dict[str, Any],
    ) -> tuple[FarFieldResult | None, float | None]:
        """Run the NF2FF transform at resonance and build a FarFieldResult.

        The scalar gain is the FDTD directivity ``10*log10(Dmax)`` from the
        transform; the E-field grid carries the real radiated-field pattern.
        """
        theta = np.array(settings.get("pattern_theta", np.arange(-180.0, 182.0, 2.0)))
        phi = np.array(settings.get("pattern_phi", [0.0, 90.0]))
        center = list(settings.get("nf2ff_center", [0.0, 0.0, 0.0]))
        try:
            res = nf2ff.CalcNF2FF(str(sim_path), f_res, theta, phi, center=center)
        except Exception:
            return None, None

        e_theta_grid = np.asarray(res.E_theta[0])  # [theta, phi]
        e_phi_grid = np.asarray(res.E_phi[0])
        e_theta = [[complex(e_theta_grid[i, j]) for j in range(len(phi))]
                   for i in range(len(theta))]
        e_phi = [[complex(e_phi_grid[i, j]) for j in range(len(phi))]
                 for i in range(len(theta))]

        far_field = FarFieldResult(
            theta=[float(t) for t in theta],
            phi=[float(p) for p in phi],
            e_theta=e_theta,
            e_phi=e_phi,
            frequency=f_res,
        )
        dmax = float(np.asarray(res.Dmax).reshape(-1)[0])
        gain_dbi = 10.0 * math.log10(max(dmax, 1e-30))
        return far_field, gain_dbi

    # ------------------------------------------------------------------
    # Structure / port resolution
    # ------------------------------------------------------------------

    def _add_structure(
        self, CSX: Any, FDTD: Any, st: dict[str, Any], mesh_res: float
    ) -> None:
        kind = st.get("kind", "metal")
        name = st.get("name", kind)
        start = list(st["start"])
        stop = list(st["stop"])
        priority = int(st.get("priority", 10))

        if kind == "metal":
            prop = CSX.AddMetal(name)
        elif kind == "material":
            mat_kw: dict[str, Any] = {"epsilon": float(st.get("epsilon", 1.0))}
            if "kappa" in st:
                mat_kw["kappa"] = float(st["kappa"])
            if "mue" in st:
                mat_kw["mue"] = float(st["mue"])
            prop = CSX.AddMaterial(name, **mat_kw)
        else:
            raise SolverError(self.name, "-", f"unknown structure kind: {kind!r}")

        prop.AddBox(priority=priority, start=start, stop=stop)

        edges = st.get("add_edges")
        if edges:
            kw: dict[str, Any] = {"dirs": edges, "properties": prop}
            if st.get("metal_edge_res"):
                kw["metal_edge_res"] = mesh_res / 2.0
            FDTD.AddEdges2Grid(**kw)

    def _resolve_structures(
        self, mesh: Mesh, settings: dict[str, Any]
    ) -> list[dict[str, Any]]:
        explicit = settings.get("structures")
        if explicit:
            return list(explicit)

        # Fallback: each triangle becomes an axis-aligned PEC box so arbitrary
        # canonical geometry still runs a genuine FDTD pass.
        out: list[dict[str, Any]] = []
        nodes = mesh.nodes
        for i, elem in enumerate(mesh.elements):
            if len(elem) < 3:
                continue
            pts = np.array([nodes[idx] for idx in elem[:3]], dtype=float)
            start = pts.min(axis=0).tolist()
            stop = pts.max(axis=0).tolist()
            out.append(
                {"kind": "metal", "name": f"metal_{i}", "start": start,
                 "stop": stop, "priority": 10}
            )
        return out

    def _resolve_ports(
        self, mesh: Mesh, spec: SimulationSpec, settings: dict[str, Any]
    ) -> list[dict[str, Any]]:
        explicit = settings.get("ports")
        if explicit:
            return list(explicit)

        out: list[dict[str, Any]] = []
        for i, port in enumerate(spec.ports, start=1):
            loc = list(port.location)
            direction = port.direction
            axis = ("x", "y", "z")[int(np.argmax(np.abs(direction)))]
            span = float(settings.get("port_span", 1.0))  # in `unit`
            start = list(loc)
            stop = list(loc)
            ai = "xyz".index(axis)
            start[ai] -= span / 2.0
            stop[ai] += span / 2.0
            out.append(
                {"nr": i, "R": float(port.impedance), "start": start, "stop": stop,
                 "dir": axis, "excite": 1.0 if i == 1 else 0.0}
            )
        return out

    @staticmethod
    def _driven_port(ports: list[tuple[Any, dict[str, Any]]]):
        for port, cfg in ports:
            if float(cfg.get("excite", 0.0)) != 0.0:
                return port, cfg
        return ports[0]

    @staticmethod
    def _air_box(
        settings: dict[str, Any],
        structures: list[dict[str, Any]],
        mesh_res: float,
    ) -> dict[str, list[float]]:
        ab = settings.get("air_box")
        if ab:
            return {axis: list(ab[axis]) for axis in ("x", "y", "z")}

        # derive from structure bounds + padding when not given explicitly
        pts = []
        for st in structures:
            pts.append(st["start"])
            pts.append(st["stop"])
        arr = np.array(pts, dtype=float)
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        pad = max(float(np.max(hi - lo)), 1.0) * 0.75 + 10 * mesh_res
        return {
            "x": [float(lo[0] - pad), float(hi[0] + pad)],
            "y": [float(lo[1] - pad), float(hi[1] + pad)],
            "z": [float(lo[2] - pad), float(hi[2] + pad)],
        }

    @staticmethod
    def _grid_cell_count(grid: Any) -> int:
        try:
            nx = len(grid.GetLines("x"))
            ny = len(grid.GetLines("y"))
            nz = len(grid.GetLines("z"))
            return int(nx * ny * nz)
        except Exception:
            return 0

    @staticmethod
    def _vswr_from_s11(s11: complex) -> float:
        mag = abs(s11)
        if mag >= 1.0:
            return float("inf")
        return (1.0 + mag) / (1.0 - mag)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_native_format(self, geometry: Geometry) -> bytes:
        """Export geometry as a minimal CSXCAD ContinuousStructure XML.

        Each triangle is exported as an axis-aligned metal box (its bounding
        box). This is a lightweight serialisation helper; the live solver path
        in :meth:`solve` builds the CSX structure directly via the bindings.
        """
        root = ET.Element("ContinuousStructure")
        ET.SubElement(root, "CoordSystem", Type="0")

        if geometry.vertices and geometry.faces:
            for i, face in enumerate(geometry.faces):
                if len(face) < 3:
                    continue
                v = [geometry.vertices[idx] for idx in face]
                metal = ET.SubElement(root, "Metal", Name=f"face_{i}")
                prop = ET.SubElement(metal, "Properties")
                box = ET.SubElement(prop, "Box")
                ET.SubElement(box, "Priority").text = "10"
                start = [min(p[k] for p in v) for k in range(3)]
                stop = [max(p[k] for p in v) for k in range(3)]
                ET.SubElement(box, "Start").text = " ".join(f"{s:.6e}" for s in start)
                ET.SubElement(box, "Stop").text = " ".join(f"{s:.6e}" for s in stop)

        return bytes(ET.tostring(root, encoding="utf-8"))

    async def from_native_result(self, raw_output: bytes) -> SimulationResult:
        raise NotImplementedError(
            "OpenEMSAdapter.from_native_result: openEMS results are HDF5 dumps "
            "read back through the port/nf2ff objects, not a single blob. Use "
            "solve() directly."
        )

    async def health_check(self) -> bool:
        return _OpenEMS is not None and _ContinuousStructure is not None
