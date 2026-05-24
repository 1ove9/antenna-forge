# ============================================================
# REFERENCE
#   仿造来源：necpp (PyNEC) @ https://github.com/tmolteno/necpp
#   对标文件：necpp/example/test.py, necpp/example/test_nec.c
#   对标函数：nec_create / nec_wire / nec_geometry_complete / nec_fr_card /
#             nec_ex_card / nec_rp_card / nec_impedance_real / nec_impedance_imag /
#             nec_gain_max
#   关键设计点：
#     - 直接通过 SWIG Python 绑定 (necpp) 调用 nec2++ C 库的 MoM 求解器
#     - 经典 NEC2 卡片体系（GW/GE/EX/FR/RP）以函数参数形式而非文本卡片
#     - 频率扫描通过多次 solve 调用而非 FR 卡片 NF>1（更易抽取每点 Z）
#   YAF 的差异化改造：
#     - 没有 necpp → SolverUnavailable，不做静默 analytical fallback（绝不返回假值）
#     - 真值从 nec_impedance_real / nec_impedance_imag / nec_gain_max 提取
#     - card_writer 仍保留用于 to_native_format（.nec 文件导出），但 solve 不走文件
# ============================================================

"""NEC2 MoM solver adapter — real necpp backend, no analytical fallback."""

from __future__ import annotations

import math
import time
import uuid
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
from yaf_solvers.nec2_adapter.card_writer import NEC2CardWriter

try:
    import necpp as _necpp  # type: ignore[import-not-found]
    _NECPP_IMPORT_ERROR: str | None = None
except Exception as _e:  # pragma: no cover
    _necpp = None  # type: ignore[assignment]
    _NECPP_IMPORT_ERROR = repr(_e)


def _require_necpp() -> Any:
    if _necpp is None:
        raise SolverUnavailable(
            "nec2",
            f"Python binding `necpp` not importable: {_NECPP_IMPORT_ERROR}. "
            "Install with `pip install necpp --break-system-packages`.",
        )
    return _necpp


def _check(rc: int, op: str) -> None:
    if rc != 0:
        msg = _necpp.nec_error_message() if _necpp is not None else "no detail"
        raise SolverError("nec2", "-", f"necpp {op} failed (rc={rc}): {msg}")


class NEC2Adapter(BaseSolverAdapter):
    """MoM solver adapter for NEC2 via necpp Python binding.

    Wire geometry is taken from mesh.nodes (x,y,z) + mesh.elements (pairs of
    node indices). Each element is one GW wire with the configured segment
    count. Excitation is a voltage source on the middle segment of wire 1.
    """

    name = "nec2"
    version = "necpp"
    supports = {"mom"}

    # Defaults — overridable via spec.solver_settings
    _default_segments = 21
    _default_radius_m = 0.0005
    _default_ex_wire_tag = 1

    def __init__(self, executable: str = "nec2c") -> None:
        super().__init__()
        # executable kept only for back-compat / health_check; solve uses necpp
        self.executable = executable

    async def capabilities(self) -> dict[str, Any]:
        caps = await super().capabilities()
        caps.update(
            {
                "methods": ["mom"],
                "frequency_range": [1e3, 100e9],
                "max_cells": 50000,
                "gpu_support": False,
                "excitation_types": ["voltage"],
                "structure_types": ["wire"],
                "backend": "necpp" if _necpp is not None else "unavailable",
            }
        )
        return caps

    async def mesh(self, geometry: Geometry, spec: SimulationSpec) -> Mesh:
        try:
            elements = [[f[0], f[1]] for f in geometry.faces if len(f) >= 2]
            return Mesh(
                geometry_id=geometry.id,
                solver_name=self.name,
                nodes=geometry.vertices,
                elements=elements,
                element_type="wire",
                metadata={},
            )
        except Exception as e:
            raise MeshError(f"NEC2 meshing failed: {e}") from e

    async def solve(
        self,
        mesh: Mesh,
        spec: SimulationSpec,
        progress_callback: Callable[[float], Any] | None = None,
    ) -> SimulationResult:
        nec_mod = _require_necpp()

        wires = self._wires_from_mesh(mesh)
        if not wires:
            raise SolverError(
                self.name,
                str(mesh.id),
                "no wire elements found in mesh (need nodes + elements)",
            )

        settings = spec.solver_settings or {}
        segments = int(settings.get("segments", self._default_segments))
        radius = float(settings.get("wire_radius", self._default_radius_m))
        ex_tag = int(settings.get("ex_tag", self._default_ex_wire_tag))
        ground_type = int(settings.get("ground_type", -1))  # -1 = free space
        n_theta = int(settings.get("n_theta", 19))
        n_phi = int(settings.get("n_phi", 37))
        if spec.far_field_request:
            n_theta = int(spec.far_field_request.get("n_theta", n_theta))
            n_phi = int(spec.far_field_request.get("n_phi", n_phi))

        f_min, f_max = spec.frequency_range
        n_pts = max(int(spec.frequency_points), 1)
        if n_pts == 1 or f_min == f_max:
            freqs_hz = [float(f_min)]
        else:
            freqs_hz = np.linspace(f_min, f_max, n_pts).tolist()

        z0 = 50.0
        s_matrix: list[list[list[complex]]] = []
        per_freq_z: list[complex] = []
        gain_max_db: list[float] = []
        t_start = time.perf_counter()

        dtheta = 180.0 / max(n_theta - 1, 1)
        dphi = 360.0 / max(n_phi, 1)

        # Pattern grid captured only at the center frequency (saves N×M extra
        # nec_gain calls per sweep point); a sweep-wide pattern is rarely
        # what the caller actually wants.
        center_idx = len(freqs_hz) // 2
        pattern_dbi: list[list[float]] | None = None

        for idx, f_hz in enumerate(freqs_hz):
            nec = nec_mod.nec_create()
            try:
                for tag, (p0, p1) in enumerate(wires, start=1):
                    _check(
                        nec_mod.nec_wire(
                            nec, tag, segments,
                            float(p0[0]), float(p0[1]), float(p0[2]),
                            float(p1[0]), float(p1[1]), float(p1[2]),
                            radius, 1.0, 1.0,
                        ),
                        f"nec_wire(tag={tag})",
                    )
                _check(nec_mod.nec_geometry_complete(nec, 0), "nec_geometry_complete")

                if ground_type >= 0:
                    _check(
                        nec_mod.nec_gn_card(nec, ground_type, 0, 0, 0, 0, 0, 0, 0),
                        "nec_gn_card",
                    )

                f_mhz = f_hz / 1e6
                _check(nec_mod.nec_fr_card(nec, 0, 1, f_mhz, 0), "nec_fr_card")

                ex_seg = max(1, segments // 2 + (segments % 2))
                _check(
                    nec_mod.nec_ex_card(
                        nec, 0, ex_tag, ex_seg, 0,
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ),
                    "nec_ex_card",
                )

                _check(
                    nec_mod.nec_rp_card(
                        nec, 0, n_theta, n_phi, 0, 5, 0, 0,
                        0.0, 0.0, dtheta, dphi, 0, 0,
                    ),
                    "nec_rp_card",
                )

                R = float(nec_mod.nec_impedance_real(nec, 0))
                X = float(nec_mod.nec_impedance_imag(nec, 0))
                G = float(nec_mod.nec_gain_max(nec, 0))

                if idx == center_idx:
                    pattern_dbi = [
                        [float(nec_mod.nec_gain(nec, 0, ti, pj))
                         for pj in range(n_phi)]
                        for ti in range(n_theta)
                    ]
            finally:
                nec_mod.nec_delete(nec)

            z = complex(R, X)
            per_freq_z.append(z)
            gain_max_db.append(G)
            gamma = (z - z0) / (z + z0)
            s_matrix.append([[gamma]])

            if progress_callback is not None:
                progress_callback((idx + 1) / len(freqs_hz))

        elapsed = time.perf_counter() - t_start

        gain_dbi = gain_max_db[center_idx]
        z_center = per_freq_z[center_idx]

        # Plumb real NEC2 per-direction gain into FarFieldResult by storing an
        # equivalent E_theta magnitude. FarFieldResult.gain_dbi() recovers G via
        #     g_dbi = 10*log10((|E_t|^2 + |E_p|^2) / (2*eta0)) + 2.15
        # so we invert: |E_t| = sqrt(2*eta0 * 10^((g_dbi - 2.15)/10)).
        # NEC sentinel for unreachable directions is -999 (poles); zero out.
        eta_0 = 377.0
        theta_deg = [i * dtheta for i in range(n_theta)]
        phi_deg = [j * dphi for j in range(n_phi)]
        e_theta: list[list[complex]] = []
        e_phi: list[list[complex]] = []
        for ti in range(n_theta):
            row_t: list[complex] = []
            row_p: list[complex] = []
            for pj in range(n_phi):
                if pattern_dbi is None:
                    row_t.append(complex(0.0, 0.0))
                    row_p.append(complex(0.0, 0.0))
                    continue
                g_dbi = pattern_dbi[ti][pj]
                if g_dbi < -100.0:  # NEC -999 sentinel
                    row_t.append(complex(0.0, 0.0))
                else:
                    amp = math.sqrt(2.0 * eta_0 * (10.0 ** ((g_dbi - 2.15) / 10.0)))
                    row_t.append(complex(amp, 0.0))
                row_p.append(complex(0.0, 0.0))
            e_theta.append(row_t)
            e_phi.append(row_p)

        try:
            job_uuid = uuid.UUID(str(mesh.id))
        except (ValueError, AttributeError):
            job_uuid = uuid.uuid4()

        return SimulationResult(
            job_id=job_uuid,
            solver_name=self.name,
            solver_version=self.version,
            status="success",
            s_params=SParamResult(
                frequency=[float(f) for f in freqs_hz],
                s_matrix=s_matrix,
                z0=z0,
            ),
            far_field=FarFieldResult(
                theta=theta_deg,
                phi=phi_deg,
                e_theta=e_theta,
                e_phi=e_phi,
                frequency=float(freqs_hz[center_idx]),
            ),
            gain_dbi=gain_dbi,
            efficiency=None,
            vswr=self._vswr_from_z(z_center, z0),
            simulation_time_sec=elapsed,
            solver_metadata={
                "backend": "necpp",
                "segments": segments,
                "wire_radius_m": radius,
                "n_wires": len(wires),
                "impedance_per_freq": [(z.real, z.imag) for z in per_freq_z],
                "gain_max_per_freq_db": gain_max_db,
            },
        )

    @staticmethod
    def _wires_from_mesh(mesh: Mesh) -> list[tuple[list[float], list[float]]]:
        out: list[tuple[list[float], list[float]]] = []
        if not mesh.nodes or not mesh.elements:
            return out
        n = len(mesh.nodes)
        for elem in mesh.elements:
            if len(elem) < 2:
                continue
            i0, i1 = int(elem[0]), int(elem[1])
            if 0 <= i0 < n and 0 <= i1 < n:
                out.append((list(mesh.nodes[i0]), list(mesh.nodes[i1])))
        return out

    @staticmethod
    def _vswr_from_z(z: complex, z0: float) -> float:
        gamma = abs((z - z0) / (z + z0))
        if gamma >= 1.0:
            return float("inf")
        return (1.0 + gamma) / (1.0 - gamma)

    def to_native_format(self, geometry: Geometry) -> bytes:
        writer = NEC2CardWriter()
        if geometry.vertices and geometry.faces:
            for ei, face in enumerate(geometry.faces):
                if len(face) >= 2:
                    i0, i1 = face[0], face[1]
                    if i0 < geometry.num_vertices and i1 < geometry.num_vertices:
                        p0 = geometry.vertices[i0]
                        p1 = geometry.vertices[i1]
                        writer.gw_card(
                            ei + 1, 11,
                            p0[0], p0[1], p0[2],
                            p1[0], p1[1], p1[2],
                            0.001,
                        )
        else:
            writer.add_dipole(length=0.5, tag=1)
        writer.cards.append(writer.ge_card(0))
        writer.cards.append(writer.gn_card(0))
        writer.cards.append(writer.ex_card(0, 1, 6))
        writer.cards.append(writer.fr_card(1000.0))
        writer.cards.append(writer.rp_card())
        return writer.to_bytes()

    async def from_native_result(self, raw_output: bytes) -> SimulationResult:
        raise NotImplementedError(
            "NEC2Adapter.from_native_result: parsing nec2c text output is not "
            "supported in the necpp-backend implementation. Use solve() directly."
        )

    async def health_check(self) -> bool:
        return _necpp is not None
