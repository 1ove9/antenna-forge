# ============================================================
# REFERENCE
#   仿造来源：Tidy3D @ https://github.com/flexcompute/tidy3d
#             + Meep @ https://github.com/NanoComp/meep
#   对标文件：tidy3d/components/geometry/base.py, meep/python/geometry.py
#   对标类/函数：td.Box, td.Cylinder, td.PolySlab, td.TriangleMesh, mp.GeometricObject
#   关键设计点：
#     - Geometry 抽象基类 → Box/Cylinder/PolySlab/TriangleMesh 多态层次
#     - Material 包含色散模型（Debye/Drude/Lorentz/Kubo）
#     - 多表示共存：mesh（三角面片）/ brep（OpenCASCADE）/ voxel（3D 网格）/ implicit（SIREN）
#     - 序列化/反序列化支持 NPZ 二进制格式
#   YAF 的差异化改造：
#     - 单一 Geometry 类承载多表示（非继承多态），简化序列化
#     - Material 作为独立类，内嵌色散参数
#     - Mesh 独立于 Geometry，求解器特定
#     - 内置 to_numpy()/serialize()/deserialize() 便捷方法
# ============================================================

"""
Geometry domain model — geometry representation, materials, and mesh.

Supports multiple geometry representations: BREP, mesh, voxel, and
implicit (neural) representations.
"""

from __future__ import annotations

import enum
import uuid
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class MaterialType(str, enum.Enum):
    """Material classification."""

    PEC = "pec"
    PMC = "pmc"
    DIELECTRIC = "dielectric"
    MAGNETIC = "magnetic"
    CONDUCTOR = "conductor"
    GRAPHENE = "graphene"
    LIQUID_METAL = "liquid_metal"
    PLASMA = "plasma"
    CUSTOM = "custom"


class Material(BaseModel):
    """Electromagnetic material definition with optional dispersive model.

    Supports:
    - Constant: epsilon_r, mu_r, sigma
    - Debye: tau, delta_eps
    - Drude: plasma_freq, collision_freq
    - Lorentz: resonant_freq, damping, delta_eps
    - Kubo (graphene): mu_c, gamma, temperature
    """

    id: str
    name: str
    type: MaterialType
    epsilon_r: float = 1.0
    mu_r: float = 1.0
    sigma: float = 0.0  # S/m
    density_kg_m3: float | None = None

    # Dispersive model parameters
    dispersion_model: str | None = None  # "debye", "drude", "lorentz", "kubo"
    dispersion_params: dict[str, float] = Field(default_factory=dict)


class Geometry(BaseModel):
    """Canonical geometry representation.

    A geometry can be one of:
    - triangular mesh (vertices + faces)
    - BREP solid (OpenCASCADE shape)
    - voxel grid (3D numpy array)
    - implicit field (SIREN / neural field)
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = ""
    representation: str = "mesh"  # "mesh", "brep", "voxel", "implicit"

    # Mesh representation
    vertices: list[list[float]] = Field(default_factory=list)
    faces: list[list[int]] = Field(default_factory=list)

    # Voxel representation
    voxel_grid: list[list[list[float]]] = Field(default_factory=list)
    voxel_resolution: tuple[int, int, int] | None = None

    # Material assignment
    material_regions: dict[str, list[int]] = Field(
        default_factory=dict,
        description="material_id -> list of face indices"
    )

    # Metadata
    bounding_box: tuple[float, float, float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert vertices/faces to numpy arrays."""
        return (
            np.array(self.vertices, dtype=np.float64),
            np.array(self.faces, dtype=np.int64),
        )

    def serialize(self) -> bytes:
        """Serialize geometry to binary (NPZ format)."""
        import io

        v, f = self.to_numpy()
        buf = io.BytesIO()
        np.savez_compressed(buf, vertices=v, faces=f)
        return buf.getvalue()

    @classmethod
    def deserialize(cls, data: bytes) -> Geometry:
        """Deserialize geometry from binary (NPZ format)."""
        import io

        buf = io.BytesIO(data)
        arr = np.load(buf)
        return cls(
            vertices=arr["vertices"].tolist(),
            faces=arr["faces"].tolist(),
            representation="mesh",
        )


class Mesh(BaseModel):
    """Simulation-ready mesh.

    Contains element nodes, element connectivity, and material assignments.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    geometry_id: uuid.UUID
    solver_name: str

    # Node coordinates [N x 3]
    nodes: list[list[float]] = Field(default_factory=list)

    # Element connectivity
    elements: list[list[int]] = Field(default_factory=list)
    element_type: str = "tet4"  # tet4, hex8, tri3, quad4, etc.

    # Material per element
    element_materials: list[str] = Field(default_factory=list)

    # Boundary conditions
    boundary_nodes: dict[str, list[int]] = Field(
        default_factory=dict,
        description="boundary_name -> list of node indices"
    )

    # Port / excitation regions
    excitation_regions: list[dict[str, Any]] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)
