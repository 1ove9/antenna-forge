# ============================================================
# REFERENCE
#   仿造来源：无（首创实现）
#   理由：DesignSpec/DesignVersion/DesignDB 为 YAF 专属领域聚合，
#        天线设计的状态机和版本管理无现成开源对标。
#   关键设计点：
#     - DesignState 状态机：Draft → Generating → Meshing → Solving → Solved|Failed|Archived
#     - DesignSpec 作为用户合同：频率/增益/极化/尺寸/材料约束
#     - PatternTarget 远场方向图目标描述（主瓣/波束宽度/旁瓣/零陷）
#     - DesignVersion 不可变快照，geometry_hash + params 双绑定
#     - BoundingBox + Region 三维约束体系
# ============================================================

"""
Design domain model.

A Design progresses through states:
Draft → Generating → Meshing → Solving → Solved | Failed
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class DesignState(str, enum.Enum):
    """Design lifecycle state machine."""

    DRAFT = "draft"
    GENERATING = "generating"
    MESHING = "meshing"
    SOLVING = "solving"
    SOLVED = "solved"
    FAILED = "failed"
    ARCHIVED = "archived"

    @property
    def is_terminal(self) -> bool:
        return self in (DesignState.SOLVED, DesignState.FAILED, DesignState.ARCHIVED)

    @property
    def is_active(self) -> bool:
        return self in (DesignState.GENERATING, DesignState.MESHING, DesignState.SOLVING)


class Polarization(str, enum.Enum):
    LINEAR = "linear"
    CIRCULAR = "circular"
    DUAL = "dual"
    ELLIPTICAL = "elliptical"


class PatternTarget(BaseModel):
    """Far-field radiation pattern target.

    Attributes:
        theta_range: Elevation angle range [degrees].
        phi_range: Azimuth angle range [degrees].
        main_lobe_direction: (theta, phi) of main lobe peak [degrees].
        beamwidth_3db: 3 dB beamwidth target [degrees].
        side_lobe_level_db: Max allowed side lobe level [dB].
        null_directions: (theta, phi) pairs for pattern nulls.
    """

    theta_range: tuple[float, float] = (0.0, 180.0)
    phi_range: tuple[float, float] = (0.0, 360.0)
    main_lobe_direction: tuple[float, float] = (90.0, 0.0)
    beamwidth_3db: float | None = None
    side_lobe_level_db: float | None = None
    null_directions: list[tuple[float, float]] = Field(default_factory=list)


class BoundingBox(BaseModel):
    """Axis-aligned 3D bounding box in meters."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return (self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min)

    @property
    def volume(self) -> float:
        dx, dy, dz = self.dimensions
        return dx * dy * dz


class Region(BaseModel):
    """Named 3D region (forbidden zone, keep-out area, etc.)."""

    name: str
    box: BoundingBox
    reason: str = ""


class DesignSpec(BaseModel):
    """User-provided design specification — the input contract for the platform.

    All downstream generation and optimization targets derive from this spec.
    """

    name: str
    frequency_range: tuple[float, float]  # Hz
    target_gain_dbi: float | None = None
    target_pattern: PatternTarget | None = None
    polarization: Polarization = Polarization.LINEAR
    bandwidth_target: float | None = None  # fractional, 0-1
    efficiency_target: float | None = None  # 0-1
    size_constraint: BoundingBox
    material_palette: list[str] = Field(default_factory=list)
    forbidden_regions: list[Region] = Field(default_factory=list)
    novel_physics: list[str] = Field(default_factory=list)
    target_vswr: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Design(BaseModel):
    """Persistent design entity — root aggregate."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    spec: DesignSpec
    state: DesignState = DesignState.DRAFT
    current_version: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)
    parent_design_id: uuid.UUID | None = None

    model_config = {"extra": "allow"}


class DesignVersion(BaseModel):
    """Immutable snapshot of a design's geometry at a point in time."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    design_id: uuid.UUID
    version: int
    geometry_hash: str
    params: dict[str, Any] = Field(default_factory=dict)
    artifact_uri: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    commit_message: str = ""
