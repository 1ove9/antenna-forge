"""YAF Core domain models."""

from yaf_core.domain.design import (
    BoundingBox,
    Design,
    DesignSpec,
    DesignState,
    DesignVersion,
    PatternTarget,
    Polarization,
    Region,
)
from yaf_core.domain.geometry import Geometry, Material, MaterialType, Mesh
from yaf_core.domain.optimization import OptimizationRun, Trial
from yaf_core.domain.simulation import (
    PortType,
    SimulationJob,
    SimulationResult,
    SimulationSpec,
    SParamResult,
)

__all__ = [
    "BoundingBox",
    "Design",
    "DesignSpec",
    "DesignState",
    "DesignVersion",
    "Geometry",
    "Material",
    "MaterialType",
    "Mesh",
    "OptimizationRun",
    "PatternTarget",
    "Polarization",
    "PortType",
    "Region",
    "SimulationJob",
    "SimulationResult",
    "SimulationSpec",
    "SParamResult",
    "Trial",
]
