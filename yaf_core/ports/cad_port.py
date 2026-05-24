"""
CADBackend Protocol — CAD import/export and geometry manipulation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from yaf_core.domain.geometry import Geometry


@runtime_checkable
class CADBackend(Protocol):
    """Protocol for CAD engine backends (FreeCAD, Blender, OpenCASCADE, etc.)."""

    name: str
    supported_formats: list[str]  # ["step", "stl", "iges", "gdsii"]

    async def import_geometry(self, file_path: str, format: str) -> Geometry:
        """Import geometry from a CAD file.

        Args:
            file_path: Path to the CAD file.
            format: File format ("step", "stl", "iges", "gdsii").

        Returns:
            Canonical Geometry.
        """
        ...

    async def export_geometry(
        self, geometry: Geometry, file_path: str, format: str
    ) -> None:
        """Export canonical geometry to a CAD file.

        Args:
            geometry: Canonical Geometry.
            file_path: Output path.
            format: Target format.
        """
        ...

    async def boolean_union(self, a: Geometry, b: Geometry) -> Geometry:
        """Compute boolean union of two geometries."""
        ...

    async def boolean_difference(self, a: Geometry, b: Geometry) -> Geometry:
        """Compute boolean difference (a - b)."""
        ...

    async def offset_surface(self, geometry: Geometry, distance: float) -> Geometry:
        """Offset a surface by distance."""
        ...

    async def health_check(self) -> bool:
        """Check CAD engine availability."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
