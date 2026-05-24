"""openEMS mesh converter — geometry to FDTD Yee grid."""

from __future__ import annotations

import numpy as np
from yaf_core.domain.geometry import Geometry, Mesh


class OpenEMSMeshConverter:
    """Converts canonical YAF geometry to openEMS-compatible mesh.

    Generates structured Yee grid with automatic mesh refinement.
    """

    @staticmethod
    def estimate_resolution(
        geometry: Geometry, points_per_wavelength: int = 20, frequency: float = 10e9
    ) -> float:
        """Estimate mesh cell size for given frequency.

        Args:
            geometry: The antenna geometry.
            points_per_wavelength: Grid points per λ_min.
            frequency: Maximum frequency of interest [Hz].

        Returns:
            Recommended cell size [m].
        """
        c0 = 3e8
        wavelength = c0 / frequency
        return wavelength / points_per_wavelength

    @staticmethod
    def compute_bounding_box(geometry: Geometry) -> tuple[float, float, float, float, float, float]:
        """Compute axis-aligned bounding box from geometry vertices."""
        if not geometry.vertices:
            return (0, 0, 0, 0, 0, 0)
        v = np.array(geometry.vertices)
        return (
            float(v[:, 0].min()), float(v[:, 0].max()),
            float(v[:, 1].min()), float(v[:, 1].max()),
            float(v[:, 2].min()), float(v[:, 2].max()),
        )

    @staticmethod
    def generate_yee_grid(
        bounds: tuple[float, float, float, float, float, float],
        cell_size: float,
        padding_cells: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D Yee grid lines.

        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax) [m].
            cell_size: Cell size [m].
            padding_cells: Extra cells on each side.

        Returns:
            (x_lines, y_lines, z_lines) numpy arrays.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        pad = padding_cells * cell_size
        x_lines = np.arange(xmin - pad, xmax + pad + cell_size, cell_size)
        y_lines = np.arange(ymin - pad, ymax + pad + cell_size, cell_size)
        z_lines = np.arange(zmin - pad, zmax + pad + cell_size, cell_size)
        return x_lines, y_lines, z_lines
