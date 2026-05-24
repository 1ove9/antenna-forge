"""
Topology optimization field representation.

Implements density-based (SIMP) and level-set topology optimization
fields for antenna design. The design domain is discretized as a 3D
grid where each voxel has a continuous density value ρ ∈ [0, 1].

Reference: Bendsoe & Sigmund, "Topology Optimization", Springer 2004.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from yaf_core.domain.geometry import Geometry


class TopologyField:
    """Density-based topology optimization field.

    Each voxel has density ρ ∈ [0, 1] representing material presence.
    The SIMP (Solid Isotropic Material with Penalization) model
    penalizes intermediate densities: E(ρ) = E_min + ρ^p * (E_0 - E_min).
    """

    def __init__(self, resolution: tuple[int, int, int]) -> None:
        """Initialize topology field.

        Args:
            resolution: Grid dimensions (nx, ny, nz).
        """
        self.resolution = resolution
        # Initialize with 0.5 density everywhere
        self.density: np.ndarray = np.full(resolution, 0.5, dtype=np.float64)
        self.penalty: float = 3.0  # SIMP penalty exponent
        self.filter_radius: float = 1.5  # density filter radius in voxels
        self.volume_fraction: float = 0.3  # target volume fraction

    @property
    def shape(self) -> tuple[int, ...]:
        return self.density.shape

    def set_uniform(self, value: float) -> None:
        """Set uniform density."""
        self.density.fill(np.clip(value, 0.0, 1.0))

    def set_sphere(
        self, center: tuple[float, float, float], radius: float, value: float = 1.0
    ) -> None:
        """Set a spherical region to a given density."""
        nx, ny, nz = self.resolution
        cx, cy, cz = center
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    d = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                    if d <= radius:
                        self.density[i, j, k] = value

    def set_box(
        self,
        x_range: tuple[int, int],
        y_range: tuple[int, int],
        z_range: tuple[int, int],
        value: float = 1.0,
    ) -> None:
        """Set a rectangular region to a given density."""
        self.density[x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]] = value

    def apply_density_filter(self) -> None:
        """Apply a density filter (smoothing) to avoid checkerboard patterns.

        Uses a simple convolution with a spherical kernel.
        """
        from scipy.ndimage import uniform_filter

        size = max(3, int(2 * self.filter_radius + 1))
        self.density = uniform_filter(self.density, size=size)

    def apply_heaviside(self, beta: float = 8.0) -> None:
        """Apply a smooth Heaviside projection to binarize densities.

        Args:
            beta: Sharpness parameter (higher = sharper transition).
        """
        # Smooth Heaviside: H_β(ρ) = tanh(βη) + tanh(β(ρ - η)) / (tanh(βη) + tanh(β(1 - η)))
        eta = 0.5
        num = np.tanh(beta * eta) + np.tanh(beta * (self.density - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        self.density = np.clip(num / den, 0.0, 1.0)

    def compute_volume(self) -> float:
        """Compute total volume fraction."""
        return float(np.mean(self.density))

    def compute_compliance(self, sensitivity: np.ndarray) -> float:
        """Compute compliance (objective) from sensitivity field.

        C = Σ (ρ^p * sensitivity)
        """
        penalized = self.density ** self.penalty
        return float(np.sum(penalized * sensitivity))

    def update_density(
        self,
        sensitivity: np.ndarray,
        learning_rate: float = 0.1,
        move_limit: float = 0.2,
    ) -> None:
        """Update density using optimality criteria (OC) method.

        Args:
            sensitivity: ∂C/∂ρ for each voxel (negative = improve).
            learning_rate: Step size for update.
            move_limit: Max density change per iteration.
        """
        # Damping factor for stability
        lagrange = 1.0
        # Updated density: ρ_new = ρ * (-∂C/∂ρ / λ)^η
        eta = 0.5
        be = (-sensitivity.clip(min=0) / lagrange) ** eta
        new_density = self.density * be.clip(min=0)

        # Apply move limits
        lower = np.maximum(0.0, self.density - move_limit)
        upper = np.minimum(1.0, self.density + move_limit)
        self.density = np.clip(new_density, lower, upper)

    def to_geometry(
        self,
        bounds: tuple[float, float, float, float, float, float],
        threshold: float = 0.5,
    ) -> "Geometry":
        """Convert density field to a mesh geometry via marching cubes.

        Args:
            bounds: Physical bounding box (xmin, xmax, ymin, ymax, zmin, zmax).
            threshold: Density threshold for material boundary.

        Returns:
            Triangular mesh Geometry.
        """
        from yaf_core.domain.geometry import Geometry

        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        nx, ny, nz = self.resolution

        try:
            from skimage.measure import marching_cubes  # type: ignore[import-not-found]

            verts, faces_arr, _normals, _values = marching_cubes(self.density, threshold)

            class _MeshShim:
                vertices = verts
                faces = faces_arr

            mesh = _MeshShim()
            v = mesh.vertices.copy()
            v[:, 0] = xmin + v[:, 0] / (nx - 1) * (xmax - xmin)
            v[:, 1] = ymin + v[:, 1] / (ny - 1) * (ymax - ymin)
            v[:, 2] = zmin + v[:, 2] / (nz - 1) * (zmax - zmin)
            return Geometry(
                name="topology_opt_geometry",
                vertices=v.tolist(),
                faces=mesh.faces.tolist(),
            )
        except ImportError:
            # Fallback: voxel-based geometry
            vertices: list[list[float]] = []
            faces: list[list[int]] = []
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            dz = (zmax - zmin) / nz

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if self.density[i, j, k] > threshold:
                            x = xmin + (i + 0.5) * dx
                            y = ymin + (j + 0.5) * dy
                            z = zmin + (k + 0.5) * dz
                            v_base = len(vertices)
                            # Simple voxel cube
                            vertices.extend([
                                [x - dx / 2, y - dy / 2, z - dz / 2],
                                [x + dx / 2, y - dy / 2, z - dz / 2],
                                [x + dx / 2, y + dy / 2, z - dz / 2],
                                [x - dx / 2, y + dy / 2, z - dz / 2],
                                [x - dx / 2, y - dy / 2, z + dz / 2],
                                [x + dx / 2, y - dy / 2, z + dz / 2],
                                [x + dx / 2, y + dy / 2, z + dz / 2],
                                [x - dx / 2, y + dy / 2, z + dz / 2],
                            ])
                            faces.extend([
                                [v_base, v_base + 1, v_base + 2],
                                [v_base, v_base + 2, v_base + 3],
                                [v_base + 4, v_base + 6, v_base + 5],
                                [v_base + 4, v_base + 7, v_base + 6],
                                [v_base, v_base + 5, v_base + 1],
                                [v_base, v_base + 4, v_base + 5],
                                [v_base + 1, v_base + 6, v_base + 2],
                                [v_base + 1, v_base + 5, v_base + 6],
                                [v_base + 2, v_base + 7, v_base + 3],
                                [v_base + 2, v_base + 6, v_base + 7],
                                [v_base + 3, v_base + 4, v_base + 0],
                                [v_base + 3, v_base + 7, v_base + 4],
                            ])

            return Geometry(
                name="topology_opt_geometry",
                vertices=vertices,
                faces=faces,
            )
