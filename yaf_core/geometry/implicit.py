"""
Implicit Neural Representation (SIREN) for antenna geometry.

Uses periodic activation functions to represent complex 3D geometry
as a continuous field f(x,y,z) -> signed distance, enabling
gradient-based topology optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import random

if TYPE_CHECKING:
    from yaf_core.domain.geometry import Geometry


class SIRENLayer:
    """A single SIREN layer: W * sin(omega_0 * x + b).

    Reference: Sitzmann et al., "Implicit Neural Representations
    with Periodic Activation Functions", NeurIPS 2020.
    """

    def __init__(
        self, in_features: int, out_features: int, omega_0: float, key: jax.Array
    ) -> None:
        self.omega_0 = omega_0
        c = jnp.sqrt(6.0 / in_features) / omega_0
        self.W = jax.random.uniform(key, (in_features, out_features), minval=-c, maxval=c)
        self.b = jnp.zeros(out_features)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(self.omega_0 * (x @ self.W) + self.b)


class SIRENGeometry:
    """SIREN network that represents antenna geometry as a signed distance field.

    The network maps a 3D coordinate (x, y, z) to a signed distance value.
    Positive = inside conductor, Negative = outside (or vice versa depending
    on convention). This enables:

    1. Continuous optimization of geometry via gradient descent
    2. Resolution-independent geometry representation
    3. Direct integration with differentiable physics simulators
    """

    def __init__(
        self,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        key: jax.Array | None = None,
    ) -> None:
        if key is None:
            key = random.PRNGKey(0)

        self.omega_0 = omega_0
        keys = random.split(key, hidden_layers + 2)

        # Input layer (3 -> hidden_features)
        self.layer0 = SIRENLayer(3, hidden_features, omega_0, keys[0])

        # Hidden layers
        self.hidden_layers: list[SIRENLayer] = []
        for i in range(hidden_layers):
            k = keys[i + 1]
            self.hidden_layers.append(
                SIRENLayer(hidden_features, hidden_features, omega_0, k)
            )

        # Output layer (hidden_features -> 1, no sin)
        c = jnp.sqrt(6.0 / hidden_features)
        k_out = keys[-1]
        self.W_out = jax.random.uniform(k_out, (hidden_features, 1), minval=-c, maxval=c)
        self.b_out = jnp.zeros(1)

        self.parameters: list[jnp.ndarray] = []

    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Evaluate SDF at given coordinates.

        Args:
            coords: (N, 3) array of (x, y, z) points.

        Returns:
            (N,) array of signed distance values.
        """
        x = coords
        x = self.layer0(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = x @ self.W_out + self.b_out
        return x.ravel()

    def to_geometry(
        self,
        bounds: tuple[float, float, float, float, float, float],
        resolution: int = 64,
        threshold: float = 0.0,
    ) -> "Geometry":
        """Marching-cubes-like extraction of mesh from SDF.

        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax).
            resolution: Grid resolution per axis.
            threshold: Surface threshold for SDF.

        Returns:
            Triangular mesh Geometry.
        """
        from yaf_core.domain.geometry import Geometry

        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        xs = jnp.linspace(xmin, xmax, resolution)
        ys = jnp.linspace(ymin, ymax, resolution)
        zs = jnp.linspace(zmin, zmax, resolution)
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        sdf = self(points).reshape(resolution, resolution, resolution)

        # Surface = level set at threshold
        # Simple marching cubes: extract cells crossing the threshold
        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        try:
            import numpy as np
            from skimage.measure import marching_cubes  # type: ignore[import-not-found]

            sdf_np = np.array(sdf)
            v, faces_arr, _normals, _values = marching_cubes(sdf_np, level=threshold)
            v[:, 0] = xmin + v[:, 0] / (resolution - 1) * (xmax - xmin)
            v[:, 1] = ymin + v[:, 1] / (resolution - 1) * (ymax - ymin)
            v[:, 2] = zmin + v[:, 2] / (resolution - 1) * (zmax - zmin)
            return Geometry(
                name="siren_geometry",
                vertices=v.tolist(),
                faces=faces_arr.tolist(),
            )
        except ImportError:
            # Fallback: simple extraction of surface points
            for i in range(resolution - 1):
                for j in range(resolution - 1):
                    for k in range(resolution - 1):
                        vals = [
                            sdf[i, j, k],
                            sdf[i + 1, j, k],
                            sdf[i + 1, j + 1, k],
                            sdf[i, j + 1, k],
                            sdf[i, j, k + 1],
                            sdf[i + 1, j, k + 1],
                            sdf[i + 1, j + 1, k + 1],
                            sdf[i, j + 1, k + 1],
                        ]
                        if min(vals) <= threshold <= max(vals):
                            cx = xmin + (i + 0.5) / (resolution - 1) * (xmax - xmin)
                            cy = ymin + (j + 0.5) / (resolution - 1) * (ymax - ymin)
                            cz = zmin + (k + 0.5) / (resolution - 1) * (zmax - zmin)
                            vertices.append([cx, cy, cz])

            # Convert to points (no real mesh, just for visualization)
            if vertices:
                return Geometry(
                    name="siren_geometry",
                    vertices=vertices,
                    faces=[],
                )
            return Geometry(name="empty_siren")
