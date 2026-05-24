"""
Parametric antenna geometry generators.

Provides standard antenna templates: dipole, patch, horn, spiral,
log-periodic, and fractal geometries — all parameterized for optimization.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from yaf_core.domain.geometry import Geometry


class ParametricGenerator:
    """Library of parametric antenna geometry generators.

    Each method returns a triangulated mesh Geometry ready for
    simulation or further CAD operations.
    """

    @staticmethod
    def dipole(
        length: float,
        radius: float = 0.001,
        gap: float = 0.001,
        segments: int = 16,
    ) -> Geometry:
        """Generate a half-wave dipole antenna.

        Args:
            length: Total dipole length [m].
            radius: Wire radius [m].
            gap: Feed gap [m].
            segments: Angular resolution per arm.

        Returns:
            Geometry of a center-fed dipole along the z-axis.
        """
        half_len = (length - gap) / 2
        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        def _add_cylinder(z_start: float, z_end: float, base_idx: int) -> int:
            """Add a cylinder segment, return next vertex index."""
            segs = max(8, segments)
            for i in range(segs + 1):
                z = z_start + (z_end - z_start) * i / segs
                for j in range(segs):
                    angle = 2 * math.pi * j / segs
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    vertices.append([x, y, z])
            # Create faces between rings
            ring_v = segs
            for i in range(segs - 1):
                r1 = base_idx + i * ring_v
                r2 = base_idx + (i + 1) * ring_v
                for j in range(ring_v):
                    j2 = (j + 1) % ring_v
                    faces.append([r1 + j, r2 + j, r2 + j2])
                    faces.append([r1 + j, r2 + j2, r1 + j2])
            return base_idx + segs * ring_v

        # Upper arm
        _add_cylinder(gap / 2, half_len + gap / 2, 0)
        # Lower arm
        _add_cylinder(-gap / 2 - half_len, -gap / 2, len(vertices))

        return Geometry(
            name="dipole",
            vertices=vertices,
            faces=faces,
            metadata={"length": length, "radius": radius, "gap": gap},
        )

    @staticmethod
    def rectangular_patch(
        width: float,
        length: float,
        substrate_thickness: float = 0.0016,
        ground_plane: bool = True,
        feed_inset: float = 0.0,
    ) -> Geometry:
        """Generate a rectangular microstrip patch antenna.

        Args:
            width: Patch width [m] (along y).
            length: Patch resonant length [m] (along x).
            substrate_thickness: Substrate height [m].
            ground_plane: Include ground plane.
            feed_inset: Inset feed depth [m].

        Returns:
            Geometry with patch, substrate, and optional ground plane.
        """
        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        # Patch (top)
        hw, hl = width / 2, length / 2
        h = substrate_thickness

        # Patch vertices: 0-3
        vertices.extend([
            [-hl, -hw, h],  # 0
            [hl, -hw, h],   # 1
            [hl, hw, h],    # 2
            [-hl, hw, h],   # 3
        ])
        faces.extend([[0, 1, 2], [0, 2, 3]])  # patch top

        # Substrate block: 4-11
        vertices.extend([
            [-hl, -hw, 0],  # 4
            [hl, -hw, 0],   # 5
            [hl, hw, 0],    # 6
            [-hl, hw, 0],   # 7
            [-hl, -hw, -h], # 8
            [hl, -hw, -h],  # 9
            [hl, hw, -h],   # 10
            [-hl, hw, -h],  # 11
        ])
        # Substrate sides
        faces.extend([
            [4, 5, 1], [4, 1, 0],  # front
            [5, 6, 2], [5, 2, 1],  # right
            [6, 7, 3], [6, 3, 2],  # back
            [7, 4, 0], [7, 0, 3],  # left
        ])

        if ground_plane:
            # Ground plane bottom: 8-11 (inverted for outward normals)
            faces.extend([
                [8, 10, 9], [8, 11, 10],  # ground bottom
                [5, 9, 10], [5, 10, 6],  # substrate bottom sides
                [4, 8, 9], [4, 9, 5],
                [7, 11, 8], [7, 8, 4],
                [6, 10, 11], [6, 11, 7],
            ])
        else:
            faces.extend([
                [5, 9, 10], [5, 10, 6],
                [4, 8, 9], [4, 9, 5],
                [6, 10, 11], [6, 11, 7],
                [7, 11, 8], [7, 8, 4],
            ])

        # Feed inset (simplified)
        if feed_inset > 0:
            inset_x = hl - feed_inset
            # Feed line vertices
            fl_w = width * 0.1
            v_base = len(vertices)
            vertices.extend([
                [inset_x, -fl_w, h],
                [hl, -fl_w, h],
                [hl, fl_w, h],
                [inset_x, fl_w, h],
            ])
            faces.extend([
                [v_base, v_base + 1, v_base + 2],
                [v_base, v_base + 2, v_base + 3],
            ])

        return Geometry(
            name="rectangular_patch",
            vertices=vertices,
            faces=faces,
            metadata={
                "width": width,
                "length": length,
                "substrate_thickness": substrate_thickness,
                "ground_plane": ground_plane,
            },
        )

    @staticmethod
    def horn_antenna(
        aperture_width: float,
        aperture_height: float,
        flare_length: float,
        waveguide_width: float,
        waveguide_height: float,
        waveguide_length: float,
    ) -> Geometry:
        """Generate a pyramidal horn antenna.

        Args:
            aperture_width, aperture_height: Aperture dimensions [m].
            flare_length: Flare section length [m].
            waveguide_width, waveguide_height: Waveguide cross-section [m].
            waveguide_length: Waveguide feed length [m].
        """
        aw2, ah2 = aperture_width / 2, aperture_height / 2
        ww2, wh2 = waveguide_width / 2, waveguide_height / 2
        fl, wl = flare_length, waveguide_length

        vertices: list[list[float]] = [
            # Aperture (larger end, at z=0)
            [-aw2, -ah2, 0], [aw2, -ah2, 0], [aw2, ah2, 0], [-aw2, ah2, 0],
            # Flare-waveguide junction (at z=fl)
            [-ww2, -wh2, fl], [ww2, -wh2, fl], [ww2, wh2, fl], [-ww2, wh2, fl],
            # Waveguide back (at z=fl+wl)
            [-ww2, -wh2, fl + wl], [ww2, -wh2, fl + wl],
            [ww2, wh2, fl + wl], [-ww2, wh2, fl + wl],
        ]

        faces: list[list[int]] = [
            # Flare walls
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
            # Waveguide walls
            [4, 5, 9], [4, 9, 8],
            [5, 6, 10], [5, 10, 9],
            [6, 7, 11], [6, 11, 10],
            [7, 4, 8], [7, 8, 11],
            # Back wall
            [8, 10, 9], [8, 11, 10],
        ]

        return Geometry(
            name="pyramidal_horn",
            vertices=vertices,
            faces=faces,
            metadata={
                "aperture_width": aperture_width,
                "aperture_height": aperture_height,
                "flare_length": flare_length,
            },
        )

    @staticmethod
    def archimedean_spiral(
        inner_radius: float,
        outer_radius: float,
        turns: float,
        arm_width: float,
        segments: int = 200,
    ) -> Geometry:
        """Generate an Archimedean spiral antenna (2-arm).

        Args:
            inner_radius: Start radius [m].
            outer_radius: End radius [m].
            turns: Number of turns.
            arm_width: Width of each arm [m].
            segments: Discretization points per arm.
        """
        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        # Generate spiral curve: r = a + b*theta
        a = inner_radius
        b = (outer_radius - inner_radius) / (turns * 2 * math.pi)

        def _spiral_arm(phase_offset: float) -> list[int]:
            idx_start = len(vertices)
            for i in range(segments):
                theta = (turns * 2 * math.pi * i / segments) + phase_offset
                r = a + b * (theta - phase_offset)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                vertices.append([x, y, 0])
            return list(range(idx_start, len(vertices)))

        arm1 = _spiral_arm(0)
        arm2 = _spiral_arm(math.pi)

        # Connect to create faces (triangle strips)
        for i in range(len(arm1) - 1):
            a1, a2 = arm1[i], arm1[i + 1]
            b1 = arm2[min(i, len(arm2) - 2)]
            b2 = arm2[min(i + 1, len(arm2) - 2)]
            if i < len(arm1) - 2:
                faces.extend([[a1, b1, a2], [a2, b1, b2]])

        # Feed points
        vertices.append([0, 0, 0])
        center_idx = len(vertices) - 1
        faces.extend([
            [center_idx, arm1[0], arm1[1]],
            [center_idx, arm2[0], arm2[1]],
        ])

        return Geometry(
            name="archimedean_spiral",
            vertices=vertices,
            faces=faces,
            metadata={"inner_radius": inner_radius, "outer_radius": outer_radius, "turns": turns},
        )

    @staticmethod
    def sierpinski_gasket(
        order: int, side_length: float, height: float = 0.0
    ) -> Geometry:
        """Generate a Sierpinski gasket fractal antenna.

        Args:
            order: Fractal iteration order (0 = triangle, 1 = first iteration).
            side_length: Side length of the base triangle [m].
            height: Z-height (thickness) [m].
        """
        # Equilateral triangle vertices
        h_tri = side_length * math.sqrt(3) / 2
        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        def _subdivide(
            v0: Any, v1: Any, v2: Any, depth: int
        ) -> None:
            if depth == 0:
                # Add triangle
                base = len(vertices)
                vertices.extend([list(v0), list(v1), list(v2)])
                if height > 0:
                    # Extrude
                    v0h = [v0[0], v0[1], v0[2] + height]
                    v1h = [v1[0], v1[1], v1[2] + height]
                    v2h = [v2[0], v2[1], v2[2] + height]
                    b2 = len(vertices)
                    vertices.extend([v0h, v1h, v2h])
                    faces.extend([
                        [base, base + 1, base + 2],  # bottom
                        [b2, b2 + 2, b2 + 1],  # top
                        [base, b2, b2 + 1], [base, b2 + 1, base + 1],  # sides
                        [base + 1, b2 + 1, b2 + 2], [base + 1, b2 + 2, base + 2],
                        [base + 2, b2 + 2, b2], [base + 2, b2, base],
                    ])
                else:
                    faces.append([base, base + 1, base + 2])
                faces.append([base + 2, base + 1, base])
                return

            # Midpoints
            m01 = (v0 + v1) / 2
            m12 = (v1 + v2) / 2
            m20 = (v2 + v0) / 2
            _subdivide(v0, m01, m20, depth - 1)
            _subdivide(m01, v1, m12, depth - 1)
            _subdivide(m20, m12, v2, depth - 1)

        v0 = np.array([0.0, h_tri * 2 / 3, 0.0])
        v1 = np.array([-side_length / 2, -h_tri / 3, 0.0])
        v2 = np.array([side_length / 2, -h_tri / 3, 0.0])

        _subdivide(v0, v1, v2, order)

        return Geometry(
            name=f"sierpinski_o{order}",
            vertices=vertices,
            faces=faces,
            metadata={"order": order, "side_length": side_length},
        )
