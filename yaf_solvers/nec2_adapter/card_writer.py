"""
NEC2 card writer — generates NEC-2 input card deck (.nec files).

Reference: NEC-2 User's Guide, Lawrence Livermore National Laboratory.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class NEC2CardWriter:
    """Generates NEC-2 input card deck for wire/structure antennas.

    Supports standard NEC cards:
      CM  — Comment
      CE  — End of comment
      GW  — Geometry wire
      GE  — End of geometry
      GN  — Ground parameters
      EX  — Excitation
      FR  — Frequency
      RP  — Radiation pattern
      EN  — End of run

    Uses standard NEC-2 formatting (FORTRAN-style fixed columns).
    """

    def __init__(self, title: str = "YAF NEC2 Simulation") -> None:
        self.title = title
        self.cards: list[str] = []

    def comment(self, text: str) -> None:
        """Add comment cards."""
        self.cards.append(f"CM {text}")

    def end_comment(self) -> None:
        """End comment section."""
        self.cards.append("CE")

    def gw_card(
        self,
        tag: int,
        segments: int,
        x1: float, y1: float, z1: float,
        x2: float, y2: float, z2: float,
        radius: float,
    ) -> str:
        """Generate a GW (geometry wire) card.

        Args:
            tag: Wire tag number.
            segments: Number of segments.
            x1..z2: Wire endpoints [m].
            radius: Wire radius [m].
        """
        return (
            f"GW {tag:>3d} {segments:>5d} "
            f"{x1:10.4f} {y1:10.4f} {z1:10.4f} "
            f"{x2:10.4f} {y2:10.4f} {z2:10.4f} "
            f"{radius:10.4f}"
        )

    def add_dipole(
        self,
        length: float,
        radius: float = 0.001,
        segments: int = 21,
        tag: int = 1,
    ) -> None:
        """Add a z-oriented half-wave dipole.

        Args:
            length: Total dipole length [m].
            radius: Wire radius [m].
            segments: Number of segments.
            tag: Wire tag.
        """
        half = length / 2
        self.cards.append(
            self.gw_card(tag, segments, 0, 0, -half, 0, 0, half, radius)
        )

    def add_loop(
        self,
        radius: float,
        wire_radius: float = 0.001,
        segments: int = 36,
        tag: int = 1,
        center_z: float = 0.0,
    ) -> None:
        """Add a circular loop in the xy-plane.

        Args:
            radius: Loop radius [m].
            wire_radius: Wire radius [m].
            segments: Number of segments.
            tag: Wire tag.
            center_z: Z-position of the loop.
        """
        for i in range(segments):
            t1 = 2 * math.pi * i / segments
            t2 = 2 * math.pi * (i + 1) / segments
            x1 = radius * math.cos(t1)
            y1 = radius * math.sin(t1)
            x2 = radius * math.cos(t2)
            y2 = radius * math.sin(t2)
            self.cards.append(
                self.gw_card(
                    tag, 1,
                    x1, y1, center_z,
                    x2, y2, center_z,
                    wire_radius,
                )
            )

    def add_yagi(
        self,
        n_elements: int = 3,
        freq: float = 1e9,
        spacing: float | None = None,
        tag_start: int = 1,
    ) -> None:
        """Add a Yagi-Uda array.

        Args:
            n_elements: Number of elements.
            freq: Design frequency [Hz].
            spacing: Element spacing [m]. Default: 0.2λ.
            tag_start: Starting wire tag.
        """
        c0 = 3e8
        wavelength = c0 / freq

        if spacing is None:
            spacing = 0.2 * wavelength

        for i in range(n_elements):
            y_pos = (i - (n_elements - 1) / 2) * spacing
            half_len = wavelength / 4 if i == (n_elements // 2) else wavelength / 4 * 0.95
            self.cards.append(
                self.gw_card(
                    tag_start + i, 11,
                    0, y_pos, -half_len,
                    0, y_pos, half_len,
                    wavelength / 200,
                )
            )

    def ge_card(self, ground_flag: int = 0) -> str:
        """GE (end of geometry) card.

        Args:
            ground_flag: -1 = no ground, 0 = perfect ground, 1 = finite ground.
        """
        return f"GE {ground_flag}"

    def gn_card(
        self,
        ground_type: int = 0,
        n_radials: int = 0,
        eps_r: float = 13.0,
        sigma: float = 0.005,
    ) -> str:
        """GN (ground parameters) card.

        Args:
            ground_type: -1 = none, 0 = perfect, 1 = finite, 2 = Sommerfeld.
            n_radials: For finite ground.
            eps_r: Relative permittivity.
            sigma: Conductivity [S/m].
        """
        return f"GN {ground_type} {n_radials} 0 0 {eps_r:.3f} {sigma:.4f}"

    def ex_card(
        self,
        excitation_type: int = 0,
        tag: int = 1,
        segment: int = 0,
        admittance: tuple[float, float] | None = None,
    ) -> str:
        """EX (excitation) card.

        Args:
            excitation_type: 0 = voltage source, 1 = incident plane wave.
            tag: Wire tag of source.
            segment: Segment number (0 = center).
            admittance: Optional (real, imag) admittance.
        """
        if admittance:
            r, x = admittance
            return f"EX {excitation_type} {tag} {segment} 0 {r:.6f} {x:.6f} 0 0 0 0"
        return f"EX {excitation_type} {tag} {segment} 0 1.0 0.0 0 0 0 0"

    def fr_card(
        self,
        frequency_range: tuple[float, float, int] | float,
    ) -> str:
        """FR (frequency) card.

        Args:
            frequency_range: If tuple: (f_min, f_max, n_steps) in MHz.
                             If float: single frequency in MHz.
        """
        if isinstance(frequency_range, tuple):
            f_min, f_max, n_steps = frequency_range
            if n_steps <= 1:
                return f"FR 0 1 0 0 {f_min:.3f} 0.0"
            step = (f_max - f_min) / (n_steps - 1) if n_steps > 1 else 0
            return f"FR 0 {n_steps} 0 0 {f_min:.3f} {step:.3f}"
        return f"FR 0 1 0 0 {frequency_range:.3f} 0.0"

    def rp_card(
        self,
        theta0: float = 0.0,
        phi0: float = 0.0,
        dtheta: float = 2.0,
        dphi: float = 2.0,
        n_theta: int = 91,
        n_phi: int = 1,
        output_format: int = 0,
    ) -> str:
        """RP (radiation pattern) card.

        Args:
            theta0, phi0: Start angles [degrees].
            dtheta, dphi: Angle increments [degrees].
            n_theta, n_phi: Number of points.
            output_format: 0 = major axis, 1 = minor axis, 2 = both.
        """
        return (
            f"RP {output_format} {n_theta} {n_phi} "
            f"0 0 {theta0:.2f} {phi0:.2f} "
            f"{dtheta:.2f} {dphi:.2f} 0.0 0.0"
        )

    def en_card(self) -> str:
        """EN (end of run) card."""
        return "EN"

    def generate(self) -> str:
        """Generate complete NEC-2 input deck as a string."""
        lines: list[str] = []
        lines.append(f"CM {self.title}")
        lines.append("CE")
        lines.extend(self.cards)
        lines.append(self.en_card())
        return "\n".join(lines)

    def to_bytes(self) -> bytes:
        """Return input deck as UTF-8 bytes."""
        return self.generate().encode("utf-8")

    def write_file(self, filepath: str) -> None:
        """Write input deck to a .nec file."""
        with open(filepath, "w") as f:
            f.write(self.generate())
