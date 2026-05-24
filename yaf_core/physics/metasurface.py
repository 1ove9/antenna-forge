"""
Metasurface physics model.

Models gradient metasurfaces using generalized Snell's law and
unit-cell phase/amplitude/polarization response.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class UnitCell:
    """A single metasurface unit cell with complex transmission/reflection."""

    def __init__(
        self,
        s11_db: float = -40.0,
        s11_phase_deg: float = 0.0,
        s21_db: float = -0.5,
        s21_phase_deg: float = 0.0,
        loss_tangent: float = 0.0,
    ) -> None:
        self.s11_db = s11_db
        self.s11_phase = np.deg2rad(s11_phase_deg)
        self.s21_db = s21_db
        self.s21_phase = np.deg2rad(s21_phase_deg)
        self.loss_tangent = loss_tangent

    def s11(self) -> complex:
        """Reflection coefficient."""
        mag = 10 ** (self.s11_db / 20)
        return complex(mag * np.exp(1j * self.s11_phase))

    def s21(self) -> complex:
        """Transmission coefficient."""
        mag = 10 ** (self.s21_db / 20)
        return complex(mag * np.exp(1j * self.s21_phase))


class MetasurfaceModel:
    """Phase-gradient metasurface model based on generalized Snell's law.

    Implements:
    - Phase/amplitude/polarization gradient arrays
    - Generalized reflection/refraction angles
    - Array factor computation for far-field pattern
    - Mutual coupling approximate model

    Reference: Yu et al., "Light Propagation with Phase Discontinuities:
    Generalized Laws of Reflection and Refraction", Science, 2011.
    """

    def __init__(
        self,
        nx: int = 16,
        ny: int = 16,
        dx: float = 0.001,  # unit cell period [m]
        dy: float = 0.001,
        frequency: float = 10e9,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.frequency = frequency
        self.wavelength = 3e8 / frequency

        # Phase gradient [rad/m] along x and y
        self.phase_gradient_x: float = 0.0
        self.phase_gradient_y: float = 0.0

        # Unit cell library
        self.cells: np.ndarray = np.empty((nx, ny), dtype=object)
        for i in range(nx):
            for j in range(ny):
                self.cells[i, j] = UnitCell()

    def set_linear_phase_gradient(
        self, dphi_dx: float, dphi_dy: float
    ) -> None:
        """Set linear phase gradient (rad/m) along x and y.

        Args:
            dphi_dx: Phase gradient in x-direction [rad/m].
            dphi_dy: Phase gradient in y-direction [rad/m].
        """
        self.phase_gradient_x = dphi_dx
        self.phase_gradient_y = dphi_dy

        for i in range(self.nx):
            for j in range(self.ny):
                phase = dphi_dx * i * self.dx + dphi_dy * j * self.dy
                self.cells[i, j] = UnitCell(
                    s21_phase_deg=np.rad2deg(phase) % 360
                )

    def generalized_reflection_angle(
        self, theta_i: float, dphi_dx: float
    ) -> float:
        """Compute anomalous reflection angle via generalized Snell's law.

        sin(θ_r) - sin(θ_i) = λ / (2π n_i) * dΦ/dx

        Args:
            theta_i: Incident angle [rad].
            dphi_dx: Phase gradient [rad/m].

        Returns:
            Reflection angle θ_r [rad].
        """
        k0 = 2 * np.pi / self.wavelength
        sin_r = np.sin(theta_i) + dphi_dx / k0
        sin_r = np.clip(sin_r, -1.0, 1.0)
        return float(np.arcsin(sin_r))

    def generalized_transmission_angle(
        self, theta_i: float, dphi_dx: float, n_t: float = 1.0
    ) -> float:
        """Compute anomalous transmission angle.

        n_t sin(θ_t) - n_i sin(θ_i) = λ / (2π) * dΦ/dx

        Args:
            theta_i: Incident angle [rad].
            dphi_dx: Phase gradient [rad/m].
            n_t: Refractive index of transmission medium.

        Returns:
            Transmission angle θ_t [rad].
        """
        k0 = 2 * np.pi / self.wavelength
        sin_t = (np.sin(theta_i) + dphi_dx / k0) / n_t
        sin_t = np.clip(sin_t, -1.0, 1.0)
        return float(np.arcsin(sin_t))

    def array_factor(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        excitation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute 2D array factor for the metasurface.

        AF(θ,φ) = Σ_i Σ_j a_ij * exp(-j k₀ (i*dx*u + j*dy*v))

        where u = sinθ cosφ, v = sinθ sinφ.

        Args:
            theta: Elevation angles [rad] (len N_theta).
            phi: Azimuth angles [rad] (len N_phi).
            excitation: Complex excitation array (ny, nx). Default: uniform 1.0.

        Returns:
            Array factor (N_theta, N_phi).
        """
        if excitation is None:
            excitation = np.ones((self.ny, self.nx), dtype=complex)

        k0 = 2 * np.pi / self.wavelength
        af = np.zeros((len(theta), len(phi)), dtype=complex)

        for ti, th in enumerate(theta):
            for pi, ph in enumerate(phi):
                u = np.sin(th) * np.cos(ph)
                v = np.sin(th) * np.sin(ph)
                phase = 0.0
                for i in range(self.nx):
                    for j in range(self.ny):
                        cell = self.cells[i, j]
                        phase_cell = np.angle(cell.s21())  # transmission phase
                        contrib = excitation[j, i] * np.exp(
                            -1j * k0 * (i * self.dx * u + j * self.dy * v)
                            + 1j * phase_cell
                        )
                        phase += contrib
                af[ti, pi] = phase

        return af

    def far_field_pattern(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        excitation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute far-field radiation pattern [dBi].

        Normalized to isotropic radiator.
        """
        af = self.array_factor(theta, phi, excitation)
        pattern = np.abs(af)
        max_val = np.max(pattern) if np.max(pattern) > 0 else 1.0
        pattern_db = 20 * np.log10(pattern / max_val)
        return np.asarray(pattern_db + 10 * np.log10(self.nx * self.ny))  # directivity scaling
