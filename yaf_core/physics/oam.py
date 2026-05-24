"""
Orbital Angular Momentum (OAM) antenna physics model.

Models OAM mode generation, purity analysis, and vortex beam
characteristics for radio-frequency OAM communications.

Reference: Thide et al., "Utilization of Photon Orbital Angular
Momentum in the Low-Frequency Radio Domain", PRL, 2007.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv  # Bessel functions


class OAMModel:
    """Orbital Angular Momentum antenna analysis model.

    An OAM beam of topological charge ℓ has azimuthal phase dependence
    exp(jℓφ), producing a donut-shaped intensity pattern with a phase
    singularity at the center.
    """

    def __init__(self, topological_charge: int = 1, frequency: float = 10e9) -> None:
        """Initialize OAM model.

        Args:
            topological_charge: OAM mode number ℓ (integer).
            frequency: Operating frequency [Hz].
        """
        self.topological_charge = topological_charge
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self.k0 = 2 * np.pi / self.wavelength

    def near_field(
        self, rho: np.ndarray, phi: np.ndarray, z: float = 0.0
    ) -> np.ndarray:
        """Compute near-field complex amplitude for vortex beam.

        E(ρ, φ, z) ∝ (ρ/w₀)^|ℓ| * exp(-ρ²/w₀²) * exp(jℓφ)

        Args:
            rho: Radial distance [m] (can be 1D or 2D).
            phi: Azimuthal angle [rad] (same shape as rho).
            z: Axial distance [m].

        Returns:
            Complex field amplitude.
        """
        # Beam waist (scales with wavelength)
        w0 = self.wavelength * 2.0

        # Laguerre-Gaussian LG_0ℓ mode
        amplitude = (rho / w0) ** abs(self.topological_charge)
        amplitude *= np.exp(-(rho**2) / w0**2)
        phase = np.exp(1j * self.topological_charge * phi)

        # Gouy phase and curvature for z > 0
        if abs(z) > 0:
            zr = np.pi * w0**2 / self.wavelength  # Rayleigh range
            wz = w0 * np.sqrt(1 + (z / zr) ** 2)
            amplitude = (rho / wz) ** abs(self.topological_charge)
            amplitude *= np.exp(-(rho**2) / wz**2)
            # Curvature phase
            Rz = z * (1 + (zr / z) ** 2) if abs(z) > 0 else float("inf")
            if np.isfinite(Rz):
                phase *= np.exp(-1j * self.k0 * rho**2 / (2 * Rz))
            # Gouy phase
            gouy = (abs(self.topological_charge) + 1) * np.arctan(z / zr)
            phase *= np.exp(1j * gouy)

        return amplitude * phase

    def far_field_pattern(
        self, theta: np.ndarray, phi: np.ndarray, radius: float = 0.05
    ) -> np.ndarray:
        """Compute far-field radiation pattern for OAM mode.

        Uses the circular aperture radiation integral:
        E(θ, φ) ∝ j^ℓ * e^{jℓφ} * ∫₀^a J_ℓ(k₀ρ sinθ) E_a(ρ) ρ dρ

        Args:
            theta: Elevation angles [rad].
            phi: Azimuth angles [rad].
            radius: Effective aperture radius [m].

        Returns:
            Complex far-field pattern.
        """
        pattern = np.zeros((len(theta), len(phi)), dtype=complex)
        ell = abs(self.topological_charge)

        for ti, th in enumerate(theta):
            # Radial integral using Bessel function
            argument = self.k0 * radius * np.sin(th)
            J_ell = jv(ell, argument)  # Bessel function of order ℓ

            # Approximate far-field for uniform circular aperture
            if argument > 0:
                radial = radius**2 * J_ell / argument
            else:
                radial = 0.0 if ell > 0 else radius**2 / 2

            for pi, ph in enumerate(phi):
                pattern[ti, pi] = radial * np.exp(1j * ell * ph)

        return pattern

    def intensity_pattern(self, theta: np.ndarray, phi: np.ndarray, radius: float = 0.05) -> np.ndarray:
        """Compute normalized intensity pattern [dB]."""
        ff = self.far_field_pattern(theta, phi, radius)
        intensity = np.abs(ff) ** 2
        max_val = np.max(intensity)
        if max_val > 0:
            return np.asarray(10 * np.log10(np.maximum(intensity / max_val, 1e-30)))
        return np.zeros_like(intensity)

    def mode_purity(self, test_phase: np.ndarray, phi: np.ndarray) -> float:
        """Compute OAM mode purity from phase measurement.

        Purity = |∫ E(φ) e^{-jℓφ} dφ|² / ∫ |E(φ)|² dφ

        Args:
            test_phase: Complex field around a ring.
            phi: Azimuthal angles [rad] (same length).

        Returns:
            Mode purity ∈ [0, 1].
        """
        signal = test_phase * np.exp(-1j * self.topological_charge * phi)
        weighted = np.trapezoid(signal, phi)
        power = np.trapezoid(np.abs(test_phase) ** 2, phi)
        if power > 0:
            purity = np.abs(weighted) ** 2 / (power * (phi[-1] - phi[0]))
            return float(np.clip(purity, 0.0, 1.0))
        return 0.0

    def phase_singularity_position(
        self, phase_field: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Locate the phase singularity from a 2D phase map.

        The singularity is where the phase circulation integral is non-zero:
        ∮ ∇Φ · dl = 2πℓ

        Args:
            phase_field: 2D phase map [rad].
            x, y: Coordinate arrays.

        Returns:
            (x_singularity, y_singularity) position.
        """
        # Compute phase gradient
        dp_dx = np.gradient(phase_field, x, axis=0)
        dp_dy = np.gradient(phase_field, y, axis=1)

        # Find minimum amplitude gradient (singularity)
        grad_mag = np.sqrt(dp_dx**2 + dp_dy**2)
        # The singularity is where phase gradient is ill-defined
        # Simplified: find point where local phase variance is highest
        from scipy.ndimage import generic_filter

        local_var = generic_filter(phase_field, np.std, size=3)
        idx = np.unravel_index(np.argmax(local_var), phase_field.shape)
        return float(x[idx[0]]), float(y[idx[1]])
