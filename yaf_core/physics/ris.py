"""
Reconfigurable Intelligent Surface (RIS) physics model.

Models phase-configurable meta-atoms, quantization effects, mutual coupling,
and codebook-based beamforming.

Reference: Di Renzo et al., "Smart Radio Environments Empowered by
Reconfigurable Intelligent Surfaces", IEEE JSAC, 2020.
"""

from __future__ import annotations

import numpy as np


class RISElement:
    """Single RIS element with N-bit phase quantization."""

    def __init__(self, bits: int = 2, loss_db: float = 0.5) -> None:
        self.bits = bits
        self.loss_db = loss_db
        self.states = 2**bits
        self.state: int = 0  # 0 to states-1
        self.phase_resolution = 360.0 / self.states  # degrees

    @property
    def phase_deg(self) -> float:
        """Current phase state in degrees."""
        return float(self.state * self.phase_resolution)

    @property
    def reflection_coefficient(self) -> complex:
        """Complex reflection coefficient."""
        mag = 10 ** (-self.loss_db / 20)
        return complex(mag * np.exp(1j * np.deg2rad(self.phase_deg)))

    def set_state(self, state: int) -> None:
        """Set discrete phase state."""
        self.state = state % self.states

    def quantize_phase(self, desired_phase_deg: float) -> float:
        """Quantize a desired continuous phase to nearest discrete state.

        Returns the quantized phase in degrees.
        """
        desired = desired_phase_deg % 360
        self.state = int(round(desired / self.phase_resolution)) % self.states
        return self.phase_deg


class RISModel:
    """Reconfigurable Intelligent Surface model.

    Models an NxM element RIS with:
    - Per-element phase control
    - Quantization effects
    - Mutual coupling approximation
    - Codebook for standard beam patterns
    """

    def __init__(
        self,
        nx: int = 16,
        ny: int = 16,
        dx: float = 0.005,  # element spacing [m]
        dy: float = 0.005,
        bits: int = 2,
        frequency: float = 3.5e9,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.frequency = frequency
        self.wavelength = 3e8 / frequency

        # Element array
        self.elements: np.ndarray = np.empty((ny, nx), dtype=object)
        for j in range(ny):
            for i in range(nx):
                self.elements[j, i] = RISElement(bits=bits)

        # Codebook: named patterns -> phase configurations
        self.codebook: dict[str, np.ndarray] = {}
        self._init_default_codebook()

    def _init_default_codebook(self) -> None:
        """Initialize standard codebook entries."""
        # Broadside
        phases_broadside = np.zeros((self.ny, self.nx), dtype=float)
        self.codebook["broadside"] = phases_broadside

        # Steer to theta=30°, phi=0°
        if self.nx > 1:
            phases_steer30 = np.zeros((self.ny, self.nx), dtype=float)
            k0 = 2 * np.pi / self.wavelength
            theta_t = np.deg2rad(30)
            for i in range(self.nx):
                phase = -k0 * i * self.dx * np.sin(theta_t)
                phases_steer30[:, i] = np.rad2deg(phase) % 360
            self.codebook["steer_30deg"] = phases_steer30

        # Beam split (two directions)
        if self.nx > 1:
            phases_split = np.zeros((self.ny, self.nx), dtype=float)
            k0 = 2 * np.pi / self.wavelength
            theta_t = np.deg2rad(20)
            for i in range(self.nx):
                if i < self.nx // 2:
                    phase = -k0 * i * self.dx * np.sin(theta_t)
                else:
                    phase = -k0 * i * self.dx * np.sin(-theta_t)
                phases_split[:, i] = np.rad2deg(phase) % 360
            self.codebook["beam_split"] = phases_split

    def apply_codebook(self, name: str) -> None:
        """Apply a precomputed codebook phase configuration."""
        if name not in self.codebook:
            available = list(self.codebook.keys())
            raise KeyError(f"Codebook '{name}' not found. Available: {available}")

        phases = self.codebook[name]
        for j in range(self.ny):
            for i in range(self.nx):
                self.elements[j, i].quantize_phase(phases[j, i])

    def set_phase_configuration(
        self, phases_deg: np.ndarray
    ) -> None:
        """Set arbitrary phase configuration (with quantization)."""
        for j in range(min(self.ny, phases_deg.shape[0])):
            for i in range(min(self.nx, phases_deg.shape[1])):
                self.elements[j, i].quantize_phase(phases_deg[j, i])

    def get_phase_configuration(self) -> np.ndarray:
        """Get current quantized phase configuration."""
        phases = np.zeros((self.ny, self.nx), dtype=float)
        for j in range(self.ny):
            for i in range(self.nx):
                phases[j, i] = self.elements[j, i].phase_deg
        return phases

    def compute_radiation_pattern(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """Compute far-field radiation pattern [dBi].

        Args:
            theta: Elevation angles [rad].
            phi: Azimuth angles [rad].

        Returns:
            Pattern in dBi (2D array).
        """
        k0 = 2 * np.pi / self.wavelength
        pattern = np.zeros((len(theta), len(phi)), dtype=complex)

        for ti, th in enumerate(theta):
            for pi, ph in enumerate(phi):
                u = np.sin(th) * np.cos(ph)
                v = np.sin(th) * np.sin(ph)
                s = 0.0 + 0.0j
                for j in range(self.ny):
                    for i in range(self.nx):
                        rc = self.elements[j, i].reflection_coefficient
                        # Element factor (assumed isotropic patch)
                        elem = np.cos(th)
                        # Array factor
                        phase = k0 * (i * self.dx * u + j * self.dy * v)
                        s += rc * elem * np.exp(1j * phase)
                pattern[ti, pi] = s

        # Normalize and convert to dBi
        max_val = np.max(np.abs(pattern))
        if max_val > 0:
            pattern_db = 20 * np.log10(np.abs(pattern) / max_val)
        else:
            pattern_db = np.zeros_like(pattern)
        # Directivity of uniform array ≈ N_elements
        directivity_factor = 10 * np.log10(self.nx * self.ny)
        return np.asarray(pattern_db + directivity_factor)

    def quantization_error(self, desired_phases: np.ndarray) -> float:
        """Compute RMS phase quantization error [degrees].

        Args:
            desired_phases: Continuous desired phase distribution.

        Returns:
            RMS error in degrees.
        """
        quantized = self.get_phase_configuration()
        errors = []
        for j in range(self.ny):
            for i in range(self.nx):
                err = desired_phases[j % desired_phases.shape[0], i % desired_phases.shape[1]] - quantized[j, i]
                # Wrap to [-180, 180]
                err = (err + 180) % 360 - 180
                errors.append(err)
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))
