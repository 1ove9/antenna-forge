"""
Space-time modulated antenna and metasurface physics.

Models non-reciprocal and frequency-converting devices through
spatio-temporal modulation of material parameters.

Reference: Caloz & Deck-Leger, "Spacetime Metamaterials",
IEEE TAP, 2020.
"""

from __future__ import annotations

import numpy as np


class SpaceTimeModulator:
    """Space-time modulated transmission line / metasurface.

    Models a structure with time-varying permittivity/permeability:
    ε(x, t) = ε_avg + Δε * cos(ω_m t - β_m x)

    This produces:
    - Non-reciprocal behavior (broken time-reversal symmetry)
    - Frequency conversion (harmonics at f ± n f_m)
    - Parametric amplification
    """

    def __init__(
        self,
        epsilon_avg: float = 2.0,
        delta_epsilon: float = 0.2,
        modulation_freq: float = 1e9,  # Hz
        modulation_wavenumber: float = 0.0,  # rad/m
        length: float = 0.1,  # m
        frequency: float = 10e9,  # operating frequency [Hz]
    ) -> None:
        self.eps_avg = epsilon_avg
        self.delta_eps = delta_epsilon
        self.fm = modulation_freq
        self.beta_m = modulation_wavenumber
        self.length = length
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.omega_m = 2 * np.pi * modulation_freq

    def permittivity(self, x: float, t: float) -> float:
        """Instantaneous permittivity at position x and time t.

        ε(x, t) = ε_avg + Δε * cos(ω_m t - β_m x)
        """
        return float(
            self.eps_avg + self.delta_eps * np.cos(self.omega_m * t - self.beta_m * x)
        )

    def floquet_harmonics(
        self, n_harmonics: int = 3
    ) -> list[tuple[float, complex]]:
        """Compute Floquet harmonic frequencies and amplitudes.

        When ε is modulated at ω_m, the field contains harmonics at
        ω_n = ω + n ω_m, with amplitudes given by Bessel functions.

        Args:
            n_harmonics: Number of harmonics to compute (±n).

        Returns:
            List of (frequency [Hz], complex amplitude) tuples.
        """
        harmonics: list[tuple[float, complex]] = []
        delta_k = self.omega_m / 3e8 * self.length  # phase modulation depth

        for n in range(-n_harmonics, n_harmonics + 1):
            f_n = self.frequency + n * self.fm
            # Amplitude from Bessel function (small modulation approx)
            from scipy.special import jv as bessel_j

            A_n = bessel_j(n, delta_k)
            harmonics.append((f_n, complex(A_n, 0.0)))

        return harmonics

    def non_reciprocity_ratio(
        self, forward_s21: complex, backward_s12: complex
    ) -> float:
        """Compute non-reciprocity isolation ratio [dB].

        NR = 20 log₁₀(|S₂₁| / |S₁₂|)

        A positive value means forward transmission is favored.
        """
        with np.errstate(divide="ignore"):
            ratio = 20 * np.log10(abs(forward_s21) / max(abs(backward_s12), 1e-30))
        return float(ratio)

    def frequency_conversion_efficiency(
        self, input_power_w: float, n_harmonic: int = 1
    ) -> float:
        """Estimate frequency conversion efficiency to n-th harmonic.

        For small modulation: η_n ∝ Δε/ε_avg * J_n(δ_k)

        Args:
            input_power_w: Input power [W].
            n_harmonic: Target harmonic order.

        Returns:
            Estimate of converted power [W].
        """
        modulation_depth = self.delta_eps / self.eps_avg
        delta_k = self.omega_m / 3e8 * self.length

        from scipy.special import jv as bessel_j

        J_n = float(abs(bessel_j(n_harmonic, delta_k)))  # type: ignore[arg-type]
        efficiency = modulation_depth**2 * J_n**2
        return input_power_w * efficiency


class TimeVaryingRIS:
    """Time-varying RIS that can perform frequency conversion.

    By switching RIS element states at rate f_switch, the reflected
    signal acquires a frequency shift:
    f_out = f_in ± m * f_switch
    """

    def __init__(
        self,
        nx: int = 8,
        ny: int = 8,
        switching_freq: float = 1e6,  # Hz
        phase_states: int = 4,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.switching_freq = switching_freq
        self.phase_states = phase_states
        self._time: float = 0.0

    def step(self, dt: float) -> None:
        """Advance time by dt seconds."""
        self._time += dt

    def current_phase_distribution(self, base_phases: np.ndarray) -> np.ndarray:
        """Compute instantaneous phase distribution including time modulation.

        Φ_ij(t) = Φ_base_ij + 2π f_switch t * m_ij

        where m_ij is the time-modulation order per element.
        """
        # Simplification: uniform temporal modulation
        t_mod = self.switching_freq * self._time * 2 * np.pi
        phases = base_phases + t_mod % (2 * np.pi)
        # Quantize to phase_states
        quantized = np.round(phases / (2 * np.pi / self.phase_states))
        quantized = quantized % self.phase_states
        return quantized * (2 * np.pi / self.phase_states)

    def harmonic_beam_pattern(
        self,
        theta: np.ndarray,
        harmonic_order: int,
        base_phases: np.ndarray,
    ) -> np.ndarray:
        """Compute radiation pattern at a specific harmonic frequency.

        The n-th harmonic beam is shifted by the array factor of the
        corresponding time-modulation waveform.

        Args:
            theta: Elevation angles [rad].
            harmonic_order: Harmonic order (0 = fundamental).
            base_phases: Base phase distribution.

        Returns:
            Pattern magnitude at harmonic frequency.
        """
        # Simplified: harmonic beam steers differently
        # due to effective phase gradient change
        wavelength = 3e8 / (self.switching_freq * harmonic_order + 1e9)
        k0 = 2 * np.pi / wavelength

        pattern = np.zeros(len(theta), dtype=complex)
        dx = dy = 0.005  # element spacing

        for i in range(self.nx):
            for j in range(self.ny):
                phase = base_phases[j, i]
                for ti, th in enumerate(theta):
                    u = np.sin(th)
                    arr_phase = k0 * (i * dx * u)
                    pattern[ti] += np.exp(1j * (phase + arr_phase))

        return np.abs(pattern)
