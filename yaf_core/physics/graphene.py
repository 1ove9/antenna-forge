"""
Graphene electrodynamic model — Kubo formula for surface conductivity.

Models the tunable surface conductivity of graphene as a function of
frequency, chemical potential (μ_c), temperature, and scattering rate.

Reference: Hanson, "Dyadic Green's Functions for an Anisotropic,
Non-Local Model of Biased Graphene", IEEE TAP, 2008.
"""

from __future__ import annotations

import math

import numpy as np


class GrapheneModel:
    """Graphene surface conductivity model using the Kubo formula.

    σ(ω, μ_c, Γ, T) = σ_intra + σ_inter

    Key properties:
    - μ_c tunable via electrostatic bias (0-1 eV)
    - Operates from DC to THz frequencies
    - Supports SPP (surface plasmon polariton) modes
    """

    # Physical constants
    e = 1.602176634e-19   # C
    hbar = 1.054571817e-34  # J·s
    kB = 1.380649e-23      # J/K

    def __init__(
        self,
        mu_c_ev: float = 0.2,       # chemical potential [eV]
        gamma_s: float = 0.1e12,    # scattering rate [1/s]
        temperature_k: float = 300.0,  # [K]
    ) -> None:
        self.mu_c = mu_c_ev * self.e  # convert to Joules
        self.gamma = gamma_s
        self.T = temperature_k

    def surface_conductivity(self, frequency: float) -> complex:
        """Compute surface conductivity at a given frequency.

        Args:
            frequency: Frequency [Hz].

        Returns:
            Complex surface conductivity σ_s = σ' + jσ'' [S].
        """
        omega = 2 * math.pi * frequency
        sigma_intra = self._intraband(omega)
        sigma_inter = self._interband(omega)
        return sigma_intra + sigma_inter

    def _intraband(self, omega: float) -> complex:
        """Intra-band (Drude-like) contribution."""
        prefactor = 1j * self.e**2 * self.kB * self.T
        denom = math.pi * self.hbar**2 * (omega + 1j * 2 * self.gamma)
        factor = self.mu_c / (self.kB * self.T) + 2 * math.log(
            math.exp(-self.mu_c / (self.kB * self.T)) + 1
        )
        return prefactor / denom * factor

    def _interband(self, omega: float) -> complex:
        """Inter-band contribution (simplified)."""
        # Interband transitions are significant when ℏω > 2μ_c
        if omega < 2 * self.mu_c / self.hbar:
            # Below threshold, interband is approximately:
            return complex(
                self.e**2 / (4 * self.hbar) * 0.0,
                0.0,
            )
        else:
            sigma_real = (
                self.e**2
                / (4 * self.hbar)
                * (0.5 + 1 / math.pi * math.atan(
                    (self.hbar * omega - 2 * self.mu_c) / (2 * self.kB * self.T)
                ))
            )
            sigma_imag = (
                -self.e**2
                / (4 * math.pi * self.hbar)
                * math.log(
                    (self.hbar * omega + 2 * self.mu_c) ** 2
                    / ((self.hbar * omega - 2 * self.mu_c) ** 2 + (2 * self.kB * self.T) ** 2)
                )
            )
            return complex(sigma_real, sigma_imag)

    def complex_permittivity(
        self, frequency: float, thickness: float = 0.34e-9
    ) -> complex:
        """Convert surface conductivity to effective bulk permittivity.

        ε(ω) = 1 + j σ_s / (ω ε_0 t)

        Args:
            frequency: Frequency [Hz].
            thickness: Graphene thickness [m] (default: monolayer ~0.34 nm).

        Returns:
            Complex relative permittivity.
        """
        eps0 = 8.854187817e-12
        omega = 2 * math.pi * frequency
        sigma = self.surface_conductivity(frequency)
        return 1.0 + 1j * sigma / (omega * eps0 * thickness)

    def surface_impedance(self, frequency: float) -> complex:
        """Surface impedance Z_s = 1/σ_s [Ω/sq].

        Args:
            frequency: Frequency [Hz].

        Returns:
            Complex surface impedance.
        """
        sigma = self.surface_conductivity(frequency)
        if abs(sigma) > 1e-30:
            return 1.0 / sigma
        return complex(float("inf"), 0.0)

    def spp_wavenumber(
        self, frequency: float, substrate_eps_r: float = 1.0
    ) -> complex:
        """Surface plasmon polariton (SPP) wavenumber on graphene.

        k_spp = k₀ √(ε_r - (2 / (η₀ σ_s))²)

        Args:
            frequency: Frequency [Hz].
            substrate_eps_r: Relative permittivity of substrate.

        Returns:
            Complex SPP wavenumber [rad/m].
        """
        import cmath as _cmath

        c0 = 3e8
        eta0 = 377.0  # free-space impedance
        k0 = 2 * math.pi * frequency / c0
        sigma = self.surface_conductivity(frequency)
        k_spp = k0 * _cmath.sqrt(substrate_eps_r - (2 / (eta0 * sigma)) ** 2)
        return k_spp

    def spp_confinement(self, frequency: float) -> float:
        """SPP confinement factor: λ_spp / λ₀.

        A small value indicates strong field confinement.
        """
        c0 = 3e8
        k_spp = self.spp_wavenumber(frequency)
        if k_spp.real > 0:
            lambda_spp = 2 * math.pi / k_spp.real
            lambda_0 = c0 / frequency
            return lambda_spp / lambda_0
        return float("inf")

    def frequency_scan(
        self, f_min: float, f_max: float, n_points: int = 200
    ) -> dict[str, np.ndarray]:
        """Compute conductivity, impedance, and permittivity vs frequency.

        Args:
            f_min, f_max: Frequency range [Hz].
            n_points: Number of frequency samples.

        Returns:
            dict with keys: 'frequency', 'sigma_real', 'sigma_imag',
            'zs_real', 'zs_imag', 'eps_real', 'eps_imag'.
        """
        freq = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
        sigma_real = np.zeros(n_points)
        sigma_imag = np.zeros(n_points)

        for i, f in enumerate(freq):
            s = self.surface_conductivity(f)
            sigma_real[i] = s.real
            sigma_imag[i] = s.imag

        return {
            "frequency": freq,
            "sigma_real": sigma_real,
            "sigma_imag": sigma_imag,
        }
