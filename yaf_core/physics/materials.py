"""
Material library — standard EM materials with dispersive models.

Includes: PEC, copper, aluminum, FR4, Rogers laminates, graphene, Galinstan.
"""

from __future__ import annotations

from yaf_core.domain.geometry import Material, MaterialType


class MaterialLibrary:
    """Curated library of common antenna and RF materials."""

    _instance = None
    _materials: dict[str, Material] = {}

    def __new__(cls) -> MaterialLibrary:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_library()
        return cls._instance

    def _init_library(self) -> None:
        """Seed the library with standard materials."""
        self._materials = {
            # --- Perfect conductors ---
            "pec": Material(
                id="pec",
                name="Perfect Electric Conductor",
                type=MaterialType.PEC,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=float("inf"),
            ),
            "pmc": Material(
                id="pmc",
                name="Perfect Magnetic Conductor",
                type=MaterialType.PMC,
                epsilon_r=1.0,
                mu_r=float("inf"),
                sigma=0.0,
            ),
            # --- Real metals ---
            "copper": Material(
                id="copper",
                name="Copper (annealed)",
                type=MaterialType.CONDUCTOR,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=5.8e7,
                density_kg_m3=8960,
            ),
            "aluminum": Material(
                id="aluminum",
                name="Aluminum",
                type=MaterialType.CONDUCTOR,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=3.5e7,
                density_kg_m3=2700,
            ),
            "silver": Material(
                id="silver",
                name="Silver",
                type=MaterialType.CONDUCTOR,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=6.3e7,
                density_kg_m3=10490,
            ),
            "gold": Material(
                id="gold",
                name="Gold",
                type=MaterialType.CONDUCTOR,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=4.1e7,
                density_kg_m3=19320,
            ),
            # --- Dielectric substrates ---
            "fr4": Material(
                id="fr4",
                name="FR-4 (standard)",
                type=MaterialType.DIELECTRIC,
                epsilon_r=4.4,
                mu_r=1.0,
                sigma=0.02,
            ),
            "rogers_ro4350b": Material(
                id="rogers_ro4350b",
                name="Rogers RO4350B",
                type=MaterialType.DIELECTRIC,
                epsilon_r=3.48,
                mu_r=1.0,
                sigma=0.0037,
            ),
            "rogers_ro4003c": Material(
                id="rogers_ro4003c",
                name="Rogers RO4003C",
                type=MaterialType.DIELECTRIC,
                epsilon_r=3.38,
                mu_r=1.0,
                sigma=0.0027,
            ),
            "rt_duroid_5880": Material(
                id="rt_duroid_5880",
                name="RT/duroid 5880",
                type=MaterialType.DIELECTRIC,
                epsilon_r=2.20,
                mu_r=1.0,
                sigma=0.0009,
            ),
            "alumina": Material(
                id="alumina",
                name="Alumina (Al₂O₃)",
                type=MaterialType.DIELECTRIC,
                epsilon_r=9.8,
                mu_r=1.0,
                sigma=0.0,
            ),
            "silicon": Material(
                id="silicon",
                name="High-resistivity Silicon",
                type=MaterialType.DIELECTRIC,
                epsilon_r=11.9,
                mu_r=1.0,
                sigma=0.0,
            ),
            # --- Advanced materials ---
            "graphene": Material(
                id="graphene",
                name="Graphene (monolayer)",
                type=MaterialType.GRAPHENE,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=0.0,
                dispersion_model="kubo",
                dispersion_params={
                    "mu_c": 0.2,  # eV
                    "gamma": 0.1e12,  # s^-1 (scattering rate)
                    "temperature": 300.0,  # K
                },
            ),
            "galinstan": Material(
                id="galinstan",
                name="Galinstan (eutectic Ga-In-Sn)",
                type=MaterialType.LIQUID_METAL,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=3.46e6,
                density_kg_m3=6440,
            ),
            "egain": Material(
                id="egain",
                name="EGaIn (eutectic Ga-In)",
                type=MaterialType.LIQUID_METAL,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=3.4e6,
                density_kg_m3=6280,
            ),
            # --- Plasma (Drude) ---
            "plasma_ar": Material(
                id="plasma_ar",
                name="Argon Plasma",
                type=MaterialType.PLASMA,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=0.0,
                dispersion_model="drude",
                dispersion_params={
                    "plasma_freq": 1.0e10,  # Hz
                    "collision_freq": 1.0e8,  # Hz
                },
            ),
            # --- Vacuum / air ---
            "vacuum": Material(
                id="vacuum",
                name="Vacuum",
                type=MaterialType.DIELECTRIC,
                epsilon_r=1.0,
                mu_r=1.0,
                sigma=0.0,
            ),
            "air": Material(
                id="air",
                name="Air (dry, 1 atm, 20°C)",
                type=MaterialType.DIELECTRIC,
                epsilon_r=1.00058986,
                mu_r=1.0,
                sigma=0.0,
                density_kg_m3=1.204,
            ),
            "teflon": Material(
                id="teflon",
                name="Teflon (PTFE)",
                type=MaterialType.DIELECTRIC,
                epsilon_r=2.1,
                mu_r=1.0,
                sigma=1e-4,
                density_kg_m3=2200,
            ),
        }

    def get(self, material_id: str) -> Material:
        """Retrieve a material by ID.

        Raises KeyError if not found.
        """
        if material_id not in self._materials:
            available = ", ".join(sorted(self._materials.keys()))
            raise KeyError(f"Material '{material_id}' not found. Available: {available}")
        return self._materials[material_id]

    def list_all(self) -> list[str]:
        """List all material IDs."""
        return sorted(self._materials.keys())

    def add(self, material: Material) -> None:
        """Register a custom material."""
        self._materials[material.id] = material

    def get_dispersive_permittivity(
        self, material_id: str, frequency: float
    ) -> complex:
        """Compute complex permittivity at a given frequency.

        Args:
            material_id: Material identifier.
            frequency: Frequency in Hz.

        Returns:
            Complex relative permittivity ε_r(ω) = ε' - j ε''.
        """
        mat = self.get(material_id)

        if mat.dispersion_model == "drude":
            wp = mat.dispersion_params.get("plasma_freq", 0.0)
            gc = mat.dispersion_params.get("collision_freq", 1.0)
            omega = 2 * 3.141592653589793 * frequency
            eps = 1.0 - wp**2 / (omega * (omega + 1j * gc))
            return eps

        if mat.dispersion_model == "debye":
            eps_s = mat.dispersion_params.get("eps_static", mat.epsilon_r)
            eps_inf = mat.dispersion_params.get("eps_inf", 1.0)
            tau = mat.dispersion_params.get("tau", 1e-12)
            omega = 2 * 3.141592653589793 * frequency
            eps = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)
            return eps

        if mat.dispersion_model == "kubo":
            return self._kubo_conductivity(mat, frequency)

        # Default: constant
        return complex(mat.epsilon_r, -mat.sigma / (2 * 3.141592653589793 * frequency * 8.854187817e-12))

    def _kubo_conductivity(self, material: Material, frequency: float) -> complex:
        """Graphene surface conductivity via Kubo formula.

        σ(ω, μ_c, Γ, T) = σ_intra + σ_inter

        Reference: Hanson, "Dyadic Green's functions for an anisotropic,
        non-local model of biased graphene", IEEE TAP, 2008.
        """
        import math as _math

        # Constants
        e = 1.602176634e-19  # C
        hbar = 1.054571817e-34  # J·s
        kB = 1.380649e-23  # J/K

        mu_c = material.dispersion_params.get("mu_c", 0.2) * e  # to Joules
        gamma = material.dispersion_params.get("gamma", 0.1e12)
        T = material.dispersion_params.get("temperature", 300.0)

        omega = 2 * _math.pi * frequency

        # Intra-band contribution
        sigma_intra = (
            1j * e**2 * kB * T
            / (_math.pi * hbar**2 * (omega + 1j * 2 * gamma))
            * (mu_c / (kB * T) + 2 * _math.log(_math.exp(-mu_c / (kB * T)) + 1))
        )

        # Inter-band contribution (simplified)
        sigma_inter: complex = 0.0 + 0.0j
        if omega > 2 * mu_c / hbar:
            sigma_inter = (
                e**2
                / (4 * hbar)
                * (0.5 + 1 / _math.pi * _math.atan((hbar * omega - 2 * mu_c) / (2 * kB * T)))
                - 1j
                * e**2
                / (4 * _math.pi * hbar)
                * _math.log(
                    (hbar * omega + 2 * mu_c) ** 2
                    / ((hbar * omega - 2 * mu_c) ** 2 + (2 * kB * T) ** 2)
                )
            )

        sigma_s = sigma_intra + sigma_inter  # surface conductivity [S]
        return complex(sigma_s.real, sigma_s.imag)
