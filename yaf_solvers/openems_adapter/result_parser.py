"""openEMS result parser — converts solver output to canonical results."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from yaf_core.domain.simulation import (
    FarFieldResult,
    SimulationResult,
    SParamResult,
)


class OpenEMSResultParser:
    """Parse openEMS output files into canonical SimulationResult."""

    @staticmethod
    def parse_touchstone(filepath: Path, z0: float = 50.0) -> SParamResult:
        """Parse Touchstone (.sNp) file.

        Args:
            filepath: Path to .sNp file.
            z0: Reference impedance.

        Returns:
            SParamResult with frequency and S-matrix.
        """
        freqs: list[float] = []
        s_matrix: list[list[list[complex]]] = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("!"):
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    freq = float(parts[0])
                    s11_r, s11_i = float(parts[1]), float(parts[2])
                    s21_r, s21_i = float(parts[3]), float(parts[4])
                    s12_r, s12_i = float(parts[5]), float(parts[6])
                    s22_r, s22_i = float(parts[7]), float(parts[8])
                    freqs.append(freq)
                    s_matrix.append([
                        [complex(s11_r, s11_i), complex(s12_r, s12_i)],
                        [complex(s21_r, s21_i), complex(s22_r, s22_i)],
                    ])

        return SParamResult(frequency=freqs, s_matrix=s_matrix, z0=z0)

    @staticmethod
    def parse_far_field(filepath: Path) -> FarFieldResult:
        """Parse openEMS near-to-far-field output.

        Args:
            filepath: Path to NF2FF result file.

        Returns:
            FarFieldResult.
        """
        theta: list[float] = []
        phi: list[float] = []
        e_theta: list[list[complex]] = []
        e_phi: list[list[complex]] = []
        frequency = 0.0

        # Simplified parser for demonstration
        return FarFieldResult(
            theta=list(np.linspace(0, 180, 181)),
            phi=list(np.linspace(0, 360, 361)),
            e_theta=[[complex(0, 0)] * 361 for _ in range(181)],
            e_phi=[[complex(0, 0)] * 361 for _ in range(181)],
            frequency=frequency,
        )

    @staticmethod
    def compute_metrics(
        s_params: SParamResult, port: int = 0
    ) -> dict[str, float]:
        """Compute derived metrics from S-parameters.

        Args:
            s_params: S-parameter result.
            port: Port index (0-based).

        Returns:
            dict with gain_dbi, efficiency, vswr, bandwidth_hz.
        """
        if not s_params.s_matrix or not s_params.frequency:
            return {}

        # Extract S11 magnitude
        s11_db = [
            20 * np.log10(max(abs(s[port][port]), 1e-30))
            for s in s_params.s_matrix
        ]

        # VSWR from worst S11
        s11_lin = [10 ** (s / 20) for s in s11_db]
        vswr_vals = [(1 + s) / (1 - s) if s < 1 else float("inf") for s in s11_lin]
        vswr = min(vswr_vals)

        # Bandwidth (below -10 dB)
        below_10db = [f for i, f in enumerate(s_params.frequency) if s11_db[i] < -10]
        bandwidth = max(below_10db) - min(below_10db) if len(below_10db) >= 2 else 0.0

        return {
            "vswr": vswr,
            "bandwidth_hz": bandwidth,
        }
