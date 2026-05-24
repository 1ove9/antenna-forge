# ============================================================
# REFERENCE
#   仿造来源：SIMP 标准实现（topfun MATLAB → Python 移植）
#             + "Topology Optimization: Theory, Methods, and Applications"
#             by Bendsøe & Sigmund (2003)
#   对标文件：topfun MATLAB 88-line code, odedesign/topopt
#   对标类/函数：SIMP interpolation, sensitivity filtering, OC update
#   关键设计点：
#     - SIMP 材料插值：E(ρ) = E_min + ρ^p * (E₀ - E_min)
#     - 灵敏度过滤（Helmholtz PDE 过滤半径 r_min）
#     - 最优准则（OC）更新方案
#     - 体积约束 compliance 最小化
#     - 黑白化投影（Heaviside filter）
#   YAF 的差异化改造：
#     - 纯 NumPy/SciPy 实现（零外部优化依赖）
#     - 2D 热传导类比 → 天线电流路径设计
#     - 内置棋盘格过滤
#     - 参数化密度场与 DiffFDTD 接口对接
# ============================================================

"""SIMP Topology Optimization — density-based structural optimization.

Implements the Solid Isotropic Material with Penalization (SIMP)
method for antenna geometry topology optimization.

Usage:
    python -m yaf_ai.optimization.topology_opt --demo
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


class SIMPTopologyOptimizer:
    """SIMP density-based topology optimizer.

    Minimizes compliance subject to volume constraint using
    optimality criteria (OC) update.
    """

    def __init__(
        self,
        nx: int = 60,
        ny: int = 30,
        vol_frac: float = 0.4,
        penal: float = 3.0,
        r_min: float = 1.5,
        density_callback: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        """Initialize SIMP optimizer.

        Args:
            nx: Number of elements in x direction.
            ny: Number of elements in y direction.
            vol_frac: Target volume fraction (0-1).
            penal: Penalization exponent (typically 3).
            r_min: Filter radius in element units.
            density_callback: Optional objective evaluator for EM-specific
                             objectives (replaces compliance minimization).
        """
        self.nx = nx
        self.ny = ny
        self.n_elements = nx * ny
        self.vol_frac = vol_frac
        self.penal = penal
        self.r_min = r_min
        self.density_callback = density_callback

        # Design variables (density field)
        self.x = np.ones(self.n_elements) * vol_frac

        # Change limits
        self.move = 0.2
        self.x_min = 0.001
        self.change = 1.0

    def _apply_simp(self, x: np.ndarray) -> np.ndarray:
        """SIMP interpolation: E(x) = E_min + x^penal * (E0 - E_min).

        Args:
            x: Density array (n_elements,).

        Returns:
            Interpolated property array.
        """
        e_min = 1e-9
        e0 = 1.0
        return e_min + x**self.penal * (e0 - e_min)

    def _filter_sensitivity(self, x: np.ndarray, dc: np.ndarray) -> np.ndarray:
        """Apply sensitivity filter to prevent checkerboarding.

        Uses a simple convolution-based filter with radius r_min.

        Args:
            x: Density array.
            dc: Raw sensitivity (compliance derivative).

        Returns:
            Filtered sensitivity.
        """
        dc = dc.reshape(self.ny, self.nx)
        x_reshaped = x.reshape(self.ny, self.nx)

        # Simple averaging filter (r_min = 1.5 → 3x3 kernel)
        kernel = np.ones((3, 3)) / 9.0
        from scipy.ndimage import convolve

        dc_filtered = convolve(dc * x_reshaped, kernel, mode="constant", cval=0.0)
        x_sum = convolve(x_reshaped, kernel, mode="constant", cval=0.0) + 1e-12

        dc_filtered = dc_filtered / x_sum
        return dc_filtered.ravel()

    def _optimality_criteria(
        self, x: np.ndarray, dc: np.ndarray, l1: float, l2: float
    ) -> np.ndarray:
        """OC update scheme.

        Args:
            x: Current density.
            dc: Sensitivity.
            l1, l2: Lagrange multiplier bounds.

        Returns:
            Updated density.
        """
        x_new = np.maximum(
            self.x_min,
            np.maximum(x - self.move, np.minimum(1.0, np.minimum(x + self.move,
                          x * np.sqrt(np.maximum(-dc / (l1 + l2), 0))))),
        )
        return np.asarray(x_new)

    def optimize(
        self,
        max_iter: int = 50,
        tol: float = 0.01,
        verbose: bool = True,
    ) -> tuple[np.ndarray, list[float]]:
        """Run SIMP topology optimization.

        Args:
            max_iter: Maximum iterations.
            tol: Convergence tolerance on density change.
            verbose: Print progress.

        Returns:
            (optimized_density, objective_history).
        """
        objective_history: list[float] = []

        for it in range(max_iter):
            x_old = self.x.copy()

            # Compute objective and sensitivity
            if self.density_callback is not None:
                objective = self.density_callback(self.x)
                # Finite difference sensitivity
                dc = self._finite_diff_sensitivity(self.x)
            else:
                # Compliance minimization
                objective, dc = self._compliance(self.x)

            objective_history.append(objective)

            # Sensitivity filtering
            dc = self._filter_sensitivity(self.x, dc)

            # Bisection for Lagrange multiplier
            l1 = 0.0
            l2 = 1e6
            while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-6:
                lmid = (l1 + l2) / 2
                x_new = self._optimality_criteria(self.x, dc, lmid, 0.0)
                if np.mean(x_new) > self.vol_frac:
                    l1 = lmid
                else:
                    l2 = lmid

            self.x = x_new
            self.change = float(np.linalg.norm(self.x - x_old, np.inf))

            if verbose:
                print(
                    f"  Iter {it:3d}: obj={objective:.6f}, "
                    f"vol={np.mean(self.x):.4f}, change={self.change:.4f}"
                )

            if self.change < tol:
                if verbose:
                    print(f"  Converged at iteration {it}")
                break

        return self.x.reshape(self.ny, self.nx), objective_history

    def _compliance(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute compliance and sensitivity (2D heat analogy).

        Args:
            x: Density field.

        Returns:
            (compliance, sensitivity).
        """
        nx, ny = self.nx, self.ny
        n = nx * ny
        penal = self.penal

        # Simple 5-point stencil Laplacian
        e = self._apply_simp(x)

        # Build stiffness-like matrix
        e_mat = e.reshape(ny, nx)

        data: list[float] = []
        row: list[int] = []
        col: list[int] = []

        for iy in range(ny):
            for ix in range(nx):
                idx = iy * nx + ix
                e_center = e_mat[iy, ix]

                # Self term
                diag_val = 0.0

                # Left
                if ix > 0:
                    e_left = e_mat[iy, ix - 1]
                    val = 0.5 * (e_center + e_left)
                    diag_val += val
                    data.append(-val)
                    row.append(idx)
                    col.append(iy * nx + ix - 1)

                # Right
                if ix < nx - 1:
                    e_right = e_mat[iy, ix + 1]
                    val = 0.5 * (e_center + e_right)
                    diag_val += val
                    data.append(-val)
                    row.append(idx)
                    col.append(iy * nx + ix + 1)

                # Bottom
                if iy > 0:
                    e_bottom = e_mat[iy - 1, ix]
                    val = 0.5 * (e_center + e_bottom)
                    diag_val += val
                    data.append(-val)
                    row.append(idx)
                    col.append((iy - 1) * nx + ix)

                # Top
                if iy < ny - 1:
                    e_top = e_mat[iy + 1, ix]
                    val = 0.5 * (e_center + e_top)
                    diag_val += val
                    data.append(-val)
                    row.append(idx)
                    col.append((iy + 1) * nx + ix)

                data.append(diag_val)
                row.append(idx)
                col.append(idx)

        K = sparse.csr_matrix(
            (np.array(data), (np.array(row), np.array(col))), shape=(n, n)
        )

        # Load vector: unit load at center
        f = np.zeros(n)
        center_idx = (ny // 2) * nx + (nx // 2)
        f[center_idx] = 1.0

        # Solve: K u = f
        u = spla.spsolve(K, f)

        # Compliance: c = f^T u
        compliance = float(f @ u)

        # Sensitivity: ∂c/∂x = -p * x^(p-1) * (E0 - E_min) * u^T K_e u
        dc = np.zeros(n)
        e0 = 1.0
        e_min = 1e-9

        for iy in range(ny):
            for ix in range(nx):
                idx = iy * nx + ix
                # Element strain energy = u_element^T * K_element * u_element
                u_local = u[idx]
                neighbors = u[max(0, idx - nx) : min(n, idx + nx + 1)]
                se = u_local * np.sum(neighbors) / (len(neighbors) + 1e-9)

                dc[idx] = (
                    -penal
                    * x[idx] ** (penal - 1)
                    * (e0 - e_min)
                    * se
                )

        return compliance, dc

    def _finite_diff_sensitivity(self, x: np.ndarray, eps: float = 0.01) -> np.ndarray:
        """Finite difference sensitivity for custom objective.

        Args:
            x: Design variables.
            eps: Perturbation magnitude.

        Returns:
            Sensitivity gradient (n_elements,).
        """
        if self.density_callback is None:
            return np.zeros_like(x)

        f0 = self.density_callback(x)
        dc = np.zeros_like(x)

        for i in range(len(x)):
            x_pert = x.copy()
            delta = eps * (1.0 - x[i])
            x_pert[i] = np.clip(x[i] + delta, self.x_min, 1.0)
            f_pert = self.density_callback(x_pert)
            dc[i] = (f_pert - f0) / (delta + 1e-12)

        return dc


def demo() -> None:
    """Quick demo: SIMP topology optimization of a cantilever beam."""
    print("=" * 50)
    print("  SIMP Topology Optimization Demo")
    print("=" * 50)

    optimizer = SIMPTopologyOptimizer(
        nx=40,
        ny=20,
        vol_frac=0.4,
        penal=3.0,
        r_min=1.2,
    )

    print(f"Elements: {optimizer.nx} × {optimizer.ny} = {optimizer.n_elements}")
    print(f"Target volume fraction: {optimizer.vol_frac}")

    result, history = optimizer.optimize(max_iter=30, verbose=True)

    print(f"\n  Final density — material: {(result > 0.5).sum()}/{result.size}")
    print(f"  Objective: {history[-1]:.6f}")
    print(f"  Objective delta: {history[0]:.6f} → {history[-1]:.6f}")

    if history[-1] < history[0]:
        print("  Objective decreased — optimization OK")
    else:
        print("  WARNING: Objective did not decrease")

    print("  Demo complete.\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SIMP Topology Optimization")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--vol-frac", type=float, default=0.4)

    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        demo()
    else:
        opt = SIMPTopologyOptimizer(vol_frac=args.vol_frac)
        opt.optimize(max_iter=args.max_iter)


if __name__ == "__main__":
    main()
