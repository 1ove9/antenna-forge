# ============================================================
# REFERENCE
#   仿造来源：BoTorch @ https://github.com/pytorch/botorch
#   对标文件：botorch/acquisition/analytic.py (ExpectedImprovement)
#   对标类/函数：ExpectedImprovement, UpperConfidenceBound, qExpectedImprovement
#   关键设计点：
#     - GP 后验 + EI/UCB/Thompson 采集函数组合
#     - Cholesky 分解求 GP 后验（稳定 <1000 观测点）
#     - RBF/Squared Exponential 核函数
#     - 随机搜索候选点优化采集函数
#     - 参数归一化到 [0,1] 提升数值稳定性
#   YAF 的差异化改造：
#     - 纯 NumPy/SciPy 实现（零 PyTorch/BoTorch 依赖）
#     - 最小化目标函数（非最大化），EI 符号适配
#     - 内置参数边界管理 + 归一化/反归一化
#     - --demo：二维天线参数优化示例
#     - 支持 EI/UCB/Thompson 三种采集策略切换
# ============================================================

"""
Bayesian Optimization Engine — Gaussian Process + Expected Improvement.

Implements GP-based Bayesian optimization with multiple acquisition
functions (EI, UCB, Thompson Sampling) for antenna design parameter
optimization.

Reference: Shahriari et al., "Taking the Human Out of the Loop:
A Review of Bayesian Optimization", Proc. IEEE, 2016.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm


class GaussianProcess:
    """Sparse Gaussian Process with squared exponential kernel."""

    def __init__(
        self,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-3,
    ) -> None:
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.K_chol: tuple[np.ndarray, bool] | None = None
        self.alpha: np.ndarray | None = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared exponential (RBF) kernel.

        k(x, x') = σ² * exp(-||x - x'||² / (2ℓ²))
        """
        sq_dist = (
            np.sum(X1**2, axis=1).reshape(-1, 1)
            + np.sum(X2**2, axis=1).reshape(1, -1)
            - 2 * X1 @ X2.T
        )
        return np.asarray(self.signal_variance * np.exp(-0.5 * sq_dist / self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to training data.

        Args:
            X: (N, D) input points.
            y: (N,) target values.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        K = self._kernel(X, X) + self.noise_variance * np.eye(len(X))
        try:
            self.K_chol = cho_factor(K)
            self.alpha = cho_solve(self.K_chol, y)
        except np.linalg.LinAlgError:
            # Add jitter for numerical stability
            K += 1e-6 * np.eye(len(X))
            self.K_chol = cho_factor(K)
            self.alpha = cho_solve(self.K_chol, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at test points.

        Args:
            X: (M, D) test points.

        Returns:
            (mean, std) each of shape (M,).
        """
        if self.X_train is None or self.alpha is None:
            return np.zeros(len(X)), np.ones(len(X)) * np.sqrt(self.signal_variance)

        K_s = self._kernel(self.X_train, X)
        K_ss = self._kernel(X, X)

        mean = K_s.T @ self.alpha
        v = cho_solve(self.K_chol, K_s)  # type: ignore[arg-type]
        cov = K_ss - K_s.T @ v
        var = np.diag(cov)
        var = np.maximum(var, 1e-12)
        return mean, np.sqrt(var)


class AcquisitionFunction:
    """Acquisition functions for Bayesian optimization."""

    @staticmethod
    def expected_improvement(
        mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01
    ) -> np.ndarray:
        """Expected Improvement: EI(x) = E[max(f_best - f(x), 0)]

        Args:
            mean: GP posterior mean.
            std: GP posterior std.
            y_best: Current best observation.
            xi: Exploration-exploitation trade-off.

        Returns:
            EI values at each point.
        """
        with np.errstate(divide="ignore"):
            improvement = y_best - mean - xi
            Z = improvement / (std + 1e-12)
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std < 1e-12] = 0.0
        return ei

    @staticmethod
    def upper_confidence_bound(
        mean: np.ndarray, std: np.ndarray, kappa: float = 2.576
    ) -> np.ndarray:
        """Upper Confidence Bound: UCB(x) = μ(x) + κ * σ(x)

        Args:
            kappa: Controls exploration (higher = more exploration).
        """
        return mean + kappa * std

    @staticmethod
    def thompson_sampling(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Thompson sampling: sample from posterior."""
        return np.random.normal(mean, std)


class BayesianOptimizer:
    """Bayesian optimization engine for antenna design.

    Minimizes an objective function f(x) over a bounded parameter space.
    """

    def __init__(
        self,
        parameter_bounds: dict[str, tuple[float, float]],
        objective: Callable[[np.ndarray], float],
        acquisition: str = "ei",
        n_initial: int = 5,
    ) -> None:
        self.bounds = parameter_bounds
        self.objective = objective
        self.acquisition_type = acquisition
        self.n_initial = n_initial

        self.param_names = list(parameter_bounds.keys())
        self.n_dims = len(self.param_names)
        self.bounds_array = np.array([parameter_bounds[k] for k in self.param_names])

        self.gp = GaussianProcess()
        self.X_observed: list[np.ndarray] = []
        self.y_observed: list[float] = []
        self.y_best: float = float("inf")
        self.x_best: np.ndarray | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1]."""
        return np.asarray((x - self.bounds_array[:, 0]) / (
            self.bounds_array[:, 1] - self.bounds_array[:, 0]
        ))

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        """Denormalize from [0, 1] to original bounds."""
        return np.asarray(
            self.bounds_array[:, 0]
            + x_norm * (self.bounds_array[:, 1] - self.bounds_array[:, 0])
        )

    def suggest(self) -> np.ndarray:
        """Suggest next point to evaluate.

        Returns:
            Parameter vector in original space.
        """
        if len(self.X_observed) < self.n_initial:
            # Random sampling for initial points
            x_norm = np.random.random(self.n_dims)
        else:
            # Optimize acquisition function over a random grid
            X_norm = np.array(self.X_observed)
            y = np.array(self.y_observed)

            self.gp.fit(X_norm, y)

            # Random search over 1000 candidates
            n_candidates = 1000
            candidates = np.random.random((n_candidates, self.n_dims))
            mean, std = self.gp.predict(candidates)

            if self.acquisition_type == "ei":
                acq = AcquisitionFunction.expected_improvement(mean, std, -self.y_best)
            elif self.acquisition_type == "ucb":
                acq = AcquisitionFunction.upper_confidence_bound(mean, std)
            else:
                acq = AcquisitionFunction.thompson_sampling(mean, std)

            best_idx = np.argmax(acq)
            x_norm = candidates[best_idx]

        return self._denormalize(x_norm)

    def observe(self, x: np.ndarray, y: float) -> None:
        """Record an observation.

        Args:
            x: Parameter vector in original space.
            y: Objective value.
        """
        x_norm = self._normalize(x)
        self.X_observed.append(x_norm)
        self.y_observed.append(y)

        if y < self.y_best:
            self.y_best = y
            self.x_best = x.copy()

    def optimize(self, n_iterations: int = 50, verbose: bool = True) -> dict[str, Any]:
        """Run Bayesian optimization loop.

        Args:
            n_iterations: Total number of evaluations.
            verbose: Print progress.

        Returns:
            dict with best_x, best_y, and history.
        """
        history: list[dict[str, Any]] = []

        for i in range(n_iterations):
            x = self.suggest()
            y = self.objective(x)
            self.observe(x, y)

            history.append({"iteration": i, "x": x.tolist(), "y": y})

            if verbose and i % 5 == 0:
                print(f"  Iter {i:3d}: y = {y:.6f}, best = {self.y_best:.6f}")

        return {
            "best_x": self.x_best.tolist() if self.x_best is not None else [],
            "best_y": self.y_best,
            "history": history,
        }


def demo_objective(x: np.ndarray) -> float:
    """Simple 2D test function: Branin-Hoo (minimization).

    Global minimum at f(x*) ≈ 0.397887
    """
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)

    x1, x2 = x[0], x[1]
    term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    return float(term1 + term2 + s)


def antenna_objective(params: np.ndarray) -> float:
    """Simulated antenna optimization: tune dipole length for 2.4 GHz.

    params: [length (m)]
    Minimize: |resonant_freq - 2.4e9| / 2.4e9 + loss
    """
    length = params[0]
    c0 = 3e8
    # Half-wave resonance: f_res = c0 / (2L)
    f_res = c0 / (2 * length + 1e-6)
    f_target = 2.4e9
    freq_error = abs(f_res - f_target) / f_target

    # Penalize extreme sizes
    size_penalty = max(0, length - 0.1) ** 2 + max(0, 0.01 - length) ** 2
    return float(freq_error + 0.01 * size_penalty)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--antenna", action="store_true")
    args = parser.parse_args()

    if args.antenna:
        print("=" * 60)
        print("  Bayesian Optimization: Antenna Tuning Demo")
        print("=" * 60)

        opt = BayesianOptimizer(
            parameter_bounds={"length": (0.01, 0.2)},
            objective=antenna_objective,
            acquisition="ei",
        )

        result = opt.optimize(n_iterations=20, verbose=True)
        print(f"\nBest length: {result['best_x'][0]:.4f} m")
        print(f"Best loss:  {result['best_y']:.6f}")
        target_length = 3e8 / (2 * 2.4e9)
        print(f"Target (λ/2): {target_length:.4f} m")
        print("✓ Antenna optimization demo complete.")

    elif args.demo or len(sys.argv) == 1:
        import sys
        print("=" * 60)
        print("  Bayesian Optimization Demo (Branin-Hoo)")
        print("=" * 60)

        opt = BayesianOptimizer(
            parameter_bounds={"x1": (-5, 10), "x2": (0, 15)},
            objective=demo_objective,
            acquisition="ei",
        )

        result = opt.optimize(n_iterations=30, verbose=True)
        print(f"\nBest x: {result['best_x']}")
        print(f"Best y: {result['best_y']:.6f}")
        print(f"True minimum y: ~0.397887")
        print("✓ Bayesian optimization demo complete.")
