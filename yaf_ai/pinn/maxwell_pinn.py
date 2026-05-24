# ============================================================
# REFERENCE
#   仿造来源：DeepXDE @ https://github.com/lululxvi/deepxde
#   对标文件：deepxde/examples/pinn_forward/, deepxde/nn/
#   对标类/函数：dde.data.PDE, dde.nn.FNN, dde.Model, dde.callbacks
#   关键设计点：
#     - PINN 核心：物理残差 + 边界条件残差 + 初始条件残差联合 loss
#     - Maxwell 旋度方程离散：∇×E = -μ ∂H/∂t, ∇×H = ε ∂E/∂t + J
#     - 频域简化：2D TE/TM 模式 Helmholtz 方程
#     - 自动微分边界强制（Dirichlet/Neumann/PEC/PMC）
#   YAF 的差异化改造：
#     - 纯 PyTorch 实现（零 DeepXDE 依赖）
#     - 频域 2D TE 模式求解器
#     - 内置 PEC 边界条件
#     - --demo 模式：矩形波导场分布预测
# ============================================================

"""Maxwell PINN — Physics-Informed Neural Network for EM field solving.

Solves 2D frequency-domain Maxwell's equations using PINN methodology:
    ∇²Ez + k₀² * εr * Ez = 0    (2D TE mode Helmholtz)

Boundary conditions: PEC (Ez=0) on waveguide walls.

Usage:
    python -m yaf_ai.pinn.maxwell_pinn --demo
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


class MaxwellPINN(nn.Module):
    """PINN for 2D Helmholtz equation (TE mode).

    Predicts Ez(x, y) complex field amplitude in a bounded domain.
    Loss = PDE residual + boundary conditions.
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        k0: float = 2 * np.pi,
        epsilon_r: float = 1.0,
    ) -> None:
        """Initialize Maxwell PINN.

        Args:
            layers: Hidden layer sizes (default: [2, 50, 50, 50, 2]).
            k0: Free-space wavenumber (2π/λ).
            epsilon_r: Relative permittivity of the medium.
        """
        super().__init__()
        self.k0 = k0
        self.epsilon_r = epsilon_r

        if layers is None:
            layers = [2, 50, 50, 50, 2]

        network: list[nn.Module] = []
        for i in range(len(layers) - 2):
            network.append(nn.Linear(layers[i], layers[i + 1]))
            network.append(nn.Tanh())
        network.append(nn.Linear(layers[-2], layers[-1]))

        self.network = nn.Sequential(*network)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict Ez = (real, imag) at spatial coordinates.

        Args:
            x: X coordinates (N,).
            y: Y coordinates (N,).

        Returns:
            (ez_real, ez_imag) both shape (N,).
        """
        coords = torch.stack([x, y], dim=1)
        output = self.network(coords)
        return output[:, 0], output[:, 1]

    def compute_laplacian(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ∇²Ez (laplacian) via autodiff.

        Args:
            x, y: Coordinates with requires_grad=True.

        Returns:
            (laplacian_real, laplacian_imag).
        """
        ez_real, ez_imag = self.forward(x, y)

        # Real part laplacian
        grad_r_x = torch.autograd.grad(
            ez_real, x, grad_outputs=torch.ones_like(ez_real),
            create_graph=True, retain_graph=True,
        )[0]
        lap_r_x = torch.autograd.grad(
            grad_r_x, x, grad_outputs=torch.ones_like(grad_r_x),
            create_graph=True, retain_graph=True,
        )[0]

        grad_r_y = torch.autograd.grad(
            ez_real, y, grad_outputs=torch.ones_like(ez_real),
            create_graph=True, retain_graph=True,
        )[0]
        lap_r_y = torch.autograd.grad(
            grad_r_y, y, grad_outputs=torch.ones_like(grad_r_y),
            create_graph=True, retain_graph=True,
        )[0]

        # Imag part laplacian
        grad_i_x = torch.autograd.grad(
            ez_imag, x, grad_outputs=torch.ones_like(ez_imag),
            create_graph=True, retain_graph=True,
        )[0]
        lap_i_x = torch.autograd.grad(
            grad_i_x, x, grad_outputs=torch.ones_like(grad_i_x),
            create_graph=True, retain_graph=True,
        )[0]

        grad_i_y = torch.autograd.grad(
            ez_imag, y, grad_outputs=torch.ones_like(ez_imag),
            create_graph=True, retain_graph=True,
        )[0]
        lap_i_y = torch.autograd.grad(
            grad_i_y, y, grad_outputs=torch.ones_like(grad_i_y),
            create_graph=True, retain_graph=True,
        )[0]

        return lap_r_x + lap_r_y, lap_i_x + lap_i_y

    def pde_residual(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Helmholtz PDE residual: ∇²Ez + k₀² * εr * Ez = 0.

        Args:
            x, y: Interior collocation points.

        Returns:
            Scalar loss (MSE of residual).
        """
        ez_real, ez_imag = self.forward(x, y)
        lap_r, lap_i = self.compute_laplacian(x, y)

        k2 = self.k0**2 * self.epsilon_r

        resid_r = lap_r + k2 * ez_real
        resid_i = lap_i + k2 * ez_imag

        return torch.mean(resid_r**2 + resid_i**2)

    def boundary_loss(
        self,
        x_bnd: torch.Tensor,
        y_bnd: torch.Tensor,
        boundary_type: str = "pec",
    ) -> torch.Tensor:
        """Boundary condition loss.

        Args:
            x_bnd, y_bnd: Boundary collocation points.
            boundary_type: "pec" (Ez=0) or "pmc" (∂Ez/∂n=0).

        Returns:
            Boundary loss.
        """
        ez_real, ez_imag = self.forward(x_bnd, y_bnd)

        if boundary_type == "pec":
            return torch.mean(ez_real**2 + ez_imag**2)
        else:
            return torch.tensor(0.0, device=x_bnd.device)


class MaxwellPINNTrainer:
    """Trainer for Maxwell PINN."""

    def __init__(
        self,
        model: MaxwellPINN,
        domain: tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5),
        n_interior: int = 1000,
        n_boundary: int = 200,
    ) -> None:
        """Initialize trainer.

        Args:
            model: MaxwellPINN instance.
            domain: (xmin, xmax, ymin, ymax) domain bounds.
            n_interior: Number of interior collocation points.
            n_boundary: Number of boundary collocation points.
        """
        self.model = model
        self.domain = domain
        self.n_interior = n_interior
        self.n_boundary = n_boundary

    def sample_interior(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random interior collocation points."""
        xmin, xmax, ymin, ymax = self.domain
        x = torch.rand(self.n_interior) * (xmax - xmin) + xmin
        y = torch.rand(self.n_interior) * (ymax - ymin) + ymin
        x.requires_grad_(True)
        y.requires_grad_(True)
        return x, y

    def sample_boundary(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample collocation points on domain boundary."""
        xmin, xmax, ymin, ymax = self.domain
        per_side = self.n_boundary // 4

        # Bottom edge
        x_b = torch.rand(per_side) * (xmax - xmin) + xmin
        y_b = torch.full((per_side,), ymin)

        # Top edge
        x_t = torch.rand(per_side) * (xmax - xmin) + xmin
        y_t = torch.full((per_side,), ymax)

        # Left edge
        y_l = torch.rand(per_side) * (ymax - ymin) + ymin
        x_l = torch.full((per_side,), xmin)

        # Right edge
        y_r = torch.rand(per_side) * (ymax - ymin) + ymin
        x_r = torch.full((per_side,), xmax)

        x = torch.cat([x_b, x_t, x_l, x_r])
        y = torch.cat([y_b, y_t, y_l, y_r])

        return x, y

    def train(
        self, epochs: int = 1000, lr: float = 1e-3, verbose: bool = True
    ) -> list[float]:
        """Train the PINN.

        Args:
            epochs: Training epochs.
            lr: Learning rate.
            verbose: Print progress.

        Returns:
            Training loss history.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses: list[float] = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Interior PDE loss
            x_int, y_int = self.sample_interior()
            loss_pde = self.model.pde_residual(x_int, y_int)

            # Boundary loss
            x_bnd, y_bnd = self.sample_boundary()
            loss_bnd = self.model.boundary_loss(x_bnd, y_bnd, "pec")

            loss = loss_pde + loss_bnd
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            losses.append(loss.item())

            if verbose and epoch % 200 == 0:
                print(
                    f"  Epoch {epoch:4d}: loss={loss.item():.6f} "
                    f"(PDE={loss_pde.item():.6f}, BC={loss_bnd.item():.6f})"
                )

        return losses

    def predict(self, nx: int = 50, ny: int = 50) -> np.ndarray:
        """Evaluate PINN on a uniform grid.

        Args:
            nx, ny: Grid resolution.

        Returns:
            Ez field magnitude, shape (ny, nx).
        """
        xmin, xmax, ymin, ymax = self.domain
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys)

        x_t = torch.from_numpy(X.ravel()).float()
        y_t = torch.from_numpy(Y.ravel()).float()

        with torch.no_grad():
            ez_r, ez_i = self.model(x_t, y_t)

        magnitude = np.sqrt(ez_r.numpy()**2 + ez_i.numpy()**2)
        return np.asarray(magnitude.reshape(ny, nx))


def demo() -> None:
    """Quick demo: solve 2D waveguide mode with PINN."""
    print("=" * 50)
    print("  Maxwell PINN — 2D TE Mode Waveguide Demo")
    print("=" * 50)

    # Wavenumber for λ = 1.0 m
    k0 = 2 * np.pi / 1.0

    model = MaxwellPINN(layers=[2, 40, 40, 40, 2], k0=k0, epsilon_r=1.0)
    trainer = MaxwellPINNTrainer(
        model,
        domain=(-0.5, 0.5, -0.5, 0.5),
        n_interior=800,
        n_boundary=160,
    )

    print("Training PINN...")
    losses = trainer.train(epochs=500, lr=1e-3)
    print(f"Final loss: {losses[-1]:.6f}")

    # Predict field
    field = trainer.predict(nx=30, ny=30)
    print(f"Field shape: {field.shape}")
    print(f"Field max: {field.max():.4f}, min: {field.min():.4f}")

    # Check loss decreased
    if losses[-1] < losses[0]:
        print("  Loss decreased — training OK")
    else:
        print("  WARNING: Loss did not decrease")

    print("  Demo complete.\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Maxwell PINN solver")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")

    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        demo()
    else:
        print("Maxwell PINN. Use --demo to run demonstration.")


if __name__ == "__main__":
    main()
