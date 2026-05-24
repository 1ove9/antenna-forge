"""Diffusion Designer — Conditional DDPM for antenna geometry (minimal)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionDesigner:
    """Minimal DDPM for antenna geometry generation."""

    def __init__(
        self, grid_size: int = 32, hidden_dim: int = 256, steps: int = 100
    ) -> None:
        self.grid_size = grid_size
        self.steps = steps
        self.input_dim = grid_size * grid_size
        self.betas = self._linear_beta_schedule(steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def _linear_beta_schedule(self, T: int) -> torch.Tensor:
        return torch.linspace(1e-4, 0.02, T)

    def sample(self, n: int = 4, device: str = "cpu") -> torch.Tensor:
        self.model.to(device)
        self.model.eval()
        x = torch.randn(n, self.input_dim, device=device)
        for t in reversed(range(self.steps)):
            beta = self.betas[t].to(device)
            alpha = self.alphas[t].to(device)
            alpha_cumprod = self.alphas_cumprod[t].to(device)
            t_tensor = torch.full((n, 1), t, device=device, dtype=torch.float)
            noise_pred = self.model(torch.cat([x, t_tensor], dim=1))
            coef = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
            x = (x - coef * noise_pred) / torch.sqrt(alpha)
            if t > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return (x > 0.5).float().reshape(-1, self.grid_size, self.grid_size)
