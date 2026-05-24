"""FNO Surrogate — Fourier Neural Operator as fast EM proxy model."""
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_c * out_c)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_c, out_c, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_c, out_c, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            b, self.out_c, h, w // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )
        return cast(torch.Tensor, torch.fft.irfft2(out_ft, s=(h, w)))


class FNO2D(nn.Module):
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32) -> None:
        super().__init__()
        self.fc0 = nn.Linear(3, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x).permute(0, 3, 1, 2)
        x1 = self.conv0(x) + self.w0(x)
        x1 = F.gelu(x1)
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)
        x2 = x2.permute(0, 2, 3, 1)
        x = self.fc1(x2)
        x = F.gelu(x)
        return cast(torch.Tensor, self.fc2(x).squeeze())


class FNOSolver:
    """FNO-based surrogate model for rapid EM prediction."""

    def __init__(self, modes: int = 12, width: int = 32, device: str = "cpu") -> None:
        self.model = FNO2D(modes, modes, width).to(device)
        self.device = device

    def predict_s11(
        self,
        geometry_grid: torch.Tensor,
        frequency_grid: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            inp = torch.stack(
                [
                    geometry_grid.unsqueeze(-1).expand(-1, -1, -1, 1),
                    torch.zeros_like(geometry_grid.unsqueeze(-1)),
                    frequency_grid.unsqueeze(-1),
                ],
                dim=-1,
            )
            return cast(torch.Tensor, self.model(inp.to(self.device)))
