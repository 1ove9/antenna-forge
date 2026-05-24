"""GAN Metasurface — GAN for high-resolution metasurface unit cell generation."""
from __future__ import annotations

import torch
import torch.nn as nn


class MetasurfaceGAN(nn.Module):
    """DCGAN-style generator for metasurface unit cells."""
    def __init__(self, latent_dim: int = 100, output_size: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, output_size * output_size), nn.Tanh(),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(output_size * output_size, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def generate(self, n: int = 4, device: str = "cpu") -> torch.Tensor:
        from typing import cast

        z = torch.randn(n, self.latent_dim, device=device)
        self.generator.to(device)
        self.generator.eval()
        with torch.no_grad():
            x = cast(torch.Tensor, self.generator(z))
        side = int(x.shape[1] ** 0.5)
        return x.reshape(-1, side, side)


class GANMetasurfaceDesigner:
    """High-level GAN designer for metasurface cells."""
    def __init__(self, resolution: int = 64, latent_dim: int = 100) -> None:
        self.gan = MetasurfaceGAN(latent_dim, resolution)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gan.to(self.device)

    def train_step(
        self,
        batch: torch.Tensor,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
    ) -> dict[str, float]:
        batch = batch.to(self.device)
        bs = len(batch)
        z = torch.randn(bs, self.gan.latent_dim, device=self.device)
        fake = self.gan.generator(z)
        real_out = self.gan.discriminator(batch)
        fake_out = self.gan.discriminator(fake.detach())
        d_loss = -torch.mean(torch.log(real_out + 1e-8) + torch.log(1 - fake_out + 1e-8))
        opt_d.zero_grad()
        d_loss.backward()  # type: ignore[no-untyped-call]
        opt_d.step()
        fake_out2 = self.gan.discriminator(fake)
        g_loss = -torch.mean(torch.log(fake_out2 + 1e-8))
        opt_g.zero_grad()
        g_loss.backward()  # type: ignore[no-untyped-call]
        opt_g.step()
        return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}
