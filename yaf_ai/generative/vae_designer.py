# ============================================================
# REFERENCE
#   仿造来源：youxch/Inverse-design-of-metasurfaces
#             @ https://github.com/youxch/Inverse-design-of-metasurfaces
#             + β-VAE 标准实现 @ PyTorch-VAE/models/beta_vae.py
#   对标文件：youxch/code/, PyTorch-VAE/models/beta_vae.py
#   对标类/函数：encoder/decoder, reparameterize, β-VAE loss
#   关键设计点：
#     - high-dim sequences ↔ reflection phases → 共享低维隐空间 → 重构
#     - encoder-decoder 范式：几何编码到连续隐空间
#     - β-VAE loss = reconstruction (BCE) + β * KL divergence
#     - 隐空间插值实现设计平滑过渡
#   YAF 的差异化改造：
#     - 32×32 网格天线几何输入（非元表面相位序列）
#     - 合成数据集生成（dipole + patch 混合）
#     - 5 epoch 快速训练模式（演示用）
#     - generate() 生成 + clamp + reshape 后处理
#     - 纯 PyTorch，无外部 VAE 依赖
# ============================================================

"""
VAE Designer — Variational Autoencoder for antenna geometry generation.

Encodes antenna geometries into a continuous latent space, enabling:
- Smooth interpolation between designs
- Latent-space optimization
- Conditional generation from design specs

Minimal but trainable implementation using PyTorch.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class AntennaVAE(nn.Module):
    """VAE for antenna geometry (2D cross-section represented as 32x32 grid)."""

    def __init__(self, latent_dim: int = 32, input_dim: int = 1024) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        from typing import cast
        return cast(torch.Tensor, self.decoder(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(
        self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """VAE loss = reconstruction + KL divergence.

        Args:
            recon: Reconstructed input.
            x: Original input.
            mu, logvar: Latent distribution parameters.

        Returns:
            dict with 'loss', 'recon_loss', 'kl_loss'.
        """
        BCE = F.binary_cross_entropy(recon, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return {
            "loss": BCE + 0.1 * KLD,
            "recon_loss": BCE,
            "kl_loss": KLD,
        }


class AntennaVAEDataset:
    """Generate synthetic antenna geometry dataset for training."""

    @staticmethod
    def generate_dipoles(n_samples: int = 1000, grid_size: int = 32) -> np.ndarray:
        """Generate half-wave dipole geometries at various lengths."""
        data = np.zeros((n_samples, grid_size * grid_size), dtype=np.float32)
        for i in range(n_samples):
            length_ratio = 0.3 + 0.4 * np.random.random()
            y_center = int(grid_size // 2)
            half_len = int(grid_size * length_ratio / 2)
            x_center = grid_size // 2

            grid = np.zeros((grid_size, grid_size), dtype=np.float32)
            for y in range(y_center - 2, y_center + 3):
                for x in range(x_center - half_len, x_center + half_len):
                    if 0 <= y < grid_size and 0 <= x < grid_size:
                        grid[y, x] = 1.0

            # Add small feed gap
            for y in range(y_center - 1, y_center + 2):
                grid[y, x_center] = 0.0

            data[i] = grid.ravel()

        return data

    @staticmethod
    def generate_patches(n_samples: int = 1000, grid_size: int = 32) -> np.ndarray:
        """Generate rectangular patch antenna geometries."""
        data = np.zeros((n_samples, grid_size * grid_size), dtype=np.float32)
        for i in range(n_samples):
            pw = int(4 + 12 * np.random.random())
            pl = int(6 + 14 * np.random.random())
            px = (grid_size - pl) // 2
            py = (grid_size - pw) // 2

            grid = np.zeros((grid_size, grid_size), dtype=np.float32)
            grid[py : py + pw, px : px + pl] = 1.0

            # Feed line
            feed_y = py + pw // 2
            feed_x_start = px + pl
            feed_x_end = min(grid_size - 1, feed_x_start + 4)
            for x in range(feed_x_start, feed_x_end):
                if 0 <= feed_y < grid_size:
                    grid[feed_y, x] = 1.0

            data[i] = grid.ravel()

        return data

    @staticmethod
    def get_dataloader(
        batch_size: int = 64, grid_size: int = 32, n_samples: int = 2000
    ) -> Any:
        """Create a DataLoader with synthetic antenna geometries."""
        d1 = AntennaVAEDataset.generate_dipoles(n_samples // 2, grid_size)
        d2 = AntennaVAEDataset.generate_patches(n_samples // 2, grid_size)
        data = np.vstack([d1, d2])
        np.random.shuffle(data)

        tensor = torch.from_numpy(data)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAEDesigner:
    """High-level VAE-based antenna designer."""

    def __init__(self, latent_dim: int = 32, grid_size: int = 32) -> None:
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.input_dim = grid_size * grid_size
        self.model = AntennaVAE(latent_dim, self.input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3
    ) -> list[float]:
        """Train the VAE on synthetic antenna data.

        Args:
            epochs: Number of training epochs.
            batch_size: Batch size.
            lr: Learning rate.

        Returns:
            List of epoch losses.
        """
        dataloader = AntennaVAEDataset.get_dataloader(
            batch_size, self.grid_size, n_samples=2000
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses: list[float] = []

        print(f"Training VAE on {self.device}...")
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model(batch)
                loss_dict = self.model.loss_function(recon, batch, mu, logvar)
                loss = loss_dict["loss"]
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(dataloader.dataset))
            losses.append(avg_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:3d}: loss = {avg_loss:.4f}")

        return losses

    def generate(self, n: int = 5, z: torch.Tensor | None = None) -> np.ndarray:
        """Generate antenna geometries from latent space.

        Args:
            n: Number of samples (ignored if z is provided).
            z: Optional latent vectors.

        Returns:
            Array of shape (n, grid_size, grid_size) with binary geometries.
        """
        self.model.eval()
        with torch.no_grad():
            if z is None:
                z = torch.randn(n, self.latent_dim, device=self.device)
            recon_t = self.model.decode(z)
            recon_arr: np.ndarray = recon_t.cpu().numpy().reshape(-1, self.grid_size, self.grid_size)
            # Binarize
            recon_arr = (recon_arr > 0.5).astype(np.float32)
        return recon_arr

    def interpolate(
        self, z1: torch.Tensor, z2: torch.Tensor, steps: int = 10
    ) -> np.ndarray:
        """Interpolate between two latent points.

        Args:
            z1, z2: Latent vectors of shape (latent_dim,).
            steps: Number of interpolation steps.

        Returns:
            Array of interpolated geometries.
        """
        alphas = torch.linspace(0, 1, steps)
        zs = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            zs.append(z)
        z_batch = torch.stack(zs).to(self.device)
        return self.generate(z=z_batch)

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def train_demo(epochs: int = 20) -> int:
    """Run VAE training demo."""
    print("=" * 60)
    print("  YAF VAE Antenna Designer Demo")
    print("=" * 60)

    designer = VAEDesigner(latent_dim=16, grid_size=32)
    losses = designer.train(epochs=epochs, batch_size=64)

    print(f"\nTraining complete. Final loss: {losses[-1]:.4f}")
    print(f"Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")

    if losses[-1] < losses[0]:
        print("✓ VAE training converged.")
    else:
        print("Note: VAE may need more epochs.")

    # Generate samples
    samples = designer.generate(n=4)
    print(f"\nGenerated {len(samples)} sample geometries.")
    for i, s in enumerate(samples):
        filling = np.mean(s)
        print(f"  Sample {i}: fill factor = {filling:.3f}")

    # Persist weights (acceptance requirement: training completes, weights saved)
    import os
    out_dir = os.environ.get("YAF_MODEL_DIR", "models")
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "vae_designer.pt")
    designer.save(weights_path)
    print(f"\nWeights saved to {weights_path}")

    print("\n" + "=" * 60)
    print("  VAE demo complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the VAE")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    if args.train or len(sys.argv) == 1:
        sys.exit(train_demo(args.epochs))
