# ============================================================
# REFERENCE
#   仿造来源：无（首创实现）
#   理由：AI 天线生成/代理/优化的统一协议为 YAF 独有设计，
#        借鉴了 PyTorch Lightning 的训练抽象和 HuggingFace 的
#        generate 范式，但针对电磁设计领域定制。
#   关键设计点：
#     - task_type 区分生成/代理/优化三大 AI 任务类型
#     - fit/predict/generate 三阶段生命周期
#     - generate 接受自然语言条件 → 返回 Geometry 列表
#     - health_check 自动检测 GPU 可用性
# ============================================================

"""
AIBackend Protocol — pluggable AI models for generation and optimization.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from yaf_core.domain.geometry import Geometry


@runtime_checkable
class AIBackend(Protocol):
    """Protocol for AI model backends.

    Supports three task types:
    - "generate": Design generation (Diffusion, VAE, GAN)
    - "surrogate": Fast proxy models (FNO, DeepONet)
    - "optimize": Optimization engines (Bayesian, Topology)
    """

    name: str
    task_type: str  # "generate" | "surrogate" | "optimize"

    async def fit(self, dataset: Any) -> None:
        """Train/fit the model on a dataset.

        Args:
            dataset: Training data (format depends on model type).
        """
        ...

    async def predict(self, x: Any) -> Any:
        """Run inference on input.

        Args:
            x: Input tensor or data.

        Returns:
            Model output.
        """
        ...

    async def generate(
        self, conditions: dict[str, Any], n: int = 1
    ) -> list[Geometry]:
        """Generate n candidate geometries given conditions.

        Args:
            conditions: Design conditions (freq, gain target, etc.).
            n: Number of candidates to generate.

        Returns:
            List of generated Geometry objects.
        """
        ...

    async def health_check(self) -> bool:
        """Check if model is loaded and GPU is available (if needed).

        Returns:
            True if healthy.
        """
        ...

    async def close(self) -> None:
        """Release model resources (GPU memory, etc.)."""
        ...
