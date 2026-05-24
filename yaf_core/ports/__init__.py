"""YAF Core ports — abstract interfaces (Protocols)."""

from yaf_core.ports.solver_port import SolverAdapter
from yaf_core.ports.ai_port import AIBackend
from yaf_core.ports.cad_port import CADBackend
from yaf_core.ports.storage_port import StorageBackend

__all__ = ["SolverAdapter", "AIBackend", "CADBackend", "StorageBackend"]
