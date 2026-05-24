"""YAF exception hierarchy — all errors inherit from YAFError."""
from __future__ import annotations
class YAFError(Exception):"""Base exception for all YAF errors."""
class SolverError(YAFError):
    def __init__(self,solver:str,job_id:str,message:str)->None:
        super().__init__(f"[{solver}] Job {job_id}: {message}")
        self.solver=solver;self.job_id=job_id
class MeshError(YAFError):"""Mesh generation failure."""
class GeometryError(YAFError):"""Geometry validation/conversion failure."""
class OptimizationError(YAFError):"""Optimization run failure."""
class StorageError(YAFError):"""Object storage I/O failure."""
class ValidationError(YAFError):"""Domain model validation failure."""
